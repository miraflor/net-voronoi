"""Exact lineal Voronoi partition of a network by point clusters.

The question this module answers: for every location on the road network,
which cluster has a point closest to it along the roads?

Definitions. The distance from cluster ``c`` to a network location ``x``,
``D(c, x)``, is the shortest-path distance from ``x`` to the nearest point
tagged ``c``. The winner at ``x`` is the cluster with the smallest
``D(c, x)``. Numerically equal distances go to the lexicographically smallest cluster
ID; the floating-point tolerance is local to the distances being compared and
is described in ``_node_partition``.

``network_voronoi`` is a short pipeline, in reading order:

1. ``_node_partition`` gives, at every graph node, the distance to the
   nearest cluster and the winning cluster;
2. ``_edge_boundary`` gives the position where the winner changes on an edge
   whose two end nodes have different winners. It is the only per-edge
   formula in the package; ``surface.py`` uses the same function;
3. ``_cut_edges`` and ``_collect_cells`` cut the edge lines at those
   positions and collect the pieces by cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse.csgraph import dijkstra
from shapely.ops import substring

from ._numeric import _tol
from .model import SpatialNetwork, _graph_matrix


@dataclass(frozen=True)
class NetworkVoronoiResult:
    segments: gpd.GeoDataFrame
    cells: gpd.GeoDataFrame
    unassigned_edges: gpd.GeoDataFrame
    node_min_distance: np.ndarray
    node_cluster: np.ndarray
    distance_unit: str | None


# ------------------------------------------------------ 1. winners at nodes


def _smallest_rank_at_nodes(point_node: np.ndarray, point_rank: np.ndarray, n_nodes: int) -> np.ndarray:
    """For each node, the smallest cluster rank among the points on it; ``-1`` where there is no point.

    Example: points on nodes ``[4, 4, 7]`` with ranks ``[2, 0, 1]`` give
    rank 0 at node 4, rank 1 at node 7 and -1 elsewhere.
    """
    none = np.iinfo(np.int64).max
    rank = np.full(n_nodes, none, dtype=np.int64)
    np.minimum.at(rank, point_node, point_rank)  # rank[node] = min(rank[node], rank of the point), for every point
    rank[rank == none] = -1
    return rank


def _tie_candidates(network: SpatialNetwork, dist: np.ndarray, label: np.ndarray, candidate_t: float) -> set[int]:
    """Clusters that may be the winner at a node where the first search chose another cluster.

    The first search in ``_node_partition`` labels every node ``v`` with the
    cluster of one nearest point, ``label[v]``. That label is a nearest
    cluster, but when two clusters are equally near, it may not be the one
    with the smaller ID.

    Example: a road 0 --- 5 --- 10 with a point of cluster "B" at 0 and a
    point of cluster "A" at 10. The node at 5 is 5 from both points, and the
    first search may label it "B". The correct winner is "A".

    Which clusters can be such a missed winner? Let ``c`` be the correct
    winner at a node ``v`` with ``c != label[v]``, so ``D_c(v)`` is within the
    local tolerance ``_tol(D_c(v), D(v))`` of ``D(v)``. That local tolerance
    is at most ``candidate_t``, the tolerance of the largest node distance
    (apart from a relative difference of about 1e-12, which the margin
    described at the end covers). Take a shortest path from ``v`` to the
    nearest point of ``c``, at node ``s``.

    * At ``s``, the label is the smallest cluster among the points on ``s``.
      That cluster is ``c``: a smaller cluster on ``s`` reaches ``v`` along
      the same path, so its distance at ``v`` is at most ``D_c(v)``, and it
      would be the correct winner at ``v`` instead of ``c``. (A smaller
      distance also passes the local test: lowering a distance by some amount
      lowers its gap to ``D(v)`` by that amount, and its tolerance by at most
      ``1e-12`` times that amount.)
    * Walking from ``v`` towards ``s`` lowers ``D_c`` by exactly the distance
      walked, while ``D`` can drop by at most that distance. So at every node
      ``p`` of the path, ``D_c(p) - D(p) <= D_c(v) - D(v) <= candidate_t``,
      and every edge ``(a, b)`` of the path, with ``a`` nearer to ``s``, is
      tight within ``candidate_t``::

          D(a) + length <= D_c(a) + length = D_c(b) <= D(b) + candidate_t

    So along the path the label changes from ``c`` (at ``s``) to another
    cluster (at ``v``) across a tight edge. This function returns the labels
    at both ends of every tight edge whose two labels differ, so ``c`` is one
    of them. In the example, the edge from 5 to 10 is tight
    (``D(10) + 5 = 0 + 5 = D(5)``) and its labels are "B" and "A", so "A"
    is a candidate.

    ``candidate_t`` is used only to find candidates; whether a candidate
    really ties at a node is decided later with the local tolerance. A
    candidate that does not win anywhere only costs one extra search, so the
    test for "tight" allows ``8 * candidate_t`` for rounding error.
    """
    u = network.edges["u"].to_numpy(np.int64)
    v = network.edges["v"].to_numpy(np.int64)
    w = network.edges["length"].to_numpy(float)
    reachable = np.isfinite(dist[u])
    u, v, w = u[reachable], v[reachable], w[reachable]
    du, dv = dist[u], dist[v]
    tight = (du + w - dv <= 8.0 * candidate_t) | (
        dv + w - du <= 8.0 * candidate_t
    )  # tight in either direction
    differ = tight & (label[u] != label[v])
    return set(label[u[differ]].tolist()) | set(label[v[differ]].tolist())


def _node_partition(network: SpatialNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-cluster distance ``D(v)`` and winning cluster rank at every node.

    ``D(v)`` is the minimum over clusters ``c`` of ``D_c(v)``, the
    shortest-path distance from ``v`` to the nearest point of ``c``. The
    winner is the smallest cluster rank (``network.cluster_ids`` order) whose
    ``D_c(v)`` is numerically equal to ``D(v)`` under the local tolerance
    ``_tol(D_c(v), D(v))``. Using a local tolerance prevents a very large,
    unrelated part of the network from making distinct distances on a short
    edge count as a tie. Unreachable nodes get ``D = inf`` and winner ``-1``.

    Method:

    1. One multi-source Dijkstra search from all points at once gives ``D``
       and, for each node, one nearest point. That point's cluster is the
       winner, unless a cluster with a smaller ID ties with it.
    2. ``_tie_candidates`` returns a safe superset of the clusters that can
       win such a tie. Candidate discovery uses the largest tolerance anywhere
       in the network, only so that no true local tie is missed. Each candidate
       then gets its own search and is allowed to replace a winner only where
       its distance is within the *local* tolerance at that node.

    A network without ties needs a single search. When ties are everywhere
    (for example, all points on the nodes of a regular grid), every cluster
    becomes a candidate and the cost approaches one search per cluster.

    ``tests/reference.py`` contains the plain version of this function, with
    one full search per cluster; the tests check that both agree.
    """
    clusters = network.cluster_ids
    rank_of = {cluster: i for i, cluster in enumerate(clusters)}
    labels = network.points_snapped["cluster_id"].astype(str)
    point_rank = np.fromiter((rank_of[c] for c in labels), dtype=np.int64, count=len(labels))
    point_node = network.points_snapped["network_node"].to_numpy(np.int64)
    rank_at_node = _smallest_rank_at_nodes(point_node, point_rank, len(network.nodes))
    source = np.flatnonzero(rank_at_node >= 0)

    # Step 1. The adjacency stores every edge in both directions (checked by
    # ``SpatialNetwork.validate``), so a directed search gives undirected
    # distances, and SciPy does not build the transposed matrix on every call.
    # ``nearest[v]`` is the source node from which ``v`` was reached.
    graph = _graph_matrix(network.adjacency)
    dist, _, nearest = dijkstra(
        graph,
        directed=True,
        indices=source,
        min_only=True,
        return_predecessors=True,
    )
    dist = np.asarray(dist, dtype=float)
    reachable = np.isfinite(dist)
    winner = np.full(len(dist), -1, dtype=np.int64)
    winner[reachable] = rank_at_node[nearest[reachable]]

    # Step 2. Candidates are visited in increasing rank, and a candidate only
    # replaces a larger rank, so each node ends with the smallest rank that
    # ties there.
    finite = dist[reachable]
    largest = float(finite.max()) if finite.size else 0.0
    # Candidate discovery needs one conservative tolerance that is valid
    # everywhere. It is deliberately *not* the tie rule itself: final tie
    # decisions below use a local tolerance based on the two distances being
    # compared, so that a very large, distant component cannot turn distinct
    # short-range distances into ties.
    candidate_t = float(_tol(largest))
    for c in sorted(_tie_candidates(network, dist, winner, candidate_t)):
        dist_c = dijkstra(
            graph,
            directed=True,
            indices=np.unique(point_node[point_rank == c]),
            min_only=True,
            limit=largest + 2.0 * candidate_t,  # safely covers every possible local tie
        )
        # First keep the few nodes where c is within 2 * candidate_t of the
        # minimum (every local tolerance is below that bound), then apply the
        # exact local test only to them. Computing the local tolerance for
        # every node instead would cost a pass over the whole network for
        # each candidate.
        near = np.flatnonzero(np.isfinite(dist_c) & (dist_c <= dist + 2.0 * candidate_t))
        local_t = _tol(dist_c[near], dist[near])
        ties = near[dist_c[near] <= dist[near] + local_t]
        winner[ties[c < winner[ties]]] = c
    return dist, winner


# ------------------------------------------------------- 2. edge boundaries


def _edge_boundary(A, B, L) -> np.ndarray:
    """Position of the cluster boundary on edges whose end nodes have different winners.

    An edge ``(u, v)`` of length ``L`` has no point in its interior, because
    every point was inserted as a node. So the nearest cluster at distance
    ``x`` from ``u`` is reached through ``u`` or through ``v``, and::

        D(x) = min(A + x, B + L - x),   where A = D(u), B = D(v).

    The first term belongs to the winner at ``u``, the second to the winner
    at ``v``. They are equal at::

        x* = (B + L - A) / 2,

    so the winner at ``u`` owns ``[0, x*]`` and the winner at ``v`` owns
    ``[x*, L]``. Example: ``A = 2``, ``B = 4``, ``L = 10`` give ``x* = 6``:
    at ``x = 6`` both terms equal 8.

    Because ``|A - B| <= L``, ``x*`` lies in ``[0, L]``; clipping only removes
    rounding error. A boundary within the tolerance of an end is moved onto
    that end, so that no piece shorter than the tolerance is produced.
    """
    A, B, L = (np.asarray(a, dtype=float) for a in (A, B, L))
    x = np.clip((B + L - A) / 2.0, 0.0, L)
    near_end = np.minimum(x, L - x) <= _tol(L, A, B)
    return np.where(near_end, np.where(x <= L - x, 0.0, L), x)


# ------------------------------------------------------ 3. pieces and cells


def _split_lines(lines: np.ndarray, fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut each LineString at ``fraction`` of its length; return the two parts.

    Networks built by ``from_geodataframes`` have only straight two-point
    edges. For those, the cut point is computed for all edges at once and the
    two parts are built directly from it. Custom networks may have edges
    with bends; those are cut one by one with ``shapely.ops.substring``, which
    keeps every bend.
    """
    first = np.empty(len(lines), dtype=object)
    second = np.empty(len(lines), dtype=object)
    straight = (shapely.get_num_coordinates(lines) == 2) & ~shapely.has_z(lines)
    if straight.any():
        ends = shapely.get_coordinates(lines[straight]).reshape(-1, 2, 2)  # [edge, start/end, x/y]
        cut = shapely.get_coordinates(
            shapely.line_interpolate_point(lines[straight], fraction[straight], normalized=True)
        )
        first[straight] = shapely.linestrings(np.stack([ends[:, 0], cut], axis=1))
        second[straight] = shapely.linestrings(np.stack([cut, ends[:, 1]], axis=1))
    for i in np.flatnonzero(~straight):
        first[i] = substring(lines[i], 0.0, fraction[i], normalized=True)
        second[i] = substring(lines[i], fraction[i], 1.0, normalized=True)
    return first, second


def _cut_edges(network: SpatialNetwork, winner: np.ndarray, dist: np.ndarray) -> gpd.GeoDataFrame:
    """One row per positive-length piece of every reachable edge, sorted by edge and part.

    The boundary position ``x`` on an edge of length ``L`` puts the edge in
    one of three cases:

    * ``x == L`` (the same winner at both ends, or the boundary at the v end):
      the whole edge goes to the winner at u;
    * ``x == 0`` (the boundary at the u end): the whole edge goes to the
      winner at v;
    * ``0 < x < L``: ``[0, x]`` goes to the winner at u (part 0) and
      ``[x, L]`` to the winner at v (part 1).

    Edges in a component without points are unreachable and get no piece.
    """
    edges = network.edges
    u = edges["u"].to_numpy(np.int64)
    v = edges["v"].to_numpy(np.int64)
    L = edges["length"].to_numpy(float)
    lines = edges.geometry.to_numpy()
    wu, wv = winner[u], winner[v]
    if np.any((wu < 0) != (wv < 0)):
        raise RuntimeError("an edge joins reachable and unreachable nodes")
    reachable = wu >= 0

    # Boundary position on every edge; x = L unless the two winners differ.
    x = L.copy()
    differ = reachable & (wu != wv)
    x[differ] = _edge_boundary(dist[u[differ]], dist[v[differ]], L[differ])
    whole_to_u = reachable & (x == L)
    whole_to_v = reachable & (x == 0.0)
    split = reachable & (x > 0.0) & (x < L)

    # The pieces come in three groups: whole edges, first parts of split
    # edges, second parts of split edges. Each column below lists the three
    # groups in that order.
    whole = np.flatnonzero(whole_to_u | whole_to_v)
    halves = np.flatnonzero(split)
    first_part, second_part = _split_lines(lines[halves], x[halves] / L[halves])
    n_whole, n_split = len(whole), len(halves)
    piece_edge = np.concatenate([whole, halves, halves])
    piece_part = np.concatenate([np.zeros(n_whole + n_split, dtype=int), np.ones(n_split, dtype=int)])
    piece_rank = np.concatenate([np.where(whole_to_u[whole], wu[whole], wv[whole]), wu[halves], wv[halves]])
    piece_start = np.concatenate([np.zeros(n_whole), np.zeros(n_split), x[halves]])
    piece_stop = np.concatenate([L[whole], x[halves], L[halves]])
    piece_geometry = np.concatenate([lines[whole], first_part, second_part])

    order = np.lexsort((piece_part, piece_edge))  # by edge, then part 0 before part 1
    clusters = network.cluster_ids
    return gpd.GeoDataFrame(
        {
            "cluster_id": clusters[piece_rank[order]].astype(str) if len(order) else np.asarray([], dtype=object),
            "edge_id": edges["edge_id"].to_numpy()[piece_edge[order]],
            "from_dist": piece_start[order],
            "to_dist": piece_stop[order],
            "segment_dist": piece_stop[order] - piece_start[order],
        },
        geometry=gpd.GeoSeries(piece_geometry[order], crs=network.crs),
        crs=network.crs,
    )


def _collect_cells(segments: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    """One row per cluster that owns network length, sorted by cluster ID.

    The geometry is the cluster's single piece, or a MultiLineString of its
    pieces in segment order. The pieces are collected, not merged with a
    union, so that coincident edges of a custom network stay distinct.
    """
    if segments.empty:
        return gpd.GeoDataFrame(columns=["cluster_id", "geometry"], geometry="geometry", crs=crs)
    labels = segments["cluster_id"].to_numpy(dtype=object)
    names, group = np.unique(labels, return_inverse=True)  # sorted IDs; group = position of each piece's ID
    group = np.asarray(group).reshape(-1)
    order = np.argsort(group, kind="stable")  # pieces grouped by cluster, segment order kept within a cluster
    group, pieces = group[order], segments.geometry.to_numpy()[order]
    first_piece = np.r_[True, group[1:] != group[:-1]]

    # ``multilinestrings(pieces, indices=group)`` builds one MultiLineString
    # per cluster from the pieces with that cluster's group number.
    multi = shapely.multilinestrings(pieces, indices=group)
    single = np.bincount(group) == 1
    geometry = np.where(single, pieces[first_piece], multi)
    return gpd.GeoDataFrame(
        {"cluster_id": names.astype(str)},
        geometry=gpd.GeoSeries(geometry, crs=crs),
        crs=crs,
    )


def network_voronoi(network: SpatialNetwork) -> NetworkVoronoiResult:
    """Partition every reachable positive-length network edge by cluster."""
    dist, winner = _node_partition(network)
    segments = _cut_edges(network, winner, dist)
    cells = _collect_cells(segments, network.crs)

    unreachable = np.flatnonzero(winner[network.edges["u"].to_numpy(np.int64)] < 0)
    if len(unreachable):
        unassigned_edges = network.edges.iloc[unreachable].copy().reset_index(drop=True)
    else:
        unassigned_edges = gpd.GeoDataFrame(
            columns=network.edges.columns, geometry="geometry", crs=network.crs
        )

    clusters = network.cluster_ids
    node_cluster = np.full(len(winner), None, dtype=object)
    node_cluster[winner >= 0] = [str(c) for c in clusters[winner[winner >= 0]]]
    return NetworkVoronoiResult(
        segments=segments,
        cells=cells,
        unassigned_edges=unassigned_edges,
        node_min_distance=dist,
        node_cluster=node_cluster,
        distance_unit=network.distance_unit,
    )
