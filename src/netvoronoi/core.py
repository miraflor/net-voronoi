"""Exact lineal network Voronoi cells and additive-epsilon memberships.

Reading order for this module
-----------------------------
1. ``network_voronoi`` (at the end of the file) is the public entry point. It
   is a three-step pipeline: the nearest-site distance at every graph node,
   one shortest-path search per site, and the exact split of every edge.
2. ``_epsilon_interval`` contains all of the per-edge mathematics. The hard
   Voronoi boundary is the same formula with ``epsilon = 0``.
3. The remaining helpers move numbers into GeoDataFrames.

All distances are in the linear unit of the projected network CRS.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse.csgraph import dijkstra
from shapely.ops import substring

from .model import SpatialNetwork


@dataclass
class NetworkVoronoiResult:
    """Exact lineal hard and additive-epsilon Voronoi outputs."""

    hard_segments: gpd.GeoDataFrame
    epsilon_segments: gpd.GeoDataFrame
    hard_cells: gpd.GeoDataFrame
    epsilon_cells: gpd.GeoDataFrame
    unassigned_edges: gpd.GeoDataFrame
    node_min_distance: np.ndarray
    epsilon: float
    distance_unit: str | None


# ---------------------------------------------------------------------------
# Numerical tolerance
# ---------------------------------------------------------------------------
#
# Shortest-path distances are sums of floating-point edge lengths, so two
# distances that are equal in exact arithmetic can differ in their last bits.
# Every equality or inequality test between network distances in this package
# allows the following small slack.

_ABSOLUTE_TOLERANCE = 1e-9  # CRS units; the lower limit for small distances
_RELATIVE_TOLERANCE = 1e-12  # fraction of the largest distance involved


def _tolerance(*values) -> np.ndarray:
    """Slack for comparing network distances of the size of ``values``.

    The result is ``max(1e-9, 1e-12 * max(1, |v| for every finite v))``.
    Arguments can be scalars or NumPy arrays; arrays are handled element by
    element. Infinite values (unreachable nodes) and NaN do not enlarge the
    slack.
    """
    scale = np.asarray(1.0)
    for value in values:
        magnitude = np.abs(np.asarray(value, dtype=float))
        scale = np.maximum(scale, np.where(np.isfinite(magnitude), magnitude, 0.0))
    return np.maximum(_ABSOLUTE_TOLERANCE, _RELATIVE_TOLERANCE * scale)


def _check_epsilon(epsilon: float) -> float:
    """Return ``epsilon`` as a float; reject negative values and NaN.

    ``math.inf`` is accepted: every site is then a member wherever it can reach
    at all. NaN needs an explicit test because every comparison with NaN is
    false: it would pass an ``epsilon < 0`` check and then silently produce an
    empty epsilon layer.
    """
    epsilon = float(epsilon)
    if not epsilon >= 0.0:  # false for negative numbers and for NaN
        raise ValueError(f"epsilon must be a non-negative number, got {epsilon!r}")
    return epsilon


def _check_batch_size(batch_size: int) -> int:
    """Return a positive integer ``batch_size`` with a clear API error.

    The CLI already parses this option as an integer. This check is for direct
    Python callers, where values such as ``1.5`` or ``nan`` would otherwise
    fail later inside ``range`` with a less helpful exception.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    return batch_size


# ---------------------------------------------------------------------------
# The mathematics on one edge
# ---------------------------------------------------------------------------


def _epsilon_interval(a, b, A, B, length, epsilon: float):
    """Exact epsilon-admissible interval of one site on site-free edges.

    Setting
    -------
    Take an edge from node ``u`` to node ``v`` with length ``L`` and no site
    in its interior. Describe a point on the edge by ``x``, its network
    distance from ``u`` (so ``0 <= x <= L``). A path from any site to that
    point enters the edge through ``u`` or through ``v``, so for one site ``s``

        d_s(x)   = min(a + x, b + L - x),  with a = d_s(u) and b = d_s(v),

    and for the nearest site

        d_min(x) = min(A + x, B + L - x),  with A = min_h d_h(u), B = min_h d_h(v).

    The site is epsilon-admissible at ``x`` when ``d_s(x) <= d_min(x) + epsilon``.

    Key fact: the excess ``e(x) = d_s(x) - d_min(x)`` is monotone on the edge
    -------------------------------------------------------------------------
    Each distance first rises with slope +1 (the route through ``u`` is
    shorter) and then falls with slope -1 (the route through ``v`` is
    shorter). Where the two slopes are equal, ``e`` is constant: it equals
    ``a - A`` near ``u`` and ``b - B`` near ``v``. Between the two slope
    changes the slopes differ, so ``e`` changes there at the constant rate +2
    or -2. Therefore ``e`` moves in one direction only, from ``a - A`` to
    ``b - B``, and the admissible set is a single interval that contains an
    endpoint (or it is empty).

    Four cases, decided at the two endpoints
    ----------------------------------------
    * admissible at ``u`` and at ``v``: the whole edge ``[0, L]``;
    * admissible at ``u`` only: ``[0, t]``. At ``t`` the site's route through
      ``u`` meets the nearest-site route through ``v`` plus epsilon:
      ``a + t = B + L - t + epsilon``, so ``t = (B + L + epsilon - a) / 2``;
    * admissible at ``v`` only: ``[t, L]`` with ``b + L - t = A + t + epsilon``,
      so ``t = (b + L - A - epsilon) / 2``;
    * admissible at neither endpoint: nowhere on the edge.

    Example: ``a = 0, b = 600, A = B = 0, L = 600, epsilon = 100`` is
    admissible at ``u`` only, so the interval is ``[0, (0 + 600 + 100 - 0) / 2]
    = [0, 350]``.

    Arguments are NumPy arrays with one element per edge (scalars also work).
    Returns ``(start, stop, keep)``: the interval bounds, clipped to
    ``[0, L]``, and a mask of the edges where the interval has positive
    length. An admissible set that is a single point is dropped on purpose,
    because the lineal output contains positive-length pieces only.
    """
    a, b, A, B, length = (np.asarray(value, dtype=float) for value in (a, b, A, B, length))
    tol = _tolerance(a, b, A, B, length, epsilon)
    admissible_at_u = np.isfinite(a) & (a <= A + epsilon + tol)
    admissible_at_v = np.isfinite(b) & (b <= B + epsilon + tol)

    # Boundary positions for the two one-endpoint cases. Each one is used only
    # where its case applies; in other elements it can be inf or NaN (for
    # example, inf - inf for an unreachable site), so those warnings are off.
    with np.errstate(invalid="ignore"):
        stop_if_only_u = np.clip((B + length + epsilon - a) / 2.0, 0.0, length)
        start_if_only_v = np.clip((b + length - A - epsilon) / 2.0, 0.0, length)

    start = np.where(admissible_at_u, 0.0, start_if_only_v)
    stop = np.where(admissible_at_v, length, stop_if_only_u)
    with np.errstate(invalid="ignore"):
        keep = (admissible_at_u | admissible_at_v) & (stop - start > tol)
    return start, stop, keep


# ---------------------------------------------------------------------------
# Array bundles used by the pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EdgeArrays:
    """Columns of ``network.edges`` as NumPy arrays (edge ``i`` is row ``i``).

    The edge table is read once. Indexing NumPy arrays is much faster than
    row-by-row pandas access, and the loops below touch edges many times.
    """

    edge_id: np.ndarray
    u: np.ndarray
    v: np.ndarray
    length: np.ndarray
    geometry: np.ndarray

    @classmethod
    def from_network(cls, network: SpatialNetwork) -> _EdgeArrays:
        edges = network.edges
        return cls(
            edge_id=edges["edge_id"].to_numpy(),
            u=edges["u"].to_numpy(dtype=int),
            v=edges["v"].to_numpy(dtype=int),
            length=edges["length"].to_numpy(dtype=float),
            geometry=edges.geometry.to_numpy(),
        )


@dataclass(frozen=True)
class _Pieces:
    """Positive-length pieces of edges, as parallel arrays.

    Element ``k`` describes one piece: ``edge[k]`` is a row position in the
    edge table, ``start[k]`` and ``stop[k]`` are network distances from that
    edge's endpoint ``u``, and ``site[k]`` is a position in the site list.
    """

    edge: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    site: np.ndarray

    @classmethod
    def concatenate(cls, parts: list[_Pieces]) -> _Pieces:
        if not parts:
            empty_int, empty_float = np.empty(0, dtype=int), np.empty(0, dtype=float)
            return cls(empty_int, empty_float, empty_float, empty_int)
        return cls(
            edge=np.concatenate([part.edge for part in parts]),
            start=np.concatenate([part.start for part in parts]),
            stop=np.concatenate([part.stop for part in parts]),
            site=np.concatenate([part.site for part in parts]),
        )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _minimum_node_distances(network: SpatialNetwork) -> np.ndarray:
    """Distance from every graph node to its nearest site (one search)."""
    site_nodes = np.unique(np.asarray(network.site_nodes, dtype=int))
    values = dijkstra(
        network.adjacency,
        directed=False,
        indices=site_nodes,
        min_only=True,
        return_predecessors=False,
    )
    return np.asarray(values, dtype=float)


def _scan_sites(
    network: SpatialNetwork,
    edges: _EdgeArrays,
    site_ids: np.ndarray,
    min_dist: np.ndarray,
    epsilon: float,
    batch_size: int,
) -> tuple[np.ndarray, _Pieces]:
    """Run one shortest-path search per site and collect two results.

    1. The hard winner at every node. Sites are visited in increasing string
       ``site_id`` order, and a node keeps the first site whose distance
       equals the node's nearest-site distance. So an exact tie at a node goes
       to the smallest ``site_id``, independent of the input order and of
       SciPy's internal tie handling.
    2. The epsilon pieces. For each site, ``_epsilon_interval`` is evaluated
       on all edges at once.

    Returns ``(winner, pieces)``. ``winner[node]`` is a position in
    ``site_ids``, or -1 for a node that no site can reach. The pieces are
    ordered by site ID and then by edge.
    """
    site_nodes = np.asarray(network.site_nodes, dtype=int)
    order = np.asarray(sorted(range(len(site_ids)), key=lambda i: site_ids[i]), dtype=int)
    winner = np.full(len(min_dist), -1, dtype=int)
    min_at_u, min_at_v = min_dist[edges.u], min_dist[edges.v]
    parts: list[_Pieces] = []

    for first in range(0, len(order), batch_size):
        batch = order[first : first + batch_size]
        distances = dijkstra(
            network.adjacency,
            directed=False,
            indices=site_nodes[batch],
            return_predecessors=False,
        )
        distances = np.atleast_2d(distances)  # one row per site in the batch

        for site_position, site_dist in zip(batch.tolist(), distances):
            # 1. Claim the still-unclaimed nodes where this site is nearest.
            #    (A finite site distance implies a finite nearest distance.)
            reached = np.isfinite(site_dist)
            at_minimum = np.zeros(len(site_dist), dtype=bool)
            at_minimum[reached] = np.abs(site_dist[reached] - min_dist[reached]) <= _tolerance(
                site_dist[reached], min_dist[reached]
            )
            winner[at_minimum & (winner < 0)] = site_position

            # 2. This site's epsilon interval on every edge.
            start, stop, keep = _epsilon_interval(
                site_dist[edges.u], site_dist[edges.v], min_at_u, min_at_v, edges.length, epsilon
            )
            kept = np.flatnonzero(keep)
            parts.append(
                _Pieces(kept, start[kept], stop[kept], np.full(len(kept), site_position, dtype=int))
            )

    if np.any(np.isfinite(min_dist) & (winner < 0)):
        raise RuntimeError("failed to resolve a hard Voronoi winner at a reachable node")
    return winner, _Pieces.concatenate(parts)


def _hard_pieces(
    edges: _EdgeArrays,
    min_dist: np.ndarray,
    winner: np.ndarray,
) -> tuple[_Pieces, np.ndarray]:
    """Split every reachable edge at its exact hard Voronoi boundary.

    If both endpoints of an edge have the same winner, that site is nearest on
    the whole edge. Otherwise the winner at ``u`` is nearest up to
    ``(B + L - A) / 2`` and the winner at ``v`` after that point: this is the
    boundary of ``_epsilon_interval`` with ``epsilon = 0``.

    The node tie rule carries over to the edge interior. Strictly between
    ``u`` and the boundary, the nearest sites are exactly the sites that are
    nearest at ``u`` (they arrive through ``u``), and the smallest ID among
    them is ``winner[u]``. The same holds on the ``v`` side.

    Returns the pieces (ordered by edge, ``u`` side first) and the row
    positions of edges that no site can reach.
    """
    winner_u, winner_v = winner[edges.u], winner[edges.v]
    reachable = winner_u >= 0
    if np.any(reachable != (winner_v >= 0)):
        # An edge joins its two endpoints, so they are in the same component.
        raise RuntimeError("an edge joins a reachable and an unreachable node")

    min_at_u, min_at_v, length = min_dist[edges.u], min_dist[edges.v], edges.length
    with np.errstate(invalid="ignore"):  # inf - inf on unreachable edges
        boundary = np.clip((min_at_v + length - min_at_u) / 2.0, 0.0, length)
    same_winner = winner_u == winner_v
    all_edges = np.arange(len(length))

    u_side = _Pieces(
        edge=all_edges[reachable],
        start=np.zeros(int(reachable.sum())),
        stop=np.where(same_winner, length, boundary)[reachable],
        site=winner_u[reachable],
    )
    split = reachable & ~same_winner
    v_side = _Pieces(
        edge=all_edges[split],
        start=boundary[split],
        stop=length[split],
        site=winner_v[split],
    )

    # Put each edge's u-side piece before its v-side piece; drop pieces of
    # zero length (a boundary that falls on an endpoint).
    both = _Pieces.concatenate([u_side, v_side])
    order = np.argsort(both.edge, kind="stable")
    both = _Pieces(both.edge[order], both.start[order], both.stop[order], both.site[order])
    positive = both.stop - both.start > _tolerance(length[both.edge])
    pieces = _Pieces(
        both.edge[positive], both.start[positive], both.stop[positive], both.site[positive]
    )
    return pieces, np.flatnonzero(~reachable)


# ---------------------------------------------------------------------------
# From pieces to geometry tables
# ---------------------------------------------------------------------------


def _cut_lines(geometry, start_fraction, stop_fraction) -> np.ndarray:
    """Cut the part between two positions out of each line.

    ``start_fraction`` and ``stop_fraction`` give positions along each line
    as fractions of its length, measured from its first vertex (endpoint
    ``u``).

    Straight two-vertex lines -- every edge built by the spaghetti adapter --
    are cut in one vectorized step. The cut points are ``(1 - f) * p + f * q``
    for the line's vertices ``p`` and ``q``; this form gives exactly ``p`` at
    ``f = 0`` and exactly ``q`` at ``f = 1``, so a piece that reaches an edge
    endpoint uses that vertex exactly. Lines with more vertices are cut one
    at a time with ``shapely.ops.substring``.
    """
    geometry = np.asarray(geometry, dtype=object)
    start_fraction = np.asarray(start_fraction, dtype=float)
    stop_fraction = np.asarray(stop_fraction, dtype=float)
    pieces = np.empty(len(geometry), dtype=object)

    straight = shapely.get_num_coordinates(geometry) == 2
    if straight.any():
        p = shapely.get_coordinates(shapely.get_point(geometry[straight], 0))
        q = shapely.get_coordinates(shapely.get_point(geometry[straight], -1))
        f0 = start_fraction[straight][:, None]
        f1 = stop_fraction[straight][:, None]
        first_points = (1.0 - f0) * p + f0 * q
        last_points = (1.0 - f1) * p + f1 * q
        pieces[straight] = shapely.linestrings(np.stack([first_points, last_points], axis=1))

    for i in np.flatnonzero(~straight):
        pieces[i] = substring(geometry[i], start_fraction[i], stop_fraction[i], normalized=True)
    return pieces


def _segments_frame(
    edges: _EdgeArrays,
    site_ids: np.ndarray,
    pieces: _Pieces,
    crs,
) -> gpd.GeoDataFrame:
    """One row per piece: site, edge, network-distance bounds, and geometry."""
    length = edges.length[pieces.edge]
    geometry = _cut_lines(
        edges.geometry[pieces.edge], pieces.start / length, pieces.stop / length
    )
    return gpd.GeoDataFrame(
        {
            "site_id": site_ids[pieces.site],
            "edge_id": edges.edge_id[pieces.edge],
            "from_dist": pieces.start,
            "to_dist": pieces.stop,
            "segment_dist": pieces.stop - pieces.start,
        },
        geometry=gpd.GeoSeries(geometry, crs=crs),
        crs=crs,
    )


def _collect_cells(segments: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    """Collect each site's line pieces without topological union.

    A unary union can collapse distinct graph edges that share the same map
    coordinates (for example stacked links). ``MultiLineString`` keeps those
    graph elements distinct, so the segment table remains faithfully additive.
    """
    if segments.empty:
        return gpd.GeoDataFrame(columns=["site_id", "geometry"], geometry="geometry", crs=crs)

    # ``site_code[k]`` numbers the site of segment ``k``: 0 for the first site
    # that appears in the table, 1 for the next new one, and so on.
    site_code, site_id = pd.factorize(segments["site_id"])
    # ``shapely.multilinestrings`` builds one MultiLineString per code and
    # needs the codes in increasing order. A stable sort keeps each site's
    # segments in their table order.
    order = np.argsort(site_code, kind="stable")
    cells = shapely.multilinestrings(
        segments.geometry.to_numpy()[order], indices=site_code[order]
    )
    return gpd.GeoDataFrame(
        {"site_id": site_id}, geometry=gpd.GeoSeries(cells, crs=crs), crs=crs
    )


def _unassigned_edges(network: SpatialNetwork, positions: np.ndarray) -> gpd.GeoDataFrame:
    if len(positions):
        return network.edges.iloc[positions].copy().reset_index(drop=True)
    return gpd.GeoDataFrame(columns=network.edges.columns, geometry="geometry", crs=network.crs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def network_voronoi(
    network: SpatialNetwork,
    epsilon: float = 0.0,
    batch_size: int = 32,
) -> NetworkVoronoiResult:
    """Construct hard and additive-epsilon Voronoi cells on a spatial network.

    ``hard_segments`` assigns every reachable positive-length network portion
    to one nearest site. Exact ties use the lexicographically smallest string
    ``site_id``. ``epsilon_segments`` contains every site satisfying

    ``d_N(site, x) <= min_h d_N(h, x) + epsilon``.

    Therefore positive ``epsilon`` can produce overlapping cells. Isolated
    zero-length membership points are not emitted as line features.

    ``epsilon`` must be ``>= 0``; ``math.inf`` is allowed and makes every site
    a member wherever it can reach. ``batch_size`` is the number of sites
    whose shortest-path distances are held in memory at the same time
    (``batch_size * number_of_nodes`` floats); it does not change the result.
    """
    # The two numbers are checked first because that is instant, while
    # ``validate`` reads the whole network.
    epsilon = _check_epsilon(epsilon)
    batch_size = _check_batch_size(batch_size)
    network.validate()

    edges = _EdgeArrays.from_network(network)
    site_ids = np.asarray([str(value) for value in network.site_ids.tolist()], dtype=object)

    # 1. The nearest-site distance at every graph node.
    min_dist = _minimum_node_distances(network)

    # 2. One search per site: hard winners at nodes and epsilon pieces on edges.
    winner, epsilon_pieces = _scan_sites(network, edges, site_ids, min_dist, epsilon, batch_size)

    # 3. The endpoint winners and minima give the exact hard split of each edge.
    hard_pieces, unassigned_positions = _hard_pieces(edges, min_dist, winner)

    hard_segments = _segments_frame(edges, site_ids, hard_pieces, network.crs)
    epsilon_segments = _segments_frame(edges, site_ids, epsilon_pieces, network.crs)
    return NetworkVoronoiResult(
        hard_segments=hard_segments,
        epsilon_segments=epsilon_segments,
        hard_cells=_collect_cells(hard_segments, network.crs),
        epsilon_cells=_collect_cells(epsilon_segments, network.crs),
        unassigned_edges=_unassigned_edges(network, unassigned_positions),
        node_min_distance=min_dist,
        epsilon=epsilon,
        distance_unit=network.distance_unit,
    )
