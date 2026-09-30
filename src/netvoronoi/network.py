"""Build a ``SpatialNetwork`` from road lines and clustered points.

What this module does, in reading order:

1. ``from_geodataframes`` checks the inputs and runs the three steps below;
2. ``_road_segments`` cuts the roads into straight segments between
   consecutive vertices, removes repeated segments, and numbers the segment
   ends so that equal coordinates become one graph node;
3. ``_snap_points`` finds, for every point, the nearest segment and the
   position on it;
4. ``_insert_points`` splits segments at those positions, so that every point
   sits on a graph node.

Topology rule: two roads are connected only where they share an exactly
equal vertex coordinate. A geometric crossing without a shared vertex is not
a junction, and inserting a point never connects two roads.

Length rule: inserting points moves points, never road. Every input segment
of positive length is kept as one or more edges, however short it is.

The steps work on whole arrays (NumPy and Shapely vectorized functions)
instead of one Python loop iteration per segment, so that networks with
millions of segments build in seconds. Each array step is annotated with what
the arrays contain.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix

from ._numeric import _count_distinct, _first_appearance_ids, _tol
from .model import SpatialNetwork

_LINESTRING = 1  # Shapely geometry type ids
_MULTILINESTRING = 5


def _require_projected(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Raise ``ValueError`` unless ``gdf`` has a projected CRS (distances in linear units)."""
    if gdf.crs is None:
        raise ValueError(f"{name} CRS is missing")
    if not gdf.crs.is_projected:
        raise ValueError(f"{name} CRS must be projected")


# ------------------------------------------------------ step 2: road segments


@dataclass(frozen=True)
class _Segments:
    """Unique straight road segments, in input order. Row ``i`` of every array describes segment ``i``."""

    start: np.ndarray  # (m, 2) coordinates of the first vertex
    end: np.ndarray  # (m, 2) coordinates of the second vertex
    source_row: np.ndarray  # (m,) row of the roads layer that the segment came from
    lines: np.ndarray  # (m,) the segment as a two-point LineString
    length: np.ndarray  # (m,) geometric length
    u: np.ndarray  # (m,) node id of ``start``
    v: np.ndarray  # (m,) node id of ``end``
    node_xy: np.ndarray  # (k, 2) coordinates of the vertex nodes; row = node id


def _road_segments(roads: gpd.GeoDataFrame) -> _Segments:
    """Explode roads into unique straight segments between consecutive vertices.

    A vertex repeated in a row (a zero-length segment) is dropped. A segment
    with the same two end coordinates as an earlier segment, in either
    direction, is dropped as a repeat. Every other segment is kept, however
    short. Segment ends with exactly equal coordinates get the same node id;
    node ids are numbered in order of first appearance.
    """
    if roads.empty:
        raise ValueError("roads layer is empty")
    geoms = roads.geometry.to_numpy()
    present = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))
    kind = shapely.get_type_id(geoms[present])
    if np.any((kind != _LINESTRING) & (kind != _MULTILINESTRING)):
        raise ValueError("roads must contain only LineString or MultiLineString geometries")

    # Split MultiLineStrings into their LineStrings. ``part_row`` is the row of
    # the roads layer that each LineString came from.
    parts, part_index = shapely.get_parts(geoms[present], return_index=True)
    part_row = np.flatnonzero(present)[part_index]

    # All vertices of all LineStrings in one (n, 2) array. ``vertex_part`` says
    # which LineString each vertex belongs to: two lines with 3 and 2 vertices
    # give vertex_part = [0, 0, 0, 1, 1].
    xy, vertex_part = shapely.get_coordinates(parts, return_index=True)
    if not np.isfinite(xy).all():
        raise ValueError("road coordinates must be finite")

    # A segment joins a vertex to the next vertex of the same LineString. With
    # vertex_part = [0, 0, 0, 1, 1], the vertex pairs (0,1), (1,2) and (3,4)
    # are segments; the pair (2,3) is not, because it joins two lines.
    same_line = vertex_part[1:] == vertex_part[:-1]
    start, end = xy[:-1][same_line], xy[1:][same_line]
    source_row = part_row[vertex_part[:-1][same_line]]

    # Drop zero-length segments (a vertex repeated in a row).
    keep = np.any(start != end, axis=1)
    start, end, source_row = start[keep], end[keep], source_row[keep]

    # Drop repeated segments. The same segment may be drawn in either
    # direction, so each segment is described by its two ends in a fixed
    # order (smaller x first; for equal x, smaller y first) before comparing.
    swap = (start[:, 0] > end[:, 0]) | ((start[:, 0] == end[:, 0]) & (start[:, 1] > end[:, 1]))
    low = np.where(swap[:, None], end, start)
    high = np.where(swap[:, None], start, end)
    _, first = _first_appearance_ids(np.c_[low, high])  # first copy of each segment, in input order
    start, end, source_row = start[first], end[first], source_row[first]
    if len(start) == 0:
        raise ValueError("roads contain no positive-length line segments")

    # Number the segment ends. ``ends`` lists start0, end0, start1, end1, ...,
    # so ``end_node[0::2]`` are the start nodes and ``end_node[1::2]`` the end
    # nodes. Equal coordinates get the same node id.
    ends = np.empty((2 * len(start), 2))
    ends[0::2], ends[1::2] = start, end
    end_node, first_end = _first_appearance_ids(ends)

    lines = shapely.linestrings(np.stack([start, end], axis=1))
    return _Segments(
        start=start,
        end=end,
        source_row=source_row,
        lines=lines,
        length=shapely.length(lines),
        u=end_node[0::2],
        v=end_node[1::2],
        node_xy=ends[first_end],
    )


# -------------------------------------------------------- step 3: snap points


@dataclass(frozen=True)
class _Snap:
    """Where each point attaches to the road segments. Row ``i`` describes point ``i``."""

    segment: np.ndarray  # chosen nearest segment
    position: np.ndarray  # distance along that segment from its start vertex; exactly 0 or the length at an end
    distance: np.ndarray  # straight-line distance from the point to the segment
    locations: np.ndarray  # number of distinct graph locations at that distance (1 = unambiguous)


def _positions(seg: _Segments, segment: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Distance along ``segment`` to the nearest location of each point.

    A position within the tolerance of an end vertex is set to exactly that
    end (0 or the segment length), so that the point uses the vertex node
    instead of a new node a tiny distance away from it.
    """
    position = shapely.line_locate_point(seg.lines[segment], points)
    length = seg.length[segment]
    t = _tol(length)
    return np.where(position <= t, 0.0, np.where(length - position <= t, length, position))


def _snap_points(seg: _Segments, points: np.ndarray) -> _Snap:
    """Nearest segment and position for every point.

    When several segments are exactly equally near, the one that comes first
    in input order is used. ``locations`` counts how many different graph
    locations those equally near segments offer:

    * segments that meet at the nearest vertex offer one location, the node
      of that vertex, so a point next to an ordinary bend or junction has
      ``locations == 1``;
    * positions inside different segments are different locations, so a point
      at a crossing without a shared vertex has ``locations == 2``.
    """
    # ``query_nearest`` returns all equally near segments as (point, segment)
    # index pairs with their distances. A point with two nearest segments
    # appears in two pairs.
    (point_idx, seg_idx), distance = shapely.STRtree(seg.lines).query_nearest(
        points, all_matches=True, return_distance=True
    )
    # Sort the pairs by point and then by segment, so that the first pair of
    # each point holds the nearest segment that comes first in input order.
    order = np.lexsort((seg_idx, point_idx))
    point_idx, seg_idx, distance = point_idx[order], seg_idx[order], distance[order]
    first = np.r_[True, point_idx[1:] != point_idx[:-1]]
    if first.sum() != len(points):
        raise RuntimeError("failed to find a nearest network segment for a point")

    position = _positions(seg, seg_idx, points[point_idx])

    # Describe each nearest location by one integer: the node id when the
    # position is at a segment end, otherwise ``-1 - segment index``, a
    # negative number that no other segment and no node can have.
    at_start = position == 0.0
    at_end = position == seg.length[seg_idx]
    location = np.where(at_start, seg.u[seg_idx], np.where(at_end, seg.v[seg_idx], -1 - seg_idx))
    return _Snap(
        segment=seg_idx[first],
        position=position[first],
        distance=distance[first],
        locations=_count_distinct(point_idx, location, len(points)),
    )


# ------------------------------------------------------ step 4: insert points


def _insert_points(seg: _Segments, snap: _Snap):
    """Split segments at the snapped positions and return the graph.

    Returns ``(node_xy, point_node, edges)``: the coordinates of all nodes
    (vertex nodes first, then inserted nodes), the node of every point, and
    the edge table as a dict of columns.

    On each segment, points are handled in order of position:

    * a point at position 0 or at the segment length uses that end vertex's
      node (``_positions`` has already moved positions within the tolerance
      onto the ends);
    * a point within the tolerance of the previous inserted node on the same
      segment shares that node;
    * any other point gets a new node.

    Consecutive nodes on a segment are therefore more than the tolerance
    apart, and no segment is ever dropped: a segment keeps at least one edge.
    """
    first_new_node = len(seg.node_xy)  # inserted nodes are numbered after the vertex nodes
    point_node = np.empty(len(snap.segment), dtype=np.int64)
    new_segment: list[int] = []  # the segment of each inserted node
    new_position: list[float] = []  # the position of each inserted node on its segment

    # Visit the points grouped by segment and, within a segment, by position.
    # Only segments that received points are visited.
    order = np.lexsort((snap.position, snap.segment))
    group_starts = np.flatnonzero(np.diff(snap.segment[order])) + 1
    for members in np.split(order, group_starts):
        s = int(snap.segment[members[0]])
        length = float(seg.length[s])
        t = float(_tol(length))
        last_position, last_node = None, None
        for p in members:
            m = float(snap.position[p])
            if m == 0.0:
                node = seg.u[s]
            elif m == length:
                node = seg.v[s]
            elif last_position is not None and m - last_position <= t:
                node = last_node
            else:
                node = first_new_node + len(new_segment)
                new_segment.append(s)
                new_position.append(m)
                last_position, last_node = m, node
            point_node[p] = node

    new_segment = np.asarray(new_segment, dtype=np.int64)
    new_position = np.asarray(new_position, dtype=float)
    if len(new_segment):
        new_xy = shapely.get_coordinates(
            shapely.line_interpolate_point(seg.lines[new_segment], new_position)
        )
    else:
        new_xy = np.empty((0, 2))
    new_node = first_new_node + np.arange(len(new_segment), dtype=np.int64)

    # Every segment becomes a chain of breakpoints ordered by position: its
    # start vertex, its inserted nodes, its end vertex. Each consecutive pair
    # of breakpoints is one edge. Example: a segment of length 10 with nodes
    # inserted at positions 3 and 7 has the chain
    #     (0, u) -> (3, n1) -> (7, n2) -> (10, v)
    # and the three edges (u, n1), (n1, n2), (n2, v). A segment without
    # inserted nodes has the chain (0, u) -> (10, v) and one edge.
    n_seg = len(seg.lines)
    every_segment = np.arange(n_seg)
    chain_segment = np.concatenate([every_segment, new_segment, every_segment])
    chain_position = np.concatenate([np.zeros(n_seg), new_position, seg.length])
    chain_node = np.concatenate([seg.u, new_node, seg.v])
    chain_xy = np.concatenate([seg.start, new_xy, seg.end])
    order = np.lexsort((chain_position, chain_segment))  # by segment, then by position
    chain_segment, chain_node, chain_xy = chain_segment[order], chain_node[order], chain_xy[order]

    # Consecutive breakpoints of the same segment form an edge.
    pair = chain_segment[1:] == chain_segment[:-1]
    edge_segment = chain_segment[:-1][pair]
    edge_u = chain_node[:-1][pair]
    edge_v = chain_node[1:][pair]
    from_xy = chain_xy[:-1][pair]
    to_xy = chain_xy[1:][pair]
    # The part number counts the edges of each segment: 0, 1, 2, ... The
    # edges are sorted by segment, so it is the edge's position minus the
    # position of the first edge of the same segment (found by searchsorted).
    part = np.arange(len(edge_segment)) - np.searchsorted(edge_segment, edge_segment, side="left")

    # An unsplit segment keeps its original LineString; a split segment gets
    # a new two-point LineString for each part.
    parts_of_segment = np.bincount(edge_segment, minlength=n_seg)
    split = parts_of_segment[edge_segment] > 1
    geometry = seg.lines[edge_segment]
    geometry[split] = shapely.linestrings(np.stack([from_xy[split], to_xy[split]], axis=1))

    edges = {
        "edge_id": [f"e{s:07d}_{k:03d}" for s, k in zip(edge_segment.tolist(), part.tolist())],
        "source_edge": edge_segment,
        "source_row": seg.source_row[edge_segment],
        "u": edge_u,
        "v": edge_v,
        "length": shapely.length(geometry),
        "geometry": geometry,
    }
    return np.concatenate([seg.node_xy, new_xy]), point_node, edges


# -------------------------------------------------------- step 1: public API


def from_geodataframes(
    roads: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    *,
    cluster_col: str,
    point_id_col: str | None = None,
    max_snap_distance: float | None = None,
) -> SpatialNetwork:
    """Build a network and snap/insert clustered points.

    Road topology is taken literally from road vertices. Geometric crossings do
    not become junctions unless the input roads share a vertex there.
    """
    # --- check the inputs
    _require_projected(roads, "roads")
    if points.crs is None:
        raise ValueError("points CRS is missing")
    if cluster_col not in points.columns:
        raise ValueError(f"cluster column not found: {cluster_col!r}")
    if point_id_col is not None and point_id_col not in points.columns:
        raise ValueError(f"point ID column not found: {point_id_col!r}")
    if points.empty:
        raise ValueError("points layer is empty")
    if max_snap_distance is not None:
        max_snap_distance = float(max_snap_distance)
        if not np.isfinite(max_snap_distance) or max_snap_distance < 0:
            raise ValueError("max_snap_distance must be finite and non-negative")

    pts = points.to_crs(roads.crs)
    bad = pts.geometry.isna() | pts.geometry.is_empty | (pts.geometry.geom_type != "Point")
    if bool(bad.any()):
        raise ValueError("points must contain only non-empty Point geometries")
    point_geoms = pts.geometry.to_numpy()
    point_xy = np.c_[shapely.get_x(point_geoms), shapely.get_y(point_geoms)]
    if not np.isfinite(point_xy).all():
        raise ValueError("point coordinates must be finite after reprojection")
    if pts[cluster_col].isna().any():
        raise ValueError("cluster labels may not be missing")

    # Cluster IDs are compared as strings, so two different labels must not
    # become the same string (for example the number 1 and the text "1").
    raw_clusters = pts[cluster_col].unique().tolist()
    if len({str(value) for value in raw_clusters}) != len(raw_clusters):
        raise ValueError("distinct cluster labels collapse to the same string representation")
    cluster_id = pts[cluster_col].astype(str).to_numpy(dtype=object)
    if any(len(c) == 0 for c in cluster_id):
        raise ValueError("cluster labels may not be empty")
    if point_id_col is None:
        point_id = np.asarray([str(i) for i in range(len(pts))], dtype=object)
    else:
        if pts[point_id_col].isna().any():
            raise ValueError("point IDs may not be missing")
        point_id = pts[point_id_col].astype(str).to_numpy(dtype=object)
    if len(set(point_id.tolist())) != len(point_id):
        raise ValueError("point IDs must be unique after conversion to string")

    # --- steps 2 to 4
    seg = _road_segments(roads)
    snap = _snap_points(seg, point_geoms)

    # ``max_snap_distance`` limits the distance from each point to the road
    # network, measured to the nearest road location (``snap.distance``).
    # The node the point finally sits on can be slightly farther: when the
    # nearest location is within the segment's tolerance (``_tol`` of the
    # segment length) of a vertex or of another inserted point, the point is
    # placed on that vertex or node. So the returned ``snap_distance`` is at
    # most ``snap.distance + _tol(segment length)``. Example: a point exactly
    # on a 60 km road, 4e-8 m from a vertex, is 0 m from the road and is
    # placed on the vertex (the tolerance of a 60 km segment is 6e-8 m); it
    # passes ``max_snap_distance=0`` and gets ``snap_distance = 4e-8``.
    if max_snap_distance is not None:
        too_far = np.flatnonzero(snap.distance > max_snap_distance + _tol(max_snap_distance))
        if len(too_far):
            examples = ", ".join(point_id[too_far[:5]].tolist())
            raise ValueError(
                f"{len(too_far)} point(s) exceed max_snap_distance={max_snap_distance}; "
                f"examples: {examples}"
            )
    node_xy, point_node, edge_columns = _insert_points(seg, snap)

    # --- assemble the tables
    crs = roads.crs
    node_geoms = shapely.points(node_xy)
    nodes = gpd.GeoDataFrame(
        {"node_id": np.arange(len(node_xy), dtype=np.int64)},
        geometry=gpd.GeoSeries(node_geoms, crs=crs),
        crs=crs,
    )
    edge_geometry = edge_columns.pop("geometry")
    edges = gpd.GeoDataFrame(edge_columns, geometry=gpd.GeoSeries(edge_geometry, crs=crs), crs=crs)

    # The adjacency matrix stores every edge in both directions (u to v and
    # v to u) with its length as the weight.
    u = edges["u"].to_numpy(np.int64)
    v = edges["v"].to_numpy(np.int64)
    length = edges["length"].to_numpy(float)
    adjacency = coo_matrix(
        (np.r_[length, length], (np.r_[u, v], np.r_[v, u])), shape=(len(nodes), len(nodes))
    ).tocsr()

    # The snapped geometry of a point is its node, exactly. When several
    # points within the tolerance share one node, each of them is placed on
    # that node, and ``snap_distance`` is the distance to that node.
    snapped_xy = node_xy[point_node]
    points_input = gpd.GeoDataFrame(
        {"point_id": point_id, "cluster_id": cluster_id},
        geometry=gpd.GeoSeries(point_geoms, crs=crs),
        crs=crs,
    )
    points_snapped = gpd.GeoDataFrame(
        {
            "point_id": point_id,
            "cluster_id": cluster_id,
            "network_node": point_node,
            "snap_distance": np.hypot(point_xy[:, 0] - snapped_xy[:, 0], point_xy[:, 1] - snapped_xy[:, 1]),
            "snap_ties": snap.locations,
        },
        geometry=gpd.GeoSeries(node_geoms[point_node], crs=crs),
        crs=crs,
    )
    return SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)
