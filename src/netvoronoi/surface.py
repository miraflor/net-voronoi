"""Approximate 2-D rendering of the exact cluster partition on the network.

The network partition in ``core.py`` is exact but one-dimensional: it lives
on the roads. This module answers a different question for the land between
the roads: if every place attaches to its nearest road location, which
cluster does it belong to? The answer is computed on a square grid, so it is
an approximation whose detail is set by ``resolution``.

Reading order:

1. ``surface_voronoi`` checks the inputs, computes the exact network
   partition, chooses the roads that land may attach to
   (``_attachment_edges``), and calls ``_render``;
2. ``_render`` runs the steps below for one grid spacing. It does not
   compute any shortest path: it receives the exact network result, so the
   adaptive renderer in ``adaptive.py`` can call it for several spacings
   with one network computation;
3. ``_grid`` cuts the boundary into square cells and gives one
   representative point per cell;
4. ``_anchor_edges`` attaches each representative point to its nearest
   allowed network location (the anchor);
5. ``_assign_anchors`` gives each anchor the exact network cluster at that
   location, using ``_edge_boundary`` from ``core.py``;
6. the cells are dissolved by cluster, and ``_point_qa`` checks that every
   input point lies in the polygon of its own cluster.

Two attachment policies decide which roads land may attach to:

* ``"nearest_network"`` (the default, and the only behaviour before this
  option existed): every road. Land whose nearest road lies on a component
  without any point stays unassigned, because that road has no owner.
* ``"nearest_reachable_network"``: only roads on components that contain
  at least one point. Land then attaches to the nearest road that has an
  owner, even when a road without an owner is nearer. This is a different
  model of the land, not a change of the network: no road is connected to
  another, and the exact network partition is the same under both policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely

from ._numeric import _count_distinct, _tol
from .core import NetworkVoronoiResult, _edge_boundary, network_voronoi
from .model import SpatialNetwork

DEFAULT_MAX_CELLS = 1_000_000

ATTACHMENT_POLICIES = ("nearest_network", "nearest_reachable_network")

_POLYGON = 3  # Shapely geometry type ids
_MULTIPOLYGON = 6


@dataclass(frozen=True)
class SurfaceVoronoiResult:
    """One grid rendering of the nearest-network attachment model.

    ``resolution`` and ``attachment`` record how the rendering was made. They
    were added after the first five fields and have defaults, so code that
    builds this object with five positional arguments still works.
    """

    cells: gpd.GeoDataFrame
    grid: gpd.GeoDataFrame
    unassigned: gpd.GeoDataFrame
    point_qa: gpd.GeoDataFrame
    network: NetworkVoronoiResult
    resolution: float | None = None
    attachment: str = "nearest_network"
    # The exact clipping domain, shared by every refinement. Optional so
    # callers constructing older five-field results keep working.
    domain_geometry: object | None = None


# ------------------------------------------------------------ input checks


def _boundary_geometry(boundary, crs):
    """The boundary as one valid Polygon or MultiPolygon in ``crs``.

    ``boundary`` is a GeoDataFrame, a GeoSeries, or one Shapely geometry
    (assumed to be in ``crs`` already). The rows of a GeoDataFrame or
    GeoSeries are reprojected and merged into one geometry; a single geometry
    is used as given. Missing and empty rows are ignored.

    Every part must be a Polygon or MultiPolygon with finite coordinates, and
    must be valid by the GEOS rules (for example, its outline may not cross
    itself). Invalid geometry is rejected here, with the reason GEOS gives,
    because the grid clipping in ``_grid`` can otherwise stop with a GEOS
    error. A tool such as ``shapely.make_valid`` can repair such geometry
    before it is passed in.
    """
    from_frame = isinstance(boundary, (gpd.GeoDataFrame, gpd.GeoSeries))
    if from_frame:
        if boundary.crs is None:
            raise ValueError("boundary CRS is missing")
        parts = boundary.to_crs(crs).geometry.to_numpy()
    else:
        parts = np.asarray([boundary], dtype=object)

    parts = parts[~(shapely.is_missing(parts) | shapely.is_empty(parts))]
    if len(parts) == 0:
        raise ValueError("boundary is empty")
    kind = shapely.get_type_id(parts)
    if np.any((kind != _POLYGON) & (kind != _MULTIPOLYGON)):
        raise ValueError("boundary must be polygonal")
    if not np.isfinite(shapely.get_coordinates(parts)).all():
        raise ValueError("boundary coordinates must be finite")
    valid = shapely.is_valid(parts)
    if not valid.all():
        raise ValueError(f"boundary geometry is invalid: {shapely.is_valid_reason(parts[~valid][0])}")
    # The union of valid polygons is again a valid Polygon or MultiPolygon.
    return shapely.union_all(parts) if from_frame else parts[0]


def _validate_resolution(resolution: float) -> float:
    """``resolution`` as a float, after checking that it is finite and positive."""
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    return resolution


def _check_attachment(attachment: str) -> str:
    """``attachment`` unchanged, after checking that it names a known policy."""
    if attachment not in ATTACHMENT_POLICIES:
        raise ValueError(f"attachment must be one of {', '.join(ATTACHMENT_POLICIES)}; got {attachment!r}")
    return attachment


def _grid_shape(boundary, resolution: float) -> tuple[int, int]:
    """Number of grid columns and rows needed to cover the bounding box of the boundary."""
    x0, y0, x1, y1 = boundary.bounds
    nx = max(1, int(np.ceil((x1 - x0) / resolution)))
    ny = max(1, int(np.ceil((y1 - y0) / resolution)))
    return nx, ny


def _candidate_count(boundary, resolution: float) -> int:
    """Number of grid squares before clipping to the boundary."""
    nx, ny = _grid_shape(boundary, resolution)
    return nx * ny


def check_grid_size(boundary, crs, resolution: float, max_cells: int | None = DEFAULT_MAX_CELLS) -> int:
    """Return the number of grid squares; raise ``ValueError`` when it exceeds ``max_cells``."""
    resolution = _validate_resolution(resolution)
    geom = _boundary_geometry(boundary, crs)
    count = _candidate_count(geom, resolution)
    if max_cells is not None:
        if isinstance(max_cells, bool):
            raise ValueError("max_cells must be a positive integer or None")
        try:
            numeric = float(max_cells)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_cells must be a positive integer or None") from exc
        if not np.isfinite(numeric) or numeric < 1 or numeric != np.floor(numeric):
            raise ValueError("max_cells must be a positive integer or None")
        limit = int(numeric)
        if count > limit:
            raise ValueError(
                f"surface grid would require {count:,} candidate cells, exceeding max_cells={limit:,}; "
                "use a coarser resolution or deliberately raise max_cells"
            )
    return count


# ------------------------------------------------------------------ 2. grid


def _grid(boundary, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Grid cells clipped to the boundary, and one representative point per cell.

    The grid has exactly the ``(nx, ny)`` shape counted by the allocation
    guard. The grid lines are computed once, and every square takes its four
    sides from these shared lines. A square's right side is therefore the
    same number as its right neighbour's left side, so dissolving adjacent
    squares leaves no gaps. (Computing ``left + resolution`` separately for
    each square does not guarantee this: the two sums can differ in the last
    bits.)
    """
    x0, y0, _, _ = boundary.bounds
    nx, ny = _grid_shape(boundary, resolution)
    xs = x0 + np.arange(nx + 1, dtype=float) * resolution
    ys = y0 + np.arange(ny + 1, dtype=float) * resolution
    # Square (col, row) spans xs[col] to xs[col + 1] and ys[row] to ys[row + 1].
    # The squares are listed column by column (x outer, y inner); ``cell_id``
    # in the output follows this order.
    col, row = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    col, row = col.ravel(), row.ravel()
    squares = shapely.box(xs[col], ys[row], xs[col + 1], ys[row + 1])
    # Keep the part of each square inside the boundary, and drop squares
    # that only touch it.
    intersects = shapely.intersects(squares, boundary)
    clipped = shapely.intersection(squares[intersects], boundary)
    positive = shapely.area(clipped) > 0
    clipped = clipped[positive]
    return clipped, shapely.point_on_surface(clipped)


# --------------------------------------------------------------- 3. anchors


def _attachment_edges(network: SpatialNetwork, result: NetworkVoronoiResult, attachment: str) -> np.ndarray:
    """Positions in ``network.edges`` of the roads that land may attach to.

    ``"nearest_network"`` allows every edge. ``"nearest_reachable_network"``
    allows only edges whose end nodes have a finite distance to some point,
    which are the edges on components that contain a point. On an undirected
    network both ends of an edge are reachable or neither is; ``core._cut_edges``
    already stops with an error otherwise, so one end is enough here.
    """
    attachment = _check_attachment(attachment)
    if attachment == "nearest_network":
        return np.arange(len(network.edges), dtype=np.int64)
    u = network.edges["u"].to_numpy(np.int64)
    allowed = np.flatnonzero(np.isfinite(result.node_min_distance[u])).astype(np.int64)
    if len(allowed) == 0:
        # Not reachable with a network built by ``from_geodataframes``, which
        # always places points on roads; kept as a clear message for custom
        # networks instead of an empty spatial index.
        raise ValueError("nearest_reachable_network needs at least one edge on a component with a point")
    return allowed


@dataclass(frozen=True)
class _AnchorIndex:
    """A spatial index over the roads that land may attach to.

    ``edge`` holds the positions in ``network.edges`` of the indexed roads;
    the index returns positions into ``edge``, which ``_anchor_edges`` maps
    back to ``network.edges``. ``id_rank`` is the string-order rank of every
    ``edge_id`` in the whole edge table. Ranking the whole table, not only the
    indexed roads, keeps the tie rule identical under both attachment
    policies: the smaller ``edge_id`` wins either way.

    Building the index once and reusing it saves rebuilding it for every grid
    spacing in ``adaptive.py``.
    """

    edge: np.ndarray
    tree: shapely.STRtree
    id_rank: np.ndarray


def _anchor_index(network: SpatialNetwork, edge: np.ndarray | None = None) -> _AnchorIndex:
    """Build the ``_AnchorIndex`` over ``edge`` (default: every edge)."""
    edges = network.edges
    if edge is None:
        edge = np.arange(len(edges), dtype=np.int64)
    edge_ids = edges["edge_id"].astype(str).to_numpy()
    id_rank = np.empty(len(edge_ids), dtype=np.int64)
    id_rank[np.argsort(edge_ids, kind="stable")] = np.arange(len(edge_ids))
    return _AnchorIndex(
        edge=np.asarray(edge, dtype=np.int64),
        tree=shapely.STRtree(edges.geometry.to_numpy()[edge]),
        id_rank=id_rank,
    )


def _anchor_edges(network: SpatialNetwork, points: np.ndarray, index: _AnchorIndex | None = None):
    """Nearest allowed network edge and position on it for every representative point.

    Returns ``(edge, offset, access, ties)``: the edge position in
    ``network.edges``, the network distance from the edge's ``u`` end to the
    anchor, the straight-line distance from the point to the anchor, and the
    number of distinct network locations at that nearest distance.

    ``index`` limits the search to the roads land may attach to; without it,
    every edge is searched (the ``"nearest_network"`` policy).

    When several edges are exactly equally near, the one with the smallest
    ``edge_id`` (string order) is used. An offset within the numerical
    tolerance of an edge end is set to exactly that end, as in point
    snapping. Edges that meet at the nearest node offer one location (that
    node), so a point next to a junction is not counted as ambiguous.
    """
    if index is None:
        index = _anchor_index(network)
    edges = network.edges
    lines = edges.geometry.to_numpy()
    # ``query_nearest`` returns all equally near edges as (point, edge) index
    # pairs with their distances. A point with two nearest edges appears in
    # two pairs. The tree numbers its roads 0, 1, 2, ...; ``index.edge`` turns
    # those numbers into positions in ``network.edges``.
    (pidx, tree_pos), distance = index.tree.query_nearest(points, all_matches=True, return_distance=True)
    eidx = index.edge[tree_pos]
    # Sort the pairs by point and then by the string order of edge_id, so that
    # the first pair of each point holds the nearest edge with the smallest
    # edge_id.
    order = np.lexsort((index.id_rank[eidx], pidx))
    pidx, eidx, distance = pidx[order], eidx[order], distance[order]
    first = np.r_[True, pidx[1:] != pidx[:-1]]
    if first.sum() != len(points):
        raise RuntimeError("failed to find a nearest network edge for a surface cell")

    # Network distance from the u end to the anchor, for every pair. The
    # geometric position is rescaled to the ``length`` column, which is the
    # network distance; the two agree within the tolerance.
    L = edges["length"].to_numpy(float)[eidx]
    measure = shapely.line_locate_point(lines[eidx], points[pidx])
    offset = L * measure / shapely.length(lines[eidx])
    t = _tol(L)
    offset = np.where(offset <= t, 0.0, np.where(L - offset <= t, L, offset))

    # Describe each anchor location by one integer: the node id at an edge
    # end, otherwise ``-1 - edge position``, a negative number that no other
    # edge and no node can have. Equally near edges that meet at one node
    # then count as one location.
    u = edges["u"].to_numpy(np.int64)[eidx]
    v = edges["v"].to_numpy(np.int64)[eidx]
    location = np.where(offset == 0.0, u, np.where(offset == L, v, -1 - eidx))
    ties = _count_distinct(pidx, location, len(points))
    return eidx[first], offset[first], distance[first], ties


def _assign_anchors(
    network: SpatialNetwork, result: NetworkVoronoiResult, edge: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    """Exact network cluster at every anchor, or ``None`` on an unreachable component.

    An anchor on an end node takes that node's winner. An anchor inside an
    edge takes the winner of the side of ``_edge_boundary`` it lies on, which
    is the same rule that cuts ``network_segments``. An anchor on the boundary
    itself (within the tolerance) is a tie and goes to the lexicographically
    smaller cluster ID.
    """
    u = network.edges["u"].to_numpy(np.int64)[edge]
    v = network.edges["v"].to_numpy(np.int64)[edge]
    L = network.edges["length"].to_numpy(float)[edge]
    A = result.node_min_distance[u]
    B = result.node_min_distance[v]
    cu = result.node_cluster[u]
    cv = result.node_cluster[v]

    # Four cases follow. An anchor on an edge of a component without points
    # keeps ``None``: its node winners are ``None`` and its distances are inf.
    at_u = offset == 0.0  # anchor on the u node
    at_v = ~at_u & (offset == L)  # anchor on the v node
    inside = ~at_u & ~at_v & np.isfinite(A)  # anchor strictly inside a reachable edge
    split = inside & (cu != cv)  # ... of an edge whose ends have different winners

    assigned = np.full(len(edge), None, dtype=object)
    assigned[at_u] = cu[at_u]
    assigned[at_v] = cv[at_v]
    same = inside & ~split
    assigned[same] = cu[same]

    # On a split edge, compare the anchor with the boundary position x. An
    # anchor within the tolerance t of x is on the boundary: a tie.
    a, b, length, o = A[split], B[split], L[split], offset[split]
    x = _edge_boundary(a, b, length)
    t = _tol(length, a, b)
    left, right = cu[split], cv[split]
    smaller = np.where(left < right, left, right)  # string comparison of the two cluster IDs
    assigned[split] = np.where(o < x - t, left, np.where(o > x + t, right, smaller))
    return assigned


# -------------------------------------------------------------------- 5. QA


def _point_qa(network: SpatialNetwork, surface: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Whether each input point is covered by the surface polygon of its own cluster."""
    polygons = dict(zip(surface["cluster_id"].astype(str), surface.geometry.to_numpy()))
    # Preparing a polygon builds a spatial index inside it, which makes the
    # repeated point-in-polygon tests below faster.
    shapely.prepare(np.asarray(list(polygons.values()), dtype=object))
    labels = network.points_input["cluster_id"].astype(str).to_numpy()
    # The polygon of each point's own cluster, or None when that cluster owns
    # no surface; ``covers(None, point)`` is False.
    own = np.asarray([polygons.get(c) for c in labels], dtype=object)
    contained = shapely.covers(own, network.points_input.geometry.to_numpy())
    return gpd.GeoDataFrame(
        {
            "point_id": network.points_input["point_id"].to_numpy(),
            "cluster_id": network.points_input["cluster_id"].to_numpy(),
            "contained": np.asarray(contained, dtype=bool),
        },
        geometry=network.points_input.geometry.copy(),
        crs=network.crs,
    )


# ---------------------------------------------------------------- 1. entry


def surface_voronoi(
    network: SpatialNetwork,
    boundary,
    *,
    resolution: float = 100.0,
    max_cells: int | None = DEFAULT_MAX_CELLS,
    attachment: str = "nearest_network",
) -> SurfaceVoronoiResult:
    """Render cluster cells over a polygon using nearest-network attachment.

    This is a grid approximation of a defined 2-D access model: each grid-cell
    representative point attaches to its nearest allowed location on the road
    network and inherits that location's exact network Voronoi cluster.
    ``attachment`` chooses the allowed roads (see the module docstring); the
    default gives the same result as before the option existed.
    """
    resolution = _validate_resolution(resolution)
    attachment = _check_attachment(attachment)
    boundary_geom = _boundary_geometry(boundary, network.crs)
    check_grid_size(boundary_geom, network.crs, resolution, max_cells)
    net = network_voronoi(network)
    index = _anchor_index(network, _attachment_edges(network, net, attachment))
    return _render(network, net, boundary_geom, resolution, index, attachment)


def _render(
    network: SpatialNetwork,
    net: NetworkVoronoiResult,
    boundary_geom,
    resolution: float,
    index: _AnchorIndex,
    attachment: str,
) -> SurfaceVoronoiResult:
    """One grid rendering, steps 3 to 6 of the module docstring.

    The inputs are already checked: ``boundary_geom`` comes from
    ``_boundary_geometry``, the grid size from ``check_grid_size`` (or the
    planner in ``adaptive.py``), and ``net`` is ``network_voronoi(network)``.
    Nothing here computes a shortest path.
    """
    cells, reps = _grid(boundary_geom, resolution)
    if len(cells) == 0:
        raise ValueError("boundary produced no positive-area grid cells")

    edge, offset, access, anchor_ties = _anchor_edges(network, reps, index)
    assigned = _assign_anchors(network, net, edge, offset)
    edge_id = network.edges["edge_id"].to_numpy()[edge]

    grid = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(len(cells), dtype=int),
            "cluster_id": assigned,
            "edge_id": edge_id,
            "edge_offset": offset,
            "access_dist": access,
            "anchor_ties": anchor_ties,
        },
        geometry=gpd.GeoSeries(cells, crs=network.crs),
        crs=network.crs,
    )

    assigned_grid = grid[grid["cluster_id"].notna()].copy()
    if assigned_grid.empty:
        surface = gpd.GeoDataFrame(columns=["cluster_id", "geometry"], geometry="geometry", crs=network.crs)
    else:
        surface = assigned_grid.dissolve(by="cluster_id", as_index=False)
        surface = surface[["cluster_id", "geometry"]].sort_values("cluster_id").reset_index(drop=True)

    missing = grid[grid["cluster_id"].isna()].copy()
    if missing.empty:
        unassigned = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=network.crs)
    else:
        unassigned = missing[["geometry"]].dissolve().reset_index(drop=True)

    return SurfaceVoronoiResult(
        cells=surface,
        grid=grid,
        unassigned=unassigned,
        point_qa=_point_qa(network, surface),
        network=net,
        resolution=resolution,
        attachment=attachment,
        domain_geometry=boundary_geom,
    )
