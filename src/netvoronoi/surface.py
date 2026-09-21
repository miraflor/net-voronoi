"""Approximate 2-D rendering of the exact network Voronoi result.

Reading order for this module
-----------------------------
1. ``surface_voronoi`` (at the end of the file) is the public entry point.
   Before any expensive work it checks its arguments and the size of the
   grid (``check_grid_size``). It then computes the exact network result and
   runs a four-step pipeline: grid cells, the nearest network location
   ("anchor") of each cell, the exact memberships at each anchor, and one
   dissolved polygon layer per membership type.
2. ``_grid_cells``, ``_anchor_edges`` with ``_anchor_positions``, and
   ``_memberships_at`` are those steps. Each one works on all cells at the
   same time with array operations; there is no loop over cells.

All distances are in the linear unit of the projected network CRS.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from .core import NetworkVoronoiResult, _tolerance, network_voronoi
from .model import SpatialNetwork


@dataclass
class SurfaceVoronoiResult:
    """Approximate 2-D rendering plus its exact network result."""

    hard: gpd.GeoDataFrame
    epsilon: gpd.GeoDataFrame
    grid: gpd.GeoDataFrame
    unassigned: gpd.GeoDataFrame
    network: NetworkVoronoiResult


# ---------------------------------------------------------------------------
# Checks made before any expensive work
# ---------------------------------------------------------------------------

# Default upper limit for the number of candidate grid squares. For scale: on
# a 4 GB test machine, with the 22,500-node benchmark network, 200 sites and
# epsilon = 150, a grid of 950,625 candidate squares (674,694 of them inside a
# 5,000-vertex boundary) took about 60 s and 0.9 GB of memory. v0.4.0, which
# had no limit, ran out of memory at 4,000,000 candidate squares.
DEFAULT_MAX_CELLS = 1_000_000


def _check_resolution(resolution) -> float:
    """Return ``resolution`` as a float; reject zero, negative, NaN and infinity.

    Written as ``not (... > 0)`` because every comparison with NaN is false:
    a plain ``resolution <= 0`` test would let NaN through, and the grid
    arithmetic would then fail with an unclear error.
    """
    resolution = float(resolution)
    if not (np.isfinite(resolution) and resolution > 0.0):
        raise ValueError(f"resolution must be a finite positive number, got {resolution!r}")
    return resolution


def _check_max_cells(max_cells) -> int | None:
    """Return ``max_cells`` as an int, or ``None`` (no limit); reject anything else.

    ``True`` is rejected on purpose, because Python treats it as the integer 1.
    """
    if max_cells is None:
        return None
    if isinstance(max_cells, bool) or not isinstance(max_cells, (int, np.integer)):
        raise ValueError(f"max_cells must be a positive integer or None, got {max_cells!r}")
    if max_cells < 1:
        raise ValueError(f"max_cells must be a positive integer or None, got {max_cells!r}")
    return int(max_cells)


def _boundary_geometry(boundary, network_crs):
    """Return one valid polygon in network CRS units.

    A bare Shapely geometry has no CRS metadata, so it is assumed to already be
    expressed in ``network_crs``.  GeoPandas inputs are reprojected explicitly.
    """
    if isinstance(boundary, gpd.GeoDataFrame):
        if boundary.crs is None:
            raise ValueError("boundary has no CRS")
        work = boundary.to_crs(network_crs) if boundary.crs != network_crs else boundary
        geom = work.geometry.union_all()
    elif isinstance(boundary, gpd.GeoSeries):
        if boundary.crs is None:
            raise ValueError("boundary has no CRS")
        work = boundary.to_crs(network_crs) if boundary.crs != network_crs else boundary
        geom = work.union_all()
    elif isinstance(boundary, BaseGeometry):
        geom = boundary
    else:
        raise TypeError(
            "boundary must be a Polygon/MultiPolygon geometry, GeoSeries, or GeoDataFrame"
        )

    if geom.is_empty:
        raise ValueError("boundary is empty")
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("boundary must resolve to a Polygon or MultiPolygon")
    if not geom.is_valid:
        raise ValueError("boundary geometry is invalid; repair it before surface rendering")
    return geom


def _grid_shape(boundary, resolution: float) -> tuple[float, float, int, int]:
    """Origin and size of the square grid that covers ``boundary``.

    Returns ``(x0, y0, nx, ny)``. The grid starts at ``(x0, y0)``, which is the
    lower-left corner of the boundary's bounding box rounded down to whole
    multiples of ``resolution``, and it has ``nx`` columns and ``ny`` rows.
    ``nx * ny`` is the number of candidate squares, counted before any square
    is clipped to the boundary or dropped. Both the size check and the grid
    itself use this function, so the number that is checked is the number
    that is built.
    """
    minx, miny, maxx, maxy = boundary.bounds
    x0 = np.floor(minx / resolution) * resolution
    y0 = np.floor(miny / resolution) * resolution
    nx = int(np.ceil((maxx - x0) / resolution))
    ny = int(np.ceil((maxy - y0) / resolution))
    return x0, y0, nx, ny


def _require_grid_size(boundary, resolution: float, max_cells: int | None) -> int:
    """Return the number of candidate squares; raise if it is above ``max_cells``."""
    _, _, nx, ny = _grid_shape(boundary, resolution)
    candidates = nx * ny
    if max_cells is not None and candidates > max_cells:
        raise ValueError(
            f"the surface grid would have {candidates:,} candidate cells "
            f"({nx:,} columns x {ny:,} rows), above max_cells={max_cells:,}. "
            "Use a larger resolution value (larger cells) or a larger max_cells."
        )
    return candidates


def check_grid_size(
    boundary, crs, resolution: float, max_cells: int | None = DEFAULT_MAX_CELLS
) -> int:
    """Check the size of the surface grid before any expensive work.

    This is the same check that ``surface_voronoi`` makes first. It needs only
    the boundary, the network CRS (the boundary is measured in its units) and
    ``resolution``, and it reads only the boundary's bounding box, so it is
    fast. The CLI calls it before it builds the road network.

    Returns the number of candidate squares. Raises ``ValueError`` when that
    number is above ``max_cells`` (``None`` means no limit).
    """
    resolution = _check_resolution(resolution)
    max_cells = _check_max_cells(max_cells)
    if crs is None:
        raise ValueError("the network CRS is missing, so the grid cannot be measured")
    return _require_grid_size(_boundary_geometry(boundary, crs), resolution, max_cells)


# ---------------------------------------------------------------------------
# Step 1: grid cells
# ---------------------------------------------------------------------------


def _grid_cells(boundary, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Square cells clipped to ``boundary``, and one interior point per cell.

    The squares have side ``resolution`` and corners on whole multiples of
    ``resolution``. Cells are numbered column by column: all cells of the
    first x position from south to north, then the next x position, and so on.
    A square that does not cover a positive area inside the boundary is left
    out, so ``cell_id`` counts only the cells that remain.

    Only the squares that the boundary line crosses need a polygon
    intersection. A square that lies completely inside the boundary is its own
    cell and is kept as it is, which matters when the boundary has many
    vertices: on a 5,000-vertex boundary with 25,000 squares this step took
    1.2 s instead of 9.5 s.
    """
    # The size of this grid was already checked by ``_require_grid_size``.
    x0, y0, nx, ny = _grid_shape(boundary, resolution)
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")  # x index outer
    left = x0 + ix.ravel() * resolution
    bottom = y0 + iy.ravel() * resolution
    squares = shapely.box(left, bottom, left + resolution, bottom + resolution)

    # ``prepare`` attaches a spatial index to the boundary for the two tests
    # below; the shape itself does not change.
    shapely.prepare(boundary)
    inside = shapely.contains_properly(boundary, squares)
    crossing = shapely.intersects(boundary, squares) & ~inside
    cells = np.where(inside, squares, None)
    cells[crossing] = shapely.intersection(squares[crossing], boundary)

    # Squares outside the boundary are still ``None`` here; a square that only
    # touches the boundary line gives a cell of zero area. Both are dropped.
    keep = inside | (crossing & (shapely.area(cells) > 0))
    cells = cells[keep]
    return cells, shapely.point_on_surface(cells)


# ---------------------------------------------------------------------------
# Step 2: the anchor of each cell
# ---------------------------------------------------------------------------


def _anchor_edges(network: SpatialNetwork, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest edge of every point, and the number of equally near edges.

    ``query_nearest`` reports every edge at the smallest distance. When
    several edges are exactly equally near, the one whose ``edge_id`` comes
    first as text is used, so edge ``10`` comes before edge ``9``. A validated
    network has at least one edge, so every point has a nearest edge.

    Returns row positions in the edge table, and the tie count per point.
    """
    lines = network.edges.geometry.to_numpy()
    point_of_pair, edge_of_pair = shapely.STRtree(lines).query_nearest(points, all_matches=True)

    # ``rank[e]`` is the position of edge ``e`` when the edge IDs are sorted
    # as text. Sorting the pairs by point and then by that rank puts each
    # point's chosen edge first among its pairs.
    text_ids = np.asarray([str(value) for value in network.edges["edge_id"].tolist()], dtype=object)
    rank = np.empty(len(lines), dtype=int)
    rank[np.argsort(text_ids, kind="stable")] = np.arange(len(lines))
    order = np.lexsort((rank[edge_of_pair], point_of_pair))
    point_of_pair, edge_of_pair = point_of_pair[order], edge_of_pair[order]
    first_of_point = np.r_[True, point_of_pair[1:] != point_of_pair[:-1]]

    edge = np.full(len(points), -1, dtype=int)
    edge[point_of_pair[first_of_point]] = edge_of_pair[first_of_point]
    if np.any(edge < 0):  # pragma: no cover - a validated network has edges
        raise ValueError("the network has no edge to attach a grid cell to")
    return edge, np.bincount(point_of_pair, minlength=len(points))


def _anchor_positions(
    network: SpatialNetwork, edge: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where each point sits on its edge, and how far it is from there.

    The position (``edge_offset``) is the network distance of the anchor from
    edge endpoint ``u``, so it can be compared directly with the ``from_dist``
    and ``to_dist`` of the exact segment tables. ``access_dist`` is the
    straight-line distance from the point to its anchor.
    """
    lines = network.edges.geometry.to_numpy()[edge]
    length = network.edges["length"].to_numpy(dtype=float)[edge]
    measure = shapely.line_locate_point(lines, points)  # geometric distance along the line
    anchors = shapely.line_interpolate_point(lines, measure)
    return length * measure / shapely.length(lines), shapely.distance(points, anchors)


def _nearest_site_distance(
    network: SpatialNetwork, node_min_distance: np.ndarray, edge: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    """Network distance from each anchor to its nearest site (NaN if none).

    A path from the anchor leaves its edge through endpoint ``u`` or through
    endpoint ``v``.
    """
    u = network.edges["u"].to_numpy(dtype=int)[edge]
    v = network.edges["v"].to_numpy(dtype=int)[edge]
    length = network.edges["length"].to_numpy(dtype=float)[edge]
    value = np.minimum(node_min_distance[u] + offset, node_min_distance[v] + length - offset)
    return np.where(np.isfinite(value), value, np.nan)


# ---------------------------------------------------------------------------
# Step 3: memberships at the anchors
# ---------------------------------------------------------------------------


def _memberships_at(
    segments: gpd.GeoDataFrame, edge_id: np.ndarray, offset: np.ndarray
) -> pd.DataFrame:
    """Every pair (cell, site_id) whose exact segment covers the cell's anchor.

    Each cell is joined to the segments that lie on its own edge, and a
    segment is kept when ``from_dist <= offset <= to_dist`` with the package's
    usual small slack for rounding. An anchor exactly on the boundary between
    two sites therefore belongs to both. The pairs are returned without
    duplicates, sorted by cell and then by ``site_id``.
    """
    columns = ["cell", "site_id"]
    if segments.empty:
        return pd.DataFrame({name: [] for name in columns})

    anchors = pd.DataFrame({"cell": np.arange(len(edge_id)), "edge_id": edge_id, "offset": offset})
    candidates = anchors.merge(
        segments[["edge_id", "from_dist", "to_dist", "site_id"]], on="edge_id", how="inner"
    )
    slack = _tolerance(candidates["offset"].to_numpy(dtype=float))
    covers = (candidates["from_dist"] - slack <= candidates["offset"]) & (
        candidates["offset"] <= candidates["to_dist"] + slack
    )
    pairs = candidates.loc[covers, columns].drop_duplicates()
    return pairs.sort_values(columns, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: cells to polygon layers
# ---------------------------------------------------------------------------


def _dissolved_by_site(site_id, geometry, crs) -> gpd.GeoDataFrame:
    """One row per site: the union of the cells that belong to it."""
    if len(site_id) == 0:
        return gpd.GeoDataFrame(columns=["site_id", "geometry"], geometry="geometry", crs=crs)
    frame = gpd.GeoDataFrame({"site_id": site_id}, geometry=gpd.GeoSeries(geometry, crs=crs))
    return frame.dissolve(by="site_id", as_index=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def surface_voronoi(
    network: SpatialNetwork,
    boundary,
    epsilon: float = 0.0,
    resolution: float = 100.0,
    batch_size: int = 32,
    max_cells: int | None = DEFAULT_MAX_CELLS,
) -> SurfaceVoronoiResult:
    """Render network Voronoi membership approximately over a 2-D polygon.

    A representative point from each clipped grid cell is attached to its
    nearest network location and inherits the exact membership at that anchor.
    The result is therefore a grid approximation to an explicit nearest-network
    access model; it is not an exact continuous 2-D network-induced Voronoi.

    ``resolution`` and all reported distances use the projected network CRS
    units.  A bare Shapely ``boundary`` is assumed to use that same CRS.
    ``max_cells`` limits the number of candidate grid squares in the boundary's
    bounding box (see ``DEFAULT_MAX_CELLS`` for what the default costs). The
    limit is checked before any expensive work; pass ``None`` to disable it
    deliberately.

    Order of work: the cheap checks (arguments and grid size), then the exact
    network result (which also validates the network, ``epsilon`` and
    ``batch_size``), then the grid and everything that depends on it. A wrong
    argument is therefore reported before the grid is built.

    The ``unassigned`` layer is one dissolved polygon of the cells whose
    anchor no site can reach; those cells are also in ``grid``, with an empty
    ``hard_site``.
    """
    resolution = _check_resolution(resolution)
    max_cells = _check_max_cells(max_cells)
    boundary_geom = _boundary_geometry(boundary, network.crs)
    _require_grid_size(boundary_geom, resolution, max_cells)

    network_result = network_voronoi(network, epsilon=epsilon, batch_size=batch_size)

    # 1. Grid cells and one representative point per cell.
    cells, points = _grid_cells(boundary_geom, resolution)
    if len(cells) == 0:
        raise ValueError("boundary produced no grid cells")

    # 2. The nearest network location of every representative point.
    edge, anchor_ties = _anchor_edges(network, points)
    edge_offset, access_dist = _anchor_positions(network, edge, points)
    edge_id = network.edges["edge_id"].to_numpy()[edge]

    # 3. Exact memberships at the anchors. For the hard layer, an anchor on
    #    the boundary between two sites goes to the smaller site_id.
    hard_pairs = _memberships_at(network_result.hard_segments, edge_id, edge_offset)
    hard_site = hard_pairs.groupby("cell")["site_id"].min()
    epsilon_pairs = _memberships_at(network_result.epsilon_segments, edge_id, edge_offset)
    n_epsilon = epsilon_pairs.groupby("cell").size()

    cell_id = np.arange(len(cells))
    site_per_cell = hard_site.reindex(cell_id).to_numpy(dtype=object)
    site_per_cell = np.where(pd.isna(site_per_cell), None, site_per_cell)
    grid = gpd.GeoDataFrame(
        {
            "cell_id": cell_id,
            "edge_id": edge_id,
            "edge_offset": edge_offset,
            "nearest_site_dist": _nearest_site_distance(
                network, network_result.node_min_distance, edge, edge_offset
            ),
            "access_dist": access_dist,
            "hard_site": list(site_per_cell),
            "n_epsilon": n_epsilon.reindex(cell_id, fill_value=0).to_numpy(dtype=int),
            "anchor_ties": anchor_ties,
        },
        geometry=gpd.GeoSeries(cells, crs=network.crs),
        crs=network.crs,
    )

    # 4. Dissolve the cells into one polygon per site, and one for the rest.
    assigned = hard_site.index.to_numpy()
    unassigned_cell = np.setdiff1d(cell_id, assigned)
    if len(unassigned_cell):
        unassigned = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(cells[unassigned_cell], crs=network.crs)
        ).dissolve()
    else:
        unassigned = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=network.crs)

    return SurfaceVoronoiResult(
        hard=_dissolved_by_site(hard_site.to_numpy(), cells[assigned], network.crs),
        epsilon=_dissolved_by_site(
            epsilon_pairs["site_id"].to_numpy(), cells[epsilon_pairs["cell"].to_numpy()], network.crs
        ),
        grid=grid,
        unassigned=unassigned,
        network=network_result,
    )
