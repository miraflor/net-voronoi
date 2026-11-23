"""Adaptive choice of the grid spacing for the 2-D surface, with measured stability.

The question. ``surface_voronoi`` renders the nearest-network attachment
model on a grid whose spacing the user chooses. A coarse grid can move the
border between two clusters by about one cell, and can miss a cluster whose
land is narrower than a cell. How fine must the grid be? A single grid cannot
answer this. This module renders a sequence of grids, each finer than the one
before, and measures how much land changes owner between two consecutive
grids. When the change is small enough under a stated policy, it stops with
the status ``converged``. When the cell budget or the number of refinements
runs out first, it stops with ``budget_limited`` or ``refinement_limit`` and
says why. It never reports a convergence it did not measure.

Example. A 10 m road with a point of cluster A at one end and a point of B at
the other has its border at 5 m. Grids with 1 m and 0.5 m spacing both put the
border at 5 m, so no land changes owner between them, and the result is
``converged`` at 0.5 m. A grid with 20 m spacing has one cell, whose
representative point lies exactly on the border; the tie goes to A, so B has
no land. B is then reported as ``grid_lost`` (see ``territory_qa``), and the
next, finer grid recovers it.

What stays fixed. The exact network partition is computed once, by
``network_voronoi``, and every grid reuses it through ``surface._render``.
Changing the grid never changes the network result.

Reading order:

1. ``adaptive_surface_voronoi`` checks every option, finds the starting
   spacing, computes the exact network partition once, and runs the loop;
2. ``_starting_resolution`` finds the finest spacing whose grid fits
   ``max_cells``, by counting squares without building any;
3. ``surface._render`` renders one grid;
4. ``_check_candidate`` checks the geometry that every rendering must have:
   valid polygons that cover the boundary, stay inside it, and do not
   overlap. A failure here is a program or input error, not a coarse grid,
   so the run stops;
5. ``territory_qa`` gives one row per cluster. It separates a cluster that is
   missing only because the grid is too coarse from a cluster that may have
   no land inside the boundary at any spacing;
6. ``compare_surface_ownership`` measures the share of land whose owner
   changed between two grids, and the overlap (IoU) of each cluster's land;
7. ``_meets_policy`` decides whether two consecutive grids are close enough
   under a ``ConvergencePolicy``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import shapely

from . import core
from ._numeric import _tol
from .model import SpatialNetwork
from .surface import (
    _MULTIPOLYGON,
    _POLYGON,
    DEFAULT_MAX_CELLS,
    SurfaceVoronoiResult,
    _anchor_index,
    _attachment_edges,
    _boundary_geometry,
    _check_attachment,
    _render,
    _validate_resolution,
)

STATUSES = ("converged", "budget_limited", "refinement_limit")

TERRITORY_STATUSES = ("present", "grid_lost", "no_road_inside", "shared_road_only", "no_network_length")

# Share of the boundary area that the geometry checks accept as rounding
# error. Grid cells are cut from the boundary and then merged again; each of
# these overlay steps moves vertices by about 1e-16 of the coordinate size,
# so their area error is many orders of magnitude below 1e-9 of the area.
_AREA_TOL = 1e-9


# ------------------------------------------------------------ errors, policy


class CandidateGeometryError(RuntimeError):
    """A rendered grid broke a rule that every rendering must satisfy.

    A coarse grid alone never causes this (it moves borders and can lose a
    cluster, but it still covers the boundary exactly once). So the run
    stops instead of trying a finer grid.
    """


class ConvergenceError(Exception):
    """Convergence was required (``require_convergence=True``) but not demonstrated.

    ``result`` holds the complete ``AdaptiveSurfaceResult``: the finest valid
    surface, every attempt with its measurements, and the territory table.
    A caller can inspect it. Nothing has been written to disk when this is
    raised.
    """

    def __init__(self, result: "AdaptiveSurfaceResult"):
        self.result = result
        super().__init__(convergence_summary(result))


def _check_fraction(name: str, value) -> float:
    """``value`` as a float after checking that it is a finite number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1]")
    return value


@dataclass(frozen=True)
class ConvergencePolicy:
    """When two consecutive grids count as close enough.

    * ``global_changed_area_max``: the largest allowed share of the boundary
      area whose owner changed between the two grids (0.005 means 0.5 %).
    * ``min_cluster_iou``: when set, every cluster with land in either grid
      must have an IoU of at least this value. IoU is the area of the land
      the cluster has in both grids divided by the area of the land it has in
      either grid. ``None`` (the default) measures and reports IoU but does
      not require it. Reason for this default: IoU of a small cluster is
      dominated by the grid itself. A square cluster of side ``s`` changes
      roughly a strip of width ``r / 4`` along its outline between spacings
      ``r`` and ``r / 2``, so its IoU is about ``1 - r / s``, and IoU 0.95
      needs ``r`` below about ``s / 20``. With 1,000,000 cells over 900 km2
      (30 m spacing), every cluster would have to be wider than about 600 m.
      This is a rough estimate, given to explain the default.
    * ``require_all_territories``: when true (the default), a cluster that is
      missing only because of the grid (territory status ``grid_lost``)
      blocks convergence. A cluster that may have no land inside the boundary
      at all never blocks it; it is reported in the territory table.

    These numbers are settings, not error bounds derived from the geometry.
    Choose them for each study area. The values used are recorded in the
    manifest.
    """

    global_changed_area_max: float = 0.005
    min_cluster_iou: float | None = None
    require_all_territories: bool = True

    def __post_init__(self) -> None:
        _check_fraction("global_changed_area_max", self.global_changed_area_max)
        if self.min_cluster_iou is not None:
            _check_fraction("min_cluster_iou", self.min_cluster_iou)
        if not isinstance(self.require_all_territories, bool):
            raise ValueError("require_all_territories must be True or False")


@dataclass(frozen=True)
class AdaptiveSurfaceResult:
    """The finest valid rendering, why the loop stopped, and every measurement.

    * ``surface``: the rendering at ``selected_resolution`` (the finest grid
      tried that passed the geometry checks);
    * ``status``: ``converged``, ``budget_limited`` or ``refinement_limit``;
    * ``territories``: ``territory_qa`` for ``surface``;
    * ``attempts``: one dictionary per grid tried, plus one record for a finer
      grid that was not built because it exceeded ``max_cells``;
    * ``manifest``: a JSON-ready summary of the method, options, attempts
      and status, for writing next to the polygons.
    """

    surface: SurfaceVoronoiResult
    status: str
    selected_resolution: float
    territories: pd.DataFrame
    attempts: tuple[dict, ...]
    manifest: dict

    @property
    def convergence_demonstrated(self) -> bool:
        return self.status == "converged"


# ------------------------------------------------------- 2. starting spacing


def _cell_count(boundary_geom, resolution: float):
    """Grid squares before clipping at ``resolution``; ``math.inf`` when too many to count.

    The formula is the one in ``surface._candidate_count``, which is the shape
    that ``surface._grid`` builds: ``ceil(width / r) * ceil(height / r)``, with
    at least one square in each direction. For every spacing that the grid
    code can use, the two give the same number (``tests/test_adaptive.py``
    checks this). The difference: a very small spacing makes ``width / r``
    infinite, which ``_candidate_count`` cannot turn into an integer; here it
    gives ``math.inf``, which is larger than any budget.
    """
    if resolution <= 0:
        return math.inf
    x0, y0, x1, y1 = boundary_geom.bounds
    count = 1
    for span in (x1 - x0, y1 - y0):
        ratio = span / resolution
        if not math.isfinite(ratio):
            return math.inf
        count *= max(1, math.ceil(ratio))
    return count


def _starting_resolution(boundary_geom, requested: float, max_cells: int) -> tuple[float, bool]:
    """The requested spacing when its grid fits ``max_cells``; otherwise the finest spacing that fits.

    Returns ``(spacing, coarsened)``. No grid is built: squares are only
    counted.

    Why a search works: a larger spacing never needs more squares, so the
    spacings that fit form one interval ``[r*, infinity)``. The search keeps
    two spacings: ``low``, whose grid is too large, and ``high``, whose grid
    fits. It starts with ``low = requested`` and ``high = max(width,
    height)``, where the grid has one square. Each step tests a spacing
    between the two and replaces one of them, so the interval around ``r*``
    shrinks. While ``high`` is more than twice ``low``, the test spacing is
    the geometric mean ``sqrt(low * high)``; this reaches the right order of
    magnitude in a few steps even for a requested spacing such as 1e-300.
    After that, it is the midpoint. The loop stops when ``high`` and ``low``
    differ by at most 1e-9 of ``high``. The answer is ``high``: it fits, and
    any spacing smaller by more than that relative amount does not.

    Example: a 10 x 2 boundary, ``max_cells = 100`` and ``requested = 0.01``
    give about 0.5 (20 x 4 = 80 squares). Any spacing below 0.5 needs at
    least 21 x 5 = 105 squares.
    """
    if _cell_count(boundary_geom, requested) <= max_cells:
        return requested, False
    x0, y0, x1, y1 = boundary_geom.bounds
    low, high = requested, max(x1 - x0, y1 - y0)
    for _ in range(200):  # about 50 steps are enough for any finite input
        if high - low <= 1e-9 * high:
            break
        if high > 2.0 * low:
            # Geometric mean through logarithms: low * high could round to 0.
            middle = math.exp(0.5 * (math.log(low) + math.log(high)))
        else:
            middle = 0.5 * (low + high)
        if _cell_count(boundary_geom, middle) <= max_cells:
            high = middle
        else:
            low = middle
    return high, True


# ------------------------------------------------------ 4. geometry checks


def _categories(surface: SurfaceVoronoiResult) -> dict:
    """Land of each ownership category: ``"cluster:<id>"`` for every cluster, ``"unassigned"`` if present."""
    land = {
        "cluster:" + str(cluster): geom
        for cluster, geom in zip(surface.cells["cluster_id"], surface.cells.geometry.to_numpy())
    }
    if not surface.unassigned.empty:
        land["unassigned"] = shapely.union_all(surface.unassigned.geometry.to_numpy())
    return land


def _check_candidate(boundary_geom, surface: SurfaceVoronoiResult) -> dict:
    """Check the geometry that every rendering must have; return the measured areas.

    With ``B`` the boundary and ``P_k`` the land of category ``k`` (every
    cluster, plus ``unassigned`` when there is any):

    * every ``P_k`` is a valid Polygon or MultiPolygon;
    * ``uncovered = area(B minus the union of all P_k)`` is about 0: no land
      inside the boundary is left out;
    * ``outside = area(the union of all P_k minus B)`` is about 0: no land
      outside the boundary is added;
    * ``overlap = sum of area(P_k) - area(the union of all P_k)`` is about 0:
      no land has two owners.

    The missing and the overlapping area are measured separately because in
    one combined number, such as ``sum of area(P_k) - area(B)``, a gap and an
    overlap of the same size cancel.

    Under ``"nearest_reachable_network"`` every representative point attaches
    to a road with an owner, so there must be no unassigned land.

    "About 0" means at most ``_AREA_TOL`` (1e-9) of the boundary area.
    """
    land = _categories(surface)
    if not land:
        raise CandidateGeometryError("rendering has no land in any category")
    geoms = np.asarray(list(land.values()), dtype=object)
    kinds = shapely.get_type_id(geoms)
    if np.any((kinds != _POLYGON) & (kinds != _MULTIPOLYGON)) or not shapely.is_valid(geoms).all():
        raise CandidateGeometryError("rendering contains a geometry that is not a valid polygon")

    domain = float(shapely.area(boundary_geom))
    union = shapely.union_all(geoms)
    uncovered = float(shapely.area(shapely.difference(boundary_geom, union)))
    outside = float(shapely.area(shapely.difference(union, boundary_geom)))
    overlap = max(0.0, float(shapely.area(geoms).sum()) - float(shapely.area(union)))
    limit = _AREA_TOL * domain
    if uncovered > limit or outside > limit or overlap > limit:
        raise CandidateGeometryError(
            f"rendering at resolution {surface.resolution} breaks the coverage rule: "
            f"uncovered={uncovered:.6g}, outside={outside:.6g}, overlap={overlap:.6g} "
            f"(limit {limit:.6g} in squared CRS units)"
        )
    if surface.attachment == "nearest_reachable_network" and not surface.unassigned.empty:
        raise CandidateGeometryError("nearest_reachable_network left unassigned land")
    return {
        "boundary_area": domain,
        "assigned_area": float(surface.cells.geometry.area.sum()) if not surface.cells.empty else 0.0,
        "unassigned_area": float(surface.unassigned.geometry.area.sum()) if not surface.unassigned.empty else 0.0,
        "uncovered_area": uncovered,
        "outside_area": outside,
        "overlap_area": overlap,
    }


# ---------------------------------------------------------- 5. territories


def _length_inside(lines: np.ndarray, polygon) -> np.ndarray:
    """Length of each line strictly inside ``polygon``, not counting parts that run along its outline.

    Lines wholly in the interior keep their whole length (a fast vectorized
    test). Only lines that touch the outline are cut, and from the cut part
    the length running along the outline is subtracted.
    """
    shapely.prepare(polygon)
    wholly_inside = shapely.contains_properly(polygon, lines)
    length = np.where(wholly_inside, shapely.length(lines), 0.0)
    touching = np.flatnonzero(~wholly_inside & shapely.intersects(polygon, lines))
    if len(touching):
        part = shapely.intersection(lines[touching], polygon)
        along = shapely.intersection(lines[touching], polygon.boundary)
        length[touching] = np.maximum(0.0, shapely.length(part) - shapely.length(along))
    return length


def _unshared_length_inside(pieces: np.ndarray, others: np.ndarray, polygon) -> float:
    """Length of ``pieces`` strictly inside ``polygon`` that no line in ``others`` covers.

    Why a positive answer proves that the cluster has land. Take a point
    ``p`` on such a piece, inside the boundary, away from the piece ends, and
    on no road of another owner. The roads of other owners are closed sets
    that do not contain ``p``, so all of them are at least some distance
    ``d > 0`` from ``p``. A land point ``g`` closer than ``d / 2`` to ``p``
    has a road location within ``d / 2`` (``p`` itself), while every road of
    another owner is farther than ``d / 2`` from it. So ``g`` attaches to a
    road of this cluster. The boundary interior contains a small disk around
    ``p``, so the cluster owns land of positive area inside the boundary, and
    a fine enough grid places a representative point in it.

    Crossing roads meet a piece in single points, which have no length, so
    they do not reduce the answer. Roads that run along the piece (the same
    line, as can happen in a custom network) do.
    """
    inside = shapely.union_all(shapely.intersection(pieces, polygon))
    if len(others):
        inside = shapely.difference(inside, shapely.union_all(others))
    inside = shapely.difference(inside, polygon.boundary)
    return float(shapely.length(inside))


def territory_qa(network: SpatialNetwork, boundary, surface: SurfaceVoronoiResult) -> pd.DataFrame:
    """One row per cluster: its network length, its land in ``surface``, and why land may be missing.

    Columns:

    * ``cluster_id``, ``source_point_count``;
    * ``owned_network_length`` and ``network_segment_count``: the cluster's
      pieces in the exact network partition;
    * ``owned_length_inside``: the part of that length strictly inside the
      boundary (``_length_inside``);
    * ``surface_area`` and ``surface_cell_count``: the cluster's land and
      grid cells in this rendering;
    * ``source_points_in_surface``: input points of the cluster covered by its
      own polygon (diagnostic only; see ``surface._point_qa``);
    * ``territory_status``, one of:

      - ``present``: the cluster has land in this rendering;
      - ``grid_lost``: no land here, but the cluster owns road inside the
        boundary that no road of another owner covers. Such a cluster has
        land of positive area (proof in ``_unshared_length_inside``), so a
        finer grid recovers it;
      - ``no_road_inside``: no land here, and all of the cluster's road lies
        outside the boundary or on its outline. Typical when the road network
        extends beyond the study area. Land near the outline may still attach
        to such a road, so this is "not proven to exist", not "proven absent";
      - ``shared_road_only``: no land here; the cluster's road inside the
        boundary is all covered by roads of other owners. Also not proven
        either way;
      - ``no_network_length``: the cluster owns no positive network length,
        for example when its points share a node with points of a cluster
        with a smaller ID. This is a property of the exact partition, not of
        the grid.

    ``territory_evidence`` separately states what the observation proves;
    ``no_road_inside`` and ``shared_road_only`` remain inconclusive. A
    convergence status only asserts consecutive-grid stability under the
    selected policy; it never certifies that every true planar territory
    was represented. ``resolution`` and ``attachment`` identify the grid.
    """
    boundary_geom = _boundary_geometry(boundary, network.crs)
    exact = surface.network
    clusters = network.cluster_ids
    table = pd.DataFrame({"cluster_id": clusters.astype(str)}).set_index("cluster_id")

    points = network.points_input["cluster_id"].astype(str)
    table["source_point_count"] = points.value_counts().reindex(table.index, fill_value=0)

    seg = exact.segments
    seg_cluster = seg["cluster_id"].astype(str).to_numpy() if len(seg) else np.asarray([], dtype=object)
    seg_geom = seg.geometry.to_numpy()
    seg_inside = _length_inside(seg_geom, boundary_geom) if len(seg) else np.asarray([], dtype=float)
    by_cluster = pd.DataFrame(
        {"cluster_id": seg_cluster, "length": seg["segment_dist"].to_numpy(float), "inside": seg_inside}
    ).groupby("cluster_id")
    table["owned_network_length"] = by_cluster["length"].sum().reindex(table.index, fill_value=0.0)
    table["network_segment_count"] = by_cluster["length"].size().reindex(table.index, fill_value=0)
    table["owned_length_inside"] = by_cluster["inside"].sum().reindex(table.index, fill_value=0.0)

    cells = surface.cells
    area = pd.Series(cells.geometry.area.to_numpy(), index=cells["cluster_id"].astype(str)) if len(cells) else None
    table["surface_area"] = 0.0 if area is None else area.reindex(table.index, fill_value=0.0)
    assigned = surface.grid["cluster_id"].dropna().astype(str)
    table["surface_cell_count"] = assigned.value_counts().reindex(table.index, fill_value=0)
    qa = surface.point_qa
    covered = qa.loc[qa["contained"], "cluster_id"].astype(str)
    table["source_points_in_surface"] = covered.value_counts().reindex(table.index, fill_value=0)

    # Status. Only clusters with no land and with road inside the boundary
    # need the slower proof; there are usually few of them.
    tol = _tol(table["owned_network_length"].to_numpy(float))
    status = np.full(len(table), "present", dtype=object)
    no_land = table["surface_area"].to_numpy(float) <= 0.0
    no_length = table["owned_network_length"].to_numpy(float) <= tol
    no_inside = table["owned_length_inside"].to_numpy(float) <= 10.0 * tol
    status[no_land & no_length] = "no_network_length"
    status[no_land & ~no_length & no_inside] = "no_road_inside"
    to_prove = np.flatnonzero(no_land & ~no_length & ~no_inside)
    if len(to_prove):
        # Roads of every other owner. Under "nearest_network" the roads of
        # components without points are owners too ("unassigned"), because
        # land may attach to them; under "nearest_reachable_network" it may not.
        owner = seg_cluster
        lines = seg_geom
        if surface.attachment == "nearest_network" and not exact.unassigned_edges.empty:
            unassigned = exact.unassigned_edges.geometry.to_numpy()
            owner = np.concatenate([owner, np.full(len(unassigned), None, dtype=object)])
            lines = np.concatenate([lines, unassigned])
        tree = shapely.STRtree(lines)
        for row in to_prove:
            cluster = table.index[row]
            pieces = seg_geom[seg_cluster == cluster]
            near = tree.query(pieces, predicate="intersects")[1]
            near = near[owner[near] != cluster]
            length = _unshared_length_inside(pieces, lines[np.unique(near)], boundary_geom)
            status[row] = "grid_lost" if length > 10.0 * tol[row] else "shared_road_only"
    table["territory_status"] = status
    # A missing cluster is not necessarily a failure of the grid. Only
    # grid_lost supplies a geometric proof that positive-area land exists
    # and was missed. no_road_inside and shared_road_only are inconclusive,
    # not proofs of no land. Even a zero-length network cell is not, by
    # itself, a complete characterization of the planar attachment map.
    evidence = {
        "present": "positive_area_observed",
        "grid_lost": "positive_area_proven_but_not_rendered",
        "no_road_inside": "presence_undetermined",
        "shared_road_only": "presence_undetermined",
        "no_network_length": "no_positive_owned_network_length",
    }
    table["territory_evidence"] = [evidence[value] for value in status]
    table["resolution"] = surface.resolution
    table["attachment"] = surface.attachment
    return table.reset_index()


# ------------------------------------------------------------ 6. comparison


def compare_surface_ownership(previous: SurfaceVoronoiResult, current: SurfaceVoronoiResult) -> dict:
    """How much land changed owner between two renderings of the same boundary.

    The categories are the clusters plus ``unassigned``. For a category ``k``
    with land ``A_k`` in ``previous`` and ``B_k`` in ``current``, the land
    that kept its owner is ``area(A_k & B_k)``, where ``&`` means the
    intersection (the land in both), and::

        changed_owner_fraction = 1 - (sum over k of area(A_k & B_k)) / area(boundary)

    Why not compare the union of all polygons: both renderings cover the
    whole boundary, so the two unions are equal even if every cluster had
    swapped its land with another. Example: two clusters that exchange all
    their land give ``changed_owner_fraction = 1``.

    For each cluster:

    * ``iou = area(A & B) / area(A | B)``, where ``|`` means the union (the
      land in either), with ``area(A | B) = area(A) + area(B) - area(A & B)``;
    * ``area_drift = |area(B) - area(A)| / max(area(A), area(B))``.

    A cluster with no land in either rendering has no IoU (``None``), not 1.
    A cluster with land in only one has IoU 0.

    Returns ``changed_owner_fraction``, ``min_cluster_iou`` (over clusters
    with land in either rendering), ``worst_clusters`` (up to 10 lowest IoU),
    ``clusters_in_one_rendering_only`` and ``per_cluster``. The source
    boundaries must describe the same geographic domain and CRS; equal
    area alone is not sufficient. This comparison measures grid-to-grid
    stability, never the exact error of either rendering.
    """
    # Equal *areas* do not imply equal *domains*. Two distant squares can
    # have identical areas while their intersections are empty. Prefer the
    # source boundary carried by _render (O(boundary complexity)); for
    # legacy/hand-built result objects, reconstruct the domain from their
    # grid cells (slower but makes the public comparison independently safe).
    from pyproj import CRS

    left_crs, right_crs = previous.grid.crs, current.grid.crs
    if (left_crs is None) != (right_crs is None):
        raise ValueError("the two renderings have different coordinate reference systems")
    if left_crs is not None and not CRS.from_user_input(left_crs).equals(
        CRS.from_user_input(right_crs), ignore_axis_order=True
    ):
        raise ValueError("the two renderings have different coordinate reference systems")
    if previous.attachment != current.attachment:
        raise ValueError("the two renderings use different attachment policies")

    left_domain = previous.domain_geometry
    right_domain = current.domain_geometry
    if left_domain is None:
        left_domain = shapely.union_all(previous.grid.geometry.to_numpy())
    if right_domain is None:
        right_domain = shapely.union_all(current.grid.geometry.to_numpy())
    domain = float(shapely.area(left_domain))
    other = float(shapely.area(right_domain))
    if not domain > 0 or not math.isfinite(domain) or not math.isfinite(other):
        raise ValueError("the two renderings have invalid or empty geographic domains")
    if not math.isclose(domain, other, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError("the two renderings do not cover the same area")
    if not (shapely.is_valid(left_domain) and shapely.is_valid(right_domain)):
        raise ValueError("the two renderings have invalid geographic domains")
    domain_difference = float(shapely.area(shapely.symmetric_difference(left_domain, right_domain)))
    domain_tolerance = 1e-9 * domain
    if not math.isfinite(domain_difference) or domain_difference > domain_tolerance:
        raise ValueError(
            "the two renderings do not cover the same geographic domain "
            f"(symmetric-difference area={domain_difference:.6g}, tolerance={domain_tolerance:.6g})"
        )
    a, b = _categories(previous), _categories(current)
    kept_total = 0.0
    per_cluster = {}
    for key in sorted(set(a) | set(b)):
        ga, gb = a.get(key), b.get(key)
        area_a = 0.0 if ga is None else float(shapely.area(ga))
        area_b = 0.0 if gb is None else float(shapely.area(gb))
        kept = 0.0 if ga is None or gb is None else float(shapely.area(shapely.intersection(ga, gb)))
        kept_total += kept
        if key == "unassigned":
            continue
        either = area_a + area_b - kept
        per_cluster[key.removeprefix("cluster:")] = {
            "iou": None if either <= 0 else min(1.0, max(0.0, kept / either)),
            "area_drift": None if max(area_a, area_b) <= 0 else abs(area_b - area_a) / max(area_a, area_b),
            "previous_area": area_a,
            "current_area": area_b,
        }
    scored = sorted(
        ((values["iou"], cluster) for cluster, values in per_cluster.items() if values["iou"] is not None)
    )
    return {
        "changed_owner_fraction": min(1.0, max(0.0, 1.0 - kept_total / domain)),
        "min_cluster_iou": scored[0][0] if scored else None,
        "worst_clusters": [{"cluster_id": cluster, "iou": iou} for iou, cluster in scored[:10]],
        "clusters_in_one_rendering_only": sum(
            1 for v in per_cluster.values() if (v["previous_area"] > 0) != (v["current_area"] > 0)
        ),
        "per_cluster": per_cluster,
    }


def _meets_policy(policy: ConvergencePolicy, comparison: dict, territories: pd.DataFrame) -> list[str]:
    """Reasons why two consecutive grids are not close enough; an empty list means converged."""
    reasons = []
    changed = comparison["changed_owner_fraction"]
    if changed > policy.global_changed_area_max:
        reasons.append(f"changed-owner area {changed:.4%} > {policy.global_changed_area_max:.4%}")
    if policy.min_cluster_iou is not None:
        iou = comparison["min_cluster_iou"]
        if iou is not None and iou < policy.min_cluster_iou:
            reasons.append(f"lowest cluster IoU {iou:.4f} < {policy.min_cluster_iou:.4f}")
    if policy.require_all_territories:
        lost = int((territories["territory_status"] == "grid_lost").sum())
        if lost:
            reasons.append(f"{lost} cluster(s) lost by the grid")
    return reasons


# --------------------------------------------------------------- 1. entry


def _check_int(name: str, value, minimum: int) -> int:
    """``value`` after checking that it is an integer (not a bool) of at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _count_for_json(count) -> int | None:
    """A square count for the manifest: ``None`` instead of infinity, which JSON cannot hold."""
    return None if count == math.inf else int(count)


def adaptive_surface_voronoi(
    network: SpatialNetwork,
    boundary,
    *,
    start_resolution: float = 50.0,
    max_cells: int = DEFAULT_MAX_CELLS,
    refine_factor: float = 0.5,
    max_refinements: int = 5,
    attachment: str = "nearest_network",
    convergence: ConvergencePolicy | None = None,
    require_convergence: bool = False,
) -> AdaptiveSurfaceResult:
    """Render the surface on finer and finer grids until two consecutive grids agree.

    * ``start_resolution``: the first spacing to try. If its grid has more
      than ``max_cells`` squares, the finest spacing that fits is used
      instead, and the manifest says so.
    * ``max_cells``: the most squares (before clipping) of any one grid. It
      limits the largest grid, not the total work of the run.
    * ``refine_factor``: each new spacing is the previous one times this
      number, which must be strictly between 0 and 1 (0.5 halves it).
    * ``max_refinements``: how many finer grids may follow the first one.
    * ``attachment``: ``"nearest_network"`` or ``"nearest_reachable_network"``
      (see ``surface.py``).
    * ``convergence``: a ``ConvergencePolicy``; the default is
      ``ConvergencePolicy()``.
    * ``require_convergence``: when true, a run that stops without converging
      raises ``ConvergenceError`` (which carries the result). When false, the
      result is returned with its status, and the caller decides.

    A failed geometry check (``CandidateGeometryError``) or any other
    exception stops the run at once. Nothing falls back to another method.
    """
    # Step 1. Check every option before any expensive work.
    requested = _validate_resolution(start_resolution)
    max_cells = _check_int("max_cells", max_cells, 1)
    max_refinements = _check_int("max_refinements", max_refinements, 0)
    if isinstance(refine_factor, bool) or not isinstance(refine_factor, (int, float, np.floating)):
        raise ValueError("refine_factor must be a number strictly between 0 and 1")
    refine_factor = float(refine_factor)
    if not math.isfinite(refine_factor) or not 0.0 < refine_factor < 1.0:
        raise ValueError("refine_factor must be a number strictly between 0 and 1")
    attachment = _check_attachment(attachment)
    policy = ConvergencePolicy() if convergence is None else convergence
    if not isinstance(policy, ConvergencePolicy):
        raise ValueError("convergence must be a ConvergencePolicy")
    if not isinstance(require_convergence, bool):
        raise ValueError("require_convergence must be True or False")
    boundary_geom = _boundary_geometry(boundary, network.crs)

    # Step 2. The first spacing, found by counting squares only.
    resolution, coarsened = _starting_resolution(boundary_geom, requested, max_cells)
    first_resolution = resolution

    # Step 3. The exact network partition and the road index, once for all grids.
    exact = core.network_voronoi(network)
    index = _anchor_index(network, _attachment_edges(network, exact, attachment))

    # Step 4. Finer and finer grids. Every grid that passes the geometry
    # checks becomes the selected one, so the result is always the finest
    # valid grid; convergence is judged on the last two.
    attempts: list[dict] = []
    previous = None
    territories = None
    status = None
    for level in range(max_refinements + 1):
        surface = _render(network, exact, boundary_geom, resolution, index, attachment)
        areas = _check_candidate(boundary_geom, surface)
        territories = territory_qa(network, boundary_geom, surface)
        counts = territories["territory_status"].value_counts()
        attempt = {
            "level": level,
            "resolution": resolution,
            "candidate_cells": _count_for_json(_cell_count(boundary_geom, resolution)),
            "clipped_cells": int(len(surface.grid)),
            **areas,
            "territory_counts": {name: int(counts.get(name, 0)) for name in TERRITORY_STATUSES},
            "grid_lost_clusters": territories.loc[
                territories["territory_status"] == "grid_lost", "cluster_id"
            ].tolist()[:50],
        }
        if previous is None:
            attempt["not_converged_because"] = ["first grid: nothing to compare with yet"]
        else:
            comparison = compare_surface_ownership(previous, surface)
            reasons = _meets_policy(policy, comparison, territories)
            attempt["comparison"] = {k: v for k, v in comparison.items() if k != "per_cluster"}
            attempt["compared_with_resolution"] = previous.resolution
            attempt["not_converged_because"] = reasons
            if not reasons:
                status = "converged"
        attempts.append(attempt)
        previous = surface
        if status is not None:
            break
        if level == max_refinements:
            status = "refinement_limit"
            break
        finer = resolution * refine_factor
        finer_cells = _cell_count(boundary_geom, finer)
        if finer_cells > max_cells:
            status = "budget_limited"
            attempts.append(
                {
                    "level": level + 1,
                    "resolution": finer,
                    "candidate_cells": _count_for_json(finer_cells),
                    "not_built": f"grid would need more than max_cells={max_cells:,} squares",
                }
            )
            break
        resolution = finer

    result = AdaptiveSurfaceResult(
        surface=previous,
        status=status,
        selected_resolution=float(previous.resolution),
        territories=territories,
        attempts=tuple(attempts),
        manifest={
            "surface_method": "grid rendering of nearest-network attachment",
            "attachment": attachment,
            "road_tie_rule": "equal network distance: smallest cluster_id in string order",
            "planar_tie_rule": "equally near roads: smallest edge_id in string order",
            "requested_resolution": requested,
            "first_resolution": first_resolution,
            "first_resolution_was_coarsened": coarsened,
            "max_cells": max_cells,
            "refine_factor": refine_factor,
            "max_refinements": max_refinements,
            "convergence_policy": asdict(policy),
            "status": status,
            "convergence_demonstrated": status == "converged",
            "convergence_scope": "two consecutive grids; not an exact planar error bound",
            "territory_completeness_proven": False,
            "uncertain_territory_count": int(territories["territory_evidence"].eq("presence_undetermined").sum()),
            "selected_resolution": float(previous.resolution),
            "attempts": attempts,
        },
    )
    if require_convergence and status != "converged":
        raise ConvergenceError(result)
    return result


def convergence_summary(result: AdaptiveSurfaceResult) -> str:
    """One line on why the run stopped, for messages and logs."""
    last = next(a for a in reversed(result.attempts) if "clipped_cells" in a)
    reasons = "; ".join(last.get("not_converged_because", [])) or "policy met"
    return (
        f"status={result.status} at resolution {result.selected_resolution:g} "
        f"after {sum('clipped_cells' in a for a in result.attempts)} grid(s): {reasons}"
    )
