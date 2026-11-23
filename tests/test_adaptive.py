"""Adaptive grid refinement: budget planning, stopping statuses, territories and ownership comparison."""

import json
import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

import netvoronoi.core as core
import netvoronoi.surface as surface_module
from netvoronoi import (
    ConvergenceError,
    ConvergencePolicy,
    adaptive_surface_voronoi,
    from_geodataframes,
    surface_voronoi,
)
from netvoronoi import adaptive
from netvoronoi.adaptive import (
    CandidateGeometryError,
    _cell_count,
    _starting_resolution,
    compare_surface_ownership,
)
from netvoronoi.surface import SurfaceVoronoiResult, _candidate_count

CRS = "EPSG:3857"


def _network(roads, points, crs=CRS):
    """Points given as (cluster, (x, y)) pairs."""
    return from_geodataframes(
        gpd.GeoDataFrame(geometry=roads, crs=crs),
        gpd.GeoDataFrame({"c": [c for c, _ in points]}, geometry=[Point(xy) for _, xy in points], crs=crs),
        cluster_col="c",
    )


def _area(geom, crs=CRS):
    return gpd.GeoDataFrame(geometry=[geom], crs=crs)


@pytest.fixture
def line():
    """A 10 m road with cluster A at one end and B at the other; the border is at 5 m."""
    network = _network([LineString([(0, 0), (10, 0)])], [("A", (0, 0)), ("B", (10, 0))])
    return network, _area(box(0, -1, 10, 1))


# ------------------------------------------------------------ budget planning


def test_cell_count_equals_the_grid_shape_count():
    rng = np.random.default_rng(3)
    for _ in range(500):
        x0, y0 = rng.uniform(-1e6, 1e6, 2)
        geom = box(x0, y0, x0 + rng.uniform(0.1, 5000), y0 + rng.uniform(0.1, 5000))
        r = float(10 ** rng.uniform(-1, 3))
        assert _cell_count(geom, r) == _candidate_count(geom, r)


def test_starting_resolution_is_the_finest_spacing_that_fits():
    geom = box(0, 0, 10, 2)
    spacing, coarsened = _starting_resolution(geom, 0.01, 100)
    assert coarsened
    assert spacing == pytest.approx(0.5, rel=1e-8)
    assert _cell_count(geom, spacing) <= 100
    assert _cell_count(geom, spacing * (1 - 1e-6)) > 100
    assert _starting_resolution(geom, 1.0, 100) == (1.0, False)


def test_starting_resolution_handles_a_tiny_request():
    geom = box(0, 0, 10, 10)
    spacing, coarsened = _starting_resolution(geom, 1e-300, 1_000_000)
    assert coarsened and math.isfinite(spacing)
    assert _cell_count(geom, spacing) <= 1_000_000 < _cell_count(geom, spacing * (1 - 1e-6))


def test_no_grid_larger_than_max_cells_is_ever_built(line, monkeypatch):
    network, boundary = line
    built = []
    original = surface_module._grid

    def recording_grid(geom, resolution):
        built.append(_candidate_count(geom, resolution))
        return original(geom, resolution)

    monkeypatch.setattr(surface_module, "_grid", recording_grid)
    result = adaptive_surface_voronoi(network, boundary, start_resolution=0.001, max_cells=100)
    assert built and max(built) <= 100
    assert result.status == "budget_limited"
    assert result.manifest["first_resolution_was_coarsened"]
    assert "not_built" in result.attempts[-1]


# ----------------------------------------------------------- loop and status


def test_exact_partition_is_computed_once(line, monkeypatch):
    network, boundary = line
    calls = []
    original = core._node_partition
    monkeypatch.setattr(core, "_node_partition", lambda net: (calls.append(1), original(net))[1])
    result = adaptive_surface_voronoi(network, boundary, start_resolution=1, max_cells=10_000)
    assert len(calls) == 1
    assert len([a for a in result.attempts if "clipped_cells" in a]) == 2


def test_line_converges_and_matches_the_fixed_renderer(line):
    network, boundary = line
    result = adaptive_surface_voronoi(network, boundary, start_resolution=1, max_cells=10_000)
    assert result.status == "converged" and result.convergence_demonstrated
    assert result.selected_resolution == pytest.approx(0.5)
    assert result.attempts[1]["comparison"]["changed_owner_fraction"] == pytest.approx(0.0)
    fixed = surface_voronoi(network, boundary, resolution=0.5)
    assert result.surface.grid["cluster_id"].tolist() == fixed.grid["cluster_id"].tolist()
    assert result.surface.cells.geometry.to_wkb().tolist() == fixed.cells.geometry.to_wkb().tolist()


def test_one_grid_is_never_called_converged(line):
    network, boundary = line
    result = adaptive_surface_voronoi(network, boundary, start_resolution=1, max_refinements=0)
    assert result.status == "refinement_limit"
    assert result.attempts[0]["not_converged_because"] == ["first grid: nothing to compare with yet"]


def test_require_convergence_raises_with_the_result_attached(line):
    network, boundary = line
    with pytest.raises(ConvergenceError) as caught:
        adaptive_surface_voronoi(network, boundary, start_resolution=1, max_refinements=0, require_convergence=True)
    assert caught.value.result.status == "refinement_limit"
    assert "refinement_limit" in str(caught.value)


def test_cluster_lost_by_a_coarse_grid_is_recovered(line):
    network, boundary = line
    # A 20 m grid has one cell; its representative point lies on the border
    # at 5 m, the tie goes to A, and B has no land.
    result = adaptive_surface_voronoi(network, boundary, start_resolution=20, max_cells=10_000, max_refinements=4)
    assert result.attempts[0]["grid_lost_clusters"] == ["B"]
    assert result.status == "converged"
    assert set(result.surface.cells["cluster_id"]) == {"A", "B"}


def test_cluster_outside_the_boundary_does_not_block_convergence():
    # The road network extends beyond the study area, and cluster C lives
    # only outside it. C has no land inside at any spacing; that must not be
    # treated as a grid that is too coarse.
    network = _network(
        [LineString([(0, 0), (100, 0)]), LineString([(100, 0), (200, 0)])],
        [("A", (0, 0)), ("B", (90, 0)), ("C", (200, 0))],
    )
    result = adaptive_surface_voronoi(network, _area(box(0, -20, 100, 20)), start_resolution=5, max_cells=100_000)
    assert result.status == "converged"
    status = result.territories.set_index("cluster_id")["territory_status"]
    assert status.to_dict() == {"A": "present", "B": "present", "C": "no_road_inside"}


def test_territory_narrower_than_any_affordable_grid_is_reported_not_fatal():
    # A at 0 and 49.8, B at 50, C at 50.2: B owns only [49.9, 50.1], 0.2 m.
    network = _network(
        [LineString([(0, 0), (100, 0)])],
        [("A", (0, 0)), ("A", (49.8, 0)), ("B", (50, 0)), ("C", (50.2, 0)), ("D", (100, 0))],
    )
    boundary = _area(box(0, -20, 100, 20))
    result = adaptive_surface_voronoi(network, boundary, start_resolution=5, max_cells=100_000)
    assert result.status == "budget_limited"
    status = result.territories.set_index("cluster_id")["territory_status"]
    assert status["B"] == "grid_lost"
    assert "1 cluster(s) lost by the grid" in result.attempts[-2]["not_converged_because"]
    # With the territory rule relaxed, the same input converges.
    relaxed = adaptive_surface_voronoi(
        network, boundary, start_resolution=5, max_cells=100_000,
        convergence=ConvergencePolicy(require_all_territories=False),
    )
    assert relaxed.status == "converged"
    assert relaxed.territories.set_index("cluster_id").loc["B", "territory_status"] == "grid_lost"


def test_cluster_without_network_length_is_classified():
    # A and Z share the node at 0; the tie goes to A, so Z owns no length.
    network = _network([LineString([(0, 0), (10, 0)])], [("A", (0, 0)), ("Z", (0, 0)), ("B", (10, 0))])
    result = adaptive_surface_voronoi(network, _area(box(0, -1, 10, 1)), start_resolution=1)
    status = result.territories.set_index("cluster_id")["territory_status"]
    assert status["Z"] == "no_network_length"
    assert result.status == "converged"


def test_min_cluster_iou_is_enforced_only_when_set():
    # A at 0 and B at 10.6 on a 20 m road: the border is at 5.3 m, so it
    # moves between grids. Grids of 1 m and 0.5 m give A [0, 5] and [0, 5.5]
    # (IoU 10/11 = 0.909); grids of 0.5 m and 0.25 m give [0, 5.5] and
    # [0, 5.25] (IoU 10.5/11 = 0.955).
    network = _network([LineString([(0, 0), (20, 0)])], [("A", (0, 0)), ("B", (10.6, 0))])
    boundary = _area(box(0, -1, 20, 1))
    loose = ConvergencePolicy(global_changed_area_max=0.05)
    strict = ConvergencePolicy(global_changed_area_max=0.05, min_cluster_iou=0.95)
    first = adaptive_surface_voronoi(network, boundary, start_resolution=1, convergence=loose)
    second = adaptive_surface_voronoi(network, boundary, start_resolution=1, convergence=strict)
    assert (first.status, first.selected_resolution) == ("converged", 0.5)
    assert (second.status, second.selected_resolution) == ("converged", 0.25)
    assert any("IoU" in reason for reason in second.attempts[1]["not_converged_because"])
    assert second.attempts[1]["comparison"]["min_cluster_iou"] == pytest.approx(10 / 11)


def test_reachable_attachment_leaves_no_unassigned_land():
    network = _network(
        [LineString([(0, 0), (10, 0)]), LineString([(100, 0), (110, 0)])], [("A", (0, 0)), ("B", (10, 0))]
    )
    boundary = _area(box(0, -5, 110, 5))
    plain = adaptive_surface_voronoi(network, boundary, start_resolution=2, max_cells=100_000)
    reach = adaptive_surface_voronoi(
        network, boundary, start_resolution=2, max_cells=100_000, attachment="nearest_reachable_network"
    )
    assert not plain.surface.unassigned.empty
    assert reach.surface.unassigned.empty
    assert reach.manifest["attachment"] == "nearest_reachable_network"


def test_large_projected_coordinates_pass_the_geometry_checks():
    x0, y0 = 500_000.0, 1_600_000.0  # UTM-sized coordinates
    roads = [LineString([(x0, y0), (x0 + 300, y0)]), LineString([(x0 + 300, y0), (x0 + 300, y0 + 250)])]
    network = _network(roads, [("A", (x0, y0)), ("B", (x0 + 300, y0 + 250))], crs="EPSG:32651")
    shape = Polygon([(x0 - 7.3, y0 - 41.1), (x0 + 333.3, y0 - 13.7), (x0 + 311.9, y0 + 271.3), (x0 + 3.1, y0 + 80.9)])
    result = adaptive_surface_voronoi(
        network, gpd.GeoDataFrame(geometry=[shape], crs="EPSG:32651"), start_resolution=7.7, max_cells=200_000
    )
    assert result.status in ("converged", "budget_limited", "refinement_limit")
    assert all(a["uncovered_area"] <= 1e-9 * a["boundary_area"] for a in result.attempts if "clipped_cells" in a)


def test_manifest_is_valid_json(line):
    network, boundary = line
    for kwargs in ({"start_resolution": 1}, {"start_resolution": 0.001, "max_cells": 100}):
        result = adaptive_surface_voronoi(network, boundary, **kwargs)
        json.dumps(result.manifest, allow_nan=False)


# --------------------------------------------------------- failures and options


def test_a_geometry_error_stops_the_run(line, monkeypatch):
    network, boundary = line
    original = adaptive._render

    def broken_render(*args, **kwargs):
        good = original(*args, **kwargs)
        half = good.cells.iloc[:1].copy()  # drop cluster B: half the boundary is left uncovered
        return SurfaceVoronoiResult(half, good.grid, good.unassigned, good.point_qa, good.network, good.resolution)

    monkeypatch.setattr(adaptive, "_render", broken_render)
    with pytest.raises(CandidateGeometryError, match="coverage rule"):
        adaptive_surface_voronoi(network, boundary, start_resolution=1)


def test_unexpected_errors_are_not_caught(line, monkeypatch):
    network, boundary = line
    monkeypatch.setattr(adaptive, "_render", lambda *a, **k: (_ for _ in ()).throw(MemoryError("simulated")))
    with pytest.raises(MemoryError, match="simulated"):
        adaptive_surface_voronoi(network, boundary, start_resolution=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"refine_factor": 0}, {"refine_factor": 1}, {"refine_factor": float("nan")}, {"refine_factor": True},
        {"max_refinements": -1}, {"max_refinements": 1.5}, {"max_cells": 0}, {"max_cells": True},
        {"start_resolution": -1}, {"attachment": "nearest"}, {"convergence": {"global_changed_area_max": 0.1}},
        {"require_convergence": "yes"},
    ],
)
def test_options_are_checked_before_the_network_is_computed(line, monkeypatch, kwargs):
    network, boundary = line
    monkeypatch.setattr(core, "network_voronoi", lambda net: pytest.fail("network computed before option check"))
    with pytest.raises(ValueError):
        adaptive_surface_voronoi(network, boundary, **kwargs)


def test_policy_thresholds_are_checked():
    for bad in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError):
            ConvergencePolicy(global_changed_area_max=bad)
        with pytest.raises(ValueError):
            ConvergencePolicy(min_cluster_iou=bad)


# --------------------------------------------------------- ownership comparison


def _rendering(parts, unassigned=None):
    """A minimal SurfaceVoronoiResult from {cluster: geometry}, enough for compare_surface_ownership."""
    cells = gpd.GeoDataFrame({"cluster_id": list(parts)}, geometry=list(parts.values()), crs=CRS)
    pieces = list(parts.values()) + ([unassigned] if unassigned is not None else [])
    grid = gpd.GeoDataFrame({"cluster_id": [None] * len(pieces)}, geometry=pieces, crs=CRS)
    loose = gpd.GeoDataFrame(geometry=[] if unassigned is None else [unassigned], crs=CRS)
    return SurfaceVoronoiResult(cells, grid, loose, None, None)


LEFT, RIGHT = box(0, 0, 1, 1), box(1, 0, 2, 1)


def test_identical_renderings_have_no_change():
    result = compare_surface_ownership(_rendering({"A": LEFT, "B": RIGHT}), _rendering({"A": LEFT, "B": RIGHT}))
    assert result["changed_owner_fraction"] == 0.0
    assert result["min_cluster_iou"] == 1.0


def test_swapped_territories_count_as_full_change():
    # Both renderings cover the same boundary; only a per-category
    # comparison sees that every point changed owner.
    result = compare_surface_ownership(_rendering({"A": LEFT, "B": RIGHT}), _rendering({"A": RIGHT, "B": LEFT}))
    assert result["changed_owner_fraction"] == 1.0
    assert result["min_cluster_iou"] == 0.0


def test_unassigned_land_that_gains_an_owner_counts_as_change():
    result = compare_surface_ownership(_rendering({"A": LEFT}, unassigned=RIGHT), _rendering({"A": box(0, 0, 2, 1)}))
    assert result["changed_owner_fraction"] == pytest.approx(0.5)
    assert result["per_cluster"]["A"]["iou"] == pytest.approx(0.5)


def test_rearranged_pieces_with_equal_area_count_as_change():
    squares = [box(i, 0, i + 1, 1) for i in range(4)]
    before = _rendering({"A": squares[0].union(squares[2]), "B": squares[1].union(squares[3])})
    after = _rendering({"A": squares[0].union(squares[3]), "B": squares[1].union(squares[2])})
    result = compare_surface_ownership(before, after)
    assert result["changed_owner_fraction"] == pytest.approx(0.5)
    assert result["per_cluster"]["A"]["area_drift"] == 0.0
    assert result["per_cluster"]["A"]["iou"] == pytest.approx(1 / 3)


def test_comparison_refuses_different_boundaries():
    with pytest.raises(ValueError, match="same area"):
        compare_surface_ownership(_rendering({"A": LEFT}), _rendering({"A": box(0, 0, 2, 1)}))
