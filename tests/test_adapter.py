import importlib.util
import sys
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from netvoronoi.core import network_voronoi
from netvoronoi.spaghetti_backend import (
    _can_skip_component_labelling,
    _convert_spaghetti_network,
    _place_sites_on_arc,
    _prepare_sites,
    from_geodataframes,
)

# Tests marked with this run the real spaghetti package. They are skipped
# where spaghetti is not installed; the other tests do not need it.
live = pytest.mark.skipif(
    importlib.util.find_spec("spaghetti") is None, reason="spaghetti is not installed"
)
CRS = "EPSG:32651"


def test_component_skip_fast_path_is_guarded_by_exact_spaghetti_version():
    assert _can_skip_component_labelling(SimpleNamespace(__version__="1.7.6"))
    assert not _can_skip_component_labelling(SimpleNamespace(__version__="1.7.7"))
    assert not _can_skip_component_labelling(SimpleNamespace(__version__="1.8.0"))
    assert not _can_skip_component_labelling(SimpleNamespace())


# ---------------------------------------------------------------------------
# The placement rule, on its own
# ---------------------------------------------------------------------------


def test_placement_worked_example_from_the_docstring():
    slot, new_offsets = _place_sites_on_arc(np.array([50.000, 50.009, 50.012]), 100.0, 0.01)
    assert slot.tolist() == [1, 1, 2]  # first two sites share a node
    assert np.allclose(new_offsets, [50.000, 50.012])  # 0.012 apart: more than 0.01


def test_placement_prefers_the_nearest_node_and_breaks_ties_backwards():
    # Arc of 0.005 with tolerance 0.01: both vertices are within tolerance.
    slot, new_offsets = _place_sites_on_arc(np.array([0.004, 0.001, 0.0025]), 0.005, 0.01)
    assert new_offsets.size == 0
    assert slot.tolist() == [1, 0, 0]  # nearer end; the exact middle goes back to vertex 0
    # Far from both ends: a new node.
    slot, new_offsets = _place_sites_on_arc(np.array([0.02]), 100.0, 0.01)
    assert slot.tolist() == [1] and new_offsets.tolist() == [0.02]


def test_placement_guarantees_on_random_input():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        length = float(rng.choice([0.004, 1.0, 100.0]))
        tolerance = float(rng.choice([1e-8, 1e-3, 0.01, 0.5]))
        offsets = rng.uniform(0, length, int(rng.integers(1, 12)))
        if rng.random() < 0.3:
            offsets[0] = offsets[-1]  # an exact duplicate
        slot, new_offsets = _place_sites_on_arc(offsets, length, tolerance)
        node_offsets = np.concatenate([[0.0], new_offsets, [length]])
        # Every site moves at most the tolerance.
        assert np.all(np.abs(node_offsets[slot] - offsets) <= tolerance)
        # Nodes are ordered along the arc, and new nodes keep more than the
        # tolerance from their neighbours, so every piece has positive length.
        gaps = np.diff(node_offsets)
        assert np.all(gaps > 0)
        if new_offsets.size:
            assert np.all(gaps > tolerance)
        # Sites at the same position share a node.
        assert len(set(slot[offsets == offsets[0]].tolist())) == 1


# ---------------------------------------------------------------------------
# Conversion of a spaghetti-like object (no spaghetti needed)
# ---------------------------------------------------------------------------


def _fake_network(dist_snapped=(5.0, 3.0, 4.0)):
    point_pattern = SimpleNamespace(
        obs_to_arc={(0, 1): {0: (20.0, 0.0), 1: (20.0, 0.0), 2: (80.0, 0.0)}},
        dist_to_vertex={
            0: {0: 20.0, 1: 80.0},
            1: {0: 20.0, 1: 80.0},
            2: {0: 80.0, 1: 20.0},
        },
        dist_snapped={i: value for i, value in enumerate(dist_snapped)},
    )
    return SimpleNamespace(
        pointpatterns={"sites": point_pattern},
        vertex_coords={0: (0.0, 0.0), 1: (100.0, 0.0)},
        arcs=[(0, 1)],
        arc_lengths={(0, 1): 100.0},
    )


def test_conversion_splits_sites_exactly_and_merges_colocated_sites():
    sites = gpd.GeoDataFrame(
        geometry=[Point(20, 5), Point(20, 3), Point(80, 4)], crs=CRS
    )
    network = _convert_spaghetti_network(
        _fake_network(),
        roads_crs=sites.crs,
        sites_work=sites,
        site_ids=np.asarray(["A", "B", "C"], dtype=object),
        snap_tolerance=1e-8,
        max_snap_distance=None,
    )

    assert np.allclose(network.edges.length.to_numpy(), [20.0, 60.0, 20.0])
    assert network.site_nodes[0] == network.site_nodes[1]
    assert network.site_nodes[2] != network.site_nodes[0]
    assert network.distance_unit == "metre"
    assert list(network.sites_snapped["snap_distance"]) == [5.0, 3.0, 4.0]


def test_conversion_enforces_max_snap_distance():
    sites = gpd.GeoDataFrame(
        geometry=[Point(20, 5), Point(20, 3), Point(80, 4)], crs=CRS
    )
    with pytest.raises(ValueError, match="max_snap_distance"):
        _convert_spaghetti_network(
            _fake_network(),
            roads_crs=sites.crs,
            sites_work=sites,
            site_ids=np.asarray(["A", "B", "C"], dtype=object),
            snap_tolerance=1e-8,
            max_snap_distance=4.0,
        )


def test_conversion_skips_self_arc_and_keeps_its_site_on_the_vertex():
    # spaghetti makes arc (1, 1) from a repeated vertex. A site snapped to it
    # belongs to vertex 1.
    point_pattern = SimpleNamespace(
        obs_to_arc={(1, 1): {0: (100.0, 0.0)}},
        dist_to_vertex={0: {1: 0.0}},
        dist_snapped={0: 2.0},
    )
    ntw = SimpleNamespace(
        pointpatterns={"sites": point_pattern},
        vertex_coords={0: (0.0, 0.0), 1: (100.0, 0.0)},
        arcs=[(1, 1), (0, 1)],
    )
    sites = gpd.GeoDataFrame(geometry=[Point(100, 2)], crs=CRS)
    network = _convert_spaghetti_network(
        ntw,
        roads_crs=sites.crs,
        sites_work=sites,
        site_ids=np.asarray(["A"], dtype=object),
        snap_tolerance=1e-8,
        max_snap_distance=None,
    )
    assert network.site_nodes.tolist() == [1]
    assert network.edges[["u", "v"]].to_numpy().tolist() == [[0, 1]]


# ---------------------------------------------------------------------------
# Input checks (no spaghetti needed: they run before spaghetti is used)
# ---------------------------------------------------------------------------


def test_adapter_rejects_missing_site_geometry(monkeypatch):
    monkeypatch.setitem(sys.modules, "spaghetti", SimpleNamespace())
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs=CRS)
    sites = gpd.GeoDataFrame(geometry=[Point(1, 0), None], crs=CRS)
    with pytest.raises(ValueError, match="null/empty"):
        from_geodataframes(roads, sites)


@pytest.mark.parametrize(
    "keywords",
    [
        {"snap_tolerance": 0.0},
        {"snap_tolerance": -1.0},
        {"snap_tolerance": float("nan")},
        {"snap_tolerance": float("inf")},
        {"max_snap_distance": -1.0},
        {"max_snap_distance": float("nan")},
    ],
)
def test_adapter_rejects_invalid_numeric_parameters(monkeypatch, keywords):
    monkeypatch.setitem(sys.modules, "spaghetti", SimpleNamespace())
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs=CRS)
    sites = gpd.GeoDataFrame(geometry=[Point(1, 0)], crs=CRS)
    with pytest.raises(ValueError, match=next(iter(keywords))):
        from_geodataframes(roads, sites, **keywords)


def test_adapter_rejects_non_finite_coordinates(monkeypatch):
    monkeypatch.setitem(sys.modules, "spaghetti", SimpleNamespace())
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (float("nan"), 5)])], crs=CRS)
    sites = gpd.GeoDataFrame(geometry=[Point(1, 0)], crs=CRS)
    with pytest.raises(ValueError, match="NaN or infinite"):
        from_geodataframes(roads, sites)


def test_geographic_sites_are_reprojected_to_projected_road_crs():
    sites = gpd.GeoDataFrame(geometry=[Point(121.05, 14.65)], crs="EPSG:4326")
    work, site_ids = _prepare_sites(sites, CRS, site_id_col=None)
    assert work.crs.to_epsg() == 32651
    assert site_ids.tolist() == ["0"]


# ---------------------------------------------------------------------------
# Real spaghetti runs: one regression test per defect found in v0.3.0
# ---------------------------------------------------------------------------


@live
def test_full_precision_utm_coordinates_are_accepted():
    # v0.3.0 raised "arc length disagrees with its geometry" here, because
    # spaghetti rounds vertices but computes arc_lengths partly unrounded.
    x0, y0 = 285_432.123456789, 1_623_456.987654321
    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(x0, y0), (x0 + 400.3333333, y0), (x0 + 1000.7777777, y0 + 0.4444444)]),
            LineString([(x0 + 400.3333333, y0), (x0 + 400.3333333, y0 + 600.1212121)]),
        ],
        crs=CRS,
    )
    sites = gpd.GeoDataFrame(
        {"sid": ["A", "B", "C"]},
        geometry=[Point(x0 + 100.5, y0 + 7.0), Point(x0 + 900.25, y0 - 5.0), Point(x0 + 395.0, y0 + 450.0)],
        crs=CRS,
    )
    network = from_geodataframes(roads, sites, site_id_col="sid")
    # Vertex rounding (at most 0.05 mm per coordinate here) changes the
    # total length only in the fifth decimal place.
    assert np.isclose(network.edges.length.sum(), roads.length.sum(), atol=1e-3)
    result = network_voronoi(network, epsilon=50)
    assert np.isclose(result.hard_segments.segment_dist.sum(), network.edges.length.sum())


@live
def test_repeated_vertex_does_not_stop_the_build():
    # v0.3.0 raised "invalid arc length" for spaghetti's zero-length arc (1, 1).
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (50, 0), (50, 0), (100, 0)])], crs=CRS)
    sites = gpd.GeoDataFrame(geometry=[Point(10, 3), Point(90, -2)], crs=CRS)
    network = from_geodataframes(roads, sites)
    assert np.isclose(network.edges.length.sum(), 100.0)


@live
def test_snap_tolerance_never_removes_road():
    # v0.3.0 dropped the 5 mm piece when snap_tolerance was 0.01, which cut
    # the road in two and left half of it unassigned.
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0), (100.005, 0), (200, 0)])], crs=CRS
    )
    sites = gpd.GeoDataFrame({"sid": ["A"]}, geometry=[Point(10, 1)], crs=CRS)
    network = from_geodataframes(roads, sites, site_id_col="sid", snap_tolerance=0.01)
    result = network_voronoi(network)
    assert np.isclose(network.edges.length.sum(), 200.0)
    assert np.isclose(result.hard_segments.segment_dist.sum(), 200.0)
    assert result.unassigned_edges.empty


@live
def test_close_sites_keep_the_network_connected():
    # v0.3.0 placed two merged-site nodes 0.0075 apart and then dropped the
    # piece between them, which disconnected the road.
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=CRS)
    sites = gpd.GeoDataFrame(
        {"sid": ["A", "B", "C"]},
        geometry=[Point(50.000, 1), Point(50.009, 1), Point(50.012, 1)],
        crs=CRS,
    )
    network = from_geodataframes(roads, sites, site_id_col="sid", snap_tolerance=0.01)
    nodes = network.site_nodes.tolist()
    assert nodes[0] == nodes[1] != nodes[2]
    result = network_voronoi(network)
    assert np.isclose(network.edges.length.sum(), 100.0)
    assert np.isclose(result.hard_segments.segment_dist.sum(), 100.0)
    assert result.unassigned_edges.empty


@live
def test_multilinestring_parts_are_not_joined():
    # Without splitting, spaghetti would add a false 40-unit arc from x=10 to x=50.
    roads = gpd.GeoDataFrame(
        geometry=[MultiLineString([[(0, 0), (10, 0)], [(50, 0), (60, 0)]])], crs=CRS
    )
    sites = gpd.GeoDataFrame(geometry=[Point(5, 1)], crs=CRS)
    network = from_geodataframes(roads, sites)
    assert np.isclose(network.edges.length.sum(), 20.0)
    result = network_voronoi(network)
    assert np.isclose(result.unassigned_edges.length.sum(), 10.0)


@live
def test_skipping_component_labelling_changes_nothing():
    # The adapter builds spaghetti's network without its component labelling
    # (``w_components=False``), because the labelling is slow and unused. This
    # builds the same input the old way, with the labelling, and compares.
    import spaghetti

    from netvoronoi.spaghetti_backend import _prepare_roads

    rng = np.random.default_rng(8)
    lines = [LineString(rng.uniform(0, 500, (int(rng.integers(2, 5)), 2))) for _ in range(12)]
    lines += [LineString([(0, 250), (500, 250)]), LineString([(250, 0), (250, 500)])]
    roads = gpd.GeoDataFrame(geometry=lines, crs=CRS)  # several separate components
    sites = gpd.GeoDataFrame(
        {"sid": [f"S{i}" for i in range(6)]},
        geometry=[Point(xy) for xy in rng.uniform(0, 500, (6, 2))],
        crs=CRS,
    )
    fast = from_geodataframes(roads, sites, site_id_col="sid")

    roads_work = _prepare_roads(roads)
    sites_work, site_ids = _prepare_sites(sites, roads_work.crs, "sid")
    labelled = spaghetti.Network(
        in_data=roads_work, unique_arcs=True, extractgraph=False, w_components=True
    )
    labelled.snapobservations(sites_work, "sites", attribute=False)
    reference = _convert_spaghetti_network(
        labelled,
        roads_crs=roads_work.crs,
        sites_work=sites_work,
        site_ids=site_ids,
        snap_tolerance=1e-8,
        max_snap_distance=None,
    )

    assert np.array_equal(fast.vertices_xy, reference.vertices_xy)
    assert fast.edges.drop(columns="geometry").equals(reference.edges.drop(columns="geometry"))
    assert fast.edges.geometry.geom_equals_exact(reference.edges.geometry, 0).all()
    assert np.array_equal(fast.site_nodes, reference.site_nodes)
    assert fast.sites_snapped.drop(columns="geometry").equals(
        reference.sites_snapped.drop(columns="geometry")
    )


@live
def test_other_spaghetti_versions_get_the_same_network(monkeypatch):
    # Any spaghetti release other than 1.7.6 takes the component-labelled path.
    import netvoronoi.spaghetti_backend as backend

    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (100, 0), (100, 0), (200, 0)]),
            LineString([(100, -50), (100, 50)]),
        ],
        crs=CRS,
    )
    sites = gpd.GeoDataFrame(
        geometry=[Point(20, 3), Point(150, -4), Point(100, 40)],
        crs=CRS,
    )

    fast = from_geodataframes(roads, sites)

    monkeypatch.setattr(
        backend,
        "_can_skip_component_labelling",
        lambda module: False,
    )
    labelled = from_geodataframes(roads, sites)

    assert labelled.edges.drop(columns="geometry").equals(
        fast.edges.drop(columns="geometry")
    )
    assert np.array_equal(labelled.site_nodes, fast.site_nodes)

    assert labelled.edges.geometry.geom_equals_exact(fast.edges.geometry, 0).all()

    assert labelled.sites_snapped.drop(columns="geometry").equals(
        fast.sites_snapped.drop(columns="geometry")
    )
    assert labelled.sites_snapped.geometry.geom_equals_exact(
        fast.sites_snapped.geometry, 0
    ).all()    