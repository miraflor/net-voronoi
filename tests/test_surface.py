import numpy as np
import pytest
from shapely.geometry import box

from netvoronoi.surface import surface_voronoi


def test_surface_has_overlap_near_network_midpoint(line_network):
    result = surface_voronoi(
        line_network,
        boundary=box(0, -50, 1000, 50),
        epsilon=100,
        resolution=50,
    )
    assert set(result.hard.site_id.astype(str)) == {"A", "B"}
    assert set(result.epsilon.site_id.astype(str)) == {"A", "B"}
    assert result.grid["n_epsilon"].max() == 2
    assert result.unassigned.empty
    assert {"edge_offset", "nearest_site_dist", "access_dist"}.issubset(result.grid.columns)
    assert "network_dist" not in result.grid.columns


def test_surface_equal_anchor_tie_is_deterministic_and_reported(make_network):
    # Two disconnected crossing links. The cell representative at the crossing
    # has two equally near anchors; edge_id 0 is the documented tie break.
    network = make_network(
        [(-10, 0), (10, 0), (0, -10), (0, 10)],
        [(0, 1), (2, 3)],
        ["A", "B"],
        [0, 2],
    )
    result = surface_voronoi(network, box(-1, -1, 1, 1), resolution=2)
    assert int(result.grid.iloc[0]["anchor_ties"]) == 2
    assert str(result.grid.iloc[0]["hard_site"]) == "A"


def test_surface_reports_actual_nearest_site_distance(line_network):
    result = surface_voronoi(
        line_network,
        boundary=box(0, -50, 50, 0),
        resolution=50,
    )
    row = result.grid.iloc[0]
    # Representative point is (25, -25), anchored at x=25 on the first edge.
    assert np.isclose(float(row["edge_offset"]), 25.0)
    assert np.isclose(float(row["nearest_site_dist"]), 175.0)
    assert np.isclose(float(row["access_dist"]), 25.0)


def test_surface_rejects_a_resolution_that_is_not_a_positive_number(line_network):
    for bad in (0.0, -10.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="resolution"):
            surface_voronoi(line_network, box(0, -50, 1000, 50), resolution=bad)


def test_unassigned_layer_carries_geometry_only(make_network):
    # Component 2 has no site, so the cells over it are unassigned. The layer
    # is one dissolved polygon; v0.3.0 also kept a misleading cell_id column
    # holding the number of the first cell only.
    network = make_network(
        [(0, 0), (100, 0), (300, 0), (400, 0)],
        [(0, 1), (2, 3)],
        ["A"],
        [0],
    )
    result = surface_voronoi(network, box(0, -50, 400, 50), resolution=50)
    assert list(result.unassigned.columns) == ["geometry"]
    assert len(result.unassigned) == 1
    assert result.grid["hard_site"].isna().sum() > 0  # the same cells are listed in the grid


def _reference_grid(network, result, cells):
    """Direct per-cell reading of the exact result, used to check the fast path.

    This is the straightforward way to do the same work: for every cell, take
    its representative point, look for the nearest edge by testing all edges,
    and read the memberships from the segment tables one row at a time.
    """
    hard_site, n_epsilon, offsets = [], [], []
    for cell in cells:
        point = cell.representative_point()
        distances = np.array([point.distance(line) for line in network.edges.geometry])
        near = np.flatnonzero(distances == distances.min())
        position = min(near.tolist(), key=lambda i: str(network.edges.iloc[i]["edge_id"]))
        edge = network.edges.iloc[position]
        offset = float(edge.length) * edge.geometry.project(point) / edge.geometry.length
        offsets.append(offset)

        members = {"hard": [], "epsilon": []}
        for name, segments in (("hard", result.hard_segments), ("epsilon", result.epsilon_segments)):
            for row in segments.itertuples(index=False):
                slack = max(1e-9, 1e-12 * max(1.0, abs(offset)))
                if row.edge_id == edge.edge_id and row.from_dist - slack <= offset <= row.to_dist + slack:
                    members[name].append(str(row.site_id))
        hard_site.append(min(members["hard"]) if members["hard"] else None)
        n_epsilon.append(len(set(members["epsilon"])))
    return hard_site, n_epsilon, offsets


def test_surface_matches_a_direct_per_cell_reading(make_network):
    rng = np.random.default_rng(5)
    for _ in range(5):
        xy = np.round(rng.uniform(0, 400, (9, 2)), 1)
        pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (0, 4), (2, 6)]
        network = make_network(xy, pairs, ["A", "B", "C"], rng.choice(9, 3, replace=False).tolist())
        boundary = box(-20, -20, 420, 420).difference(box(150, 150, 260, 260))
        result = surface_voronoi(network, boundary, epsilon=40.0, resolution=70.0)

        hard_site, n_epsilon, offsets = _reference_grid(
            network, result.network, result.grid.geometry.to_list()
        )
        assert result.grid["hard_site"].where(result.grid["hard_site"].notna(), None).to_list() == hard_site
        assert result.grid["n_epsilon"].to_list() == n_epsilon
        assert np.allclose(result.grid["edge_offset"].to_numpy(), offsets)


def test_surface_grid_guard_runs_before_large_allocation(line_network, monkeypatch):
    # 100 x 10 candidate squares at resolution 10 over this bounding box.
    # The guard should fire before even the exact network calculation starts.
    import netvoronoi.surface as surface_module

    def should_not_run(*args, **kwargs):
        raise AssertionError("network_voronoi ran before the surface grid preflight")

    monkeypatch.setattr(surface_module, "network_voronoi", should_not_run)
    with pytest.raises(ValueError, match="candidate cells"):
        surface_module.surface_voronoi(
            line_network,
            box(0, -50, 1000, 50),
            resolution=10,
            max_cells=999,
        )

    # Restore the real function before checking that the guard can be disabled.
    monkeypatch.undo()

    # A caller can deliberately raise or disable the guard.
    result = surface_voronoi(
        line_network,
        box(0, -50, 1000, 50),
        resolution=50,
        max_cells=None,
    )
    assert not result.grid.empty


@pytest.mark.parametrize("bad", [0, -1, 1.5, float("nan"), True])
def test_surface_max_cells_must_be_positive_integer_or_none(line_network, bad):
    with pytest.raises(ValueError, match="max_cells"):
        surface_voronoi(
            line_network,
            box(0, -50, 1000, 50),
            resolution=50,
            max_cells=bad,
        )


def test_surface_reports_a_bad_epsilon_before_building_the_grid(line_network, monkeypatch):
    # v0.4.1 built the whole grid first and only then noticed epsilon=nan.
    import netvoronoi.surface as surface_module

    def should_not_run(*args, **kwargs):
        raise AssertionError("the grid was built before the arguments were checked")

    monkeypatch.setattr(surface_module, "_grid_cells", should_not_run)
    with pytest.raises(ValueError, match="epsilon"):
        surface_module.surface_voronoi(
            line_network, box(0, -50, 1000, 50), epsilon=float("nan"), resolution=50
        )


def test_check_grid_size_counts_candidate_squares():
    import geopandas as gpd

    from netvoronoi.surface import check_grid_size

    # x from 0 to 1000 and y from -50 to 50 at resolution 10: 100 columns x 10 rows.
    assert check_grid_size(box(0, -50, 1000, 50), "EPSG:32651", 10) == 1000
    with pytest.raises(ValueError, match="100 columns x 10 rows"):
        check_grid_size(box(0, -50, 1000, 50), "EPSG:32651", 10, max_cells=999)
    assert check_grid_size(box(0, -50, 1000, 50), "EPSG:32651", 10, max_cells=None) == 1000

    # A GeoPandas boundary is measured after reprojection to the network CRS.
    lonlat = gpd.GeoSeries([box(121.00, 14.60, 121.01, 14.61)], crs="EPSG:4326")
    count = check_grid_size(lonlat, "EPSG:32651", 100)
    assert 100 <= count <= 169  # about 1.1 km x 1.1 km in metres, at 100 m

    with pytest.raises(ValueError, match="CRS is missing"):
        check_grid_size(lonlat, None, 100)
