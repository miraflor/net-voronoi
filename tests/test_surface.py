import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box

from netvoronoi import from_geodataframes, surface_voronoi
from netvoronoi.surface import check_grid_size


def test_surface_covers_boundary(line_roads, end_points, boundary):
    net = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    result = surface_voronoi(net, boundary, resolution=1.0)
    assert set(result.cells.cluster_id) == {"A", "B"}
    assert result.grid.geometry.area.sum() == pytest.approx(boundary.geometry.area.sum())
    assert result.unassigned.empty


def test_surface_point_qa(line_roads, end_points, boundary):
    net = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    result = surface_voronoi(net, boundary, resolution=1.0)
    assert result.point_qa.contained.all()


def test_grid_size_guard(boundary, crs):
    with pytest.raises(ValueError, match="exceeding max_cells"):
        check_grid_size(boundary, crs, resolution=0.01, max_cells=100)


def test_surface_unassigned_for_nearest_empty_component(crs):
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(100, 0), (110, 0)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(0, 0)], crs=crs)
    boundary = gpd.GeoDataFrame(geometry=[box(95, -2, 115, 2)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    result = surface_voronoi(net, boundary, resolution=1.0)
    assert not result.unassigned.empty
    assert result.grid.cluster_id.isna().all()


# --- review regressions -----------------------------------------------------


def test_dissolved_surface_has_no_gaps_between_grid_cells(crs):
    # At UTM-sized coordinates, a square's right side computed as
    # ``left + resolution`` differed from its neighbour's left side in the
    # last bits, so dissolving left thin gaps and split the polygon.
    x0, y0 = 500000.123, 1600000.377
    roads = gpd.GeoDataFrame(geometry=[LineString([(x0, y0 + 50), (x0 + 1000, y0 + 50)])], crs=crs)
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(x0, y0 + 50)], crs=crs)
    area = box(x0, y0, x0 + 997.3, y0 + 101.9)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    result = surface_voronoi(net, gpd.GeoDataFrame(geometry=[area], crs=crs), resolution=7.7)
    cell = result.cells.geometry.iloc[0]
    assert cell.geom_type == "Polygon"
    assert len(cell.interiors) == 0
    assert cell.area == pytest.approx(area.area, rel=1e-12)


def test_cluster_and_unassigned_surfaces_fill_a_curved_boundary_without_gaps(crs):
    import numpy as np
    import shapely

    x0, y0 = 500000.123, 1600000.377
    rng = np.random.default_rng(1)
    roads = gpd.GeoDataFrame(
        geometry=[LineString(np.c_[x0 + rng.uniform(0, 1000, 3), y0 + rng.uniform(0, 1000, 3)]) for _ in range(12)],
        crs=crs,
    )
    points = gpd.GeoDataFrame(
        {"cluster": rng.choice(["A", "B", "C"], 30)},
        geometry=[Point(x0 + a, y0 + b) for a, b in rng.uniform(0, 1000, (30, 2))],
        crs=crs,
    )
    area = Point(x0 + 500, y0 + 500).buffer(480, 64)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    result = surface_voronoi(net, gpd.GeoDataFrame(geometry=[area], crs=crs), resolution=7.7)
    # Some roads carry no point, so some cells are unassigned; together with
    # the cluster polygons they must fill the boundary exactly.
    assert not result.unassigned.empty
    pieces = np.r_[result.cells.geometry.to_numpy(), result.unassigned.geometry.to_numpy()]
    union = shapely.union_all(pieces)
    assert union.geom_type == "Polygon"
    assert len(union.interiors) == 0
    assert union.area == pytest.approx(area.area, rel=1e-9)
    assert sum(shapely.area(pieces)) == pytest.approx(area.area, rel=1e-9)  # no overlaps


def test_anchor_ties_is_one_at_a_shared_road_vertex(crs):
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0), (10, 10)])], crs=crs)
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(0, 0)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    one_cell = gpd.GeoDataFrame(geometry=[box(10.5, -1.5, 11.5, -0.5)], crs=crs)  # centre (11, -1)
    result = surface_voronoi(net, one_cell, resolution=1.0)
    assert list(result.grid.anchor_ties) == [1]


def test_anchor_ties_is_two_at_a_crossing_without_a_shared_vertex(crs):
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(5, -5), (5, 5)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A", "B"]}, geometry=[Point(0, 0), Point(5, 5)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    one_cell = gpd.GeoDataFrame(geometry=[box(4.5, -0.5, 5.5, 0.5)], crs=crs)  # centre (5, 0)
    result = surface_voronoi(net, one_cell, resolution=1.0)
    assert list(result.grid.anchor_ties) == [2]


def test_every_grid_cell_takes_the_network_cluster_at_its_anchor(crs):
    import numpy as np

    rng = np.random.default_rng(7)
    pool = rng.uniform(0, 300, (15, 2))
    roads = gpd.GeoDataFrame(
        geometry=[LineString(pool[rng.choice(15, 3, replace=False)]) for _ in range(10)], crs=crs
    )
    points = gpd.GeoDataFrame(
        {"cluster": rng.choice(["A", "B", "C", "D"], 12)},
        geometry=[Point(p) for p in rng.uniform(0, 300, (12, 2))],
        crs=crs,
    )
    net = from_geodataframes(roads, points, cluster_col="cluster")
    result = surface_voronoi(net, gpd.GeoDataFrame(geometry=[box(0, 0, 300, 300)], crs=crs), resolution=5.0)
    edges = net.edges.set_index("edge_id")
    pieces = {k: g for k, g in result.network.segments.groupby("edge_id")}
    checked = 0
    for cell in result.grid.itertuples(index=False):
        edge = edges.loc[cell.edge_id]
        if cell.edge_offset == 0.0 or cell.edge_offset == edge["length"]:
            node = int(edge["u"] if cell.edge_offset == 0.0 else edge["v"])
            assert cell.cluster_id == result.network.node_cluster[node]
            continue
        if cell.edge_id not in pieces:
            assert cell.cluster_id is None
            continue
        piece = pieces[cell.edge_id]
        inside = piece[(piece.from_dist + 1e-6 < cell.edge_offset) & (cell.edge_offset < piece.to_dist - 1e-6)]
        if len(inside) == 1:
            assert cell.cluster_id == inside.cluster_id.iloc[0]
            checked += 1
    assert checked > 1000


def test_invalid_boundary_is_rejected_cleanly(crs):
    from shapely.geometry import Polygon

    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    with pytest.raises(ValueError, match="boundary geometry is invalid"):
        check_grid_size(bowtie, crs, resolution=1.0)


def test_nonfinite_max_cells_is_rejected(boundary, crs):
    with pytest.raises(ValueError, match="positive integer or None"):
        check_grid_size(boundary, crs, resolution=1.0, max_cells=float("inf"))
