import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from netvoronoi import from_geodataframes, network_voronoi


def test_two_clusters_split_line_at_midpoint(line_roads, end_points):
    net = from_geodataframes(line_roads, end_points, cluster_col="cluster", point_id_col="pid")
    result = network_voronoi(net)
    assert list(result.segments.cluster_id) == ["A", "B"]
    assert list(result.segments.segment_dist) == pytest.approx([5.0, 5.0])
    assert result.segments.iloc[0].to_dist == pytest.approx(5.0)
    assert result.segments.iloc[1].from_dist == pytest.approx(5.0)


def test_cluster_distance_uses_nearest_member_not_cluster_center(line_roads, crs):
    points = gpd.GeoDataFrame(
        {"pid": ["a0", "a1", "b"], "cluster": ["A", "A", "B"]},
        geometry=[Point(0, 0), Point(10, 0), Point(6, 0)],
        crs=crs,
    )
    net = from_geodataframes(line_roads, points, cluster_col="cluster", point_id_col="pid")
    result = network_voronoi(net)
    # A owns near both ends; B owns around its own source at x=6.
    lengths = result.segments.groupby("cluster_id").segment_dist.sum().to_dict()
    assert lengths["A"] == pytest.approx(5.0)
    assert lengths["B"] == pytest.approx(5.0)
    b = result.segments[result.segments.cluster_id == "B"]
    assert len(b) == 2


def test_tie_uses_lexicographically_smallest_cluster(crs):
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs=crs)
    points = gpd.GeoDataFrame(
        {"cluster": ["Z", "A"]}, geometry=[Point(0, 0), Point(0, 0)], crs=crs
    )
    net = from_geodataframes(roads, points, cluster_col="cluster")
    result = network_voronoi(net)
    assert set(result.segments.cluster_id) == {"A"}


def test_disconnected_component_without_points_is_unassigned(crs):
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(100, 0), (110, 0)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(0, 0)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    result = network_voronoi(net)
    assert result.segments.segment_dist.sum() == pytest.approx(10.0)
    assert result.unassigned_edges.length.sum() == pytest.approx(10.0)


def test_every_reachable_edge_length_is_partitioned(line_roads, end_points):
    net = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    result = network_voronoi(net)
    assert result.segments.segment_dist.sum() == pytest.approx(net.edges.length.sum())


def test_cells_one_row_per_winning_cluster(line_roads, end_points):
    net = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    result = network_voronoi(net)
    assert list(result.cells.cluster_id) == ["A", "B"]
    assert result.cells.geometry.notna().all()


def test_each_source_node_is_owned_by_its_cluster_without_cross_cluster_colocation(line_roads, crs):
    points = gpd.GeoDataFrame(
        {"cluster": ["A", "A", "B"]},
        geometry=[Point(1, 0), Point(3, 0), Point(9, 0)],
        crs=crs,
    )
    net = from_geodataframes(line_roads, points, cluster_col="cluster")
    result = network_voronoi(net)
    for row in net.points_snapped.itertuples(index=False):
        assert result.node_cluster[int(row.network_node)] == row.cluster_id


def test_random_line_matches_direct_nearest_cluster(crs):
    rng = np.random.default_rng(42)
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=crs)
    for _ in range(30):
        xs = np.sort(rng.uniform(0, 100, size=8))
        labels = np.asarray(["A", "B", "C", "A", "B", "C", "A", "B"], dtype=object)
        rng.shuffle(labels)
        points = gpd.GeoDataFrame(
            {"cluster": labels}, geometry=[Point(float(x), 0) for x in xs], crs=crs
        )
        net = from_geodataframes(roads, points, cluster_col="cluster")
        result = network_voronoi(net)

        samples = np.linspace(0.001, 99.999, 301)
        direct = []
        for x in samples:
            by_cluster = {
                c: min(abs(x - px) for px, lab in zip(xs, labels) if lab == c)
                for c in sorted(set(labels))
            }
            m = min(by_cluster.values())
            direct.append(min(c for c, d in by_cluster.items() if abs(d - m) <= 1e-9))

        predicted = []
        for x in samples:
            hits = []
            for seg in result.segments.itertuples(index=False):
                minx, _, maxx, _ = seg.geometry.bounds
                if minx - 1e-9 <= x <= maxx + 1e-9:
                    hits.append(seg.cluster_id)
            assert hits
            predicted.append(min(hits))
        assert predicted == direct


def test_custom_curved_edge_is_cut_along_its_geometry(crs):
    from scipy.sparse import csr_matrix
    from netvoronoi.model import SpatialNetwork

    line = LineString([(0, 0), (5, 5), (10, 0)])
    length = float(line.length)
    nodes = gpd.GeoDataFrame(
        {"node_id": [0, 1]}, geometry=[Point(0, 0), Point(10, 0)], crs=crs
    )
    edges = gpd.GeoDataFrame(
        {"edge_id": ["e0"], "u": [0], "v": [1], "length": [length]},
        geometry=[line],
        crs=crs,
    )
    points_input = gpd.GeoDataFrame(
        {"point_id": ["a", "b"], "cluster_id": ["A", "B"]},
        geometry=[Point(0, 0), Point(10, 0)],
        crs=crs,
    )
    points_snapped = gpd.GeoDataFrame(
        {
            "point_id": ["a", "b"],
            "cluster_id": ["A", "B"],
            "network_node": [0, 1],
            "snap_distance": [0.0, 0.0],
            "snap_ties": [1, 1],
        },
        geometry=[Point(0, 0), Point(10, 0)],
        crs=crs,
    )
    adjacency = csr_matrix([[0.0, length], [length, 0.0]])
    net = SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)
    result = network_voronoi(net)

    assert len(result.segments) == 2
    assert result.segments.geometry.iloc[0].coords[-1] == pytest.approx((5.0, 5.0))
    assert result.segments.geometry.iloc[1].coords[0] == pytest.approx((5.0, 5.0))
    assert result.segments.geometry.length.sum() == pytest.approx(length)
