import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from netvoronoi import from_geodataframes


def test_inserts_interior_point_as_node(line_roads, crs):
    points = gpd.GeoDataFrame(
        {"cluster": ["A"]}, geometry=[Point(5, 1)], crs=crs
    )
    net = from_geodataframes(line_roads, points, cluster_col="cluster")
    assert len(net.edges) == 2
    assert len(net.nodes) == 3
    assert net.points_snapped.iloc[0].geometry.equals(Point(5, 0))
    assert net.points_snapped.iloc[0].snap_distance == pytest.approx(1.0)
    interior = int(net.points_snapped.iloc[0].network_node)
    assert net.nodes.iloc[interior].geometry.equals(Point(5, 0))


def test_multiple_points_same_location_share_node(line_roads, crs):
    points = gpd.GeoDataFrame(
        {"cluster": ["A", "B"]}, geometry=[Point(5, 0), Point(5, 0)], crs=crs
    )
    net = from_geodataframes(line_roads, points, cluster_col="cluster")
    assert net.points_snapped.network_node.nunique() == 1
    assert len(net.edges) == 2


def test_multilinestring_is_supported(crs):
    roads = gpd.GeoDataFrame(
        geometry=[MultiLineString([[(0, 0), (5, 0)], [(5, 0), (10, 0)]])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(0, 0)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    assert len(net.edges) == 2


def test_duplicate_segment_removed(crs):
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (0, 0)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(0, 0)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    assert len(net.edges) == 1


def test_crossing_without_shared_vertex_stays_disconnected(crs):
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(5, -5), (5, 5)])], crs=crs
    )
    points = gpd.GeoDataFrame(
        {"cluster": ["A", "B"]}, geometry=[Point(0, 0), Point(5, 5)], crs=crs
    )
    net = from_geodataframes(roads, points, cluster_col="cluster")
    # Each road remains a two-endpoint component; no node at the mere crossing.
    assert len(net.nodes) == 4


def test_max_snap_distance_guard(line_roads, crs):
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(5, 2)], crs=crs)
    with pytest.raises(ValueError, match="exceed max_snap_distance"):
        from_geodataframes(line_roads, points, cluster_col="cluster", max_snap_distance=1)


def test_point_ids_must_be_unique(line_roads, crs):
    points = gpd.GeoDataFrame(
        {"pid": [1, 1], "cluster": ["A", "A"]}, geometry=[Point(0, 0), Point(10, 0)], crs=crs
    )
    with pytest.raises(ValueError, match="point IDs must be unique"):
        from_geodataframes(line_roads, points, cluster_col="cluster", point_id_col="pid")


def test_geographic_roads_rejected(end_points):
    roads = gpd.GeoDataFrame(geometry=[LineString([(120, 14), (120.1, 14)])], crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected"):
        from_geodataframes(roads, end_points.to_crs("EPSG:4326"), cluster_col="cluster")


def test_cluster_labels_that_collapse_to_same_string_are_rejected(line_roads, crs):
    points = gpd.GeoDataFrame(
        {"cluster": [1, "1"]}, geometry=[Point(0, 0), Point(10, 0)], crs=crs
    )
    with pytest.raises(ValueError, match="collapse to the same string"):
        from_geodataframes(line_roads, points, cluster_col="cluster")


def test_snap_ties_are_exposed_at_ambiguous_crossing(crs):
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(5, -5), (5, 5)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(5, 0)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    assert int(net.points_snapped.iloc[0].snap_ties) == 2


@pytest.mark.filterwarnings("ignore:invalid value encountered in linestrings:RuntimeWarning")
def test_nonfinite_road_coordinate_rejected(crs):
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (float("nan"), 1)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(0, 0)], crs=crs)
    with pytest.raises(ValueError, match="road coordinates must be finite"):
        from_geodataframes(roads, points, cluster_col="cluster")


def test_nonfinite_point_coordinate_rejected(line_roads, crs):
    points = gpd.GeoDataFrame(
        {"cluster": ["A"]}, geometry=[Point(float("nan"), 0)], crs=crs
    )
    with pytest.raises(ValueError, match="point coordinates must be finite"):
        from_geodataframes(line_roads, points, cluster_col="cluster")


# --- review regressions -----------------------------------------------------


def test_segment_shorter_than_tolerance_is_kept(crs):
    # A 5e-10 m segment between two longer ones used to be dropped, which
    # disconnected the road and left the far half unassigned.
    x = 500_000.0
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(x, 0), (x + 100, 0), (x + 100 + 5e-10, 0), (x + 200, 0)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(x, 0)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    assert len(net.edges) == 3
    assert net.edges.length.min() < 1e-9

    from netvoronoi import network_voronoi

    result = network_voronoi(net)
    assert result.unassigned_edges.empty
    assert result.segments.segment_dist.sum() == pytest.approx(200.0)


def test_inserting_points_never_removes_road_length(crs):
    rng = np.random.default_rng(3)
    x = 500_000.0
    lines = []
    for _ in range(20):
        xy = x + rng.uniform(0, 1000, (4, 2))
        xy[2] = xy[1] + rng.uniform(-1e-9, 1e-9, 2)  # a segment shorter than the tolerance
        lines.append(LineString(xy))
    roads = gpd.GeoDataFrame(geometry=lines, crs=crs)
    points = gpd.GeoDataFrame(
        {"cluster": rng.choice(["A", "B", "C"], 60)},
        geometry=[Point(p) for p in x + rng.uniform(0, 1000, (60, 2))],
        crs=crs,
    )
    net = from_geodataframes(roads, points, cluster_col="cluster")
    # total length of the distinct input segments (repeated vertices and
    # repeated segments are not road length)
    pairs = set()
    for line in lines:
        c = [tuple(p) for p in line.coords]
        pairs |= {tuple(sorted((a, b))) for a, b in zip(c[:-1], c[1:]) if a != b}
    expected = sum(float(np.hypot(a[0] - b[0], a[1] - b[1])) for a, b in pairs)
    assert net.edges["source_edge"].nunique() == len(pairs)  # no segment is missing
    assert net.edges.length.sum() == pytest.approx(expected, rel=1e-12)


def test_snap_ties_is_one_at_an_ordinary_bend(crs):
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0), (10, 10)])], crs=crs)
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(11, -1)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    assert int(net.points_snapped.iloc[0].snap_ties) == 1


def test_snap_ties_is_one_at_a_junction_of_three_roads(crs):
    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
            LineString([(10, 0), (10, 10)]),
        ],
        crs=crs,
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(10, -1)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    assert int(net.points_snapped.iloc[0].snap_ties) == 1


def test_snap_ties_is_two_between_parallel_roads(crs):
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(0, 2), (10, 2)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(5, 1)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    assert int(net.points_snapped.iloc[0].snap_ties) == 2


def test_snap_ties_is_two_where_a_road_end_touches_another_road_without_a_vertex(crs):
    # The end of the second road lies inside the first road, which has no
    # vertex there, so the two roads are not connected. A point at that place
    # can go to either road.
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(5, 0), (5, 10)])], crs=crs
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(5, 0)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster")
    assert int(net.points_snapped.iloc[0].snap_ties) == 2


def test_snapped_geometry_is_exactly_the_node(crs):
    x = 500_000.123
    roads = gpd.GeoDataFrame(geometry=[LineString([(x, 0), (x + 10, 0)])], crs=crs)
    points = gpd.GeoDataFrame(
        {"cluster": ["A", "B"]},
        geometry=[Point(x + 3.0, 1.0), Point(x + 3.0 + 4e-10, 1.0)],  # closer than the tolerance
        crs=crs,
    )
    net = from_geodataframes(roads, points, cluster_col="cluster")
    snapped = net.points_snapped
    assert snapped.network_node.nunique() == 1
    node = net.nodes.geometry.iloc[int(snapped.network_node.iloc[0])]
    assert all(geom.equals_exact(node, 0.0) for geom in snapped.geometry)
    observed = net.points_input.geometry.distance(snapped.geometry).to_numpy()
    assert snapped.snap_distance.to_numpy() == pytest.approx(observed, abs=1e-12)


def test_max_snap_distance_accepts_a_point_on_a_long_road_next_to_a_vertex(crs):
    # The point lies exactly on the road, 4e-8 m from a vertex. The builder
    # places it on the vertex, because the tolerance of a 60 km segment is
    # 6e-8 m. The point is 0 m from the road, so max_snap_distance=0 holds.
    x, y = 500_000.0, 1_600_000.0
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(x, y), (x + 60_000, y)]), LineString([(x + 60_000, y), (x + 60_000, y + 10)])],
        crs=crs,
    )
    points = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(x + 60_000 - 4e-8, y)], crs=crs)
    net = from_geodataframes(roads, points, cluster_col="cluster", max_snap_distance=0)
    snapped = net.points_snapped.iloc[0]
    assert snapped.geometry.equals(Point(x + 60_000, y))  # on the vertex node
    assert snapped.snap_distance == pytest.approx(4e-8, abs=1e-9)  # 4e-8 up to the rounding of x + 60000 - 4e-8


def test_max_snap_distance_limits_the_distance_to_the_road(line_roads, crs):
    on_limit = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(5, 1.0)], crs=crs)
    from_geodataframes(line_roads, on_limit, cluster_col="cluster", max_snap_distance=1.0)
    beyond = gpd.GeoDataFrame({"cluster": ["A"]}, geometry=[Point(5, 1.000001)], crs=crs)
    with pytest.raises(ValueError, match="exceed max_snap_distance"):
        from_geodataframes(line_roads, beyond, cluster_col="cluster", max_snap_distance=1.0)


def test_snap_distance_exceeds_the_distance_to_the_road_by_at_most_the_segment_tolerance(crs):
    # Points near vertices and near each other are moved onto an existing
    # node when they are within the segment tolerance of it; the returned
    # snap_distance can grow by at most that tolerance.
    import shapely

    rng = np.random.default_rng(11)
    x0, y0 = 500_000.0, 1_600_000.0
    pool = rng.uniform(0, 5000, (25, 2)) + [x0, y0]
    lines = [LineString(pool[rng.choice(25, 3, replace=False)]) for _ in range(15)]
    roads = gpd.GeoDataFrame(geometry=lines, crs=crs)
    vertices = shapely.get_coordinates(roads.geometry.to_numpy())
    near_vertex = vertices[rng.integers(0, len(vertices), 40)] + rng.normal(0, 2e-9, (40, 2))
    anywhere = rng.uniform(0, 5000, (40, 2)) + [x0, y0]
    xy = np.r_[near_vertex, anywhere, anywhere[:10] + rng.normal(0, 1e-10, (10, 2))]
    points = gpd.GeoDataFrame(
        {"cluster": rng.choice(["A", "B"], len(xy))}, geometry=[Point(p) for p in xy], crs=crs
    )
    net = from_geodataframes(roads, points, cluster_col="cluster")
    to_road = shapely.distance(points.geometry.to_numpy(), shapely.union_all(roads.geometry.to_numpy()))
    longest = float(net.edges.groupby("source_edge").length.sum().max())
    allowance = max(1e-9, 1e-12 * longest) + 1e-9  # segment tolerance, plus rounding of coordinates
    assert np.all(net.points_snapped.snap_distance.to_numpy() <= to_road + allowance)
