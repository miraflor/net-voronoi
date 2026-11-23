"""The ``attachment`` option of ``surface_voronoi``: the default is unchanged, the alternative is explicit."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from netvoronoi import from_geodataframes, network_voronoi, surface_voronoi

CRS = "EPSG:3857"


def _two_components():
    """Road A-B carries the points; a second road 90 m away carries none."""
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)]), LineString([(100, 0), (110, 0)])], crs=CRS)
    points = gpd.GeoDataFrame({"c": ["A", "B"]}, geometry=[Point(0, 0), Point(10, 0)], crs=CRS)
    return from_geodataframes(roads, points, cluster_col="c")


def test_default_is_nearest_network(line_roads, end_points, boundary):
    network = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    default = surface_voronoi(network, boundary, resolution=1.0)
    explicit = surface_voronoi(network, boundary, resolution=1.0, attachment="nearest_network")
    assert default.attachment == "nearest_network" and default.resolution == 1.0
    assert default.grid.drop(columns="geometry").equals(explicit.grid.drop(columns="geometry"))


def test_reachable_attachment_gives_owners_to_land_near_roads_without_points():
    network = _two_components()
    area = gpd.GeoDataFrame(geometry=[box(95, -2, 115, 2)], crs=CRS)
    plain = surface_voronoi(network, area, resolution=1.0)
    reach = surface_voronoi(network, area, resolution=1.0, attachment="nearest_reachable_network")
    assert plain.cells.empty and not plain.unassigned.empty
    assert reach.unassigned.empty
    assert set(reach.cells["cluster_id"]) == {"B"}  # the reachable road end nearest to this land is B's
    # Every anchor lies on an edge whose end nodes have a finite distance to a point.
    reachable = np.isfinite(reach.network.node_min_distance[network.edges["u"].to_numpy()])
    assert set(reach.grid["edge_id"]) <= set(network.edges.loc[reachable, "edge_id"])


def test_attachment_never_changes_the_exact_partition():
    network = _two_components()
    area = gpd.GeoDataFrame(geometry=[box(0, -2, 115, 2)], crs=CRS)
    exact = network_voronoi(network).segments
    for policy in ("nearest_network", "nearest_reachable_network"):
        rendered = surface_voronoi(network, area, resolution=1.0, attachment=policy).network.segments
        assert rendered.drop(columns="geometry").equals(exact.drop(columns="geometry"))


def test_both_policies_keep_the_edge_id_tie_rule():
    # Two roads cross at (5, 0) without a shared vertex; both carry points,
    # so both policies allow both roads. The representative point (5, 0) is
    # on both roads, and the smaller edge_id wins under either policy.
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)]), LineString([(5, -5), (5, 5)])], crs=CRS)
    points = gpd.GeoDataFrame({"c": ["B", "A"]}, geometry=[Point(0, 0), Point(5, 5)], crs=CRS)
    network = from_geodataframes(roads, points, cluster_col="c")
    area = gpd.GeoDataFrame(geometry=[box(4.5, -0.5, 5.5, 0.5)], crs=CRS)
    plain = surface_voronoi(network, area, resolution=1.0)
    reach = surface_voronoi(network, area, resolution=1.0, attachment="nearest_reachable_network")
    assert plain.grid["anchor_ties"].tolist() == [2]
    assert plain.grid["edge_id"].iloc[0] == min(network.edges["edge_id"].astype(str))
    assert reach.grid.drop(columns="geometry").equals(plain.grid.drop(columns="geometry"))


def test_output_columns_do_not_depend_on_the_policy():
    network = _two_components()
    area = gpd.GeoDataFrame(geometry=[box(0, -2, 115, 2)], crs=CRS)
    plain = surface_voronoi(network, area, resolution=1.0)
    reach = surface_voronoi(network, area, resolution=1.0, attachment="nearest_reachable_network")
    for name in ("cells", "grid", "point_qa"):
        assert list(getattr(plain, name).columns) == list(getattr(reach, name).columns)


def test_unknown_policy_is_rejected(line_roads, end_points, boundary):
    network = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    with pytest.raises(ValueError, match="attachment must be one of"):
        surface_voronoi(network, boundary, resolution=1.0, attachment="nearest")
