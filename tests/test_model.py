import geopandas as gpd
import pytest
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, Point

from netvoronoi.model import SpatialNetwork


def _base_layers(crs):
    nodes = gpd.GeoDataFrame(
        {"node_id": [0, 1, 2]},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 0)],
        crs=crs,
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["e0", "e1"],
            "u": [0, 1],
            "v": [1, 2],
            "length": [10.0, 10.0],
        },
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])],
        crs=crs,
    )
    points_input = gpd.GeoDataFrame(
        {"point_id": ["a"], "cluster_id": ["A"]}, geometry=[Point(0, 0)], crs=crs
    )
    points_snapped = gpd.GeoDataFrame(
        {
            "point_id": ["a"],
            "cluster_id": ["A"],
            "network_node": [0],
            "snap_distance": [0.0],
            "snap_ties": [1],
        },
        geometry=[Point(0, 0)],
        crs=crs,
    )
    return nodes, edges, points_input, points_snapped


def test_adjacency_rejects_extra_connection_even_if_weight_is_tiny(crs):
    nodes, edges, points_input, points_snapped = _base_layers(crs)
    adjacency = csr_matrix(
        [[0.0, 10.0, 1e-12], [10.0, 0.0, 10.0], [1e-12, 10.0, 0.0]]
    )
    with pytest.raises(ValueError, match="adjacency connections"):
        SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)


def test_edge_length_must_match_geometry(crs):
    nodes, edges, points_input, points_snapped = _base_layers(crs)
    edges = edges.copy()
    edges.loc[0, "length"] = 9.0
    adjacency = csr_matrix([[0.0, 9.0, 0.0], [9.0, 0.0, 10.0], [0.0, 10.0, 0.0]])
    with pytest.raises(ValueError, match="edge length must equal"):
        SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)


# --- review regressions -----------------------------------------------------
# SciPy's shortest-path routines read every stored entry of a sparse matrix
# as a connection. The two tests below build matrices that the previous check
# accepted, although SciPy reads them as a different graph.


def test_adjacency_rejects_explicitly_stored_zero_between_unconnected_nodes(crs):
    import numpy as np

    nodes, edges, points_input, points_snapped = _base_layers(crs)
    data = np.array([10.0, 0.0, 10.0, 10.0, 0.0, 10.0])
    rows = np.array([0, 0, 1, 1, 2, 2])
    cols = np.array([1, 2, 0, 2, 0, 1])
    adjacency = csr_matrix((data, (rows, cols)), shape=(3, 3))  # stores 0.0 at (0, 2) and (2, 0)
    assert (adjacency.data == 0).sum() == 2
    with pytest.raises(ValueError, match="adjacency connections"):
        SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)


def test_adjacency_rejects_two_entries_for_one_node_pair(crs):
    import numpy as np

    nodes, edges, points_input, points_snapped = _base_layers(crs)
    # (0, 1) is stored twice, as 1.0 and 9.0. Their sum equals the edge
    # length, but SciPy uses the shorter entry, 1.0.
    indptr = np.array([0, 2, 4, 5])
    indices = np.array([1, 1, 0, 2, 1])
    data = np.array([1.0, 9.0, 10.0, 10.0, 10.0])
    adjacency = csr_matrix((data, indices, indptr), shape=(3, 3))
    with pytest.raises(ValueError, match="more than one entry"):
        SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)


def test_adjacency_in_coordinate_format_is_accepted(crs):
    from scipy.sparse import coo_matrix

    nodes, edges, points_input, points_snapped = _base_layers(crs)
    adjacency = coo_matrix(
        ([10.0, 10.0, 10.0, 10.0], ([0, 1, 1, 2], [1, 0, 2, 1])), shape=(3, 3)
    )
    net = SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)
    from netvoronoi import network_voronoi

    assert network_voronoi(net).segments.segment_dist.sum() == pytest.approx(20.0)


def test_adjacency_weight_tolerance_is_local_not_global(crs):
    import numpy as np

    nodes = gpd.GeoDataFrame(
        {"node_id": [0, 1, 2, 3]},
        geometry=[Point(0, 0), Point(0.0005, 0), Point(10, 0), Point(1_000_000_010, 0)],
        crs=crs,
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": ["small", "huge"],
            "u": [0, 2],
            "v": [1, 3],
            "length": [0.0005, 1_000_000_000.0],
        },
        geometry=[
            LineString([(0, 0), (0.0005, 0)]),
            LineString([(10, 0), (1_000_000_010, 0)]),
        ],
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
            "network_node": [0, 2],
            "snap_distance": [0.0, 0.0],
            "snap_ties": [1, 1],
        },
        geometry=[Point(0, 0), Point(10, 0)],
        crs=crs,
    )
    # Correct topology, but the 0.0005 edge has been changed to zero weight.
    indptr = np.array([0, 1, 2, 3, 4])
    indices = np.array([1, 0, 3, 2])
    data = np.array([0.0, 0.0, 1_000_000_000.0, 1_000_000_000.0])
    adjacency = csr_matrix((data, indices, indptr), shape=(4, 4))
    with pytest.raises(ValueError, match="adjacency weights"):
        SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)


def test_custom_cluster_ids_may_not_collapse_to_same_string(crs):
    nodes, edges, _, _ = _base_layers(crs)
    points_input = gpd.GeoDataFrame(
        {"point_id": ["a", "b"], "cluster_id": [1, "1"]},
        geometry=[Point(0, 0), Point(20, 0)],
        crs=crs,
    )
    points_snapped = gpd.GeoDataFrame(
        {
            "point_id": ["a", "b"],
            "cluster_id": [1, "1"],
            "network_node": [0, 2],
            "snap_distance": [0.0, 0.0],
            "snap_ties": [1, 1],
        },
        geometry=[Point(0, 0), Point(20, 0)],
        crs=crs,
    )
    adjacency = csr_matrix([[0.0, 10.0, 0.0], [10.0, 0.0, 10.0], [0.0, 10.0, 0.0]])
    with pytest.raises(ValueError, match="collapse to the same string"):
        SpatialNetwork(nodes, edges, points_input, points_snapped, adjacency)
