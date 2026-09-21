import numpy as np

from netvoronoi.core import network_voronoi


def test_unreachable_network_is_absent_from_cells(make_network):
    network = make_network(
        [(0, 0), (100, 0), (300, 0), (400, 0)],
        [(0, 1), (2, 3)],
        ["A"],
        [0],
    )
    result = network_voronoi(network, epsilon=50)
    assert np.isclose(result.hard_cells.geometry.length.sum(), 100.0)
    assert np.isclose(result.epsilon_cells.geometry.length.sum(), 100.0)
    assert len(result.unassigned_edges) == 1
    assert np.isclose(result.unassigned_edges.geometry.length.sum(), 100.0)


def test_sites_on_separate_components_do_not_cross_assign(make_network):
    network = make_network(
        [(0, 0), (100, 0), (300, 0), (400, 0)],
        [(0, 1), (2, 3)],
        ["A", "B"],
        [0, 2],
    )
    result = network_voronoi(network, epsilon=10, batch_size=1)
    by_edge = {
        int(edge_id): set(group.site_id.astype(str))
        for edge_id, group in result.epsilon_segments.groupby("edge_id")
    }
    assert by_edge[0] == {"A"}
    assert by_edge[1] == {"B"}
    assert result.unassigned_edges.empty
