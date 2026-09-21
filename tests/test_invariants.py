import warnings
import numpy as np
import pytest
from shapely.geometry import LineString

from netvoronoi.core import network_voronoi
from netvoronoi.model import SpatialNetwork


def test_hard_partition_covers_reachable_network_once(line_network):
    result = network_voronoi(line_network, epsilon=250)
    network_length = float(line_network.edges.geometry.length.sum())
    hard_length = float(result.hard_segments.geometry.length.sum())
    assert np.isclose(hard_length, network_length)


def test_epsilon_cells_are_superset_of_hard_cells_by_site(line_network):
    result = network_voronoi(line_network, epsilon=100)
    hard = {
        str(row.site_id): row.geometry.length
        for row in result.hard_cells.itertuples(index=False)
    }
    eps = {
        str(row.site_id): row.geometry.length
        for row in result.epsilon_cells.itertuples(index=False)
    }
    for site_id, length in hard.items():
        assert eps[site_id] + 1e-9 >= length


def test_model_rejects_reversed_geometry_orientation(line_network):
    edges = line_network.edges.copy()
    edges.at[0, "geometry"] = LineString(list(edges.at[0, "geometry"].coords)[::-1])
    bad = SpatialNetwork(
        line_network.vertices_xy,
        edges,
        line_network.adjacency,
        line_network.site_ids,
        line_network.site_nodes,
        line_network.sites_snapped,
        line_network.crs,
    )
    with pytest.raises(ValueError, match="start coordinate"):
        bad.validate()


def test_model_rejects_adjacency_that_disagrees_with_edges(line_network):
    adjacency = line_network.adjacency.copy().tolil()
    adjacency[0, 1] = 999
    adjacency[1, 0] = 999
    bad = SpatialNetwork(
        line_network.vertices_xy,
        line_network.edges,
        adjacency.tocsr(),
        line_network.site_ids,
        line_network.site_nodes,
        line_network.sites_snapped,
        line_network.crs,
    )
    with pytest.raises(ValueError, match="adjacency"):
        bad.validate()


def test_model_rejects_snapped_site_that_does_not_match_its_node(line_network):
    sites = line_network.sites_snapped.copy()
    # Move the first snapped point one map unit while keeping network_node=1.
    from shapely.geometry import Point

    sites.at[0, "geometry"] = Point(
        float(sites.geometry.iloc[0].x) + 1.0,
        float(sites.geometry.iloc[0].y),
    )
    bad = SpatialNetwork(
        line_network.vertices_xy,
        line_network.edges,
        line_network.adjacency,
        line_network.site_ids,
        line_network.site_nodes,
        sites,
        line_network.crs,
    )
    with pytest.raises(ValueError, match="does not match its network node"):
        bad.validate()


def test_model_rejects_a_geographic_crs_given_as_a_string(line_network):
    # v0.3.0 only looked for an ``is_projected`` attribute, which a plain
    # string does not have, so "EPSG:4326" passed and distances would have
    # been degrees.
    edges = line_network.edges.copy().set_crs("EPSG:4326", allow_override=True)
    sites = line_network.sites_snapped.copy().set_crs("EPSG:4326", allow_override=True)
    bad = SpatialNetwork(
        line_network.vertices_xy,
        edges,
        line_network.adjacency,
        line_network.site_ids,
        line_network.site_nodes,
        sites,
        "EPSG:4326",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="projected"):
            bad.validate()
    assert not any("geographic CRS" in str(item.message) for item in caught)


def test_model_error_names_the_first_offending_edge(line_network):
    edges = line_network.edges.copy()
    edges.at[1, "length"] = float(edges.at[1, "length"]) + 5.0
    bad = SpatialNetwork(
        line_network.vertices_xy,
        edges,
        line_network.adjacency,
        line_network.site_ids,
        line_network.site_nodes,
        line_network.sites_snapped,
        line_network.crs,
    )
    with pytest.raises(ValueError, match="edge_id 1"):
        bad.validate()


def test_model_rejects_missing_crs_metadata_on_geometry_layers(line_network):
    # Declaring network.crs is not enough: raw coordinates with missing layer
    # CRS metadata are ambiguous and must not be silently treated as projected.
    edges = line_network.edges.copy()
    edges.set_crs(None, allow_override=True, inplace=True)
    bad_edges = SpatialNetwork(
        line_network.vertices_xy,
        edges,
        line_network.adjacency,
        line_network.site_ids,
        line_network.site_nodes,
        line_network.sites_snapped,
        line_network.crs,
    )
    with pytest.raises(ValueError, match="edges must have CRS metadata"):
        bad_edges.validate()

    sites = line_network.sites_snapped.copy()
    sites.set_crs(None, allow_override=True, inplace=True)
    bad_sites = SpatialNetwork(
        line_network.vertices_xy,
        line_network.edges,
        line_network.adjacency,
        line_network.site_ids,
        line_network.site_nodes,
        sites,
        line_network.crs,
    )
    with pytest.raises(ValueError, match="sites_snapped must have CRS metadata"):
        bad_sites.validate()


def test_crs_error_precedes_edge_geometry_checks(line_network):
    """A known-invalid CRS should fail before any geometry measurement."""
    edges = line_network.edges.copy().set_crs("EPSG:4326", allow_override=True)
    edges.at[0, "length"] = float(edges.at[0, "length"]) + 123.0
    sites = line_network.sites_snapped.copy().set_crs("EPSG:4326", allow_override=True)
    bad = SpatialNetwork(
        line_network.vertices_xy,
        edges,
        line_network.adjacency,
        line_network.site_ids,
        line_network.site_nodes,
        sites,
        "EPSG:4326",
    )
    with pytest.raises(ValueError, match="network CRS must be projected"):
        bad.validate()
