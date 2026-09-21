import pytest

from netvoronoi.core import network_voronoi
from netvoronoi.io import write_network_gpkg


def test_writer_refuses_to_overwrite_by_default(tmp_path, line_network):
    result = network_voronoi(line_network)
    output = tmp_path / "result.gpkg"
    write_network_gpkg(output, line_network, result)
    with pytest.raises(FileExistsError):
        write_network_gpkg(output, line_network, result)
    write_network_gpkg(output, line_network, result, overwrite=True)


def test_writer_requires_gpkg_suffix(tmp_path, line_network):
    result = network_voronoi(line_network)
    with pytest.raises(ValueError, match=".gpkg"):
        write_network_gpkg(tmp_path / "wrong.shp", line_network, result)


def test_check_output_path_reports_problems_without_writing(tmp_path):
    from netvoronoi.io import check_output_path

    good = tmp_path / "result.gpkg"
    assert check_output_path(good) == good
    assert not good.exists()  # checking creates nothing

    good.write_bytes(b"")
    with pytest.raises(FileExistsError):
        check_output_path(good)
    assert check_output_path(good, overwrite=True) == good  # allowed, still not deleted
    assert good.exists()
    with pytest.raises(ValueError, match=".gpkg"):
        check_output_path(tmp_path / "result.shp")


def test_site_qa_layers_do_not_require_snap_distance(line_network):
    import geopandas as gpd

    from netvoronoi.io import _site_qa_layers
    from netvoronoi.model import SpatialNetwork

    sites = line_network.sites_snapped.copy()
    sites["source_x"] = sites.geometry.x + 2.0
    sites["source_y"] = sites.geometry.y + 3.0
    assert "snap_distance" not in sites.columns

    network = SpatialNetwork(
        line_network.vertices_xy,
        line_network.edges,
        line_network.adjacency,
        line_network.site_ids,
        line_network.site_nodes,
        sites,
        line_network.crs,
    )
    network.validate()
    inputs, connectors = _site_qa_layers(network)
    assert isinstance(inputs, gpd.GeoDataFrame)
    assert isinstance(connectors, gpd.GeoDataFrame)
    assert "snap_distance" not in inputs.columns
    assert "snap_distance" not in connectors.columns
    assert len(inputs) == len(sites) == len(connectors)
