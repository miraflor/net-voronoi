import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point
from typer.testing import CliRunner

import netvoronoi.cli as cli


@pytest.fixture
def inputs(tmp_path):
    """Write a minimal road and site file and return the CLI arguments."""
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0), (100, 100)])], crs="EPSG:32651"
    )
    sites = gpd.GeoDataFrame(
        {"sid": ["A", "B"]}, geometry=[Point(10, 2), Point(98, 90)], crs="EPSG:32651"
    )
    roads.to_file(tmp_path / "roads.gpkg")
    sites.to_file(tmp_path / "sites.gpkg")
    return [
        "build",
        str(tmp_path / "roads.gpkg"),
        str(tmp_path / "sites.gpkg"),
        "--site-id-col",
        "sid",
        "-o",
        str(tmp_path / "out.gpkg"),
    ]


def _refuse(*args, **kwargs):
    raise AssertionError("the CLI started working before checking its arguments")


def test_existing_output_is_refused_before_any_work(tmp_path, inputs, monkeypatch):
    (tmp_path / "out.gpkg").write_bytes(b"")
    monkeypatch.setattr(cli, "from_geodataframes", _refuse)
    result = CliRunner().invoke(cli.app, inputs)
    assert isinstance(result.exception, FileExistsError)


def test_wrong_output_suffix_is_refused_before_any_work(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(cli, "from_geodataframes", _refuse)
    arguments = inputs[:-1] + [str(tmp_path / "out.shp")]
    result = CliRunner().invoke(cli.app, arguments)
    assert isinstance(result.exception, ValueError)
    assert ".gpkg" in str(result.exception)


@pytest.mark.parametrize(
    "option", [["--epsilon", "nan"], ["--max-snap-distance", "nan"], ["--resolution", "nan"], ["--resolution", "inf"]]
)
def test_not_a_number_options_are_refused(inputs, monkeypatch, option):
    # Typer's own ``min`` check passes NaN, because every comparison with NaN
    # is false.
    monkeypatch.setattr(cli, "from_geodataframes", _refuse)
    result = CliRunner().invoke(cli.app, inputs + option)
    assert result.exit_code == 2  # a usage error, not a crash
    assert "Invalid value" in result.output


def test_too_fine_grid_is_refused_before_the_road_network_is_built(tmp_path, inputs, monkeypatch):
    # v0.4.1 built the whole road network first; for a large road file that
    # was the slowest step of the run.
    from shapely.geometry import box

    gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32651").to_file(
        tmp_path / "boundary.gpkg"
    )
    monkeypatch.setattr(cli, "from_geodataframes", _refuse)
    arguments = inputs + [
        "--boundary", str(tmp_path / "boundary.gpkg"), "--resolution", "1", "--max-cells", "100"
    ]
    result = CliRunner().invoke(cli.app, arguments)
    assert isinstance(result.exception, ValueError)
    assert "max_cells" in str(result.exception)
