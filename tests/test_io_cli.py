from pathlib import Path

import geopandas as gpd
import pytest
from typer.testing import CliRunner

from netvoronoi import from_geodataframes, network_voronoi
from netvoronoi.cli import app
from netvoronoi.io import check_output_path, write_network_gpkg


def test_write_network_gpkg(tmp_path, line_roads, end_points):
    net = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    result = network_voronoi(net)
    out = tmp_path / "out.gpkg"
    write_network_gpkg(out, net, result)
    assert out.exists()
    layers = set(gpd.list_layers(out).name)
    assert {"network_edges", "points_input", "points_snapped", "network_segments", "network_cells"}.issubset(layers)


def test_output_overwrite_guard(tmp_path):
    out = tmp_path / "x.gpkg"
    out.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        check_output_path(out)


def test_cli_has_no_epsilon_option():
    runner = CliRunner()
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0
    assert "--epsilon" not in result.stdout
    assert "--cluster-col" in result.stdout


def test_cli_reader_dispatches_parquet_to_read_parquet(monkeypatch, crs):
    from netvoronoi.cli import _read

    expected = gpd.GeoDataFrame(geometry=[], crs=crs)
    called = {}

    def fake_read_parquet(path):
        called["path"] = Path(path)
        return expected

    monkeypatch.setattr(gpd, "read_parquet", fake_read_parquet)
    result = _read(Path("points.parquet"), None)
    assert result is expected
    assert called["path"] == Path("points.parquet")


def test_cli_reader_rejects_layer_for_parquet():
    from netvoronoi.cli import _read

    with pytest.raises(Exception, match="layer selection is not supported for Parquet"):
        _read(Path("points.parquet"), "points")


# --- review additions ---------------------------------------------------------


def _write_inputs(tmp_path, line_roads, end_points, boundary):
    roads = tmp_path / "roads.gpkg"
    points = tmp_path / "points.gpkg"
    area = tmp_path / "boundary.gpkg"
    line_roads.to_file(roads, layer="roads")
    end_points.to_file(points, layer="points")
    boundary.to_file(area, layer="boundary")
    return roads, points, area


def test_cli_build_with_boundary_writes_network_and_surface_layers(tmp_path, line_roads, end_points, boundary):
    roads, points, area = _write_inputs(tmp_path, line_roads, end_points, boundary)
    out = tmp_path / "out.gpkg"
    result = CliRunner().invoke(
        app,
        ["build", str(roads), str(points), "--cluster-col", "cluster", "--point-id-col", "pid",
         "--boundary", str(area), "--resolution", "1", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    layers = set(gpd.list_layers(out).name)
    assert {
        "network_edges", "points_input", "points_snapped", "snap_connectors",
        "network_segments", "network_cells", "surface_cells", "surface_grid_debug", "surface_point_qa",
    } <= layers
    # Layers with no rows are not written: every edge is reachable and every
    # surface cell is assigned, so these two layers are absent.
    assert "network_unassigned" not in layers
    assert "surface_unassigned" not in layers
    segments = gpd.read_file(out, layer="network_segments")
    assert list(segments.cluster_id) == ["A", "B"]


def test_cli_reports_input_errors_in_one_line(tmp_path, line_roads, end_points, boundary):
    roads, points, _ = _write_inputs(tmp_path, line_roads, end_points, boundary)
    result = CliRunner().invoke(
        app, ["build", str(roads), str(points), "--cluster-col", "missing", "--output", str(tmp_path / "o.gpkg")]
    )
    assert result.exit_code == 1
    assert "Error: cluster column not found: 'missing'" in result.output
    assert "Traceback" not in result.output


def test_cli_existing_output_without_force_is_an_error(tmp_path, line_roads, end_points, boundary):
    roads, points, _ = _write_inputs(tmp_path, line_roads, end_points, boundary)
    out = tmp_path / "o.gpkg"
    out.write_bytes(b"x")
    result = CliRunner().invoke(app, ["build", str(roads), str(points), "--cluster-col", "cluster", "--output", str(out)])
    assert result.exit_code == 1
    assert "pass --force to replace it" in result.output


def test_cli_reader_dispatches_geoparquet_to_read_parquet(monkeypatch, crs):
    from netvoronoi.cli import _read

    expected = gpd.GeoDataFrame(geometry=[], crs=crs)
    monkeypatch.setattr(gpd, "read_parquet", lambda path: expected)
    assert _read(Path("points.geoparquet"), None) is expected


def test_snap_connectors_join_each_input_point_to_its_node(line_roads, crs):
    from shapely.geometry import Point

    from netvoronoi.io import _snap_connectors

    points = gpd.GeoDataFrame({"cluster": ["A", "B"]}, geometry=[Point(2, 3), Point(7, -1)], crs=crs)
    net = from_geodataframes(line_roads, points, cluster_col="cluster")
    lines = _snap_connectors(net)
    for line, a, b, d in zip(lines.geometry, net.points_input.geometry, net.points_snapped.geometry, lines.snap_distance):
        assert line.coords[0] == a.coords[0]
        assert line.coords[-1] == b.coords[0]
        assert line.length == pytest.approx(d)
