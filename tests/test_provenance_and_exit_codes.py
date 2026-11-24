"""Input hashing of multi-file datasets, the code hash in the manifest, early output checks, and exit codes."""

import hashlib
import json

import geopandas as gpd
import pyogrio
import pytest
from shapely.geometry import LineString, Point
from typer.testing import CliRunner

import netvoronoi.cli as cli
from netvoronoi import io
from netvoronoi.cli import InputChangedError, _code_fingerprint, _file_record, _verify_input_records, app

CRS = "EPSG:32651"


def _write_inputs(folder, *, points_name="points.shp", labels=("A", "B")):
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs=CRS)
    roads.to_file(folder / "roads.gpkg", layer="roads")
    points = gpd.GeoDataFrame({"cluster": list(labels)}, geometry=[Point(0, 0), Point(10, 0)], crs=CRS)
    points.to_file(folder / points_name)
    return folder / "roads.gpkg", folder / points_name


def _manifest(path):
    return json.loads(pyogrio.read_dataframe(path, layer="netvoronoi_manifest")["json"].iloc[0])


# -------------------------------------------------------- multi-file datasets


def test_shapefile_record_covers_every_part_and_nothing_else(tmp_path):
    _, points = _write_inputs(tmp_path)
    (tmp_path / "points.csv").write_text("not part of the shapefile")
    (tmp_path / "points.gpkg").write_bytes(b"an output beside the input")
    record = _file_record(points, None)
    assert record["kind"] == "file_set"
    names = [member["name"] for member in record["members"]]
    assert {"points.shp", "points.shx", "points.dbf", "points.prj"} <= set(names)
    assert "points.csv" not in names and "points.gpkg" not in names
    assert record["bytes"] == sum(member["bytes"] for member in record["members"])


def test_label_change_in_the_dbf_alone_changes_the_hash_and_blocks_publication(tmp_path):
    _, points = _write_inputs(tmp_path)
    snapshot = {"roads": None, "points": _file_record(points, None), "boundary": None}
    shp_before = (tmp_path / "points.shp").read_bytes()
    relabelled = gpd.read_file(points)
    relabelled["cluster"] = ["B", "A"]
    relabelled.to_file(points)
    assert (tmp_path / "points.shp").read_bytes() == shp_before  # only the .dbf changed
    assert _file_record(points, None)["sha256"] != snapshot["points"]["sha256"]
    with pytest.raises(InputChangedError, match="points input changed"):
        _verify_input_records(snapshot)


def test_shapefile_parts_are_found_in_any_letter_case(tmp_path):
    (tmp_path / "POINTS.SHP").write_bytes(b"shp")
    (tmp_path / "Points.DBF").write_bytes(b"dbf")
    record = _file_record(tmp_path / "POINTS.SHP", None)
    assert [m["name"] for m in record["members"]] == ["Points.DBF", "POINTS.SHP"]


def test_geopackage_write_ahead_log_is_part_of_the_dataset(tmp_path):
    main = tmp_path / "roads.gpkg"
    main.write_bytes(b"database")
    assert _file_record(main, None)["kind"] == "file"
    (tmp_path / "roads.gpkg-wal").write_bytes(b"recent changes")
    record = _file_record(main, None)
    assert record["kind"] == "file_set"
    assert [m["name"] for m in record["members"]] == ["roads.gpkg", "roads.gpkg-wal"]


def test_single_file_record_is_unchanged(tmp_path):
    source = tmp_path / "points.geojson"
    source.write_bytes(b'{"type": "FeatureCollection", "features": []}')
    record = _file_record(source, "layer")
    assert record == {
        "path": str(source),
        "layer": "layer",
        "kind": "file",
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def test_cli_records_different_hashes_for_different_labels(tmp_path):
    roads, points = _write_inputs(tmp_path)
    first, second = tmp_path / "first.gpkg", tmp_path / "second.gpkg"
    base = ["build", str(roads), str(points), "--cluster-col", "cluster", "--output"]
    assert CliRunner().invoke(app, base + [str(first)]).exit_code == 0
    relabelled = gpd.read_file(points)
    relabelled["cluster"] = ["B", "A"]
    relabelled.to_file(points)
    assert CliRunner().invoke(app, base + [str(second)]).exit_code == 0
    hashes = [_manifest(path)["inputs"]["points"]["sha256"] for path in (first, second)]
    owners = [
        pyogrio.read_dataframe(path, layer="network_segments")["cluster_id"].tolist() for path in (first, second)
    ]
    assert owners[0] != owners[1]
    assert hashes[0] != hashes[1]


# ------------------------------------------------------------------ code hash


def test_manifest_identifies_the_code(tmp_path):
    roads, points = _write_inputs(tmp_path, points_name="points.gpkg")
    out = tmp_path / "out.gpkg"
    result = CliRunner().invoke(
        app, ["build", str(roads), str(points), "--cluster-col", "cluster", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    software = _manifest(out)["software"]
    assert software["netvoronoi_code_sha256"] == _code_fingerprint()
    assert len(software["netvoronoi_code_sha256"]) == 64


def test_code_hash_ignores_line_endings_but_not_content(tmp_path):
    unix, windows, edited, empty = (tmp_path / name for name in ("unix", "windows", "edited", "empty"))
    for folder in (unix, windows, edited, empty):
        folder.mkdir()
    (unix / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (windows / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (edited / "a.py").write_bytes(b"x = 1\ny = 3\n")
    assert _code_fingerprint(unix) == _code_fingerprint(windows)
    assert _code_fingerprint(unix) != _code_fingerprint(edited)
    assert _code_fingerprint(empty) is None


# ---------------------------------------------------- output checks and exits


def test_output_in_use_fails_in_one_line_before_any_input_is_read(tmp_path, monkeypatch):
    roads, points = _write_inputs(tmp_path, points_name="points.gpkg")
    out = tmp_path / "out.gpkg"
    out.write_bytes(b"old output")
    (tmp_path / "out.gpkg-wal").write_bytes(b"QGIS is editing")
    monkeypatch.setattr(cli, "_read", lambda *a, **k: pytest.fail("input read before the output check"))
    result = CliRunner().invoke(
        app, ["build", str(roads), str(points), "--cluster-col", "cluster", "-o", str(out), "--force"]
    )
    assert result.exit_code == cli.EXIT_NOTHING_WRITTEN == 1
    assert "Error: refusing to replace GeoPackage with existing SQLite sidecar(s)" in result.output
    assert "Close the GeoPackage in QGIS" in result.output
    assert "Traceback" not in result.output
    assert out.read_bytes() == b"old output"


def test_output_in_use_error_is_still_a_runtime_error(tmp_path):
    target = tmp_path / "out.gpkg"
    (tmp_path / "out.gpkg-shm").write_bytes(b"busy")
    with pytest.raises(io.OutputInUseError) as caught:
        io.check_output_target(target)
    assert isinstance(caught.value, RuntimeError)


def test_failure_after_replacement_has_its_own_exit_code(tmp_path, monkeypatch):
    roads, points = _write_inputs(tmp_path, points_name="points.gpkg")
    out = tmp_path / "out.gpkg"
    out.write_bytes(b"old output")

    def fail(folder):
        raise OSError("simulated folder sync failure")

    monkeypatch.setattr(io, "_flush_folder", fail)
    result = CliRunner().invoke(
        app, ["build", str(roads), str(points), "--cluster-col", "cluster", "-o", str(out), "--force"]
    )
    assert result.exit_code == cli.EXIT_WRITTEN_DURABILITY_UNCONFIRMED == 3
    assert "output was replaced" in result.output
    assert out.read_bytes() != b"old output"  # the new file is in place
    assert "network_segments" in set(pyogrio.list_layers(out)[:, 0])


def test_ordinary_failure_keeps_exit_code_one(tmp_path):
    roads, points = _write_inputs(tmp_path, points_name="points.gpkg")
    result = CliRunner().invoke(
        app, ["build", str(roads), str(points), "--cluster-col", "missing", "-o", str(tmp_path / "o.gpkg")]
    )
    assert result.exit_code == 1


def test_shapefile_xml_sidecar_is_hashed_and_changes_detected(tmp_path):
    # GDAL 3.13+ can use .shp.xml for extended attribute names.
    _, points = _write_inputs(tmp_path)
    xml = tmp_path / "points.shp.xml"
    xml.write_bytes(b"<metadata>original</metadata>")
    original = _file_record(points, None)
    assert "points.shp.xml" in {member["name"] for member in original["members"]}
    xml.write_bytes(b"<metadata>modified</metadata>")
    assert _file_record(points, None)["sha256"] != original["sha256"]
    with pytest.raises(InputChangedError):
        _verify_input_records({"roads": None, "points": original, "boundary": None})


def test_output_in_use_does_not_advise_deleting_sqlite_sidecars(tmp_path):
    roads, points = _write_inputs(tmp_path, points_name="points.gpkg")
    target = tmp_path / "out.gpkg"
    target.write_bytes(b"old")
    (tmp_path / "out.gpkg-wal").write_bytes(b"committed data that might not be checkpointed")
    result = CliRunner().invoke(app, [
        "build", str(roads), str(points), "--cluster-col", "cluster", "--output", str(target), "--force"
    ])
    assert result.exit_code == cli.EXIT_NOTHING_WRITTEN == 1
    assert "do not delete SQLite" in result.output
    assert "can be removed" not in result.output
    assert target.read_bytes() == b"old"


def test_cleanup_failure_after_replacement_exits_three(tmp_path, monkeypatch):
    roads, points = _write_inputs(tmp_path, points_name="points.gpkg")
    target = tmp_path / "out.gpkg"
    target.write_bytes(b"old")
    real_replace = io.os.replace
    real_unlink = io.Path.unlink

    def replace_then_leave_staging_sidecar(source, destination):
        real_replace(source, destination)
        io.Path(str(source) + "-wal").write_bytes(b"simulated stale staging WAL")

    def fail_staging_sidecar_cleanup(path, *args, **kwargs):
        if path.name.startswith(".netvoronoi-staging-") and path.name.endswith(".gpkg-wal"):
            raise OSError("simulated post-replace staging cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(io.os, "replace", replace_then_leave_staging_sidecar)
    monkeypatch.setattr(io.Path, "unlink", fail_staging_sidecar_cleanup)
    result = CliRunner().invoke(app, [
        "build", str(roads), str(points), "--cluster-col", "cluster", "--output", str(target), "--force"
    ])
    assert result.exit_code == cli.EXIT_WRITTEN_DURABILITY_UNCONFIRMED == 3
    assert "staging cleanup failed" in result.output
    assert "output was replaced" in result.output
    assert "network_segments" in set(pyogrio.list_layers(target)[:, 0])
    assert target.read_bytes() != b"old"


def test_cleanup_failure_before_replacement_never_changes_old_output(tmp_path, monkeypatch):
    roads, points = _write_inputs(tmp_path, points_name="points.gpkg")
    target = tmp_path / "out.gpkg"
    target.write_bytes(b"old")
    original_check = io._check_staged
    real_unlink = io.Path.unlink

    def fail_before_replacement(*args, **kwargs):
        raise ValueError("simulated staging validation failure")

    def fail_staged_unlink(path, *args, **kwargs):
        if path.name.startswith(".netvoronoi-staging-") and path.suffix == ".gpkg":
            # The initial reservation unlink also occurs before the try block.
            # Only fail after the file has been populated by the writer.
            if path.exists() and path.stat().st_size > 0:
                raise OSError("simulated cleanup failure before publication")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(io, "_check_staged", fail_before_replacement)
    monkeypatch.setattr(io.Path, "unlink", fail_staged_unlink)
    result = CliRunner().invoke(app, [
        "build", str(roads), str(points), "--cluster-col", "cluster", "--output", str(target), "--force"
    ])
    assert result.exit_code == 1
    assert "simulated staging validation failure" in result.output
    assert target.read_bytes() == b"old"
    monkeypatch.setattr(io, "_check_staged", original_check)
