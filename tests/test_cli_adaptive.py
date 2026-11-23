"""Command line: manifest in every output, strict adaptive publication, early option checks."""

import json

import pyogrio
import pytest
from typer.testing import CliRunner

from netvoronoi import io
from netvoronoi.cli import app


@pytest.fixture
def inputs(tmp_path, line_roads, end_points, boundary):
    roads, points, area = tmp_path / "roads.gpkg", tmp_path / "points.gpkg", tmp_path / "boundary.gpkg"
    line_roads.to_file(roads, layer="roads")
    end_points.to_file(points, layer="points")
    boundary.to_file(area, layer="boundary")
    return roads, points, area


def _run(inputs, *extra):
    roads, points, _ = inputs
    return CliRunner().invoke(app, ["build", str(roads), str(points), "--cluster-col", "cluster", *extra])


def _manifest(path):
    return json.loads(pyogrio.read_dataframe(path, layer="netvoronoi_manifest")["json"].iloc[0])


def test_network_only_output_has_a_manifest(tmp_path, inputs):
    out = tmp_path / "net.gpkg"
    result = _run(inputs, "--output", str(out))
    assert result.exit_code == 0, result.output
    manifest = _manifest(out)
    assert manifest["surface"] is None
    assert manifest["publication"]["accepted_status"] == "exact_network_only"
    assert len(manifest["inputs"]["roads"]["sha256"]) == 64


def test_fixed_surface_output_records_its_settings(tmp_path, inputs):
    out = tmp_path / "fixed.gpkg"
    result = _run(inputs, "--boundary", str(inputs[2]), "--resolution", "1", "--output", str(out))
    assert result.exit_code == 0, result.output
    surface = _manifest(out)["surface"]
    assert surface["mode"] == "fixed" and surface["resolution"] == 1.0
    assert surface["attachment"] == "nearest_network" and surface["convergence_demonstrated"] is False
    assert "surface_territory_qa" in set(pyogrio.list_layers(out)[:, 0])


def test_adaptive_converged_output(tmp_path, inputs):
    out = tmp_path / "adaptive.gpkg"
    result = _run(inputs, "--boundary", str(inputs[2]), "--resolution", "1", "--adaptive", "--output", str(out))
    assert result.exit_code == 0, result.output
    assert "converged at resolution 0.5" in result.output
    surface = _manifest(out)["surface"]
    assert surface["mode"] == "adaptive" and surface["status"] == "converged"
    assert surface["convergence_demonstrated"] is True


def test_strict_adaptive_run_that_does_not_converge_writes_nothing(tmp_path, inputs):
    out = tmp_path / "keep.gpkg"
    assert _run(inputs, "--output", str(out)).exit_code == 0
    before = out.read_bytes()
    # 100 squares allow one grid of about 0.45 m on the 10 x 4 boundary; the
    # next, halved spacing would need about 400 squares.
    result = _run(
        inputs, "--boundary", str(inputs[2]), "--resolution", "0.01", "--max-cells", "100",
        "--adaptive", "--force", "--output", str(out),
    )
    assert result.exit_code == 1
    assert "convergence was not demonstrated" in result.output and "budget_limited" in result.output
    assert "Wrote" not in result.output
    assert out.read_bytes() == before


def test_accepted_nonconverged_output_says_so(tmp_path, inputs):
    out = tmp_path / "bounded.gpkg"
    result = _run(
        inputs, "--boundary", str(inputs[2]), "--resolution", "0.01", "--max-cells", "100",
        "--adaptive", "--accept-nonconverged", "--output", str(out),
    )
    assert result.exit_code == 0, result.output
    assert "NOT verified as converged" in result.output
    manifest = _manifest(out)
    assert manifest["surface"]["status"] == "budget_limited"
    assert manifest["publication"]["accepted_status"] == "budget_limited (accepted by --accept-nonconverged)"


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--refine-factor", "0.5"], "need --adaptive"),
        (["--accept-nonconverged"], "need --adaptive"),
        (["--adaptive"], "need --boundary"),
        (["--attachment", "nearest_reachable_network"], "need --boundary"),
        (["--attachment", "nearest"], "--attachment must be one of"),
    ],
)
def test_wrong_options_fail_before_any_input_is_read(tmp_path, inputs, monkeypatch, extra, message):
    import netvoronoi.cli as cli

    monkeypatch.setattr(cli, "_read", lambda *a, **k: pytest.fail("input read before the option check"))
    result = _run(inputs, *extra, "--output", str(tmp_path / "o.gpkg"))
    assert result.exit_code == 1
    assert message in result.output


def test_bad_adaptive_threshold_fails_before_any_input_is_read(tmp_path, inputs, monkeypatch):
    import netvoronoi.cli as cli

    monkeypatch.setattr(cli, "_read", lambda *a, **k: pytest.fail("input read before the option check"))
    result = _run(
        inputs, "--boundary", str(inputs[2]), "--adaptive", "--changed-area-max", "2", "--output", str(tmp_path / "o.gpkg")
    )
    assert result.exit_code == 1 and "global_changed_area_max" in result.output


def test_failed_write_with_force_keeps_the_previous_output(tmp_path, inputs, monkeypatch):
    out = tmp_path / "keep.gpkg"
    assert _run(inputs, "--output", str(out)).exit_code == 0
    before = out.read_bytes()
    calls = []
    original = io._write

    def fail_third(table, path, layer):
        calls.append(layer)
        if len(calls) == 3:
            raise OSError("simulated disk full")
        original(table, path, layer)

    monkeypatch.setattr(io, "_write", fail_third)
    result = _run(inputs, "--force", "--output", str(out))
    assert result.exit_code != 0
    assert "Wrote" not in result.output
    assert out.read_bytes() == before
