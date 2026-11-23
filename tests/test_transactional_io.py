"""The GeoPackage writer replaces an existing file only after the new one is complete and checked.

Every failure test follows the same pattern: write a valid file, record its
bytes, make one step of a second write fail, and check that the exception
reaches the caller, that the old bytes are unchanged, and that no temporary
file is left in the folder.
"""

import json

import geopandas as gpd
import pyogrio
import pytest

from netvoronoi import from_geodataframes, network_voronoi, surface_voronoi, territory_qa
from netvoronoi import io


@pytest.fixture
def built(line_roads, end_points, boundary):
    network = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    surface = surface_voronoi(network, boundary, resolution=1.0)
    return network, surface, territory_qa(network, boundary, surface)


def _leftovers(folder):
    return sorted(p.name for p in folder.iterdir() if p.name.startswith(".netvoronoi-staging-"))


def _surface_write(path, built, **kwargs):
    network, surface, territories = built
    return io.write_surface_gpkg(
        path, network, surface, manifest={"run": "test"}, territories=territories, **kwargs
    )


def test_successful_write_contains_every_layer_and_the_manifest(tmp_path, built):
    out = _surface_write(tmp_path / "out.gpkg", built)
    layers = dict(pyogrio.list_layers(out))
    assert {"network_edges", "network_segments", "surface_cells", "surface_grid_debug"} <= set(layers)
    # Tables without geometry are listed like layers, with no geometry type.
    assert layers["netvoronoi_manifest"] is None
    assert layers["surface_territory_qa"] is None
    manifest = json.loads(pyogrio.read_dataframe(out, layer="netvoronoi_manifest")["json"].iloc[0])
    assert manifest == {"run": "test"}
    assert len(pyogrio.read_dataframe(out, layer="surface_territory_qa")) == 2
    assert _leftovers(tmp_path) == []


# The fixture writes 11 layers: 6 network layers (no network_unassigned,
# because every edge is reachable), 3 surface layers (no surface_unassigned),
# the territory table and the manifest. The first test checks this number,
# so that the parametrized test below makes each of the 11 writes fail once.
_LAYERS_WRITTEN = 11


def test_fixture_writes_the_expected_number_of_layers(tmp_path, built, monkeypatch):
    written = []
    original = io._write
    monkeypatch.setattr(io, "_write", lambda table, path, layer: (written.append(layer), original(table, path, layer)))
    _surface_write(tmp_path / "out.gpkg", built)
    assert len(written) == _LAYERS_WRITTEN


@pytest.mark.parametrize("failing_call", range(1, _LAYERS_WRITTEN + 1))
def test_failure_at_any_layer_keeps_the_previous_file(tmp_path, built, monkeypatch, failing_call):
    target = _surface_write(tmp_path / "out.gpkg", built)
    before = target.read_bytes()
    calls = []
    original = io._write

    def fail_once(table, path, layer):
        calls.append(layer)
        if len(calls) == failing_call:
            raise OSError(f"simulated disk error while writing {layer}")
        original(table, path, layer)

    monkeypatch.setattr(io, "_write", fail_once)
    with pytest.raises(OSError, match="simulated disk error"):
        _surface_write(target, built, overwrite=True)
    assert target.read_bytes() == before
    assert _leftovers(tmp_path) == []


def test_failed_read_back_keeps_the_previous_file(tmp_path, built, monkeypatch):
    target = _surface_write(tmp_path / "out.gpkg", built)
    before = target.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated read-back failure")

    monkeypatch.setattr(io, "_check_staged", fail)
    with pytest.raises(RuntimeError, match="simulated read-back failure"):
        _surface_write(target, built, overwrite=True)
    assert target.read_bytes() == before
    assert _leftovers(tmp_path) == []


def test_failed_replace_keeps_the_previous_file(tmp_path, built, monkeypatch):
    target = _surface_write(tmp_path / "out.gpkg", built)
    before = target.read_bytes()

    def fail(source, destination):
        raise PermissionError("simulated locked destination")

    monkeypatch.setattr(io.os, "replace", fail)
    with pytest.raises(PermissionError, match="simulated locked destination"):
        _surface_write(target, built, overwrite=True)
    assert target.read_bytes() == before
    assert _leftovers(tmp_path) == []


def test_read_back_detects_a_layer_with_missing_rows(tmp_path, built, monkeypatch):
    target = _surface_write(tmp_path / "out.gpkg", built)
    before = target.read_bytes()
    original = io._write

    def drop_a_row(table, path, layer):
        original(table.iloc[1:] if layer == "network_segments" else table, path, layer)

    monkeypatch.setattr(io, "_write", drop_a_row)
    with pytest.raises(RuntimeError, match="network_segments: 1 rows written, 2 expected"):
        _surface_write(target, built, overwrite=True)
    assert target.read_bytes() == before


def test_existing_file_without_overwrite_is_untouched(tmp_path, built):
    target = _surface_write(tmp_path / "out.gpkg", built)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        _surface_write(target, built)
    assert target.read_bytes() == before


def test_manifest_with_nan_is_refused_and_the_previous_file_kept(tmp_path, built):
    network, surface, _ = built
    target = _surface_write(tmp_path / "out.gpkg", built)
    before = target.read_bytes()
    with pytest.raises(ValueError):
        io.write_surface_gpkg(target, network, surface, overwrite=True, manifest={"bad": float("nan")})
    assert target.read_bytes() == before
    assert _leftovers(tmp_path) == []


def test_network_only_write_is_also_transactional(tmp_path, line_roads, end_points, monkeypatch):
    network = from_geodataframes(line_roads, end_points, cluster_col="cluster")
    result = network_voronoi(network)
    target = io.write_network_gpkg(tmp_path / "net.gpkg", network, result)
    before = target.read_bytes()
    monkeypatch.setattr(io, "_write", lambda *a, **k: (_ for _ in ()).throw(OSError("simulated")))
    with pytest.raises(OSError, match="simulated"):
        io.write_network_gpkg(target, network, result, overwrite=True)
    assert target.read_bytes() == before


def test_flush_opens_the_staged_file_for_writing(tmp_path, built, monkeypatch):
    """Windows refuses to flush (``os.fsync``) a file opened only for reading.

    This cannot be tested on Linux directly, so the test checks the mode in
    which the writer opens the file for flushing.
    """
    modes = []
    real_open = open

    def recording_open(file, mode="r", *args, **kwargs):
        modes.append(mode)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", recording_open, raising=False)
    _surface_write(tmp_path / "out.gpkg", built)
    assert modes == ["r+b"]


def test_layers_keep_their_crs(tmp_path, built, crs):
    out = _surface_write(tmp_path / "out.gpkg", built)
    for layer in ("network_edges", "surface_cells"):
        assert gpd.read_file(out, layer=layer).crs == crs
