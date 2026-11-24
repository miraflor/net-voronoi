"""V3 regression tests: domain identity, source consistency, and atomic publication.

The tests deliberately inject failures at the last possible pre-replace step.
No failure may corrupt a previously valid destination. A directory-sync failure
is different: replacement already happened, and the typed error must say so.
"""

from dataclasses import replace

import geopandas as gpd
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from netvoronoi import adaptive_surface_voronoi, from_geodataframes, surface_voronoi
from netvoronoi.adaptive import compare_surface_ownership, territory_qa
from netvoronoi.cli import InputChangedError, _file_record, _verify_input_records
from netvoronoi import io
from netvoronoi.surface import SurfaceVoronoiResult

CRS = "EPSG:3857"


def _rendering(geometry, crs=CRS):
    cells = gpd.GeoDataFrame({"cluster_id": ["A"]}, geometry=[geometry], crs=crs)
    grid = cells.copy()
    missing = gpd.GeoDataFrame(geometry=[], crs=crs)
    return SurfaceVoronoiResult(cells, grid, missing, None, None)


def _line():
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs=CRS)
    points = gpd.GeoDataFrame(
        {"cluster": ["A", "B"]}, geometry=[Point(0, 0), Point(10, 0)], crs=CRS
    )
    network = from_geodataframes(roads, points, cluster_col="cluster")
    boundary = gpd.GeoDataFrame(geometry=[box(0, -1, 10, 1)], crs=CRS)
    return network, boundary


def _no_staging(folder):
    return not any(item.name.startswith(".netvoronoi-staging-") for item in folder.iterdir())


# Geographic-domain safety, including the fallback for legacy dataclasses.

def test_equal_area_disjoint_domains_are_rejected():
    with pytest.raises(ValueError, match="same geographic domain"):
        compare_surface_ownership(_rendering(box(0, 0, 1, 1)), _rendering(box(5, 0, 6, 1)))


def test_equal_area_shifted_overlapping_domains_are_rejected():
    with pytest.raises(ValueError, match="same geographic domain"):
        compare_surface_ownership(_rendering(box(0, 0, 1, 1)), _rendering(box(0.5, 0, 1.5, 1)))


def test_comparison_rejects_different_crs():
    with pytest.raises(ValueError, match="coordinate reference systems"):
        compare_surface_ownership(_rendering(box(0, 0, 1, 1)), _rendering(box(0, 0, 1, 1), "EPSG:32651"))


def test_comparison_rejects_different_attachment_policies():
    a = _rendering(box(0, 0, 1, 1))
    with pytest.raises(ValueError, match="attachment policies"):
        compare_surface_ownership(a, replace(a, attachment="nearest_reachable_network"))


def test_rendering_carries_domain_for_fast_comparison(monkeypatch):
    network, boundary = _line()
    a = surface_voronoi(network, boundary, resolution=1)
    b = surface_voronoi(network, boundary, resolution=0.5)
    assert a.domain_geometry.equals(b.domain_geometry)

    # If the source boundary is available, no union over the whole grid is
    # needed. The comparison does not create a 2-D union of all grid cells.
    def forbid_union(*args, **kwargs):
        raise AssertionError("grid union should not be required")

    monkeypatch.setattr(shapely, "union_all", forbid_union)
    assert compare_surface_ownership(a, b)["changed_owner_fraction"] == pytest.approx(0.0)


def test_convergence_manifest_disclaims_exactness_and_territory_statuses():
    network, boundary = _line()
    result = adaptive_surface_voronoi(network, boundary, start_resolution=1)
    assert result.manifest["convergence_scope"].startswith("two consecutive grids")
    assert result.manifest["territory_completeness_proven"] is False
    assert set(result.territories.territory_evidence) == {"positive_area_observed"}


def test_undersampled_road_is_proven_lost_but_not_invalid():
    network, boundary = _line()
    surface = surface_voronoi(network, boundary, resolution=20)
    table = territory_qa(network, boundary, surface).set_index("cluster_id")
    assert table.loc["B", "territory_status"] == "grid_lost"
    assert table.loc["B", "territory_evidence"] == "positive_area_proven_but_not_rendered"


def test_external_road_is_not_proven_absent():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(100, 0), (200, 0)])], crs=CRS
    )
    points = gpd.GeoDataFrame(
        {"cluster": ["A", "B", "C"]},
        geometry=[Point(0, 0), Point(90, 0), Point(200, 0)], crs=CRS,
    )
    network = from_geodataframes(roads, points, cluster_col="cluster")
    boundary = gpd.GeoDataFrame(geometry=[box(0, -20, 100, 20)], crs=CRS)
    surface = surface_voronoi(network, boundary, resolution=5)
    table = territory_qa(network, boundary, surface).set_index("cluster_id")
    assert table.loc["C", "territory_status"] == "no_road_inside"
    assert table.loc["C", "territory_evidence"] == "presence_undetermined"


# Input snapshots are based on actual content, not just mtime or file size.

def _snapshots(path):
    return {
        "roads": _file_record(path, None),
        "points": None,
        "boundary": None,
    }


def test_file_snapshot_detects_equal_length_replacement(tmp_path):
    source = tmp_path / "roads.gpkg"
    source.write_bytes(b"aaaa")
    snapshot = _snapshots(source)
    source.write_bytes(b"bbbb")
    with pytest.raises(InputChangedError, match="roads input changed"):
        _verify_input_records(snapshot)


def test_directory_snapshot_is_order_independent_and_detects_changes(tmp_path):
    root = tmp_path / "source.gdb"
    root.mkdir()
    (root / "b.bin").write_bytes(b"AB")
    (root / "a.bin").write_bytes(b"CD")
    snapshot = _snapshots(root)
    assert snapshot["roads"]["kind"] == "folder"
    assert snapshot["roads"]["files"] == 2
    _verify_input_records(snapshot)
    (root / "a.bin").write_bytes(b"CE")
    with pytest.raises(InputChangedError, match="roads input changed"):
        _verify_input_records(snapshot)


def test_directory_snapshot_detects_added_file(tmp_path):
    root = tmp_path / "source.gdb"
    root.mkdir()
    (root / "a.bin").write_bytes(b"AB")
    snapshot = _snapshots(root)
    (root / "new.bin").write_bytes(b"X")
    with pytest.raises(InputChangedError):
        _verify_input_records(snapshot)


# Publication checks happen before os.replace and cleanup must always run.

def test_sidecar_blocks_replacement_without_touching_old_file(tmp_path):
    target = tmp_path / "old.gpkg"
    target.write_bytes(b"old file")
    sidecar = tmp_path / "old.gpkg-wal"
    sidecar.write_bytes(b"active WAL")
    with pytest.raises(RuntimeError, match="SQLite sidecar"):
        io._publish(target, [("metadata", pd.DataFrame({"value": [1]}))], CRS, overwrite=True)
    assert target.read_bytes() == b"old file"
    assert sidecar.read_bytes() == b"active WAL"
    assert _no_staging(tmp_path)


@pytest.mark.parametrize("suffix", ["-shm", "-journal"])
def test_other_sqlite_sidecars_block_replacement(tmp_path, suffix):
    target = tmp_path / "old.gpkg"
    target.write_bytes(b"old file")
    (tmp_path / ("old.gpkg" + suffix)).write_bytes(b"busy")
    with pytest.raises(RuntimeError, match="SQLite sidecar"):
        io._publish(target, [("metadata", pd.DataFrame({"value": [1]}))], CRS, overwrite=True)
    assert target.read_bytes() == b"old file"


def test_failed_precommit_input_verification_preserves_old_output(tmp_path):
    target = tmp_path / "old.gpkg"
    target.write_bytes(b"old file")
    source = tmp_path / "source.dat"
    source.write_bytes(b"initial")
    snapshots = _snapshots(source)
    source.write_bytes(b"changed")
    with pytest.raises(InputChangedError):
        io._publish(
            target, [("metadata", pd.DataFrame({"value": [1]}))], CRS, overwrite=True,
            before_replace=lambda: _verify_input_records(snapshots),
        )
    assert target.read_bytes() == b"old file"
    assert _no_staging(tmp_path)


def test_post_replace_directory_failure_has_explicit_published_state(tmp_path, monkeypatch):
    target = tmp_path / "old.gpkg"
    target.write_bytes(b"old file")

    def fail(_folder):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(io, "_flush_folder", fail)
    with pytest.raises(io.PublicationDurabilityError) as caught:
        io._publish(target, [("metadata", pd.DataFrame({"value": [1]}))], CRS, overwrite=True)
    assert caught.value.published is True
    assert caught.value.target == target
    assert target.read_bytes() != b"old file"
    assert _no_staging(tmp_path)


def test_successful_publication_remains_transactional(tmp_path):
    target = tmp_path / "new.gpkg"
    io._publish(target, [("metadata", pd.DataFrame({"value": [1]}))], CRS, overwrite=False)
    assert target.exists() and _no_staging(tmp_path)
