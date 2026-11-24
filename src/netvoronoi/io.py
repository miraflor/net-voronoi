"""GeoPackage output, published as one transaction.

A layer with no rows is not written at all. For example, a network whose
every edge is reachable from a point has no ``network_unassigned`` layer.
Readers should treat a missing layer as an empty one.

Why a transaction. A GeoPackage is written one layer at a time. If the
writer deleted the old file first and then failed at the third layer (a full
disk, a locked file, a library error), the previous valid output would be
gone and a partial file would be left. So the layers are written to a new
file with a temporary name in the same folder, that file is read back and
checked, and only then is it moved onto the target name with one
``os.replace``. Moving a file within one folder replaces the target in a
single step: a reader sees either the old complete file or the new complete
file. Until that step the old file is not touched, even with ``--force``.

Reading order:

1. ``write_network_gpkg`` and ``write_surface_gpkg`` collect the layers;
2. ``_publish`` writes them to the temporary file, calls ``_check_staged``,
   and replaces the target;
3. ``_check_staged`` reads the temporary file back.

Besides the map layers, a file can hold tables without geometry (GeoPackage
"attributes" tables, listed by QGIS and GDAL like any layer):
``netvoronoi_manifest`` (one row, one JSON text: how the file was made) and
``surface_territory_qa`` (one row per cluster, see ``adaptive.territory_qa``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import CRS

from .core import NetworkVoronoiResult
from .model import SpatialNetwork
from .surface import SurfaceVoronoiResult


def check_output_path(path: str | Path, *, overwrite: bool = False) -> Path:
    """Return ``path`` as a ``Path`` after checking that it ends in .gpkg and, unless ``overwrite``, does not exist."""
    path = Path(path)
    if path.suffix.lower() != ".gpkg":
        raise ValueError("output path must end in .gpkg")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    if path.exists() and not path.is_file():
        raise ValueError(f"output path exists and is not a file: {path}")
    return path


class PublicationDurabilityError(RuntimeError):
    """The new output was already published, but directory durability failed.

    Unlike a staged-write failure, this exception means the destination
    has been replaced. Do not assume the old file remains available or
    automatically retry a destructive operation without inspection.
    """

    def __init__(self, target: Path):
        self.target = target
        self.published = True
        super().__init__(
            f"output was replaced at {target}, but directory sync failed; "
            "the new output exists and its crash durability is unconfirmed"
        )


class PublicationCleanupError(RuntimeError):
    """The destination was replaced, but removal of a staging sidecar failed.

    The new output is already in place; its final state must be inspected
    before any retry. This remains separate from directory-sync failure.
    """

    def __init__(self, target: Path):
        self.target = target
        self.published = True
        super().__init__(
            f"output was replaced at {target}, but staging cleanup failed; "
            "inspect the destination and the staging files before retrying"
        )


class OutputInUseError(RuntimeError):
    """The target GeoPackage has SQLite side files, so another program may be using it.

    A GeoPackage is a SQLite database. While a program such as QGIS has it
    open for editing, SQLite keeps ``-wal`` and ``-shm`` files beside it; an
    interrupted write can leave a ``-journal`` file. Those files belong to
    the old database, so the old file is not replaced while they exist.
    Nothing is written. This is a subclass of ``RuntimeError``, as before it
    had its own name, so code that caught ``RuntimeError`` still works.
    """


def _check_target_safe(target: Path) -> None:
    """Refuse a symlink or SQLite sidecar at the destination.

    SQLite WAL/journal files can belong to the *old* GeoPackage. Replacing
    only the main database while leaving those files can corrupt the view
    obtained by the next reader. Never delete sidecars belonging to a
    possibly active process; require the caller to close the database and
    resolve them before publication.
    """
    if target.is_symlink():
        raise ValueError(f"refusing to publish through a symbolic link: {target}")
    sidecars = [Path(str(target) + suffix) for suffix in ("-wal", "-shm", "-journal")]
    occupied = [path for path in sidecars if path.exists() or path.is_symlink()]
    if occupied:
        raise OutputInUseError(
            "refusing to replace GeoPackage with existing SQLite sidecar(s): "
            + ", ".join(str(path) for path in occupied)
        )


def check_output_target(path: str | Path) -> Path:
    """Run the destination checks of ``_publish`` early: no symbolic link, no SQLite side files.

    ``_publish`` runs the same checks again just before it replaces the
    file. Calling this first lets the command line fail in a second, before
    it reads inputs and computes for minutes, instead of at the end.
    """
    path = Path(path)
    _check_target_safe(path)
    return path


def _write(table: pd.DataFrame, path: Path, layer: str) -> None:
    """Write one table as one layer of the GeoPackage at ``path``.

    A GeoDataFrame becomes a map layer; a plain DataFrame becomes a table
    without geometry. Every layer goes through this function, so a test can
    replace it to simulate a failure at any layer.
    """
    if isinstance(table, gpd.GeoDataFrame):
        table.to_file(path, layer=layer, driver="GPKG")
    else:
        pyogrio.write_dataframe(table, path, layer=layer, driver="GPKG")


def _snap_connectors(network: SpatialNetwork) -> gpd.GeoDataFrame:
    """One straight line per point, from its input location to its snapped node."""
    start = shapely.get_coordinates(network.points_input.geometry.to_numpy())
    stop = shapely.get_coordinates(network.points_snapped.geometry.to_numpy())
    lines = shapely.linestrings(np.stack([start, stop], axis=1))
    return gpd.GeoDataFrame(
        {
            "point_id": network.points_input["point_id"].to_numpy(),
            "cluster_id": network.points_input["cluster_id"].to_numpy(),
            "snap_distance": network.points_snapped["snap_distance"].to_numpy(),
        },
        geometry=gpd.GeoSeries(lines, crs=network.crs),
        crs=network.crs,
    )


def _json_value(value):
    """Turn NumPy numbers into Python numbers for ``json.dumps``; refuse anything else."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"manifest value of type {type(value).__name__} cannot be written as JSON")


def _manifest_table(manifest: dict) -> pd.DataFrame:
    """The manifest as a one-row table with one JSON text column.

    ``allow_nan=False`` refuses NaN and infinity, which are not valid JSON;
    such a value means a measurement failed, and the file is not written.
    """
    text = json.dumps(manifest, allow_nan=False, sort_keys=True, indent=1, default=_json_value)
    return pd.DataFrame({"json": [text]})


def _network_layers(network: SpatialNetwork, result: NetworkVoronoiResult) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("network_edges", network.edges),
        ("points_input", network.points_input),
        ("points_snapped", network.points_snapped),
        ("snap_connectors", _snap_connectors(network)),
        ("network_segments", result.segments),
        ("network_cells", result.cells),
        ("network_unassigned", result.unassigned_edges),
    ]


def write_network_gpkg(
    path: str | Path,
    network: SpatialNetwork,
    result: NetworkVoronoiResult,
    *,
    overwrite: bool = False,
    manifest: dict | None = None,
    before_replace: Callable[[], None] | None = None,
) -> Path:
    """Write the network as built and the exact network partition to a GeoPackage, as one transaction."""
    layers = _network_layers(network, result)
    if manifest is not None:
        layers.append(("netvoronoi_manifest", _manifest_table(manifest)))
    return _publish(path, layers, network.crs, overwrite=overwrite, before_replace=before_replace)


def write_surface_gpkg(
    path: str | Path,
    network: SpatialNetwork,
    result: SurfaceVoronoiResult,
    *,
    overwrite: bool = False,
    manifest: dict | None = None,
    territories: pd.DataFrame | None = None,
    before_replace: Callable[[], None] | None = None,
) -> Path:
    """Write everything that ``write_network_gpkg`` writes, plus the surface layers, as one transaction."""
    layers = _network_layers(network, result.network) + [
        ("surface_cells", result.cells),
        ("surface_grid_debug", result.grid),
        ("surface_unassigned", result.unassigned),
        ("surface_point_qa", result.point_qa),
    ]
    if territories is not None:
        layers.append(("surface_territory_qa", pd.DataFrame(territories)))
    if manifest is not None:
        layers.append(("netvoronoi_manifest", _manifest_table(manifest)))
    return _publish(path, layers, network.crs, overwrite=overwrite, before_replace=before_replace)


def _publish(
    path: str | Path,
    layers: list[tuple[str, pd.DataFrame]],
    crs,
    *,
    overwrite: bool,
    before_replace: Callable[[], None] | None = None,
) -> Path:
    """Write ``layers`` to a temporary GeoPackage, check it, then replace ``path`` in one step.

    Steps:

    1. check the target name (``check_output_path``) before touching any file;
    2. reserve a unique temporary name in the target's folder. The same
       folder matters: ``os.replace`` is a single step only within one file
       system, and a temporary folder elsewhere may be on another one;
    3. write every non-empty layer, then read the file back
       (``_check_staged``);
    4. flush the file to disk, then ``os.replace`` it onto the target. This is
       the only step that changes the target. On Windows it fails if another
       program has the target open; that error is raised, and the old file is
       kept. The old file is never deleted to make room;
    5. attempt cleanup of the temporary file and any SQLite side files of
       its temporary name. Cleanup failures after replacement are reported
       distinctly; an earlier pre-replacement error takes precedence if
       cleanup also fails. Files of the target are never removed.
    """
    target = check_output_path(path, overwrite=overwrite)
    _check_target_safe(target)
    layers = [(name, table) for name, table in layers if table is not None and not table.empty]
    if not layers:
        raise ValueError("refusing to write a GeoPackage without any rows")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".netvoronoi-staging-", suffix=".gpkg", dir=target.parent)
    os.close(handle)
    staged = Path(name)
    # The GeoPackage writer creates its own file; the reserved empty file is
    # removed so that the writer does not try to open it as a database. The
    # name stays unique to this call.
    staged.unlink()
    replaced = False
    primary_error: BaseException | None = None
    try:
        for layer, table in layers:
            _write(table, staged, layer)
        _check_staged(staged, layers, crs)
        # Opened for writing because Windows refuses to flush a file opened
        # only for reading.
        with open(staged, "r+b") as stream:
            os.fsync(stream.fileno())
        # The callback is run only after the staged database has passed all
        # checks and been fsynced, but *before* the existing output changes.
        # The CLI uses it to hash every input again and fail if a source was
        # modified during this run. A failure leaves the old output intact.
        if before_replace is not None:
            before_replace()
        _check_target_safe(target)
        if target.exists() and not overwrite:
            raise FileExistsError(f"output appeared while it was being written: {target}")
        os.replace(staged, target)
        replaced = True  # Any subsequent failure MUST be reported as post-publication.
        try:
            _flush_folder(target.parent)
        except Exception as exc:
            raise PublicationDurabilityError(target) from exc
        return target
    except BaseException as exc:
        # Preserve the primary error if cleanup also fails before publication.
        # This includes KeyboardInterrupt and SystemExit, which are re-raised.
        primary_error = exc
        raise
    finally:
        cleanup_error: Exception | None = None
        for leftover in (staged, *(Path(str(staged) + s) for s in ("-journal", "-wal", "-shm"))):
            try:
                if leftover.exists() or leftover.is_symlink():
                    leftover.unlink()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None:
            if replaced:
                # Even if a directory-fsync error was already being raised,
                # never let this become a generic exit-1 failure.
                raise PublicationCleanupError(target) from cleanup_error
            if primary_error is None:
                raise cleanup_error


def _flush_folder(folder: Path) -> None:
    """Ask a POSIX system to store the new folder entry on disk; Windows has no equivalent call.

    This runs after ``os.replace``, when the new file is already in place.
    A failure is never suppressed: ``_publish`` turns it into a typed
    ``PublicationDurabilityError`` that explicitly says replacement already
    happened. Windows does not offer this POSIX directory-fsync operation;
    the staged file itself is still fsynced before replacement.
    """
    if os.name != "posix":
        return
    handle = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _check_staged(staged: Path, layers: list[tuple[str, pd.DataFrame]], crs) -> None:
    """Read the temporary GeoPackage back and compare it with what was written.

    Checks, in order:

    * SQLite's own ``integrity_check`` and ``foreign_key_check`` on a
      read-only connection, and no write-ahead log left beside the file
      (a GeoPackage must be complete without side files);
    * the layer names are exactly the written ones;
    * for each layer: the row count, the column names, and, for a map
      layer, the coordinate reference system (``crs``).

    Any difference raises ``RuntimeError``; the target is then not replaced.
    """
    if Path(str(staged) + "-wal").exists():
        raise RuntimeError("staged GeoPackage still has a write-ahead log beside it")
    uri = staged.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        integrity = db.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise RuntimeError(f"staged GeoPackage failed the SQLite integrity check: {integrity}")
        broken = db.execute("PRAGMA foreign_key_check").fetchone()
        if broken is not None:
            raise RuntimeError(f"staged GeoPackage failed the SQLite foreign key check: {broken}")

    found = {str(row[0]) for row in pyogrio.list_layers(staged)}
    expected = {layer for layer, _ in layers}
    if found != expected:
        raise RuntimeError(f"staged GeoPackage layers differ: found {sorted(found)}, expected {sorted(expected)}")

    expected_crs = CRS.from_user_input(crs)
    for layer, table in layers:
        info = pyogrio.read_info(staged, layer=layer)
        if int(info["features"]) != len(table):
            raise RuntimeError(f"layer {layer}: {info['features']} rows written, {len(table)} expected")
        is_map = isinstance(table, gpd.GeoDataFrame)
        columns = [c for c in table.columns if not (is_map and c == table.geometry.name)]
        if list(info["fields"]) != [str(c) for c in columns]:
            raise RuntimeError(f"layer {layer}: columns {list(info['fields'])}, expected {columns}")
        if is_map:
            same = info["crs"] is not None and CRS.from_user_input(info["crs"]).equals(
                expected_crs, ignore_axis_order=True
            )
            if not same:
                raise RuntimeError(f"layer {layer}: CRS {info['crs']} differs from the network CRS")
