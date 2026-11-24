"""The ``netvoronoi build`` command.

Reading order:

1. ``build`` declares the options and turns input errors into one
   ``Error: ...`` line with exit code 1;
2. ``_check_options`` checks every option that can be checked without
   reading a file, so a wrong option fails in a second, not after a large
   file has been read and the network built;
3. ``_build`` reads the inputs, builds the network, runs one of three modes
   (exact network only; fixed-spacing surface; adaptive surface), and writes
   one GeoPackage as a transaction (see ``io.py``);
4. ``_manifest`` records how the file was made; it is written into the same
   GeoPackage as the table ``netvoronoi_manifest``.

Exit codes. A script needs to know whether the target file changed, so the
codes separate "nothing was written" from "the file was written":

* 0: the output was written and checked;
* 1: nothing was written, and an existing output is unchanged (an input or
  option error, convergence not demonstrated, the output in use by another
  program, an input that changed during the run, or any failure before the
  replacement step);
* 2: a command-line usage error found by Typer before the command ran
  (for example an unknown option); nothing was written;
* 3: the output WAS replaced, but directory sync or post-publication
  cleanup failed (``PublicationDurabilityError`` or
  ``PublicationCleanupError``). Inspect the new destination before retrying;
  crash durability may be unconfirmed when folder sync failed.
"""

from __future__ import annotations

import hashlib
import math
import platform
from pathlib import Path

import geopandas as gpd
import numpy
import pyogrio
import scipy
import shapely
import typer

from . import __version__
from .adaptive import ConvergenceError, ConvergencePolicy, adaptive_surface_voronoi, territory_qa
from .core import network_voronoi
from .io import (
    OutputInUseError,
    PublicationCleanupError,
    PublicationDurabilityError,
    check_output_path,
    check_output_target,
    write_network_gpkg,
    write_surface_gpkg,
)
from .network import from_geodataframes
from .surface import ATTACHMENT_POLICIES, DEFAULT_MAX_CELLS, check_grid_size, surface_voronoi

app = typer.Typer(add_completion=False, no_args_is_help=True)

_PARQUET_SUFFIXES = {".parquet", ".geoparquet"}

EXIT_NOTHING_WRITTEN = 1
EXIT_WRITTEN_DURABILITY_UNCONFIRMED = 3  # Post-replacement error; 2 is used for Typer usage errors

# Defaults of the adaptive options. The options themselves default to None,
# which means "not given", so that ``_check_options`` can refuse them when
# ``--adaptive`` is absent.
_REFINE_FACTOR = 0.5
_MAX_REFINEMENTS = 5


def _read(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    """Read a vector file: ``.parquet`` and ``.geoparquet`` with ``read_parquet``, all others with ``read_file``."""
    if path.suffix.lower() in _PARQUET_SUFFIXES:
        if layer is not None:
            raise typer.BadParameter("layer selection is not supported for Parquet input")
        try:
            return gpd.read_parquet(path)
        except ImportError as exc:
            raise typer.BadParameter(
                "Parquet input requires pyarrow; install netvoronoi[parquet] or pyarrow"
            ) from exc
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def _finite_nonnegative(value: float | None) -> float | None:
    """Option check: accept ``None`` or a finite number that is not negative."""
    if value is not None and (not math.isfinite(value) or value < 0):
        raise typer.BadParameter("must be finite and non-negative")
    return value


def _finite_positive(value: float) -> float:
    """Option check: accept a finite number greater than zero."""
    if not math.isfinite(value) or value <= 0:
        raise typer.BadParameter("must be finite and positive")
    return value


@app.callback()
def main() -> None:
    """Cluster-based Voronoi regions informed by shortest-path distance on a spatial network."""


@app.command("build")
def build(
    roads_path: Path = typer.Argument(..., exists=True, readable=True),
    points_path: Path = typer.Argument(..., exists=True, readable=True),
    cluster_col: str = typer.Option(..., "--cluster-col", help="Point attribute containing the cluster label."),
    point_id_col: str | None = typer.Option(None, "--point-id-col"),
    output: Path = typer.Option(Path("netvoronoi.gpkg"), "--output", "-o"),
    roads_layer: str | None = typer.Option(None, "--roads-layer"),
    points_layer: str | None = typer.Option(None, "--points-layer"),
    max_snap_distance: float | None = typer.Option(None, "--max-snap-distance", callback=_finite_nonnegative),
    boundary_path: Path | None = typer.Option(None, "--boundary", exists=True, readable=True),
    boundary_layer: str | None = typer.Option(None, "--boundary-layer"),
    resolution: float = typer.Option(
        100.0, "--resolution", callback=_finite_positive,
        help="Grid spacing in CRS units. With --adaptive: the first spacing to try.",
    ),
    max_cells: int = typer.Option(
        DEFAULT_MAX_CELLS, "--max-cells", min=1, help="Most grid squares (before clipping) of any one grid.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace an existing GeoPackage, only after the new one is written and checked.",
    ),
    attachment: str = typer.Option(
        "nearest_network", "--attachment",
        help="Roads that land may attach to: nearest_network (every road) or "
        "nearest_reachable_network (only roads on components that contain a point). Needs --boundary.",
    ),
    adaptive: bool = typer.Option(
        False, "--adaptive",
        help="Refine the grid until two consecutive grids agree, or a limit is reached. Needs --boundary.",
    ),
    refine_factor: float | None = typer.Option(
        None, "--refine-factor",
        help="With --adaptive: next spacing = spacing x factor, 0 < factor < 1. [default: 0.5]",
    ),
    max_refinements: int | None = typer.Option(
        None, "--max-refinements", help="With --adaptive: finer grids allowed after the first. [default: 5]",
    ),
    changed_area_max: float | None = typer.Option(
        None, "--changed-area-max",
        help="With --adaptive: largest share of land that may change owner between the last two grids. "
        "[default: 0.005]",
    ),
    min_cluster_iou: float | None = typer.Option(
        None, "--min-cluster-iou",
        help="With --adaptive: smallest IoU every cluster must have between the last two grids. "
        "[default: not required]",
    ),
    allow_missing_territories: bool = typer.Option(
        False, "--allow-missing-territories",
        help="With --adaptive: clusters lost only because of the grid do not block convergence (still reported).",
    ),
    accept_nonconverged: bool = typer.Option(
        False, "--accept-nonconverged",
        help="With --adaptive: write the finest valid grid even when convergence was not demonstrated. "
        "The status is recorded in the file; the surface is NOT verified as converged.",
    ),
) -> None:
    """Build the exact network partition and optionally a 2-D surface rendering."""
    options = dict(
        output=output, force=force, boundary_path=boundary_path, resolution=resolution, max_cells=max_cells,
        attachment=attachment, adaptive=adaptive, refine_factor=refine_factor, max_refinements=max_refinements,
        changed_area_max=changed_area_max, min_cluster_iou=min_cluster_iou,
        allow_missing_territories=allow_missing_territories, accept_nonconverged=accept_nonconverged,
    )
    try:
        policy = _check_options(**options)
        _build(
            roads_path,
            points_path,
            cluster_col=cluster_col,
            point_id_col=point_id_col,
            roads_layer=roads_layer,
            points_layer=points_layer,
            max_snap_distance=max_snap_distance,
            boundary_layer=boundary_layer,
            policy=policy,
            **options,
        )
    except (ValueError, FileExistsError) as exc:
        # Input problems are reported as one line, not as a traceback.
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_NOTHING_WRITTEN) from exc
    except OutputInUseError as exc:
        typer.echo(
            f"Error: {exc}. Nothing was written. Close the GeoPackage in QGIS or any other program; "
            "do not delete SQLite -wal/-journal files manually. Close the applications and use "
            "SQLite-aware recovery or checkpointing before retrying.",
            err=True,
        )
        raise typer.Exit(code=EXIT_NOTHING_WRITTEN) from exc
    except (PublicationDurabilityError, PublicationCleanupError) as exc:
        # Both failures happen after the destination has been replaced. A
        # cleanup error must not be mistaken for a pre-publication failure.
        typer.echo(f"Error: {exc}. Inspect the published destination before retrying.", err=True)
        raise typer.Exit(code=EXIT_WRITTEN_DURABILITY_UNCONFIRMED) from exc
    except ConvergenceError as exc:
        typer.echo(
            f"Error: convergence was not demonstrated ({exc}). Nothing was written. Options: raise --max-cells "
            "or --max-refinements, relax --changed-area-max, or pass --accept-nonconverged to write this "
            "result with its status recorded.",
            err=True,
        )
        raise typer.Exit(code=EXIT_NOTHING_WRITTEN) from exc


def _check_options(
    *,
    output: Path,
    force: bool,
    boundary_path: Path | None,
    resolution: float,
    max_cells: int,
    attachment: str,
    adaptive: bool,
    refine_factor: float | None,
    max_refinements: int | None,
    changed_area_max: float | None,
    min_cluster_iou: float | None,
    allow_missing_territories: bool,
    accept_nonconverged: bool,
) -> ConvergencePolicy | None:
    """Check the options that need no file; return the convergence policy (``None`` without ``--adaptive``)."""
    check_output_path(output, overwrite=force)
    # The output must not be a symbolic link or have SQLite side files.
    # ``io._publish`` checks this again just before replacing the file.
    check_output_target(output)
    if attachment not in ATTACHMENT_POLICIES:
        raise ValueError(f"--attachment must be one of {', '.join(ATTACHMENT_POLICIES)}")
    if boundary_path is None and (adaptive or attachment != "nearest_network"):
        raise ValueError("--adaptive and --attachment need --boundary")
    adaptive_only = {
        "--refine-factor": refine_factor is not None,
        "--max-refinements": max_refinements is not None,
        "--changed-area-max": changed_area_max is not None,
        "--min-cluster-iou": min_cluster_iou is not None,
        "--allow-missing-territories": allow_missing_territories,
        "--accept-nonconverged": accept_nonconverged,
    }
    given = [name for name, used in adaptive_only.items() if used]
    if given and not adaptive:
        raise ValueError(f"{', '.join(given)} need --adaptive")
    if not adaptive:
        return None
    factor = _REFINE_FACTOR if refine_factor is None else refine_factor
    if not math.isfinite(factor) or not 0.0 < factor < 1.0:
        raise ValueError("--refine-factor must be strictly between 0 and 1")
    if max_refinements is not None and max_refinements < 0:
        raise ValueError("--max-refinements must be 0 or more")
    # ConvergencePolicy checks that both thresholds lie in [0, 1].
    return ConvergencePolicy(
        global_changed_area_max=0.005 if changed_area_max is None else changed_area_max,
        min_cluster_iou=min_cluster_iou,
        require_all_territories=not allow_missing_territories,
    )


def _build(
    roads_path: Path,
    points_path: Path,
    *,
    cluster_col: str,
    point_id_col: str | None,
    roads_layer: str | None,
    points_layer: str | None,
    max_snap_distance: float | None,
    boundary_layer: str | None,
    policy: ConvergencePolicy | None,
    output: Path,
    force: bool,
    boundary_path: Path | None,
    resolution: float,
    max_cells: int,
    attachment: str,
    adaptive: bool,
    refine_factor: float | None,
    max_refinements: int | None,
    changed_area_max: float | None,
    min_cluster_iou: float | None,
    allow_missing_territories: bool,
    accept_nonconverged: bool,
) -> None:
    """Read the inputs, build the network, compute the partition, and write the GeoPackage."""
    # Never overwrite an input in-place: that would invalidate provenance
    # and destroy the user's source data (even if --force was supplied).
    for input_path in (roads_path, points_path, boundary_path):
        if input_path is None:
            continue
        resolved_output = output.resolve(strict=False)
        resolved_input = input_path.resolve(strict=False)
        if resolved_output == resolved_input or (
            input_path.is_dir() and resolved_input in resolved_output.parents
        ):
            raise ValueError(f"output would overwrite or modify input source {input_path}")

    # Snapshot *before* the data are read. v2 hashed only after loading,
    # which could record different bytes from those parsed during a race.
    # The same snapshots are rechecked after loading and immediately before
    # atomic replacement; all three checkpoints must agree.
    inputs = {
        "roads": _file_record(roads_path, roads_layer),
        "points": _file_record(points_path, points_layer),
        "boundary": None if boundary_path is None else _file_record(boundary_path, boundary_layer),
        "cluster_col": cluster_col,
        "point_id_col": point_id_col,
        "max_snap_distance": max_snap_distance,
    }
    roads = _read(roads_path, roads_layer)
    points = _read(points_path, points_layer)
    boundary = None if boundary_path is None else _read(boundary_path, boundary_layer)
    _verify_input_records(inputs)

    # A fixed grid that is too large is refused before the network is built.
    # The adaptive mode chooses a spacing that fits instead.
    if boundary is not None and not adaptive and roads.crs is not None and roads.crs.is_projected:
        check_grid_size(boundary, roads.crs, resolution, max_cells)

    network = from_geodataframes(
        roads,
        points,
        cluster_col=cluster_col,
        point_id_col=point_id_col,
        max_snap_distance=max_snap_distance,
    )
    # Only publish an output if every input still has its snapshotted
    # content after the potentially long graph and rendering computation.
    verify_before_publish = lambda: _verify_input_records(inputs)

    if boundary is None:
        result = network_voronoi(network)
        manifest = _manifest(inputs, network, surface=None, accepted="exact_network_only")
        write_network_gpkg(
            output, network, result, overwrite=force, manifest=manifest,
            before_replace=verify_before_publish,
        )
        described = "exact network partition"
    elif not adaptive:
        surface = surface_voronoi(
            network, boundary, resolution=resolution, max_cells=max_cells, attachment=attachment
        )
        section = {
            "mode": "fixed",
            "surface_method": "grid rendering of nearest-network attachment",
            "attachment": attachment,
            "resolution": resolution,
            "max_cells": max_cells,
            "road_tie_rule": "equal network distance: smallest cluster_id in string order",
            "planar_tie_rule": "equally near roads: smallest edge_id in string order",
            "convergence_demonstrated": False,
            "note": "one grid; no stability between grids was measured",
        }
        manifest = _manifest(inputs, network, surface=section, accepted="fixed_resolution")
        territories = territory_qa(network, boundary, surface)
        write_surface_gpkg(
            output, network, surface, overwrite=force, manifest=manifest,
            territories=territories, before_replace=verify_before_publish,
        )
        described = f"surface at fixed resolution {resolution:g}"
    else:
        # ConvergenceError (when not accepted) is raised here, before any file is written.
        result = adaptive_surface_voronoi(
            network,
            boundary,
            start_resolution=resolution,
            max_cells=max_cells,
            refine_factor=_REFINE_FACTOR if refine_factor is None else refine_factor,
            max_refinements=_MAX_REFINEMENTS if max_refinements is None else max_refinements,
            attachment=attachment,
            convergence=policy,
            require_convergence=not accept_nonconverged,
        )
        section = {"mode": "adaptive", **result.manifest}
        accepted = result.status
        if not result.convergence_demonstrated:
            accepted += " (accepted by --accept-nonconverged)"
        manifest = _manifest(inputs, network, surface=section, accepted=accepted)
        write_surface_gpkg(
            output, network, result.surface, overwrite=force, manifest=manifest,
            territories=result.territories, before_replace=verify_before_publish,
        )
        described = f"surface {result.status} at resolution {result.selected_resolution:g}"
        if not result.convergence_demonstrated:
            described += " (NOT verified as converged)"

    unit = network.distance_unit or "network CRS units"
    typer.echo(f"Wrote {output} (distance unit: {unit}; {described})")


class InputChangedError(ValueError):
    """A source changed between ingestion and publication; no output was replaced."""


def _digest_file(path: Path) -> tuple[int, str]:
    """SHA-256 and size from one file read, without trusting mutable metadata."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _combined_digest(entries: list[tuple[str, int, str]]) -> str:
    """One SHA-256 for several files, from ``(name, size, sha256)`` triples in the given order.

    Each name is preceded by its length, so that two different lists of
    names can never produce the same sequence of bytes.
    """
    digest = hashlib.sha256()
    for name, size, file_hash in entries:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest()


# Files that GDAL reads together when the user names only one of them.
# A shapefile keeps geometry in .shp, the record index in .shx, the
# attributes (including the cluster column) in .dbf, the CRS in .prj and
# the text encoding in .cpg. GDAL 3.13+ can read extended DBF field names
# from .shp.xml, so those sidecars affect interpreted inputs too. Spatial
# indexes (.qix, .sbn, .sbx) do not change feature values and are omitted.
_SHAPEFILE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".shp.xml")
# A SQLite database in WAL mode (QGIS uses it for GeoPackages it edits)
# keeps recently committed changes in a "-wal" file beside the main file
# until they are copied back; a reader sees those changes, so they are data.
_SQLITE_SUFFIXES = (".gpkg", ".sqlite", ".db")


def _dataset_files(path: Path) -> list[Path]:
    """The files that make up the dataset named by ``path``, in a fixed order.

    * ``points.shp``: every file in the same folder named ``points`` plus
      one of ``_SHAPEFILE_PARTS`` (including ``points.shp.xml``), in any
      letter case (``POINTS.DBF``
      counts, because GDAL also finds it);
    * ``roads.gpkg`` (or ``.sqlite``, ``.db``): the file, and
      ``roads.gpkg-wal`` when it exists;
    * any other file: the file alone.

    Other files with the same base name, such as ``points.csv`` or an output
    ``points.gpkg`` written beside the input, are not part of the dataset and
    are not included: an output must never count as an input.

    Example: a folder holding ``points.shp``, ``points.shx``, ``points.dbf``,
    ``points.prj`` and ``points.csv`` gives the first four for ``points.shp``.
    """
    suffix = path.suffix.lower()
    if suffix == ".shp":
        wanted = {path.stem.lower() + part for part in _SHAPEFILE_PARTS}
        files = [p for p in path.parent.iterdir() if p.name.lower() in wanted]
        return sorted(files, key=lambda p: (p.name.lower(), p.name))
    if suffix in _SQLITE_SUFFIXES:
        wal = path.with_name(path.name + "-wal")
        return [path, wal] if wal.exists() or wal.is_symlink() else [path]
    return [path]


def _file_record(path: Path, layer: str | None) -> dict:
    """Snapshot a vector input by content: a single file, a set of files, or a folder.

    * A single file gives ``kind="file"`` and the SHA-256 of its bytes.
    * A dataset of several files (a shapefile; a GeoPackage with a ``-wal``
      file; see ``_dataset_files``) gives ``kind="file_set"``, one SHA-256
      over all members, and a ``members`` list with each file's name, size
      and SHA-256. Before this, only the named file was hashed, so a change
      to the cluster labels in a shapefile's ``.dbf`` went unnoticed.
    * A folder dataset (for example a File Geodatabase) gives
      ``kind="folder"``: every file in it, in sorted relative-path order.

    Symlinks are rejected rather than followed, since they could point
    outside the snapshot or change targets. A changed member blocks
    publication (``_verify_input_records``).
    """
    path = Path(path)
    record = {"path": str(path), "layer": layer}
    if path.is_symlink():
        raise ValueError(f"input source must not be a symbolic link: {path}")
    if path.is_dir():
        files = []
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"directory input contains a symbolic link: {child}")
            if child.is_file():
                files.append(child)
        files.sort(key=lambda child: child.relative_to(path).as_posix())
        entries = [(child.relative_to(path).as_posix(), *_digest_file(child)) for child in files]
        record.update(
            kind="folder",
            files=len(entries),
            bytes=sum(size for _, size, _ in entries),
            sha256=_combined_digest(entries),
        )
        return record
    if not path.is_file():
        raise ValueError(f"input source is not a readable regular file: {path}")
    members = _dataset_files(path)
    for member in members:
        if member.is_symlink():
            raise ValueError(f"input dataset contains a symbolic link: {member}")
        if not member.is_file():
            raise ValueError(f"input dataset member is not a regular file: {member}")
    if len(members) == 1:
        size, file_hash = _digest_file(path)
        record.update(kind="file", bytes=size, sha256=file_hash)
        return record
    entries = [(member.name, *_digest_file(member)) for member in members]
    record.update(
        kind="file_set",
        files=len(entries),
        bytes=sum(size for _, size, _ in entries),
        sha256=_combined_digest(entries),
        members=[{"name": name, "bytes": size, "sha256": file_hash} for name, size, file_hash in entries],
    )
    return record


def _code_fingerprint(package: Path | None = None) -> str | None:
    """SHA-256 of the package's Python source, identical on Windows and Linux.

    Why: between releases, ``main`` keeps the version string of the last
    release, so the version alone cannot tell the released code from later
    code. This hash can. Every ``.py`` file of the package contributes its
    name and its content, taken in name order. Windows line endings (CRLF)
    are turned into LF first, so a Windows checkout and a Linux checkout of
    the same commit give the same hash. Returns ``None`` when no source file
    is found (for example an installation with compiled files only).
    """
    folder = Path(__file__).resolve().parent if package is None else Path(package)
    sources = sorted(folder.glob("*.py"), key=lambda p: p.name)
    if not sources:
        return None
    entries = []
    for source in sources:
        text = source.read_bytes().replace(b"\r\n", b"\n")
        entries.append((source.name, len(text), hashlib.sha256(text).hexdigest()))
    return _combined_digest(entries)


def _verify_input_records(inputs: dict) -> None:
    """Fail before output publication if any source differs from its first snapshot."""
    for name in ("roads", "points", "boundary"):
        original = inputs[name]
        if original is None:
            continue
        try:
            current = _file_record(Path(original["path"]), original["layer"])
        except (OSError, ValueError) as exc:
            raise InputChangedError(f"{name} input became unreadable or changed: {exc}") from exc
        if current != original:
            raise InputChangedError(
                f"{name} input changed after its initial snapshot; refusing to publish a stale result"
            )


def _manifest(inputs: dict, network, *, surface: dict | None, accepted: str) -> dict:
    """How the output was made, as one JSON-ready dictionary."""
    return {
        "manifest_schema": "netvoronoi.manifest.v1",
        "software": {
            "netvoronoi": __version__,
            # The version stays at the last release on main; this hash tells
            # the exact code apart (see _code_fingerprint).
            "netvoronoi_code_sha256": _code_fingerprint(),
            "python": platform.python_version(),
            "geopandas": gpd.__version__,
            "numpy": numpy.__version__,
            "pyogrio": pyogrio.__version__,
            "scipy": scipy.__version__,
            "shapely": shapely.__version__,
        },
        "inputs": inputs,
        "network": {
            "crs": network.crs.to_string(),
            "distance_unit": network.distance_unit,
            "nodes": len(network.nodes),
            "edges": len(network.edges),
            "points": len(network.points_snapped),
            "clusters": len(network.cluster_ids),
        },
        "exact_network": {
            "method": "multi-source shortest paths and exact split of every edge (network Voronoi diagram)",
            "road_tie_rule": "equal network distance: smallest cluster_id in string order",
        },
        "surface": surface,
        "publication": {
            "accepted_status": accepted,
            "input_consistency": "SHA-256 of every file of each input dataset, checked before reading, "
            "after reading and before atomic replacement",
            "durability": "staged file fsync; directory fsync on POSIX; not guaranteed on Windows",
        },
    }


if __name__ == "__main__":
    app()
