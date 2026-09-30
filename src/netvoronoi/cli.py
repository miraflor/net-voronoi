from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import typer

from .core import network_voronoi
from .io import check_output_path, write_network_gpkg, write_surface_gpkg
from .network import from_geodataframes
from .surface import DEFAULT_MAX_CELLS, check_grid_size, surface_voronoi

app = typer.Typer(add_completion=False, no_args_is_help=True)

_PARQUET_SUFFIXES = {".parquet", ".geoparquet"}


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
    resolution: float = typer.Option(100.0, "--resolution", callback=_finite_positive),
    max_cells: int = typer.Option(DEFAULT_MAX_CELLS, "--max-cells", min=1),
    force: bool = typer.Option(False, "--force", help="Replace an existing GeoPackage."),
) -> None:
    """Build the exact network partition and optionally a 2-D surface rendering."""
    try:
        _build(
            roads_path,
            points_path,
            cluster_col=cluster_col,
            point_id_col=point_id_col,
            output=output,
            roads_layer=roads_layer,
            points_layer=points_layer,
            max_snap_distance=max_snap_distance,
            boundary_path=boundary_path,
            boundary_layer=boundary_layer,
            resolution=resolution,
            max_cells=max_cells,
            force=force,
        )
    except (ValueError, FileExistsError) as exc:
        # Input problems are reported as one line, not as a traceback.
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _build(
    roads_path: Path,
    points_path: Path,
    *,
    cluster_col: str,
    point_id_col: str | None,
    output: Path,
    roads_layer: str | None,
    points_layer: str | None,
    max_snap_distance: float | None,
    boundary_path: Path | None,
    boundary_layer: str | None,
    resolution: float,
    max_cells: int,
    force: bool,
) -> None:
    """Read the inputs, build the network, compute the partition, and write the GeoPackage."""
    check_output_path(output, overwrite=force)
    roads = _read(roads_path, roads_layer)
    points = _read(points_path, points_layer)
    boundary = None if boundary_path is None else _read(boundary_path, boundary_layer)

    if boundary is not None and roads.crs is not None and roads.crs.is_projected:
        check_grid_size(boundary, roads.crs, resolution, max_cells)

    network = from_geodataframes(
        roads,
        points,
        cluster_col=cluster_col,
        point_id_col=point_id_col,
        max_snap_distance=max_snap_distance,
    )
    if boundary is None:
        result = network_voronoi(network)
        write_network_gpkg(output, network, result, overwrite=force)
    else:
        result = surface_voronoi(network, boundary, resolution=resolution, max_cells=max_cells)
        write_surface_gpkg(output, network, result, overwrite=force)

    unit = network.distance_unit or "network CRS units"
    typer.echo(f"Wrote {output} (distance unit: {unit})")


if __name__ == "__main__":
    app()
