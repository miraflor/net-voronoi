"""Command-line interface: one ``build`` command from road and site files."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import typer

from .core import network_voronoi
from .io import check_output_path, write_network_gpkg, write_surface_gpkg
from .spaghetti_backend import from_geodataframes
from .surface import DEFAULT_MAX_CELLS, check_grid_size, surface_voronoi

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Network-distance Voronoi diagrams and additive-epsilon catchments."""


def _read(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def _reject_nan(value: float | None) -> float | None:
    """Reject NaN. Typer's ``min`` check lets it through, because every
    comparison with NaN is false."""
    if value is not None and math.isnan(value):
        raise typer.BadParameter("must be a number, not NaN")
    return value


def _require_finite(value: float | None) -> float | None:
    """Reject NaN and infinity."""
    if value is not None and not math.isfinite(value):
        raise typer.BadParameter("must be a finite number")
    return value


@app.command("build")
def build(
    roads_path: Path = typer.Argument(..., exists=True, readable=True),
    sites_path: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("netvoronoi.gpkg"), "--output", "-o"),
    roads_layer: str | None = typer.Option(None, "--roads-layer"),
    sites_layer: str | None = typer.Option(None, "--sites-layer"),
    site_id_col: str | None = typer.Option(None, "--site-id-col"),
    epsilon: float = typer.Option(0.0, "--epsilon", min=0.0, callback=_reject_nan),
    max_snap_distance: float | None = typer.Option(
        None, "--max-snap-distance", min=0.0, callback=_reject_nan
    ),
    boundary_path: Path | None = typer.Option(None, "--boundary", exists=True, readable=True),
    boundary_layer: str | None = typer.Option(None, "--boundary-layer"),
    resolution: float = typer.Option(
        100.0, "--resolution", min=0.001, callback=_require_finite
    ),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    max_cells: int = typer.Option(
        DEFAULT_MAX_CELLS,
        "--max-cells",
        min=1,
        help="Largest number of candidate grid squares for the 2-D surface "
        "(used only with --boundary).",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing output GeoPackage."),
) -> None:
    """Build exact network Voronoi cells and optional approximate 2-D surfaces."""
    # The output path is checked before anything is read or computed, so that
    # a wrong suffix or an existing file is reported at once.
    check_output_path(output, overwrite=force)

    roads = _read(roads_path, roads_layer)
    sites = _read(sites_path, sites_layer)
    boundary = None if boundary_path is None else _read(boundary_path, boundary_layer)

    # A too-fine surface grid is also refused now, before the road network is
    # built, because building it is the slowest step for a large road file.
    # (A road layer without a CRS is reported by ``from_geodataframes`` below,
    # with a clearer message than the grid check could give.)
    if boundary is not None and roads.crs is not None:
        check_grid_size(boundary, roads.crs, resolution, max_cells)

    network = from_geodataframes(
        roads,
        sites,
        site_id_col=site_id_col,
        max_snap_distance=max_snap_distance,
    )

    if boundary is None:
        result = network_voronoi(network, epsilon=epsilon, batch_size=batch_size)
        write_network_gpkg(output, network, result, overwrite=force)
    else:
        result = surface_voronoi(
            network,
            boundary,
            epsilon=epsilon,
            resolution=resolution,
            batch_size=batch_size,
            max_cells=max_cells,
        )
        write_surface_gpkg(output, network, result, overwrite=force)
    unit = network.distance_unit or "network CRS units"
    typer.echo(f"Wrote {output} (distance unit: {unit})")


if __name__ == "__main__":
    app()
