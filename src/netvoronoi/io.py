from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point

from .core import NetworkVoronoiResult
from .model import SpatialNetwork
from .surface import SurfaceVoronoiResult


def _write(gdf: gpd.GeoDataFrame | None, path: Path, layer: str) -> None:
    if gdf is None or gdf.empty:
        return
    gdf.to_file(path, layer=layer, driver="GPKG")


def check_output_path(path: str | Path, overwrite: bool = False) -> Path:
    """Check the output path without writing anything, and return it.

    Callers that do a long computation before writing (the CLI, for example)
    can use this first, so that a wrong suffix or an existing file is reported
    immediately instead of after the computation.
    """
    path = Path(path)
    if path.suffix.lower() != ".gpkg":
        raise ValueError("GeoPackage output path must end in .gpkg")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass overwrite=True or --force")
    return path


def _prepare_output(path: str | Path, overwrite: bool) -> Path:
    """Check the path, create the parent directory, and remove an old file."""
    path = check_output_path(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return path


def _site_qa_layers(
    network: SpatialNetwork,
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    cols = {"source_x", "source_y", "geometry"}
    if not cols.issubset(network.sites_snapped.columns):
        return None, None

    # ``source_x/source_y`` are enough to reconstruct the QA geometry. A
    # manually constructed SpatialNetwork may reasonably omit the optional
    # ``snap_distance`` column, so include it only when it is available.
    has_snap_distance = "snap_distance" in network.sites_snapped.columns
    rows_input = []
    rows_connect = []
    for row in network.sites_snapped.itertuples(index=False):
        source = Point(float(row.source_x), float(row.source_y))
        snapped = row.geometry
        base = {
            "site_id": str(row.site_id),
            "network_node": int(row.network_node),
        }
        if has_snap_distance:
            base["snap_distance"] = float(row.snap_distance)
        rows_input.append({**base, "geometry": source})
        rows_connect.append({**base, "geometry": LineString([source, snapped])})
    return (
        gpd.GeoDataFrame(rows_input, geometry="geometry", crs=network.crs),
        gpd.GeoDataFrame(rows_connect, geometry="geometry", crs=network.crs),
    )


def write_network_gpkg(
    path: str | Path,
    network: SpatialNetwork,
    result: NetworkVoronoiResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the network and its exact Voronoi result to one GeoPackage.

    Layers: ``network_edges``, ``sites_input`` and ``snap_connectors`` (only
    for a network built by the spaghetti adapter), ``sites_snapped``,
    ``hard_segments``, ``epsilon_segments``, ``hard_cells_network``,
    ``epsilon_cells_network``, and ``network_unassigned``. A layer with no
    rows is not written at all.
    """
    path = _prepare_output(path, overwrite=overwrite)
    _write(network.edges, path, "network_edges")
    sites_input, snap_connectors = _site_qa_layers(network)
    _write(sites_input, path, "sites_input")
    _write(network.sites_snapped, path, "sites_snapped")
    _write(snap_connectors, path, "snap_connectors")
    _write(result.hard_segments, path, "hard_segments")
    _write(result.epsilon_segments, path, "epsilon_segments")
    _write(result.hard_cells, path, "hard_cells_network")
    _write(result.epsilon_cells, path, "epsilon_cells_network")
    _write(result.unassigned_edges, path, "network_unassigned")
    return path


def write_surface_gpkg(
    path: str | Path,
    network: SpatialNetwork,
    result: SurfaceVoronoiResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the network layers plus the four 2-D surface layers.

    Surface layers: ``hard_surface``, ``epsilon_surface``,
    ``surface_grid_debug`` and ``surface_unassigned``.
    """
    path = write_network_gpkg(path, network, result.network, overwrite=overwrite)
    _write(result.hard, path, "hard_surface")
    _write(result.epsilon, path, "epsilon_surface")
    _write(result.grid, path, "surface_grid_debug")
    _write(result.unassigned, path, "surface_unassigned")
    return path
