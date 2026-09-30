"""GeoPackage output.

A layer with no rows is not written at all. For example, a network whose
every edge is reachable from a point has no ``network_unassigned`` layer.
Readers should treat a missing layer as an empty one.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

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
    return path


def _write(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    """Write ``gdf`` as one layer of the GeoPackage; a table with no rows is not written."""
    if gdf is not None and not gdf.empty:
        gdf.to_file(path, layer=layer, driver="GPKG")


def _prepare(path: str | Path, overwrite: bool) -> Path:
    """Check the output path, create its folder, and delete an existing file that may be replaced."""
    path = check_output_path(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return path


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


def write_network_gpkg(
    path: str | Path,
    network: SpatialNetwork,
    result: NetworkVoronoiResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the network as built and the exact network partition to a new GeoPackage."""
    path = _prepare(path, overwrite)
    _write(network.edges, path, "network_edges")
    _write(network.points_input, path, "points_input")
    _write(network.points_snapped, path, "points_snapped")
    _write(_snap_connectors(network), path, "snap_connectors")
    _write(result.segments, path, "network_segments")
    _write(result.cells, path, "network_cells")
    _write(result.unassigned_edges, path, "network_unassigned")
    return path


def write_surface_gpkg(
    path: str | Path,
    network: SpatialNetwork,
    result: SurfaceVoronoiResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Write everything that ``write_network_gpkg`` writes, plus the four surface layers."""
    path = write_network_gpkg(path, network, result.network, overwrite=overwrite)
    _write(result.cells, path, "surface_cells")
    _write(result.grid, path, "surface_grid_debug")
    _write(result.unassigned, path, "surface_unassigned")
    _write(result.point_qa, path, "surface_point_qa")
    return path
