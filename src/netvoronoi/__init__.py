"""Cluster-based network-informed Voronoi regions."""

from .adaptive import (
    AdaptiveSurfaceResult,
    ConvergenceError,
    ConvergencePolicy,
    adaptive_surface_voronoi,
    territory_qa,
)
from .core import NetworkVoronoiResult, network_voronoi
from .model import SpatialNetwork
from .network import from_geodataframes
from .surface import SurfaceVoronoiResult, surface_voronoi

__all__ = [
    "AdaptiveSurfaceResult",
    "ConvergenceError",
    "ConvergencePolicy",
    "NetworkVoronoiResult",
    "SpatialNetwork",
    "SurfaceVoronoiResult",
    "adaptive_surface_voronoi",
    "from_geodataframes",
    "network_voronoi",
    "surface_voronoi",
    "territory_qa",
]

__version__ = "0.2.0"
