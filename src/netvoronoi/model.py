"""The ``SpatialNetwork`` data contract and its validation."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS
from pyproj.exceptions import CRSError
from scipy.sparse import coo_matrix, csr_matrix

# Coordinates are compared with an absolute slack (given at each check) plus
# this fraction of the coordinate value. The relative part matters for large
# projected coordinates: near a UTM northing of 1.6e6, adjacent float64 values
# are about 2e-10 apart, and in Web Mercator (up to 2e7) about 4e-9 apart.
_COORDINATE_RTOL = 1e-12


@dataclass(frozen=True)
class SpatialNetwork:
    """Spatial graph used by the continuous network-Voronoi engine.

    The class is intentionally small.  It stores the graph, its geometries, and
    the site-to-node mapping; :meth:`validate` checks the assumptions required
    by the exact edge calculations.

    Invariants
    ----------
    * each edge geometry runs from vertex ``u`` to vertex ``v``;
    * ``length`` equals geometric length in the network CRS units;
    * ``adjacency`` represents the same undirected simple graph as ``edges``;
    * every site is located at a graph node;
    * ``crs`` is a projected CRS (a pyproj ``CRS``, or anything pyproj accepts,
      such as ``"EPSG:32651"``).
    """

    vertices_xy: np.ndarray
    edges: gpd.GeoDataFrame
    adjacency: csr_matrix
    site_ids: np.ndarray
    site_nodes: np.ndarray
    sites_snapped: gpd.GeoDataFrame
    crs: object
    distance_unit: str | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` when a core network invariant is violated.

        CRS is checked before any geometric length calculation.  Besides
        failing earlier, this avoids GeoPandas warnings from measuring a
        network that is already known to use a geographic CRS.
        """
        xy = _validate_vertices(self.vertices_xy)
        _validate_crs(self.crs, self.edges, self.sites_snapped)
        _validate_edges(self.edges, xy)
        _validate_adjacency(self.adjacency, self.edges, len(xy))
        _validate_sites(self.site_ids, self.site_nodes, self.sites_snapped, xy)


def _validate_vertices(vertices_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(vertices_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("vertices_xy must be an (n, 2) array")
    if len(xy) == 0:
        raise ValueError("network must contain at least one vertex")
    if not np.isfinite(xy).all():
        raise ValueError("vertices_xy must contain only finite coordinates")
    return xy


def _validate_edges(edges: gpd.GeoDataFrame, xy: np.ndarray) -> None:
    required = {"edge_id", "u", "v", "length", "geometry"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"edges missing required columns: {sorted(missing)}")
    if edges.empty:
        raise ValueError("network must contain at least one edge")
    if edges["edge_id"].isna().any() or edges["edge_id"].duplicated().any():
        raise ValueError("edge_id values must be non-null and unique")

    uv_raw = edges[["u", "v"]].to_numpy(dtype=float)
    if not np.isfinite(uv_raw).all() or not np.allclose(uv_raw, np.round(uv_raw)):
        raise ValueError("edge node indices u and v must be finite integers")
    uv = uv_raw.astype(int)
    n = len(xy)
    if np.any(uv < 0) or np.any(uv >= n):
        raise ValueError("edges contain an out-of-range node index")
    if np.any(uv[:, 0] == uv[:, 1]):
        raise ValueError("self-loop edges are not supported")
    if len(uv) != len(np.unique(np.sort(uv, axis=1), axis=0)):
        raise ValueError("parallel edges between the same node pair are not supported")

    lengths = edges["length"].to_numpy(dtype=float)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("all edge lengths must be finite and positive")

    _validate_edge_geometries(edges, xy, uv, lengths)


def _raise_for_edges(bad: np.ndarray, edges: gpd.GeoDataFrame, message: str) -> None:
    """Raise ``ValueError(message)`` naming the first bad edge, if any is bad."""
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        edge_id = edges["edge_id"].iloc[first]
        if hasattr(edge_id, "item"):  # a NumPy scalar would print as np.int64(3)
            edge_id = edge_id.item()
        raise ValueError(f"{message} (edge_id {edge_id!r}; {int(bad.sum())} edge(s) affected)")


def _validate_edge_geometries(
    edges: gpd.GeoDataFrame,
    xy: np.ndarray,
    uv: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """Check the geometry assumptions used by exact line cutting.

    Every edge geometry must be a non-empty LineString that starts at vertex
    ``u``, ends at vertex ``v``, and has geometric length equal to ``length``.
    All edges are checked together as arrays.
    """
    geometry = edges.geometry
    is_line = (geometry.geom_type == "LineString").to_numpy(dtype=bool)
    is_empty = geometry.is_empty.to_numpy(dtype=bool)
    _raise_for_edges(
        ~is_line | is_empty, edges, "every edge geometry must be a non-empty LineString"
    )

    # The slack is one part in 10^9 of the larger of the two lengths (and at
    # least 1e-9), plus one part in 10^10 of ``length``. These are the v0.3.0
    # tolerances, unchanged.
    geometric = geometry.length.to_numpy(dtype=float)
    length_slack = 1e-9 * np.maximum(1.0, np.maximum(geometric, lengths))
    _raise_for_edges(
        ~np.isclose(geometric, lengths, rtol=1e-10, atol=length_slack),
        edges,
        "edge length must equal its geometric length in v0.x",
    )

    # Endpoints agree with their vertices up to the same absolute slack plus
    # a part relative to the coordinate value.
    lines = geometry.to_numpy()
    first_xy = shapely.get_coordinates(shapely.get_point(lines, 0))
    last_xy = shapely.get_coordinates(shapely.get_point(lines, -1))
    slack = length_slack[:, None]
    starts_at_u = np.isclose(first_xy, xy[uv[:, 0]], rtol=_COORDINATE_RTOL, atol=slack).all(axis=1)
    ends_at_v = np.isclose(last_xy, xy[uv[:, 1]], rtol=_COORDINATE_RTOL, atol=slack).all(axis=1)
    _raise_for_edges(~starts_at_u, edges, "edge geometry start coordinate does not match vertex u")
    _raise_for_edges(~ends_at_v, edges, "edge geometry end coordinate does not match vertex v")


def _validate_adjacency(adjacency: csr_matrix, edges: gpd.GeoDataFrame, n: int) -> None:
    if adjacency.shape != (n, n):
        raise ValueError("adjacency shape does not match vertices_xy")
    if np.any(~np.isfinite(adjacency.data)) or np.any(adjacency.data < 0):
        raise ValueError("adjacency weights must be finite and non-negative")

    uv = edges[["u", "v"]].to_numpy(dtype=int)
    lengths = edges["length"].to_numpy(dtype=float)
    rows = np.concatenate([uv[:, 0], uv[:, 1]])
    cols = np.concatenate([uv[:, 1], uv[:, 0]])
    weights = np.concatenate([lengths, lengths])
    expected = coo_matrix((weights, (rows, cols)), shape=(n, n), dtype=float).tocsr()

    diff = (adjacency.astype(float) - expected).tocsr()
    diff.eliminate_zeros()
    if not diff.nnz:
        return

    max_err = float(np.max(np.abs(diff.data)))
    scale = max(1.0, float(np.max(lengths)))
    if max_err > max(1e-9, 1e-10 * scale):
        raise ValueError("adjacency does not match the undirected edge table")


def _validate_sites(
    site_ids: np.ndarray,
    site_nodes: np.ndarray,
    sites_snapped: gpd.GeoDataFrame,
    xy: np.ndarray,
) -> None:
    ids = np.asarray(site_ids, dtype=object)
    nodes_raw = np.asarray(site_nodes, dtype=float)
    if len(ids) != len(nodes_raw):
        raise ValueError("site_ids and site_nodes lengths differ")
    if len(ids) == 0:
        raise ValueError("at least one site is required")
    if any(value is None for value in ids.tolist()):
        raise ValueError("site IDs must not be null")

    keys = [str(value) for value in ids.tolist()]
    if len(keys) != len(set(keys)):
        raise ValueError("site IDs must be unique after string conversion")
    if not np.isfinite(nodes_raw).all() or not np.allclose(nodes_raw, np.round(nodes_raw)):
        raise ValueError("site_nodes must contain finite integer node indices")
    nodes = nodes_raw.astype(int)
    if np.any(nodes < 0) or np.any(nodes >= len(xy)):
        raise ValueError("site_nodes contains an out-of-range node index")

    required = {"site_id", "network_node", "geometry"}
    if not required.issubset(sites_snapped.columns):
        raise ValueError("sites_snapped must contain site_id, network_node, and geometry")
    if len(sites_snapped) != len(ids):
        raise ValueError("sites_snapped must contain exactly one row per site")
    if sites_snapped.geometry.isna().any() or sites_snapped.geometry.is_empty.any():
        raise ValueError("sites_snapped contains null or empty geometry")
    if not set(sites_snapped.geometry.geom_type.unique()).issubset({"Point"}):
        raise ValueError("sites_snapped geometries must be Points")

    snapped_ids = sites_snapped["site_id"].astype(str).tolist()
    snapped_nodes = sites_snapped["network_node"].to_numpy(dtype=int)
    if snapped_ids != keys or not np.array_equal(snapped_nodes, nodes):
        raise ValueError("sites_snapped rows must match site_ids and site_nodes in order")
    point_xy = shapely.get_coordinates(sites_snapped.geometry.to_numpy())
    at_node = np.isclose(point_xy, xy[nodes], rtol=_COORDINATE_RTOL, atol=1e-8).all(axis=1)
    if not at_node.all():
        first = int(np.flatnonzero(~at_node)[0])
        raise ValueError(
            f"sites_snapped geometry does not match its network node (site_id {keys[first]!r})"
        )


def _validate_crs(crs, edges: gpd.GeoDataFrame, sites_snapped: gpd.GeoDataFrame) -> None:
    """The network CRS must be projected, and the layers must use it.

    ``crs`` is converted with ``pyproj.CRS.from_user_input`` first, so a
    string or EPSG code is checked in the same way as a pyproj ``CRS``. (Up to
    v0.3.0 a string such as ``"EPSG:4326"`` skipped the projected check.)
    """
    if crs is None:
        raise ValueError("network CRS must not be null")
    try:
        network_crs = CRS.from_user_input(crs)
    except CRSError as exc:
        raise ValueError(f"network CRS is not a valid CRS: {crs!r}") from exc
    if not network_crs.is_projected:
        raise ValueError("network CRS must be projected")

    # Geometry without CRS metadata is ambiguous even when ``network.crs`` is
    # present: downstream GeoPandas operations would otherwise treat raw
    # coordinates as if they already used the declared network CRS.
    if edges.crs is None:
        raise ValueError("edges must have CRS metadata matching network CRS")
    if sites_snapped.crs is None:
        raise ValueError("sites_snapped must have CRS metadata matching network CRS")

    edge_crs = CRS.from_user_input(edges.crs)
    site_crs = CRS.from_user_input(sites_snapped.crs)
    if edge_crs != network_crs:
        raise ValueError("edges CRS does not match network CRS")
    if site_crs != network_crs:
        raise ValueError("sites_snapped CRS does not match network CRS")
