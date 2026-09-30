"""The ``SpatialNetwork`` data contract.

A ``SpatialNetwork`` is the only input of the Voronoi computation in
``core.py``. The exact edge formula there is only correct when the node
table, the edge table, the sparse adjacency matrix and the two point tables
describe the same undirected simple graph, so ``validate()`` checks that they
do. It runs when a ``SpatialNetwork`` is created.

The checks work on whole columns (NumPy and Shapely vectorized functions)
instead of one Python loop iteration per row, so that networks with millions
of edges validate in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix, csr_matrix, issparse

from ._numeric import _tol

_POINT = 0  # Shapely geometry type ids
_LINESTRING = 1


def _integer_array(values, message: str, *, minimum: int, maximum: int | None = None) -> np.ndarray:
    """Return ``values`` as an int64 array, or raise ``ValueError(message)``.

    Every value must be a finite whole number in ``[minimum, maximum]``. For
    example ``[0, 2.0, 5]`` is accepted, while ``[0, 2.5]`` and ``[0, None]``
    are rejected. Booleans are rejected too, although Python treats ``True``
    as the integer 1.
    """
    arr = np.asarray(values)
    if arr.dtype.kind == "b" or (
        arr.dtype == object and any(isinstance(x, (bool, np.bool_)) for x in arr)
    ):
        raise ValueError(message)
    try:
        numeric = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    bad = ~np.isfinite(numeric) | (numeric != np.round(numeric)) | (numeric < minimum)
    if maximum is not None:
        bad |= numeric > maximum
    if bad.any():
        raise ValueError(f"{message}; first invalid value: {arr[np.flatnonzero(bad)[0]]!r}")
    return numeric.astype(np.int64)


def _point_xy(geoms: np.ndarray, what: str) -> np.ndarray:
    """Coordinates of an array of Points, after checking type, emptiness and finiteness."""
    if np.any(shapely.get_type_id(geoms) != _POINT) or np.any(shapely.is_empty(geoms)):
        raise ValueError(f"every {what} geometry must be a non-empty Point")
    xy = np.c_[shapely.get_x(geoms), shapely.get_y(geoms)]
    if not np.isfinite(xy).all():
        raise ValueError(f"{what} coordinates must be finite")
    return xy


def _graph_matrix(adjacency):
    """The adjacency as the float64 CSR matrix that SciPy's shortest-path functions read.

    This is the same conversion that ``scipy.sparse.csgraph`` applies to its
    input, so validation and the shortest-path searches see the same matrix.
    No copy is made when ``adjacency`` is already a float64 CSR matrix.
    """
    return adjacency.tocsr().astype(float, copy=False)


@dataclass(frozen=True)
class SpatialNetwork:
    """A projected undirected spatial network with clustered points inserted as nodes.

    Edge ``length`` is geometric length and therefore also the network
    impedance. ``u`` and ``v`` are positional indices into ``nodes``. The
    adjacency matrix must describe exactly the same undirected simple graph.
    Every point in ``points_snapped`` sits on its node ``network_node``.
    """

    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    points_input: gpd.GeoDataFrame
    points_snapped: gpd.GeoDataFrame
    adjacency: csr_matrix

    def __post_init__(self) -> None:
        """Validate when the object is created."""
        self.validate()

    @property
    def crs(self):
        """The coordinate reference system of the network (taken from the edge table)."""
        return self.edges.crs

    @property
    def distance_unit(self) -> str | None:
        """Name of the linear unit of the CRS, for example "metre", or ``None`` when unknown."""
        try:
            axis = self.crs.axis_info
            return axis[0].unit_name if axis else None
        except Exception:
            return None

    @property
    def cluster_ids(self) -> np.ndarray:
        """Distinct cluster IDs as strings, in sorted (lexicographic) order.

        The position of a cluster in this array is its rank. ``core.py``
        works with ranks, so "smaller rank" means "smaller string", and exact
        ties go to the smaller rank.
        """
        return np.asarray(
            sorted(self.points_snapped["cluster_id"].astype(str).unique()), dtype=object
        )

    # ------------------------------------------------------------------ checks

    def validate(self) -> None:
        """Raise ``ValueError`` unless all tables describe the same graph.

        The checks run in this order: coordinate system, required columns and
        IDs, node geometry, adjacency shape, edges, adjacency contents, points.
        """
        crs = self.edges.crs
        if crs is None:
            raise ValueError("network CRS is missing")
        if not crs.is_projected:
            raise ValueError("network CRS must be projected")
        if (
            self.nodes.crs != crs
            or self.points_input.crs != crs
            or self.points_snapped.crs != crs
        ):
            raise ValueError("all network layers must use the same CRS")
        if self.nodes.empty:
            raise ValueError("network has no nodes")
        if self.edges.empty:
            raise ValueError("network has no positive-length edges")
        if self.points_snapped.empty:
            raise ValueError("at least one clustered point is required")

        self._check_columns()
        node_xy = _point_xy(self.nodes.geometry.to_numpy(), "network node")
        self._check_adjacency_shape()
        u, v, length = self._check_edges(node_xy)
        self._check_adjacency_matches_edges(u, v, length)
        self._check_points(node_xy)

    def _check_columns(self) -> None:
        """Required columns exist, IDs are unique, and the two point tables list the same points."""
        required = {
            "edges": (self.edges, {"edge_id", "u", "v", "length", "geometry"}),
            "points_input": (self.points_input, {"point_id", "cluster_id", "geometry"}),
            "points_snapped": (
                self.points_snapped,
                {"point_id", "cluster_id", "network_node", "snap_distance", "snap_ties", "geometry"},
            ),
        }
        for name, (frame, columns) in required.items():
            missing = sorted(columns - set(frame.columns))
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")
        if self.edges["edge_id"].isna().any():
            raise ValueError("edge_id values may not be missing")
        edge_id = self.edges["edge_id"].astype(str)
        if (edge_id.str.len() == 0).any():
            raise ValueError("edge_id values may not be empty")
        if not edge_id.is_unique:
            raise ValueError("edge_id values must be unique after conversion to string")

        for name, frame in (("points_input", self.points_input), ("points_snapped", self.points_snapped)):
            if frame["point_id"].isna().any():
                raise ValueError(f"{name} point_id values may not be missing")
            point_id = frame["point_id"].astype(str)
            if (point_id.str.len() == 0).any():
                raise ValueError(f"{name} point_id values may not be empty")
            if not point_id.is_unique:
                raise ValueError(f"{name} point_id values must be unique after conversion to string")

        if self.points_input["cluster_id"].isna().any() or self.points_snapped["cluster_id"].isna().any():
            raise ValueError("cluster_id values may not be missing")
        raw_clusters = self.points_input["cluster_id"].unique().tolist()
        cluster_strings = [str(value) for value in raw_clusters]
        if any(len(value) == 0 for value in cluster_strings):
            raise ValueError("cluster_id values may not be empty")
        if len(set(cluster_strings)) != len(cluster_strings):
            raise ValueError("distinct cluster_id values collapse to the same string representation")

        # Row i of points_input and row i of points_snapped must be the same point.
        same_ids = np.array_equal(
            self.points_input["point_id"].astype(str).to_numpy(),
            self.points_snapped["point_id"].astype(str).to_numpy(),
        )
        same_clusters = np.array_equal(
            self.points_input["cluster_id"].astype(str).to_numpy(),
            self.points_snapped["cluster_id"].astype(str).to_numpy(),
        )
        if not (same_ids and same_clusters):
            raise ValueError("points_input and points_snapped rows must describe the same points")

    def _check_adjacency_shape(self) -> None:
        """The adjacency is a sparse square matrix with one row and one column per node."""
        n = len(self.nodes)
        if not issparse(self.adjacency) or self.adjacency.shape != (n, n):
            raise ValueError("adjacency shape does not match node count")

    def _check_edges(self, node_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Edge geometry, length, end nodes and graph simplicity. Returns ``(u, v, length)``."""
        n = len(node_xy)
        geoms = self.edges.geometry.to_numpy()
        if np.any(shapely.get_type_id(geoms) != _LINESTRING) or np.any(shapely.is_empty(geoms)):
            raise ValueError("every network edge must be a non-empty LineString")

        # The length column is the impedance used by the shortest-path search,
        # and must equal the length of the drawn line.
        try:
            length = self.edges["length"].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("every network edge must have a finite positive length") from exc
        if not np.isfinite(length).all() or np.any(length <= 0):
            raise ValueError("every network edge must have a finite positive length")
        geom_length = shapely.length(geoms)
        if not np.isfinite(geom_length).all() or np.any(
            np.abs(geom_length - length) > _tol(geom_length, length)
        ):
            raise ValueError("edge length must equal its LineString geometric length")

        # A simple graph: no edge from a node to itself, and at most one edge
        # between two nodes. Each undirected pair {u, v} is encoded as the
        # single integer min(u, v) * n + max(u, v), so that repeated pairs can
        # be found with np.unique.
        message = "edge u/v must contain integer node positions within the node table"
        u = _integer_array(self.edges["u"], message, minimum=0, maximum=n - 1)
        v = _integer_array(self.edges["v"], message, minimum=0, maximum=n - 1)
        if np.any(u == v):
            raise ValueError("self-loop edges are not supported")
        pair = np.minimum(u, v) * n + np.maximum(u, v)
        if np.unique(pair).size != pair.size:
            raise ValueError("parallel edges between the same node pair are not supported")

        # The first vertex of each edge line must be its u node, the last
        # vertex its v node.
        start = shapely.get_point(geoms, 0)
        stop = shapely.get_point(geoms, -1)
        gap_u = np.hypot(shapely.get_x(start) - node_xy[u, 0], shapely.get_y(start) - node_xy[u, 1])
        gap_v = np.hypot(shapely.get_x(stop) - node_xy[v, 0], shapely.get_y(stop) - node_xy[v, 1])
        limit = _tol(length)
        if np.any(gap_u > limit) or np.any(gap_v > limit):
            raise ValueError("edge geometry endpoints must match its u and v node geometries")
        return u, v, length

    def _check_adjacency_matches_edges(self, u: np.ndarray, v: np.ndarray, length: np.ndarray) -> None:
        """The adjacency matrix holds exactly the edges, in both directions, with their lengths.

        The matrix is read the way SciPy's shortest-path functions read it
        (``_graph_matrix``): every stored entry is a connection. An explicitly
        stored zero is therefore a connection of length zero, and two stored
        entries for the same node pair are two connections, of which SciPy
        uses the shorter. So stored entries are compared as they are, without
        adding repeated entries together and without dropping zeros.
        """
        n = len(self.nodes)
        adjacency = _graph_matrix(self.adjacency).copy()  # copy: sort_indices changes the matrix
        adjacency.sort_indices()
        if adjacency.data.size and (
            not np.isfinite(adjacency.data).all() or (adjacency.data < 0).any()
        ):
            raise ValueError("adjacency weights must be finite and non-negative")

        # After sorting, two entries for the same (row, column) pair are neighbours.
        row = np.repeat(np.arange(n), np.diff(adjacency.indptr))
        repeated = (row[1:] == row[:-1]) & (adjacency.indices[1:] == adjacency.indices[:-1])
        if repeated.any():
            raise ValueError("adjacency stores more than one entry for the same node pair")

        # The matrix the edge table implies: entry (u, v) and entry (v, u),
        # both equal to the edge length. Same stored positions, then same values.
        expected = coo_matrix(
            (np.r_[length, length], (np.r_[u, v], np.r_[v, u])), shape=(n, n)
        ).tocsr()
        expected.sort_indices()
        if not np.array_equal(adjacency.indptr, expected.indptr) or not np.array_equal(
            adjacency.indices, expected.indices
        ):
            raise ValueError("adjacency connections do not match network edges")
        # Weight comparison is elementwise. A billion-unit edge elsewhere in
        # the graph must not make a visibly wrong weight on a millimetre edge
        # pass validation merely because a global relative tolerance is large.
        if np.any(np.abs(adjacency.data - expected.data) > _tol(adjacency.data, expected.data)):
            raise ValueError("adjacency weights do not match network edge lengths")

    def _check_points(self, node_xy: np.ndarray) -> None:
        """Every snapped point sits on its node, and ``snap_distance`` is its distance from the input point."""
        n = len(node_xy)
        input_xy = _point_xy(self.points_input.geometry.to_numpy(), "input point")
        snapped_xy = _point_xy(self.points_snapped.geometry.to_numpy(), "snapped point")
        node = _integer_array(
            self.points_snapped["network_node"],
            "point network_node must contain integer node positions within the node table",
            minimum=0,
            maximum=n - 1,
        )
        try:
            snap = self.points_snapped["snap_distance"].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("snap distances must be finite and non-negative") from exc
        if not np.isfinite(snap).all() or np.any(snap < 0):
            raise ValueError("snap distances must be finite and non-negative")
        _integer_array(self.points_snapped["snap_ties"], "snap_ties must contain positive integers", minimum=1)

        gap = np.hypot(snapped_xy[:, 0] - node_xy[node, 0], snapped_xy[:, 1] - node_xy[node, 1])
        if np.any(gap > _tol(snap)):
            raise ValueError("snapped point geometry must coincide with its network node")
        observed = np.hypot(input_xy[:, 0] - snapped_xy[:, 0], input_xy[:, 1] - snapped_xy[:, 1])
        if np.any(np.abs(observed - snap) > _tol(observed, snap)):
            raise ValueError("snap_distance must equal input-to-snapped geometric distance")
