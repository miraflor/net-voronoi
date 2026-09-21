"""Build a :class:`SpatialNetwork` from GeoPandas road and site layers.

Reading order for this module
-----------------------------
1. ``from_geodataframes`` (at the end of the file) is the public entry point.
   It checks the inputs, lets PySAL ``spaghetti`` build the network and snap
   the sites, and then calls ``_convert_spaghetti_network``.
2. ``_convert_spaghetti_network`` turns every ``spaghetti`` arc into graph
   edges. An arc that carries sites is cut at the sites, so that every site
   becomes a graph node.
3. ``_place_sites_on_arc`` holds the only decision rule of the adapter: which
   graph node each site becomes.

What ``spaghetti`` does, and three consequences for this adapter
-----------------------------------------------------------------
``spaghetti`` splits every road LineString into straight arcs between
consecutive vertices. Before it compares vertices, it rounds each coordinate
to 11 significant digits (its ``vertex_sig`` default). At a UTM northing near
1.6 million metres, that keeps 4 decimal places, so a vertex can move by up to
0.05 mm.

* Arc geometry and arc length both come from the rounded vertex coordinates.
  ``spaghetti`` also reports ``arc_lengths``, but it computes them from one
  rounded and one unrounded vertex, so they can disagree with the rounded
  geometry (by about 3.4e-5 CRS units in a projected-coordinate regression
  case). Up to v0.3.0 this adapter
  compared the two and rejected ordinary projected data.
* A repeated consecutive vertex (for example ``(50, 0), (50, 0)``) gives an arc
  from a vertex to itself. That arc has no length, so it is skipped. The same
  happens to a straight road piece whose two ends round to the same point.
* ``snap_tolerance`` only decides where sites go. It never removes road. (Up
  to v0.3.0, a road piece shorter than ``snap_tolerance`` was dropped, which
  could disconnect the network.)

All distances are in the linear unit of the projected road CRS.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from shapely.geometry import Point

from .model import SpatialNetwork

# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


def _require_projected(gdf: gpd.GeoDataFrame, name: str) -> None:
    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS")
    if not gdf.crs.is_projected:
        raise ValueError(f"{name} must use a projected CRS before network-distance analysis")


def _require_finite_coordinates(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Reject NaN or infinite coordinates with a clear message.

    Without this check, ``spaghetti`` fails later with an unclear error such
    as "cannot convert float NaN to integer".
    """
    coordinates = shapely.get_coordinates(gdf.geometry.to_numpy())
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} contains NaN or infinite coordinates")


def _check_parameters(snap_tolerance: float, max_snap_distance: float | None) -> None:
    """Validate the two numeric parameters.

    The tests are written as ``not (... > 0)`` rather than ``... <= 0`` because
    every comparison with NaN is false: ``nan <= 0`` would let NaN through.
    """
    if not (np.isfinite(snap_tolerance) and snap_tolerance > 0):
        raise ValueError(f"snap_tolerance must be a positive finite number, got {snap_tolerance!r}")
    if max_snap_distance is not None and not max_snap_distance >= 0:
        raise ValueError(f"max_snap_distance must be >= 0 or None, got {max_snap_distance!r}")


def _distance_unit(crs) -> str | None:
    """Return the CRS linear-unit name when PyProj exposes one."""
    try:
        return crs.axis_info[0].unit_name or None
    except (AttributeError, IndexError):
        return None


def _can_skip_component_labelling(spaghetti_module) -> bool:
    """Whether the verified fast snapping path may be used.

    ``spaghetti`` 1.7.6 can snap points without building its expensive
    connected-component tables when an empty ``network_component2arc`` is
    supplied.  That relies on an internal detail of that exact release.
    Unknown or later 1.x releases therefore use the ordinary public path with
    component labelling enabled.  Correctness wins over speed when the
    dependency version has not been audited.
    """
    return str(getattr(spaghetti_module, "__version__", "")) == "1.7.6"


def _prepare_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate the road layer and split MultiLineStrings into LineStrings."""
    if roads.empty:
        raise ValueError("roads is empty")
    _require_projected(roads, "roads")

    bad_empty = roads.geometry.isna() | roads.geometry.is_empty
    if bad_empty.any():
        raise ValueError(
            f"roads contains {int(bad_empty.sum())} null/empty geometries; "
            "clean them explicitly before analysis"
        )

    bad_type = ~roads.geometry.geom_type.isin(["LineString", "MultiLineString"])
    if bad_type.any():
        kinds = sorted(roads.loc[bad_type].geometry.geom_type.unique().tolist())
        raise ValueError(f"roads must contain only line geometries; found {kinds}")
    _require_finite_coordinates(roads, "roads")

    # A MultiLineString must be split before ``spaghetti`` reads it: through
    # libpysal, ``spaghetti`` joins the vertex lists of all parts into one
    # list, which adds a false arc from the end of one part to the start of
    # the next part.
    work = roads.copy()
    if (work.geometry.geom_type == "MultiLineString").any():
        work = work.explode(index_parts=False, ignore_index=True)
    work = work.reset_index(drop=True)

    zero_length = work.geometry.length <= 0
    if zero_length.any():
        raise ValueError(
            f"roads contains {int(zero_length.sum())} zero-length LineStrings; "
            "clean them explicitly before analysis"
        )
    return work


def _prepare_sites(
    sites: gpd.GeoDataFrame,
    road_crs,
    site_id_col: str | None,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Validate sites, reproject them to the road CRS, and build stable IDs."""
    if sites.empty:
        raise ValueError("sites is empty")
    if sites.crs is None:
        raise ValueError("sites has no CRS")

    bad_geometry = sites.geometry.isna() | sites.geometry.is_empty
    if bad_geometry.any():
        raise ValueError(
            f"sites contains {int(bad_geometry.sum())} null/empty geometries; "
            "clean them explicitly before analysis"
        )
    if not set(sites.geometry.geom_type.unique()).issubset({"Point"}):
        raise ValueError("sites must contain Point geometries")

    work = sites.to_crs(road_crs) if sites.crs != road_crs else sites.copy()
    work = work.reset_index(drop=True)
    _require_finite_coordinates(work, "sites")

    if site_id_col is None:
        ids = np.asarray([str(i) for i in range(len(work))], dtype=object)
    else:
        if site_id_col not in work.columns:
            raise ValueError(f"site_id_col {site_id_col!r} not found")
        if work[site_id_col].isna().any():
            raise ValueError("site ids must not be null")
        ids = work[site_id_col].astype(str).to_numpy(dtype=object)

    if len(ids) != len(set(ids.tolist())):
        raise ValueError("site ids must be unique after string conversion")
    return work, ids


# ---------------------------------------------------------------------------
# Reading the snap result of ``spaghetti``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Snaps:
    """Where ``spaghetti`` snapped each site, as arrays in site-row order.

    ``arc[k]`` is the ``(a, b)`` vertex pair of the arc that holds site ``k``
    (``spaghetti`` stores every arc with ``a <= b``), ``offset[k]`` is the
    distance along that arc from vertex ``a``, and ``snap_distance[k]`` is the
    straight-line distance from the input point to the snapped point.
    """

    arc: np.ndarray
    offset: np.ndarray
    snap_distance: np.ndarray


def _read_snaps(point_pattern, n_sites: int) -> _Snaps:
    """Translate ``spaghetti``'s point-pattern dictionaries into arrays.

    ``spaghetti`` numbers the points of a GeoDataFrame 0, 1, ..., n - 1 in row
    order, so its point number is the site row.
    """
    arc = np.full((n_sites, 2), -1, dtype=int)
    for arc_key, observations in point_pattern.obs_to_arc.items():
        for point_index in observations:
            arc[int(point_index)] = arc_key

    missing = np.flatnonzero(arc[:, 0] < 0)
    if len(missing):
        raise RuntimeError(f"spaghetti did not snap site rows: {missing[:10].tolist()}")

    offset = np.array(
        [point_pattern.dist_to_vertex[k][int(arc[k, 0])] for k in range(n_sites)], dtype=float
    )
    snap_distance = np.array([point_pattern.dist_snapped[k] for k in range(n_sites)], dtype=float)
    invalid = ~np.isfinite(offset) | ~np.isfinite(snap_distance) | (snap_distance < 0)
    if invalid.any():
        raise RuntimeError(
            f"spaghetti returned invalid snap distances for site row {int(np.flatnonzero(invalid)[0])}"
        )
    return _Snaps(arc=arc, offset=offset, snap_distance=snap_distance)


def _check_max_snap_distance(
    snaps: _Snaps, site_ids: np.ndarray, max_snap_distance: float | None
) -> None:
    """Raise when a site was snapped farther than ``max_snap_distance``."""
    if max_snap_distance is None:
        return
    too_far = snaps.snap_distance > max_snap_distance
    if too_far.any():
        first = int(np.flatnonzero(too_far)[0])
        raise ValueError(
            f"site {site_ids[first]!r} snapped {snaps.snap_distance[first]:g} network-CRS units; "
            f"max_snap_distance={max_snap_distance:g} "
            f"({int(too_far.sum())} site(s) exceed it)"
        )


# ---------------------------------------------------------------------------
# The decision rule: which graph node does each site become?
# ---------------------------------------------------------------------------


def _place_sites_on_arc(
    offsets: np.ndarray, length: float, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Assign the sites on one arc to graph nodes.

    The arc runs from its first vertex (offset 0) to its last vertex (offset
    ``length``). Sites are taken in order of offset. Each site goes to the
    nearest *existing* node when that node is within ``tolerance`` along the
    arc; otherwise the site becomes a new node at its own offset. The existing
    nodes are the two arc vertices and the new nodes that earlier sites
    created. Because sites are taken in order, the nearest of those behind a
    site is the most recent new node (or the first vertex when there is
    none), and the only one ahead is the last vertex. An exact tie goes to the
    node behind.

    Guarantees:

    * every site moves at most ``tolerance`` along the arc;
    * sites at the same offset share a node;
    * consecutive nodes on the arc are more than ``tolerance`` apart, except
      the two arc vertices when the arc has no new node. So every piece of the
      cut arc has positive length, and no road is lost.

    Example with ``tolerance = 0.01`` on an arc of length 100: offsets 50.000,
    50.009 and 50.012 give two new nodes, at 50.000 (shared by the first two
    sites) and at 50.012. The two nodes are 0.012 apart.

    Returns ``(slot, new_offsets)``. Along the arc, slot 0 is the first
    vertex, slots ``1 .. m`` are the ``m`` new nodes (at ``new_offsets``, in
    increasing order), and slot ``m + 1`` is the last vertex. ``slot[k]`` is
    the slot of site ``k``.
    """
    offsets = np.clip(np.asarray(offsets, dtype=float), 0.0, length)
    new_offsets: list[float] = []
    slot = np.empty(len(offsets), dtype=int)
    last_vertex = -1  # placeholder; replaced by m + 1 once m is known

    for k in np.argsort(offsets, kind="stable"):
        behind = new_offsets[-1] if new_offsets else 0.0
        gap_behind = offsets[k] - behind
        gap_ahead = length - offsets[k]
        if gap_behind <= tolerance and gap_behind <= gap_ahead:
            slot[k] = len(new_offsets)  # the most recent new node, or the first vertex
        elif gap_ahead <= tolerance:
            slot[k] = last_vertex
        else:
            new_offsets.append(float(offsets[k]))
            slot[k] = len(new_offsets)

    slot[slot == last_vertex] = len(new_offsets) + 1
    return slot, np.asarray(new_offsets, dtype=float)


# ---------------------------------------------------------------------------
# Conversion of the whole network
# ---------------------------------------------------------------------------


def _adjacency_from_edges(u: np.ndarray, v: np.ndarray, length: np.ndarray, n_vertices: int):
    """Build the undirected sparse adjacency represented by the edge table."""
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    weights = np.concatenate([length, length])
    return coo_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()


def _convert_spaghetti_network(
    ntw,
    *,
    roads_crs,
    sites_work: gpd.GeoDataFrame,
    site_ids: np.ndarray,
    snap_tolerance: float,
    max_snap_distance: float | None,
) -> SpatialNetwork:
    """Convert a snapped ``spaghetti.Network`` into the package's exact-site graph.

    Node numbers: ``spaghetti`` numbers its vertices 0, 1, ..., n - 1, and
    those numbers are used unchanged. New nodes for sites get the numbers
    n, n + 1, ... in the order in which they are created.

    Edge order: arcs are processed in sorted ``(a, b)`` order, and the pieces
    of one arc from vertex ``a`` towards vertex ``b``. ``edge_id`` is the row
    number in that order.
    """
    n_sites = len(sites_work)
    snaps = _read_snaps(ntw.pointpatterns["sites"], n_sites)
    _check_max_snap_distance(snaps, site_ids, max_snap_distance)

    n_vertices = len(ntw.vertex_coords)
    if sorted(ntw.vertex_coords) != list(range(n_vertices)):
        raise RuntimeError("spaghetti vertex IDs are not 0, 1, ..., n - 1")
    # ``vertices`` grows as site nodes are added; entry i is node i.
    vertices = [tuple(float(c) for c in ntw.vertex_coords[i][:2]) for i in range(n_vertices)]

    sites_on_arc: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, (a, b) in enumerate(snaps.arc.tolist()):
        sites_on_arc[(a, b)].append(k)

    site_node = np.full(n_sites, -1, dtype=int)
    edge_u: list[int] = []
    edge_v: list[int] = []
    edge_arc: list[str] = []

    for a, b in sorted(tuple(map(int, arc)) for arc in ntw.arcs):
        on_arc = np.asarray(sites_on_arc.get((a, b), []), dtype=int)
        if a == b:  # a repeated input vertex: the arc has no length
            site_node[on_arc] = a
            continue

        p, q = np.asarray(vertices[a]), np.asarray(vertices[b])
        arc_length = float(np.hypot(*(q - p)))
        offsets = snaps.offset[on_arc]
        # An offset is measured on this same arc, so it can pass the arc end
        # by rounding error only. A clearly larger value means that the snap
        # record does not belong to this arc.
        if np.any(offsets > arc_length + 1e-6 * max(1.0, arc_length)):
            raise RuntimeError(f"spaghetti returned an out-of-range snap offset on arc {(a, b)}")
        slot, new_offsets = _place_sites_on_arc(offsets, arc_length, snap_tolerance)

        # New node coordinates, written as (1 - f) * p + f * q like the cuts
        # in ``core._cut_lines``.
        first_new = len(vertices)
        for f in new_offsets / arc_length:
            vertices.append(tuple(float(c) for c in (1.0 - f) * p + f * q))
        chain = [a, *range(first_new, len(vertices)), b]  # the nodes along the arc

        site_node[on_arc] = np.asarray(chain)[slot]
        edge_u.extend(chain[:-1])
        edge_v.extend(chain[1:])
        edge_arc.extend([f"{a}:{b}"] * (len(chain) - 1))

    if (site_node < 0).any():
        raise RuntimeError("spaghetti snapped a site to an arc that is not in the network")
    if not edge_u:
        raise ValueError("spaghetti produced no positive-length network arcs")

    xy = np.asarray(vertices, dtype=float)
    u, v = np.asarray(edge_u, dtype=int), np.asarray(edge_v, dtype=int)
    # Every edge is a straight line between its two nodes, and its length is
    # measured from that same geometry.
    geometry = shapely.linestrings(np.stack([xy[u], xy[v]], axis=1))
    length = shapely.length(geometry)
    edges = gpd.GeoDataFrame(
        {
            "edge_id": np.arange(len(u)),
            "u": u,
            "v": v,
            "length": length,
            "parent_arc": edge_arc,
        },
        geometry=gpd.GeoSeries(geometry, crs=roads_crs),
        crs=roads_crs,
    )

    source = shapely.get_coordinates(sites_work.geometry.to_numpy())
    site_arc = [f"{a}:{b}" for a, b in snaps.arc.tolist()]
    snapped = gpd.GeoDataFrame(
        {
            "site_id": [str(value) for value in site_ids],
            "network_node": site_node,
            "parent_arc": site_arc,
            "snap_distance": snaps.snap_distance,
            "source_x": source[:, 0],
            "source_y": source[:, 1],
        },
        geometry=gpd.GeoSeries([Point(xy[node]) for node in site_node], crs=roads_crs),
        crs=roads_crs,
    )

    network = SpatialNetwork(
        vertices_xy=xy,
        edges=edges,
        adjacency=_adjacency_from_edges(u, v, length, len(xy)),
        site_ids=np.asarray([str(value) for value in site_ids], dtype=object),
        site_nodes=site_node,
        sites_snapped=snapped,
        crs=roads_crs,
        distance_unit=_distance_unit(roads_crs),
    )
    network.validate()
    return network


def from_geodataframes(
    roads: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    site_id_col: str | None = None,
    snap_tolerance: float = 1e-8,
    max_snap_distance: float | None = None,
) -> SpatialNetwork:
    """Build a validated :class:`SpatialNetwork` from line and point layers.

    ``spaghetti`` extracts the network and snaps each site to its nearest
    straight arc. This adapter then inserts every snapped site as a graph
    node, so the continuous-edge Voronoi calculation is exact along the
    resulting network.

    Parameters
    ----------
    roads
        LineString or MultiLineString layer in a projected CRS.
    sites
        Point layer in any valid CRS; it is reprojected to the road CRS.
    site_id_col
        Column with unique site IDs. When ``None``, the IDs are "0", "1", ...
        in row order.
    snap_tolerance
        Largest distance along an arc by which a snapped site may be moved so
        that it shares a graph node with an arc vertex or with another site
        (see ``_place_sites_on_arc``). It never removes road. It must be a
        positive finite number; the default only merges positions that are
        equal up to floating-point rounding.
    max_snap_distance
        When given, raise ``ValueError`` if any site is farther than this from
        the network. ``math.inf`` is the same as ``None``.

    All distances are in the linear unit of the projected road CRS. Vertex
    coordinates in the output are ``spaghetti``'s rounded coordinates (see
    the module docstring).
    """
    try:
        import spaghetti
    except ImportError as exc:  # pragma: no cover - exercised in installed environments
        raise ImportError(
            "The GeoDataFrame adapter requires PySAL spaghetti. "
            "Install netvoronoi with its normal dependencies."
        ) from exc

    _check_parameters(snap_tolerance, max_snap_distance)
    roads_work = _prepare_roads(roads)
    sites_work, site_ids = _prepare_sites(sites, roads_work.crs, site_id_col)

    # Only two things are taken from spaghetti: the straight arcs (built by
    # ``Network``) and the snapped site positions (``snapobservations``).
    # ``extractgraph=False`` skips spaghetti's contracted graph, which this
    # adapter does not use.
    #
    # Component labelling is different.  In spaghetti 1.7.6 it is by far the
    # slowest part of the build, yet the adapter never consumes those labels.
    # We verified that 1.7.6 produces identical snapping when the labelling is
    # skipped and an empty ``network_component2arc`` is supplied.  Because
    # that workaround relies on 1.7.6 internals, it is deliberately guarded
    # by the exact dependency version.  Any other 1.x version falls back to
    # spaghetti's ordinary public path (``w_components=True``): slower, but
    # safe until that version is audited.
    #
    # Why an empty table is safe in 1.7.6: ``snapobservations`` calls
    # ``_snap_to_link``, which reads ``network_component2arc`` only in one loop
    # that fills ``pointpattern.component_to_obs`` (spaghetti/network.py, lines
    # 1270-1280 in 1.7.6). This adapter never reads ``component_to_obs``. Before
    # adding another release to ``_can_skip_component_labelling``, check that this
    # is still the only use of the table.

    skip_components = _can_skip_component_labelling(spaghetti)
    network = spaghetti.Network(
        in_data=roads_work,
        unique_arcs=True,
        extractgraph=False,
        w_components=not skip_components,
    )
    if skip_components:
        network.network_component2arc = {}
    network.snapobservations(sites_work, "sites", attribute=False)
    return _convert_spaghetti_network(
        network,
        roads_crs=roads_work.crs,
        sites_work=sites_work,
        site_ids=site_ids,
        snap_tolerance=float(snap_tolerance),
        max_snap_distance=max_snap_distance,
    )
