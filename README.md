# netvoronoi

`netvoronoi` is a small Python package for **Voronoi analysis on spatial
networks**. It deliberately separates two different objects:

1. an **exact lineal network Voronoi diagram**, where distance is shortest-path
   distance along a network and outputs are network segments; and
2. an optional **2-D surface rendering**, where off-network locations are
   attached to their nearest network location and membership is sampled on a
   finite grid.

The package also implements an additive-tolerance or **epsilon network
Voronoi** construction. For site `i` and network location `x`, site `i` is a
member when

```text
d_N(i, x) <= min_j d_N(j, x) + epsilon
```

At `epsilon = 0`, this is ordinary nearest-site membership. For `epsilon > 0`,
cells overlap by design. A separate hard layer gives one stable nearest-site
assignment for every positive-length reachable part of the network.

## Scope

`netvoronoi` operates on spatial networks and point sites. It does not assume
a particular application domain.

The package computes exact lineal network Voronoi assignments and
additive-epsilon memberships, with an optional grid-based 2-D surface
rendering.

## Dependencies and division of responsibility

The package reuses established scientific-Python components rather than
reimplementing them:

- **PySAL `spaghetti`**: extraction of a spatial network and point-to-network
  snapping;
- **SciPy sparse graph algorithms**: shortest-path distances;
- **GeoPandas / Pyogrio**: geospatial I/O;
- **Shapely**: geometry operations, exact line substrings, and spatial indexing;
- **Typer**: command-line interface.

`netvoronoi` implements the continuous-edge hard/epsilon Voronoi logic and the
optional 2-D rendering layer.

## Code map

The package is intentionally small and layered. A useful reading order is:

1. `core.py` — the mathematical algorithm; start with `network_voronoi()`.
2. `model.py` — the small `SpatialNetwork` data contract and its invariants.
3. `spaghetti_backend.py` — GIS input, snapping, and insertion of sites as nodes.
4. `surface.py` — the optional approximate 2-D renderer.
5. `io.py` and `cli.py` — GeoPackage output and command-line plumbing.

The public `network_voronoi()` routine is deliberately kept as a short pipeline:
nearest-node distances -> site scan -> exact edge splitting. All of the
per-edge mathematics stays in one well-documented function, `_epsilon_interval()`,
so the derivation and implementation stay together; the hard boundary is that
same function with `epsilon = 0`.

## Exact network-domain construction

`spaghetti` represents an input polyline as straight network arcs between
consecutive input vertices. Sites are snapped to those arcs. `netvoronoi` then
splits an affected arc at each snapped site position, making every site a true
graph node.

Three properties of that step are worth knowing before reading results:

- **Vertex rounding.** `spaghetti` rounds every vertex coordinate to 11
  significant digits before it compares vertices. Near a UTM northing of 1.6
  million metres this keeps four decimal places, so a vertex can move by up to
  0.05 mm. Edge geometry *and* edge length are both taken from those rounded
  coordinates, so the two always agree. (`spaghetti` also reports
  `arc_lengths`, which it computes from one rounded and one unrounded vertex;
  those values are not used.)
- **Arcs without length.** A repeated vertex in a road, or a road piece whose
  two ends round to the same point, gives an arc from a vertex to itself.
  Such an arc carries no length and is skipped; a site snapped to it stays on
  that vertex.
- **`snap_tolerance` moves sites, never road.** A site within
  `snap_tolerance` along its arc of an arc vertex, or of a node that an
  earlier site created, shares that node; otherwise it becomes a new node at
  its own position. Consecutive nodes on an arc are therefore always more than
  `snap_tolerance` apart, and no piece of road is removed. Raising
  `snap_tolerance` merges more sites; it never shortens the network.

The adapter takes only two things from `spaghetti`: the straight arcs and the
snapped site positions. With `spaghetti` 1.7.6 it skips the package's connected-component labelling, because the Voronoi core handles disconnected
networks itself and the labelling is slow: on a 44,700-segment road network,
building the network took 174.5 s with it and 4.4 s without it, with an
identical result. Other 1.x releases use the normal component-labelled path
until that release is explicitly verified.

For a site-free edge `(u, v)` of length `L`, with coordinate `x` measured from
`u`, the shortest distance from site `s` is

```text
f_s(x) = min(d_s(u) + x, d_s(v) + L - x).
```

If

```text
A = min_s d_s(u)
B = min_s d_s(v),
```

then the nearest-site distance is

```text
f_min(x) = min(A + x, B + L - x).
```

Both are piecewise linear with at most one kink, so the excess

```text
e(x) = f_s(x) - f_min(x)
```

is piecewise linear with slopes 0, +2 or -2 only. Its slope cannot change sign
along the edge, so `e` moves in one direction, from `d_s(u) - A` to
`d_s(v) - B`. The admissible set `{x : e(x) <= epsilon}` is therefore a single
interval that touches an endpoint of the edge, or it is empty, and it can be
read off the two endpoints:

```text
admissible at u and at v : [0, L]
admissible at u only     : [0, (B + L + epsilon - d_s(u)) / 2]
admissible at v only     : [(d_s(v) + L - A - epsilon) / 2, L]
admissible at neither    : empty
```

Hard boundaries (`epsilon = 0`) and `epsilon`-admissible intervals are solved
with those four cases on each edge; no road-node densification and no interval
merging is used by the network-domain algorithm. The monotonicity claim and the
closed form were also checked numerically against dense sampling on 200,000
random edge configurations, including exact ties.

### Tie convention

For the hard layer, exact positive-length ties are assigned to the
lexicographically smallest string `site_id`. The implementation determines
this itself rather than inheriting the unspecified tie behavior of a shortest-
path backend.

Adjacent hard LineStrings still share their single geometric boundary point,
as normal line features do. The hard result is therefore exclusive in
**positive length**, not as disjoint closed point sets.

The epsilon line layer contains positive-length admissible intervals. An
isolated membership that exists only at one exact vertex/boundary point is not
emitted as a separate zero-length geometry.

## Distance units

All network distances, `epsilon`, snap thresholds, and surface resolutions use
the **linear units of the projected network CRS**. They are not assumed to be
meters. Output records the CRS unit when it can be determined.

The road/network CRS itself must be projected. The internal network edge and
snapped-site layers must carry CRS metadata matching that network CRS; missing
metadata is rejected rather than guessed. Input site and boundary GeoPandas
layers may use another valid CRS and are reprojected to the road CRS.

## 2-D surface rendering

A network Voronoi diagram naturally partitions a **one-dimensional network**,
not all land in a city polygon. `surface_voronoi()` is therefore a separate,
explicitly approximate operation.

For each clipped grid-cell representative point, the renderer:

1. finds its nearest network edge/location;
2. attaches the representative point to that network location;
3. reads the exact hard/epsilon membership at the anchor; and
4. dissolves grid cells by site.

This is an explicit **nearest-network attachment model**. It is not the more
general metric that minimizes over every possible off-network connector.

If several edges are exactly equally near, the surface renderer uses the
smallest string `edge_id` as a deterministic anchor tie-break and records the
number of equally near anchors in `anchor_ties`.

Cells are numbered column by column, from west to east and from south to north
inside a column; cells with no area inside the boundary are left out, so
`cell_id` counts only the cells that are produced. `surface_unassigned` is one
dissolved polygon of the cells whose anchor no site can reach; the individual
cells stay in `surface_grid_debug` with an empty `hard_site`.

Only the final 2-D boundary is grid-approximated. For publication-quality
surface maps, compare more than one resolution. The renderer also guards
against accidental huge allocations: by default it refuses a bounding-box
grid above 1,000,000 candidate cells (`DEFAULT_MAX_CELLS`). For scale, a grid
of 950,625 candidate cells took about 60 s and 0.9 GB of memory on a 4 GB
test machine. Python callers can raise that limit or pass ``max_cells=None``
deliberately; the CLI exposes ``--max-cells``. The limit is checked before any
expensive work, and in the CLI before the road network is built;
`surface.check_grid_size` makes the same check on its own.

## Current scope: v0.4.3

Supported:

- projected, undirected spatial networks;
- geometric edge length as network impedance;
- LineString and MultiLineString road input (`MultiLineString` is exploded);
- arbitrary point sites snapped to network arcs;
- site layers in any valid CRS (reprojected to the projected road CRS);
- insertion of snapped sites as exact graph nodes;
- exact positive-length hard network Voronoi segments;
- exact positive-length additive-epsilon memberships;
- deterministic hard tie-breaking;
- disconnected networks, including explicit `network_unassigned` output;
- optional maximum site snap-distance guard;
- approximate 2-D surface output with a configurable candidate-grid size guard;
- deterministic surface-anchor tie handling;
- GeoPackage output for QGIS;
- GeoPackage layer selection from the CLI;
- refusal to overwrite an existing output unless `--force` is supplied,
  checked before any computation;
- rejection of NaN and infinite parameters in the API and the CLI.

Not currently supported:

- directed/one-way networks;
- turn restrictions;
- travel-time or arbitrary edge impedance;
- parallel edges between the same graph-node pair;
- exact zero-dimensional tie-point output;
- multiple/off-network access connectors in the surface metric;
- exact continuous 2-D polygons under a network-induced metric.

## Install locally

```powershell
python -m pip install -e ".[dev]"
pytest -q
```

For the cleanest geospatial installation on Windows, installing `spaghetti`
and the geospatial stack from `conda-forge` first is reasonable.

## Python API

```python
import geopandas as gpd

from netvoronoi import from_geodataframes, network_voronoi

roads = gpd.read_file("roads.shp")
sites = gpd.read_file("sites.gpkg")

network = from_geodataframes(
    roads,
    sites,
    site_id_col="site_id",
    max_snap_distance=500,
)
result = network_voronoi(network, epsilon=700)

result.hard_cells.to_file(
    "result.gpkg", layer="hard_network", driver="GPKG"
)
result.epsilon_cells.to_file(
    "result.gpkg", layer="epsilon_network", driver="GPKG"
)
```

The numeric values above are in the road CRS units.

## CLI: network only

```powershell
netvoronoi build `
  "data\roads.gpkg" `
  "data\sites.gpkg" `
  --roads-layer roads `
  --sites-layer sites `
  --site-id-col site_id `
  --epsilon 700 `
  --max-snap-distance 500 `
  --output "output\netvoronoi.gpkg"
```

Network layers can include:

- `network_edges`
- `sites_input` (when built through the `spaghetti` adapter)
- `sites_snapped`
- `snap_connectors` (when built through the `spaghetti` adapter)
- `hard_segments`
- `epsilon_segments`
- `hard_cells_network`
- `epsilon_cells_network`
- `network_unassigned` when a component contains no site

A layer with no rows is not written at all.

The `*_segments` layers are the authoritative analytical representation. The
`*_cells_network` layers collect each site's pieces into a `MultiLineString`
without unary union. That preserves coincident edges that already exist as
distinct elements of a `SpatialNetwork`. The default `spaghetti` adapter,
however, builds a simple geometric network with unique arcs, so truly distinct
stacked links with identical coordinates require preprocessing or a custom
`SpatialNetwork`.

Re-running against an existing output requires `--force`. The output path is
checked before any reading or computing, so an existing file is reported
immediately.

## CLI: network + 2-D surface

```powershell
netvoronoi build `
  "data\roads.gpkg" `
  "data\sites.gpkg" `
  --roads-layer roads `
  --sites-layer sites `
  --site-id-col site_id `
  --epsilon 700 `
  --boundary "data\boundary.gpkg" `
  --boundary-layer boundary `
  --resolution 100 `
  --max-cells 1000000 `
  --output "output\netvoronoi.gpkg"
```

Additional layers:

- `hard_surface`
- `epsilon_surface`
- `surface_grid_debug`
- `surface_unassigned` when the chosen nearest network component is unreachable

`surface_grid_debug` contains `access_dist`, `edge_offset`,
`nearest_site_dist`, `hard_site`, `n_epsilon`, and `anchor_ties` for QA.
`edge_offset` is the position of the anchor measured from edge endpoint `u`;
`nearest_site_dist` is the actual network distance from that anchor to its
nearest site.

## Important topology caveat

A result is only as valid as the supplied network topology. In particular:

- geometric road crossings do not automatically imply an at-grade connection;
- overpasses/underpasses can be falsely connected if the source data itself
  contains a common vertex;
- crossings can remain disconnected if lines cross geometrically but do not
  share a source vertex;
- small endpoint gaps can create false disconnected components;
- duplicated or malformed source segments can affect topology.

`netvoronoi` delegates topology extraction to `spaghetti`; it does not silently
infer uncertain road connectivity. `spaghetti` represents input polylines as
straight arcs between consecutive vertices and, by default, keeps unique arcs.
That is appropriate for an ordinary simple road graph, but duplicated/coincident
source segments should be reviewed before analysis.

## Testing status

The test suite covers analytical line cases, epsilon overlap, co-located sites,
disconnected networks, deterministic ties, model and CRS invariants, API and
CLI validation, output overwrite safety, surface-anchor ties, live spatial-
network adapter cases when `spaghetti` is installed, and randomized comparisons
against slower reference implementations.

The per-edge interval solver is checked against dense sampling and a slower
piecewise reference implementation. The surface renderer is checked against
direct per-cell readings of the exact network result.

## References

See [`REFERENCES.md`](REFERENCES.md).
