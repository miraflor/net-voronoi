# net-voronoi

`net-voronoi` builds **cluster-based Voronoi regions informed by shortest-path distance on a spatial network**.

Input points already have cluster labels. The package does not create clusters and does not replace a cluster with a centroid or medoid.

There is **no additive tolerance / epsilon Voronoi mode**. Cells are exclusive in positive network length.

The [visual algorithm guide](docs/index.html) illustrates the exact lineal construction. This README documents the current `main` branch, including adaptive features not yet included in a new tagged release.

## Method

### Distance from a location to a cluster

For a network location $`x`$ and cluster $`c`$, it uses

```math
D(c, x) = \min_{p \in c} d_N(p, x)
```

where $`d_N`$ is shortest-path distance on the supplied network. The winning cluster is the one with the smallest $`D(c, x)`$. Exact ties are resolved by the lexicographically smallest string cluster ID.

In symbols: with $`D(x) = \min_c D(c, x)`$, the distance from $`x`$ to the nearest cluster, the winner at $`x`$ is

```math
W(x) = \min \bigl\{\, c \;:\; D(c, x) - D(x) \le \tau\bigl(D(c, x),\, D(x)\bigr) \,\bigr\},
```

where the minimum is taken in string order of the cluster IDs and $`\tau`$ is the local tolerance defined under [Ties and cluster IDs](#ties-and-cluster-ids). A location that no cluster can reach ($`D(x) = \infty`$) is unassigned.

### Exact split of one edge

Every snapped point is a graph node (see [Why the point clusters are preserved](#why-the-point-clusters-are-preserved)), so no point lies inside an edge. On an edge $`(u, v)`$ of length $`L`$, a shortest path from the location at distance $`x`$ from $`u`$ to the nearest cluster leaves the edge through $`u`$ or through $`v`$:

```math
D(x) = \min\bigl(D(u) + x,\; D(v) + L - x\bigr), \qquad 0 \le x \le L.
```

If the winners at $`u`$ and $`v`$ are the same cluster, that cluster owns the whole edge. Otherwise the two terms are equal at

```math
x^* = \frac{D(v) + L - D(u)}{2},
```

clipped to $`[0, L]`$. The winner at $`u`$ owns $`[0, x^*]`$ and the winner at $`v`$ owns $`[x^*, L]`$. For example, $`D(u) = 2`$, $`D(v) = 4`$ and $`L = 10`$ give $`x^* = 6`$, where both terms equal 8.

A boundary within the tolerance $`\tau`$ of an edge end is moved onto that end, so that no piece shorter than the tolerance is produced. The pieces of every reachable edge therefore cover $`[0, L]`$ without gaps or overlaps. This formula is written once, in `core._edge_boundary`, and the 2-D surface uses the same function.

## What the package produces

The exact analytical object is one-dimensional and lives on the network:

- `network_segments` — exact positive-length pieces assigned to one cluster;
- `network_cells` — those pieces collected by cluster;
- `network_unassigned` — road components containing no input point.

With a polygon boundary, `net-voronoi` can also render an approximate 2-D surface:

- `surface_cells` — grid cells dissolved by cluster;
- `surface_grid_debug` — the underlying grid assignment and nearest-network anchor QA;
- `surface_unassigned` — cells whose nearest network component has no clustered point;
- `surface_point_qa` — whether each source point is covered by the rendered polygon for its own cluster;
- `surface_territory_qa` (a table without geometry) — one row per cluster: its network length, the part of it inside the boundary, its land and grid cells, and why land is missing where it is (see [Choosing the grid spacing](#choosing-the-grid-spacing)).

The 2-D layer uses a clear attachment model: each grid-cell representative point attaches to its nearest location on the network and inherits that network location's exact cluster. If several distinct network locations are exactly equally near, the location on the lexicographically smallest string `edge_id` is used and `anchor_ties` records the ambiguity. Finer `--resolution` values give a finer surface approximation.

In symbols: for a grid cell with representative point $`g`$, the anchor $`a(g)`$ and the cell's cluster are

```math
a(g) = \arg\min_{z \in N} \lVert g - z \rVert, \qquad \mathrm{cluster}(g) = W\bigl(a(g)\bigr),
```

where $`N`$ is the set of all locations on the network, $`\lVert g - z \rVert`$ is straight-line distance in the CRS, and $`W`$ is the winner defined under [Method](#method). An anchor inside an edge therefore takes the winner of the side of the boundary $`x^*`$ on which it lies, and an anchor exactly on $`x^*`$ is a tie, which goes to the smaller cluster ID.

The GeoPackage also contains the inputs as used: `network_edges`, `points_input`, `points_snapped`, and `snap_connectors` (a straight line from each input point to its snapped node).

Every file written by the CLI also contains `netvoronoi_manifest`, a table without geometry with one JSON text: software versions, input files (size and SHA-256), network size and CRS, tie rules, and for a surface the attachment policy, the spacing and, for an adaptive run, every grid tried and the final status.

A layer with no rows is not written. For example, when every road component contains a point, the file has no `network_unassigned` layer. Treat a missing layer as an empty one.

The file is written as one transaction: all layers go to a temporary file in the same folder, which is read back and checked before it replaces the target in one step. A failure **before** the atomic replacement leaves an existing output unchanged, including with `--force`. A failure **after** replacement, such as a POSIX directory-sync error, raises `PublicationDurabilityError(published=True)`: the new file has already replaced the old one, and crash durability is unconfirmed. The staged file is fsynced first, but cross-platform crash durability is not guaranteed.

### Which roads land attaches to

`--attachment` (Python: `attachment=`) chooses the roads that land may attach to:

- `nearest_network` (default): the nearest road of any component. Land whose nearest road is on a component without points is unassigned (`surface_unassigned`).
- `nearest_reachable_network`: the nearest road on a component that contains a point. All land then has an owner. The two components are not connected by this; land near a road without points simply attaches to a farther road that has an owner.

The exact network partition is the same under both rules.

## Why the point clusters are preserved

Every snapped input point is inserted as a true graph node. A cluster may therefore have many source nodes. Its distance field is a multi-source shortest-path field, not a distance from an invented center. A cluster can have several disconnected pieces if the network geometry makes that correct.

Cluster labels are taken literally. In particular, a clustering noise label such as DBSCAN/HDBSCAN `-1` is treated as one ordinary cluster containing all points with that label. If noise should not compete for Voronoi territory, remove or relabel those points before calling `net-voronoi`.

## Road topology

Road topology is taken literally from input vertices:

- a shared road vertex is a graph junction;
- a geometric crossing without a shared vertex is **not** turned into a junction;
- duplicate straight segments between the same endpoint coordinates are removed;
- every other segment is kept, however short: inserting points moves points, never road;
- points are snapped to their nearest segment and inserted without creating new junctions on unrelated crossing roads;
- `points_snapped.snap_ties` is the number of distinct network locations at the nearest distance. It is 1 for a point next to an ordinary bend or junction (all nearest segments meet at one node) and 2 or more where the snap is ambiguous, for example at a crossing without a shared vertex.

`max_snap_distance` (`--max-snap-distance` in the CLI) limits the distance from each point to its nearest road location. A point is placed on an existing vertex or inserted node when its nearest location is within the segment's tolerance of it, `max(1e-9, 1e-12 × segment length)`, so `snap_distance` can exceed the distance to the road by at most that tolerance.

In symbols, for a nearest segment of length $`\ell`$:

```math
\tau_{\mathrm{seg}} = \max\bigl(10^{-9},\; 10^{-12}\,\ell\bigr), \qquad \text{snap distance} \le \text{distance to the road} + \tau_{\mathrm{seg}}.
```

Use a projected CRS. All network distances and `--resolution` values use its linear units. Road and point coordinates must be finite. The graph is an undirected simple graph: parallel edges between the same graph-node pair are not supported.

## Ties and cluster IDs

Cluster labels are compared as strings. In string order `"c10"` comes before `"c9"`, and the numeric label `10` (string `"10"`) comes before `9` (string `"9"`). Numerical equality uses a local tolerance `max(1e-9, 1e-12 × m)`, where `m` is the magnitude of the distances being compared, in CRS units. A very large distance elsewhere in the network therefore does not turn distinct short-range distances into a tie.

In symbols, two distances $`a`$ and $`b`$ are treated as equal when

```math
|a - b| \le \tau(a, b), \qquad \tau(a, b) = \max\bigl(10^{-9},\; 10^{-12}\, m\bigr), \quad m = \max\bigl(|a|, |b|\bigr).
```

For example, two distances of about 1,000,000 m are equal when they differ by at most $`10^{-6}`$ m, and two distances of about 10 m when they differ by at most $`10^{-9}`$ m.

## Choosing the grid spacing

A grid can move a border between two clusters by about one cell and can miss land narrower than a cell. With `--adaptive` (Python: `adaptive_surface_voronoi`), the surface is rendered on finer and finer grids, starting from `--resolution` and multiplying the spacing by `--refine-factor` (default 0.5), and each grid is compared with the one before it. The exact network partition is computed once for all grids.

The comparison is the share of land whose owner changed: $`1 - \sum_k \mathrm{area}(A_k \cap B_k) / \mathrm{area}(\text{boundary})`$, where $`A_k`$ and $`B_k`$ are the land of cluster $`k`$ (or of the unassigned category) in the two grids. Per cluster, the overlap $`\mathrm{IoU} = \mathrm{area}(A \cap B) / \mathrm{area}(A \cup B)`$ is reported.

The run stops with one of three statuses:

- `converged`: the last two grids changed at most `--changed-area-max` of the land (default 0.005), no cluster classified as `grid_lost` remains **under the default policy** (this condition can be disabled with `--allow-missing-territories`), and, if `--min-cluster-iou` is given, every cluster met that IoU;
- `budget_limited`: the next grid would exceed `--max-cells`;
- `refinement_limit`: `--max-refinements` finer grids (default 5) were tried.

One grid alone is never `converged`. If the first spacing needs more than `--max-cells` squares, the finest spacing that fits is used, found by counting squares without building a grid. The result is always the finest grid tried.

By default the CLI writes an adaptive result only when it is `converged`; otherwise it writes nothing, reports the measurements, and exits with code 1. `--accept-nonconverged` writes the finest grid anyway, records that decision in the manifest, and says on screen that the surface is not verified as converged.

Expect `budget_limited` on city-sized inputs. On a synthetic 15 km × 15 km road grid with 100 m blocks and 1,500 clusters, the share of land that changed owner was 24 % between 100 m and 50 m, 7 % between 50 m and 25 m, and 4.5 % between 25 m and 12.5 m; 0.5 % would need a spacing of a few metres. The measured share quantifies disagreement **between the two tested grids**; it does not bound the error against the unknown exact planar surface or predict disagreements at still finer spacings. `converged` means that two grids agreed within the policy, not that the polygons are exact.

`surface_territory_qa` explains clusters without land. `grid_lost` means the cluster owns road inside the boundary that no other owner's road covers, so it has land and a finer grid recovers it. `no_road_inside` (its road lies outside the boundary, common when the network extends beyond the study area), `shared_road_only` and `no_network_length` are reported but do not block convergence. The separate `territory_evidence` column marks `no_road_inside` and `shared_road_only` as `presence_undetermined`: lack of a road inside the boundary is **not** proof of no land there. Section 12 of `DESIGN.md` gives the reasoning.

## Publication and reproducibility

The CLI takes SHA-256 snapshots of each input **before reading it, immediately after reading it, and immediately before replacing the output**. Directory datasets are hashed by their member paths and contents. A dataset stored in several files is hashed as a whole: a shapefile with its `.shx`, `.dbf`, `.prj`, `.cpg` and `.shp.xml` files (labels are in `.dbf`; newer GDAL versions may use `.shp.xml` for extended field metadata), and a GeoPackage with its `-wal` file when one exists. The manifest then lists each member file with its own size and SHA-256. Changed inputs, source/output path aliasing, symlinks in input datasets, and destination SQLite `-wal`, `-shm` or `-journal` files stop publication. These checkpoints detect changes; they are not immutable snapshots and cannot eliminate every concurrent modification race.

Each staged GeoPackage undergoes SQLite integrity and foreign-key checks, plus verification of layer names, row counts, field names and CRS before publication. Attribute tables such as `netvoronoi_manifest` and `surface_territory_qa` are included in the transaction and are visible to GeoPackage-aware applications. The CLI writes these tables automatically; direct Python calls to the writers require callers to supply the optional `manifest` and `territories` arguments.

The package version stays `0.2.0` on `main` until the next release, so the manifest also records `netvoronoi_code_sha256`, a SHA-256 of the package's Python source that is the same on Windows and Linux. Two files made by different code have different values even when the version is the same.

If a run fails **after** `os.replace`, the new output may already be present. The typed `PublicationDurabilityError` and `PublicationCleanupError` explicitly mark `published=True`; inspect the destination instead of assuming the old file survived. On Windows, the package fsyncs the staged file but does not perform POSIX directory fsync. See [DESIGN.md, sections 13–15](DESIGN.md#13-transactional-output) for the full failure contract.

Exit codes of `netvoronoi build`, so that a script can tell whether the output changed. Never manually delete SQLite `-wal` or `-journal` files while recovering an occupied destination: close all database users and use SQLite-aware checkpointing or recovery instead.

| Code | Meaning | Output file |
|---|---|---|
| 0 | Written and checked. | New file in place. |
| 1 | Failure before replacing the destination, including invalid inputs, convergence not demonstrated, source changes, or occupied destination sidecars. | Existing destination unchanged; transient staging files may have been attempted. |
| 2 | Command-line usage error found by Typer before the command ran, for example an unknown option. | Unchanged. |
| 3 | The output was replaced, but directory sync (`PublicationDurabilityError`) or staging cleanup (`PublicationCleanupError`) failed. | New destination in place; inspect before retrying. Crash durability may be unconfirmed if sync failed. |

## Install

```powershell
python -m pip install -e ".[dev]"
pytest -q
```

If you want to use GeoParquet/Parquet directly from the CLI, install the optional Arrow dependency as well:

```powershell
python -m pip install -e ".[dev,parquet]"
```

## CLI

Exact network partition:

```powershell
netvoronoi build `
  "data\roads.gpkg" `
  "data\clustered_points.parquet" `
  --roads-layer roads `
  --cluster-col cluster_id `
  --point-id-col canonical_id `
  --max-snap-distance 500 `
  --output "output\netvoronoi.gpkg"
```

Network-informed polygon rendering:

```powershell
netvoronoi build `
  "data\roads.gpkg" `
  "data\clustered_points.parquet" `
  --roads-layer roads `
  --cluster-col cluster_id `
  --point-id-col canonical_id `
  --boundary "data\boundary.gpkg" `
  --boundary-layer boundary `
  --resolution 100 `
  --output "output\netvoronoi.gpkg"
```

Adaptive spacing, with land attached only to roads that a point can reach:

```powershell
netvoronoi build `
  "data\roads.gpkg" `
  "data\clustered_points.parquet" `
  --cluster-col cluster_id `
  --boundary "data\boundary.gpkg" `
  --resolution 100 `
  --adaptive `
  --attachment nearest_reachable_network `
  --output "output\netvoronoi.gpkg"
```

Options that only apply to `--adaptive` (`--refine-factor`, `--max-refinements`, `--changed-area-max`, `--min-cluster-iou`, `--allow-missing-territories`, `--accept-nonconverged`) are refused without it, and `--adaptive` and `--attachment` are refused without `--boundary`, before any input file is read.

Files ending in `.parquet` or `.geoparquet` are read with `geopandas.read_parquet`; all other files with `geopandas.read_file`. Re-running against an existing output requires `--force`. Input problems (a missing column, a point too far from the roads, an existing output file, an output that another program has open) are reported as one `Error: ...` line with exit code 1; the output is then unchanged. See [Publication and reproducibility](#publication-and-reproducibility) for all exit codes.

## Python API

```python
import geopandas as gpd
from netvoronoi import from_geodataframes, network_voronoi, surface_voronoi

roads = gpd.read_file("roads.gpkg", layer="roads")
points = gpd.read_parquet("clustered_points.parquet")
boundary = gpd.read_file("boundary.gpkg", layer="boundary")

network = from_geodataframes(
    roads,
    points,
    cluster_col="cluster_id",
    point_id_col="canonical_id",
    max_snap_distance=500,
)

# Exact network partition only:
exact = network_voronoi(network)
print(exact.segments, exact.cells, exact.unassigned_edges)

# Or, with the 2-D surface. surface_voronoi computes the exact network
# partition itself; it is available as surface.network.
surface = surface_voronoi(network, boundary, resolution=100)
exact = surface.network

# Or let the grid spacing be refined until two grids agree.
from netvoronoi import ConvergencePolicy, adaptive_surface_voronoi

result = adaptive_surface_voronoi(
    network,
    boundary,
    start_resolution=100,
    max_cells=1_000_000,
    convergence=ConvergencePolicy(global_changed_area_max=0.005),
)
print(result.status, result.selected_resolution)  # e.g. "budget_limited 12.5"
surface = result.surface                         # the finest grid tried
print(result.territories.territory_status.value_counts())
```

With `require_convergence=True`, a run that does not converge raises `ConvergenceError`; its `result` attribute holds the same measurements.

## Code map

The modules, in reading order:

1. `src/netvoronoi/network.py` — builds a `SpatialNetwork` from road lines and points: road segments, shared vertices, snapping, point insertion;
2. `src/netvoronoi/model.py` — the `SpatialNetwork` data contract and its validation;
3. `src/netvoronoi/core.py` — the exact partition: winners at nodes, the one per-edge boundary formula (`_edge_boundary`), and the cut pieces;
4. `src/netvoronoi/surface.py` — the approximate 2-D rendering on a grid, and the two attachment policies;
5. `src/netvoronoi/adaptive.py` — refinement of the grid spacing, the comparison between grids, and the territory table;
6. `src/netvoronoi/io.py` and `src/netvoronoi/cli.py` — transactional GeoPackage output and the command line;
7. `src/netvoronoi/_numeric.py` — the shared tolerance rule and two array helpers.

`DESIGN.md` explains the method and the reasons behind it.

## Tests

`pytest -q` runs the whole suite. Besides tests of single cases, `tests/test_core_reference.py` compares `network_voronoi` with the slow reference in `tests/reference.py` (one full shortest-path search per cluster, winner taken over all clusters) on random road networks and on integer grids where exact ties are everywhere.

## Performance

Measured on one machine with a regular grid of roads with 50 m spacing and randomly placed points:

- 127,500 edges, 3,000 points in 400 clusters: building the network about 0.6 s, `network_voronoi` about 0.3 s;
- about 1,000,000 edges, 50,000 points in 5,000 clusters: building about 6.4 s, `network_voronoi` about 4.8 s.

The slowest case for `network_voronoi` is when exact ties are everywhere (for example, all points on grid nodes): every cluster then needs its own shortest-path search. With about 1,000,000 edges and 5,000 clusters this took about 23 s.

## Scope of current `main` (unreleased additions since v0.2.0)

Adaptive rendering, the attachment selector, provenance and transactional output are merged into `main`, but the package metadata and latest tagged release still report `0.2.0`. These additions have not yet received their own versioned release.

Supported: projected undirected simple line networks, geometric edge length as impedance, LineString/MultiLineString road data, clustered point input, exact insertion of snapped points, multi-source shortest paths by cluster, disconnected components, deterministic ties, transactional GeoPackage output with an embedded manifest, optional Parquet input (with `netvoronoi[parquet]`), and optional grid-rendered polygon cells with two attachment policies and adaptive grid spacing.

Not yet supported: directed/one-way routing, turn restrictions, arbitrary travel-time impedance, polygon cells that are mathematically exact off-network, or automatic repair of uncertain road topology.

## Provenance

`net-voronoi` implements a published construction. It does not propose a new kind of Voronoi diagram. The object it computes is the network Voronoi diagram (Okabe et al., 2000; Okabe et al., 2008), in which every location on a network belongs to the nearest generator by shortest-path distance. Here each generator is a cluster of points.

The cluster rule reduces to the ordinary diagram of the points. At every location $`x`$, the smallest $`D(c, x)`$ over all clusters equals the smallest network distance from $`x`$ to any single point, so the winning cluster is the cluster of the nearest point. The cluster cells are therefore the ordinary network Voronoi cells of the individual points, merged by cluster label. The only addition is the rule for exact ties between clusters, which is a convention (see [Ties and cluster IDs](#ties-and-cluster-ids)).

### Source of each step

| Step in `net-voronoi` | Published source or earlier implementation |
|---|---|
| Each network location belongs to the nearest generator by shortest-path distance | Network Voronoi diagram: Okabe et al. (2000), section on network Voronoi diagrams; Okabe et al. (2008). Partition of a network that includes locations inside edges: Hakimi, Labbé and Schmeichel (1992). Node-based graph version: Erwig (2000). |
| A generator is a set of points, and the distance to it is the distance to its nearest member | Equivalent to merging the ordinary point cells by label, as shown above. Voronoi diagrams whose generators are sets rather than single points, such as areas, are a standard generalization (Okabe et al., 2000). |
| One multi-source shortest-path search from all points | Dijkstra (1959) started from several sources at once. NetworkX `voronoi_cells` uses the same construction and cites Erwig (2000) (Hagberg et al., 2008; NetworkX documentation). Computed here with SciPy `csgraph.dijkstra` and `min_only=True` (Virtanen et al., 2020). |
| Exact boundary inside an edge, $`x^* = (D(v) + L - D(u))/2`$ | Follows from the distance along an edge, $`\min(D(u) + x,\ D(v) + L - x)`$, which is the edge function of network location theory (Hakimi, 1964) and of the network Voronoi partition (Hakimi et al., 1992; Okabe et al., 2008). |
| Snapped points inserted as graph nodes, splitting the edge | Points moved to the nearest road segment, with the segment split at that location (Wang et al., 2019); objects located on network edges (Yiu and Mamoulis, 2004). |
| 2-D surface | A rendering step, not a Voronoi construction: each grid cell takes the exact network result at its nearest network location, and the result is labeled as an approximation. |
| Adaptive grid spacing | Engineering, not a Voronoi construction: the same rendering repeated at finer spacings with a stated stopping rule. The `grid_lost` classification rests only on roads being closed sets (`DESIGN.md`, section 12). |

### What is specific to this package

**Tie-candidate search.** Section 8 of [`DESIGN.md`](DESIGN.md) finds the winners with one multi-source search, plus extra bounded searches only for clusters that could change a winner under an exact tie. No published source for this procedure was found in the provenance review of September 2026. It changes only the amount of computation, not the result: `DESIGN.md` gives the argument that no true tie can be missed, and `tests/test_core_reference.py` checks that it gives the same winners as the direct method (one full search per cluster) on random networks and on grids where ties are everywhere.

The other package-specific parts are engineering and conventions: road topology from shared vertices, the local numerical tolerance, snapping diagnostics, validation of `SpatialNetwork` objects, and the output layers. None of them changes the definition of the diagram.

## References

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.

Erwig, M. (2000). The graph Voronoi diagram with applications. *Networks*, 36(3), 156–163. https://doi.org/10.1002/1097-0037(200010)36:3%3C156::AID-NET2%3E3.0.CO;2-L

Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In *Proceedings of the 7th Python in Science Conference (SciPy 2008)*, 11–15.

Hakimi, S. L. (1964). Optimum locations of switching centers and the absolute centers and medians of a graph. *Operations Research*, 12(3), 450–459. https://doi.org/10.1287/opre.12.3.450

Hakimi, S. L., Labbé, M., & Schmeichel, E. (1992). The Voronoi partition of a network and its implications in location theory. *ORSA Journal on Computing* (now *INFORMS Journal on Computing*), 4(4), 412–417.

NetworkX documentation. *Voronoi cells*. https://networkx.org/documentation/stable/reference/algorithms/voronoi.html

Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N. (2000). *Spatial Tessellations: Concepts and Applications of Voronoi Diagrams* (2nd ed.). Wiley. ISBN 978-0-471-98635-5.

Okabe, A., Satoh, T., Furuta, T., Suzuki, A., & Okano, K. (2008). Generalized network Voronoi diagrams: Concepts, computational methods, and applications. *International Journal of Geographical Information Science*, 22(9), 965–994.

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2

Wang, T., Ren, C., Luo, Y., & Tian, J. (2019). NS-DBSCAN: A density-based clustering algorithm in network space. *ISPRS International Journal of Geo-Information*, 8(5), 218. https://doi.org/10.3390/ijgi8050218

Yiu, M. L., & Mamoulis, N. (2004). Clustering objects on a spatial network. In *Proceedings of the 2004 ACM SIGMOD International Conference on Management of Data*, 443–454. https://doi.org/10.1145/1007568.1007619

## License

MIT.

## Author and Citation

**James Matthew Miraflor**  
[Scientific Computing Laboratory](https://scl.dcs.upd.edu.ph/)  
[Department of Computer Science](https://dcs.upd.edu.ph/)  
College of Engineering  
University of the Philippines Diliman

**net-voronoi** is authored and maintained by James Matthew Miraflor, with AI assistance in software development, debugging, testing, and documentation. Responsibility for the project's methodological decisions, scientific validity, and published outputs rests with the author.
