# net-voronoi

`net-voronoi` builds **cluster-based Voronoi regions informed by shortest-path distance on a spatial network**.

Input points already have cluster labels. The package does not create clusters and does not replace a cluster with a centroid or medoid.

There is **no additive tolerance / epsilon Voronoi mode**. Cells are exclusive in positive network length.

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
- `surface_point_qa` — whether each source point is covered by the rendered polygon for its own cluster.

The 2-D layer uses a clear attachment model: each grid-cell representative point attaches to its nearest location on the network and inherits that network location's exact cluster. If several distinct network locations are exactly equally near, the location on the lexicographically smallest string `edge_id` is used and `anchor_ties` records the ambiguity. Finer `--resolution` values give a finer surface approximation.

In symbols: for a grid cell with representative point $`g`$, the anchor $`a(g)`$ and the cell's cluster are

```math
a(g) = \arg\min_{z \in N} \lVert g - z \rVert, \qquad \mathrm{cluster}(g) = W\bigl(a(g)\bigr),
```

where $`N`$ is the set of all locations on the network, $`\lVert g - z \rVert`$ is straight-line distance in the CRS, and $`W`$ is the winner defined under [Method](#method). An anchor inside an edge therefore takes the winner of the side of the boundary $`x^*`$ on which it lies, and an anchor exactly on $`x^*`$ is a tie, which goes to the smaller cluster ID.

The GeoPackage also contains the inputs as used: `network_edges`, `points_input`, `points_snapped`, and `snap_connectors` (a straight line from each input point to its snapped node).

A layer with no rows is not written. For example, when every road component contains a point, the file has no `network_unassigned` layer. Treat a missing layer as an empty one.

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

Files ending in `.parquet` or `.geoparquet` are read with `geopandas.read_parquet`; all other files with `geopandas.read_file`. Re-running against an existing output requires `--force`. Input problems (a missing column, a point too far from the roads, an existing output file) are reported as one `Error: ...` line with exit code 1.

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

# Or, with the 2-D surface. surface_voronoi computes the exact network
# partition itself; it is available as surface.network.
surface = surface_voronoi(network, boundary, resolution=100)
exact = surface.network
```

## Code map

The modules, in reading order:

1. `src/netvoronoi/network.py` — builds a `SpatialNetwork` from road lines and points: road segments, shared vertices, snapping, point insertion;
2. `src/netvoronoi/model.py` — the `SpatialNetwork` data contract and its validation;
3. `src/netvoronoi/core.py` — the exact partition: winners at nodes, the one per-edge boundary formula (`_edge_boundary`), and the cut pieces;
4. `src/netvoronoi/surface.py` — the approximate 2-D rendering on a grid;
5. `src/netvoronoi/io.py` and `src/netvoronoi/cli.py` — GeoPackage output and the command line;
6. `src/netvoronoi/_numeric.py` — the shared tolerance rule and two array helpers.

`DESIGN.md` explains the method and the reasons behind it.

## Tests

`pytest -q` runs the whole suite. Besides tests of single cases, `tests/test_core_reference.py` compares `network_voronoi` with the slow reference in `tests/reference.py` (one full shortest-path search per cluster, winner taken over all clusters) on random road networks and on integer grids where exact ties are everywhere.

## Performance

Measured on one machine with a regular grid of roads with 50 m spacing and randomly placed points:

- 127,500 edges, 3,000 points in 400 clusters: building the network about 0.6 s, `network_voronoi` about 0.3 s;
- about 1,000,000 edges, 50,000 points in 5,000 clusters: building about 6.4 s, `network_voronoi` about 4.8 s.

The slowest case for `network_voronoi` is when exact ties are everywhere (for example, all points on grid nodes): every cluster then needs its own shortest-path search. With about 1,000,000 edges and 5,000 clusters this took about 23 s.

## Scope of v0.2.0

Supported: projected undirected simple line networks, geometric edge length as impedance, LineString/MultiLineString road data, clustered point input, exact insertion of snapped points, multi-source shortest paths by cluster, disconnected components, deterministic ties, GeoPackage output, optional Parquet input (with `netvoronoi[parquet]`), and optional grid-rendered polygon cells.

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
