# Design

## 1. Cluster distance

For cluster `c` with snapped source nodes `S_c`, define the node distance

```text
D_c(v) = min d_N(s, v).
         s in S_c
```

The global nearest-cluster distance is

```text
D(v) = min_c D_c(v).
```

The node winner is the lexicographically smallest cluster ID whose `D_c(v)` is numerically equal to `D(v)` under the local tolerance in section 9. Section 8 explains how the winners are found with one shortest-path search in most networks, instead of one search per cluster.

## 2. Why points are inserted into the graph

A Voronoi boundary may lie on the same road segment as an input point. Treating a snapped point only as metadata on an unsplit road would violate the site-free-edge assumption used by the exact edge formula. Therefore every interior snapped location becomes a graph node and the road segment is split there before shortest paths are computed.

## 3. Exact partition of one site-free edge

For edge `(u, v)` of length `L`, measure `x` from `u`. The nearest-cluster distance along the edge is

```text
min(D(u) + x, D(v) + L - x).
```

If the deterministic winner at `u` and `v` is the same cluster, that cluster owns the entire positive-length edge. Otherwise the transition is at

```text
x* = (D(v) + L - D(u)) / 2.
```

clipped to `[0, L]`. The `u` winner owns `[0, x*]` and the `v` winner owns `[x*, L]`. Example: `D(u) = 2`, `D(v) = 4`, `L = 10` give `x* = 6`, where both terms equal 8.

A boundary within the tolerance of an edge end is moved onto that end, so that no piece shorter than the tolerance is produced; the whole edge then goes to one cluster. The pieces of every reachable edge therefore cover `[0, L]` exactly, without gaps or overlaps. This formula is written once, in `core._edge_boundary`, and the 2-D surface uses the same function.

## 4. 2-D surface

The network partition is exact; a land polygon is not intrinsically part of a network metric. The optional surface renderer therefore defines an explicit nearest-network attachment model. It partitions a boundary into grid cells, attaches one representative point from each cell to its nearest network location (the anchor), and assigns the exact cluster at that anchor. The resulting polygons are an approximation controlled by `resolution`.

The cluster at an anchor follows the network partition exactly: an anchor on a node takes that node's winner; an anchor inside an edge takes the winner of the side of the section 3 boundary it lies on; an anchor on the boundary itself is a tie and goes to the smaller cluster ID. `surface_grid_debug.anchor_ties` is the number of distinct network locations at the nearest distance, counted the same way as `snap_ties` (section 5).

The grid lines are computed once, and each square takes its sides from these shared lines. A square's right side is then exactly the same number as its right neighbour's left side, so dissolving the squares of one cluster leaves no gaps between them. (Computing each square's right side as `left + resolution` does not guarantee this at large coordinates such as UTM eastings: the two sums can differ in the last bits.)

## 5. Topology rule

Only shared input vertices create road junctions. Inserting a snapped point inside one segment does not connect that segment to another line that merely crosses at the same coordinate.

Every input segment of positive length is kept as at least one edge, however short. Points are moved to the network, the network is not changed to fit the points.

`max_snap_distance` limits the distance from each point to its nearest road location. The node where the point is finally placed can be slightly farther: when the nearest location is within the segment's tolerance (`max(1e-9, 1e-12 × segment length)`) of a vertex or of another inserted point, the point is placed on that vertex or node. So `snap_distance` can exceed the distance to the road by at most that tolerance. Example: a point exactly on a 60 km road, 4e-8 m from a vertex, is placed on the vertex, passes `max_snap_distance = 0`, and has `snap_distance = 4e-8`.

`snap_ties` counts distinct network locations at the nearest distance, not nearest segments. Segments that meet at the nearest vertex offer one location, the node of that vertex, so a point next to an ordinary bend or junction has `snap_ties = 1`. Positions inside different segments are different locations: a point at a crossing without a shared vertex has `snap_ties = 2`, and so does a point exactly between two parallel roads.

## 6. SpatialNetwork invariants

The public `SpatialNetwork` object validates that edge geometry, edge length, node endpoints, and the sparse adjacency matrix describe the same undirected simple graph. This matters because the shortest-path distances and the exact edge split are only consistent when those representations agree. Curved custom LineStrings are permitted; Voronoi pieces are cut along the full LineString geometry rather than replaced by straight chords.

The adjacency matrix is checked the way SciPy's shortest-path functions read it: every stored entry is a connection. An explicitly stored zero is a connection of length zero, and two stored entries for the same node pair are two connections, of which SciPy uses the shorter. So the check neither adds repeated entries together nor drops stored zeros. A matrix with a stored zero between unconnected nodes, or with two entries for one pair, is rejected. Weight equality is checked with an elementwise tolerance based on each stored/expected weight; a billion-unit edge elsewhere cannot hide a wrong weight on a millimetre edge.

## 7. Surface allocation guard

The surface grid is generated from the same integer `(nx, ny)` shape used by the `max_cells` preflight check. This keeps the memory guard and the actual candidate allocation identical even for awkward floating-point origins and resolutions.

## 8. Finding the winners with few searches

The direct method runs one multi-source shortest-path search per cluster and compares the results at every node. Its cost grows with the number of clusters times the size of the network. `core._node_partition` gets the same result as follows.

One multi-source search from all points at once gives `D(v)` at every node and, for each node, one nearest point. The cluster of that point is a nearest cluster, and it is the winner unless another cluster with a smaller ID is equally near. When two clusters are equally near, the search picks one of them without a fixed rule.

Example: a road `0 --- 5 --- 10` with a point of cluster `"B"` at 0 and a point of cluster `"A"` at 10. The node at 5 is 5 from both points, and the search may label it `"B"`. The correct winner is `"A"`.

Which clusters can be missed in this way? Suppose `c` is the correct winner at node `v`, but the search labelled `v` with another cluster. Follow a shortest path from `v` to the nearest point of `c`, at node `s`.

- At `s`, the search label is the smallest cluster among the points on `s`, and this is `c`: a smaller cluster on `s` would reach `v` along the same path, so its distance at `v` would be at most `D_c(v)`, and it would be the correct winner at `v` instead of `c`. (A smaller distance also passes the local test of section 9: lowering a distance by some amount lowers its gap to `D(v)` by that amount, and its tolerance by at most `1e-12` times that amount.)
- `c` is within the local tolerance of the minimum at `v`, and every local tolerance is at most `T`, the tolerance of the largest node distance in the network (apart from a relative difference of about `1e-12`, which the implementation's margin covers). Walking from `v` towards `s` lowers `D_c` by exactly the distance walked, while `D` can drop by at most that distance, so at every node `p` of the path `D_c(p) - D(p) <= D_c(v) - D(v) <= T`. It follows that every edge `(a, b)` on the path, with `a` nearer to `s`, is tight within `T`: `D(a) + length(a, b) <= D_c(a) + length(a, b) = D_c(b) <= D(b) + T`.

So along the path the label changes from `c` to another cluster across a tight edge. The candidates are therefore the labels at both ends of every tight edge whose two labels differ. In the example, the edge from 5 to 10 is tight (`D(10) + 5 = 0 + 5 = D(5)`) with labels `"B"` and `"A"`, so `"A"` is a candidate.

Only the candidates get a search of their own. Candidate discovery and the search limit use the conservative network-wide bound `T`, so no true local tie can be missed. Final tie decisions do **not** use `T`: at each node a candidate replaces a larger label only when its distance is within the local tolerance `_tol(D_c(v), D(v))`. This separation keeps the argument above valid while preventing a very large, unrelated component from turning distinct short-range distances into ties. The exact local test is applied only at the nodes where the candidate is within `2T` of the minimum, which is a small set; every local tie is among them, because every local tolerance is below `2T`. This keeps the cost of a candidate close to the cost of its search.

When exact ties are rare, which is usual for roads with measured coordinates, there are few or no candidates, and the whole partition needs one search. When ties are everywhere, for example with all points on the nodes of a regular grid, every cluster becomes a candidate and the cost approaches that of the direct method. `tests/reference.py` implements the direct method, and the tests check that both give the same winners on random networks and on tie-heavy grids.

## 9. Numerical tolerance

Floating-point distances and positions are compared with `_numeric._tol`: two numbers of magnitude up to `m` are treated as equal when they differ by at most `max(1e-9, 1e-12 × m)`. The relative part grows with the numbers compared, because a number of magnitude `m` is stored with a relative error of about `1e-16`, and sums of many such numbers collect more error.

At nodes, the final equality test is local: `D_c(v)` and `D(v)` are compared with `_tol(D_c(v), D(v))`. Section 8 still uses one conservative network-wide value `T = _tol(largest finite D)` for candidate discovery only. `T` is a superset bound, not the semantic tie rule.

## 10. Attachment policies for the 2-D surface

The exact network result gives owners only to locations on roads that a point can reach. Land needs one more rule: which road a land point attaches to. Two rules are offered, and the chosen one is recorded with the output.

- `nearest_network` (default, and the only rule before the `attachment` option existed): the nearest road of any component. Land whose nearest road lies on a component without points stays unassigned, because that road has no owner.
- `nearest_reachable_network`: the nearest road on a component that contains at least one point. Land near a road without points then attaches to a farther road that has an owner.

Example: a land point 10 m from a road without points and 100 m from a road of cluster A is unassigned under the first rule and belongs to A under the second. The second rule does not connect the two roads and does not claim that A's road can be reached from the nearer one; it only chooses a different road for attaching land. The exact network partition is the same under both rules. Equally near roads are decided by the smaller `edge_id` in string order under both rules, because the rank of `edge_id` is taken over the whole edge table.

## 11. Adaptive grid spacing and what it measures

A grid with spacing `r` can place a border between two clusters wrongly by up to about one cell, and can miss land narrower than a cell. `adaptive_surface_voronoi` renders grids with spacing `r0, r0·f, r0·f², …` (`f = refine_factor`) and compares each grid with the one before it.

**Starting spacing.** A larger spacing never needs more grid squares, so the spacings whose grid fits `max_cells` form an interval `[r*, ∞)`. `_starting_resolution` finds `r*` by bisection on square counts only, without building a grid; it uses the geometric mean while the bracket spans more than a factor of 2, so that a request such as `1e-300` takes a few steps.

**Comparison.** For each ownership category `k` (every cluster, and `unassigned`), let `A_k` and `B_k` be its land in the coarser and the finer grid. The share of land that changed owner is

    changed = 1 − Σ_k area(A_k ∩ B_k) / area(boundary).

Comparing the union of all polygons would show nothing, because both grids cover the whole boundary: two clusters that swap all their land give `changed = 1` here and an unchanged union. Per cluster, `IoU = area(A ∩ B) / area(A ∪ B)` is reported; a cluster with land in neither grid has no IoU (not 1), and a cluster with land in one grid only has IoU 0.

**Stopping.** The loop stops with `converged` when two consecutive grids meet the `ConvergencePolicy` (by default: `changed ≤ 0.005`, and no cluster in the state `grid_lost`, section 12). It stops with `budget_limited` when the next grid would exceed `max_cells`, and with `refinement_limit` when `max_refinements` finer grids have been tried. One grid alone is never `converged`. The result is always the finest grid that passed the geometry checks below.

**What `converged` means.** Two consecutive grids agreed within the policy. It does not mean that the polygons are exact: a different grid origin or a third, finer grid can still change them, especially for narrow territories. The thresholds are settings, not error bounds derived from the geometry.

**How far from convergence real inputs are.** On a synthetic city (a 15 km × 15 km road grid with 100 m blocks, 1,500 clusters of 1–4 points), the share of land that changed owner was 24 % between 100 m and 50 m, 7 % between 50 m and 25 m, and 4.5 % between 25 m and 12.5 m. Grid roads make the change large: inside each block, land is divided along the diagonals between the four surrounding roads, and every such line is drawn as a staircase of cells. Reaching 0.5 % would need a spacing of a few metres, that is tens of millions of cells. With the default `max_cells`, such inputs end as `budget_limited`. The measured `changed` value quantifies disagreement between the two tested grid resolutions. It does not estimate or upper-bound error relative to the exact planar attachment map.

**Geometry checks.** Every grid must satisfy, with `B` the boundary and `P_k` the land of category `k`: each `P_k` is a valid polygon; `area(B − ∪P_k)`, `area(∪P_k − B)` and `Σ area(P_k) − area(∪P_k)` are each at most `1e-9 · area(B)`. The missing and overlapping areas are checked separately, because in one combined sum a gap and an overlap of equal size cancel. Under `nearest_reachable_network` there must be no unassigned land. A coarse grid never breaks these rules (it covers the boundary exactly once), so a failure stops the run instead of starting a finer grid.

## 12. Why a cluster can lack land, and which cases a finer grid can fix

A cluster can have no land in a rendering for different reasons. Only the `grid_lost` classification proves that positive-area land exists and that a sufficiently fine grid can represent it; other categories may remain undecidable using these diagnostics. `territory_qa` separates them:

- `no_network_length`: the cluster owns no positive network length, for example when its points share a node with points of a cluster with a smaller ID. This is a property of the exact partition.
- `no_road_inside`: all of its road lies outside the boundary or on its outline. Typical when the road network extends beyond the study area. Land near the outline may still attach to such a road, so this means "not proven to exist", not "proven absent".
- `shared_road_only`: its road inside the boundary is covered by roads of other owners (possible in custom networks with coincident edges). Also not proven either way.
- `grid_lost`: it owns road inside the boundary that no road of another owner covers. Such a cluster has land of positive area, so a finer grid will recover it.

The argument for `grid_lost`: take a point `p` on such a road, strictly inside the boundary, away from piece ends and from the tie band of `_assign_anchors`, and on no road of another owner. Roads are closed sets, so every road of another owner is at least some distance `d > 0` from `p`. A land point `g` with `|g − p| < d/2` has a road location within `d/2` (namely `p`), while every road of another owner is farther than `d/2` from `g`. So `g` attaches to a road of this cluster. The boundary interior contains a disk around `p`, so the cluster owns a disk of land with positive area. Crossing roads meet the piece in single points, which have no length; they do not affect the argument.

Before this classification, the only evidence for "the grid lost this cluster" would be that the cluster owns network length somewhere. That evidence is wrong for a road network that extends beyond the boundary: a cluster outside the study area owns length but may have no land inside at any spacing, and treating it as a grid problem makes refinement continue until the budget is exhausted.

## 13. Transactional output

A GeoPackage is written one layer at a time. Deleting the old file first and then failing at a later layer would lose the previous valid output. `io._publish` therefore writes every layer to a new file with a unique temporary name in the target's folder, reads it back (`_check_staged`: SQLite integrity and foreign keys, the exact set of layers, and per layer the row count, column names and CRS), flushes it to disk, and only then moves it onto the target name with one `os.replace`. Within one folder this replaces the target in a single step. Any failure before that step leaves the old file byte for byte unchanged and removes the temporary file. The old file is never deleted to make room; on Windows, a target held open by another program makes `os.replace` fail, and that error is reported.

The file is flushed through a handle opened for reading and writing, because Windows refuses to flush (`FlushFileBuffers`) a handle opened only for reading.

How the file was made is recorded inside the same file, in the table `netvoronoi_manifest` (one row, one JSON text): software versions, a SHA-256 of the package's own Python source (section 15), input files with size and SHA-256, network size and CRS, tie rules, and for a surface the attachment policy, spacing, and for an adaptive run every attempt with its measurements and the final status. Keeping it in the same file means it cannot be separated from the layers it describes.

## 14. Comparing two renderings, and what the territory evidence proves

The changed-owner-area statistic is meaningful only for two renderings of the **same geographic domain** in the **same CRS** with the **same attachment policy**. Equal total area is not enough: two distant polygons may have equal areas but no common land. Every regular rendering therefore carries its boundary geometry. The comparison checks that the symmetric difference of the domains is below a small relative numeric tolerance. Older result objects without stored boundaries are checked using the union of their rendered grid cells; this slower path preserves compatibility.

Territory diagnostics separate evidence from inference. `present` means positive-area land was observed in the current rendering. `grid_lost` means a positive-area territory can be demonstrated from an unshared road piece, even though the current grid missed it. `no_road_inside` and `shared_road_only` do **not** mean absence of true land; they mean its existence could not be established with this test. The `territory_evidence` field states this separately. A reported `converged` result means only two successive grids passed the selected changed-owner and optional IoU thresholds. It is **not** a certified error bound or proof that all true planar territories were represented.

## 15. Input checks, outputs in use, and failures after replacement

For command-line runs, input content is hashed before reading, immediately after reading, and again after the staged GeoPackage is validated but before replacing the destination. Folder datasets are hashed as ordered collections of relative file names and file contents. A dataset stored in several files is hashed as a whole: for a shapefile, every file with the same base name and one of the extensions `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg` and `.shp.xml`, in any letter case; for a GeoPackage (or another SQLite database), the file and its `-wal` file when one exists, because a reader also sees the changes kept there. Hashing only the named file would miss changes to the attributes: the cluster labels of a shapefile are in its `.dbf`. Other files with the same base name, such as an output written beside the input, are not included. Symlinks are rejected. If an input differs between checkpoints, the command aborts and retains the old destination. This is a *multi-checkpoint integrity guard*, not an immutable filesystem snapshot; adversarial changes between a check and the subsequent operation are outside its guarantee.

The transaction refuses an output path that aliases an input, or one with live SQLite `-wal`, `-shm`, or `-journal` sidecar files (`OutputInUseError`, a subclass of `RuntimeError`). Such files usually mean that a program such as QGIS has the old file open for editing. The command line checks this before it reads any input, so it fails in a second rather than after the whole computation, and `_publish` checks it again just before the replacement. It never attempts to delete the destination sidecars of a possibly active connection.

On POSIX, a directory fsync follows replacement. If this fails, the new file has already been published. A distinct `PublicationDurabilityError` reports `published=True`, instead of pretending that the old file is intact or suppressing a durability problem, and the command line exits with code 3. Every other failure exits with code 1 (or 2 for a usage error found by Typer), and code 1 therefore always means that the old file is unchanged. Windows does not provide this POSIX directory fsync; the staged file's fsync still occurs. Platform-level crash durability cannot be promised merely from successful `os.replace` on every filesystem.

The package version stays at the last release while `main` changes, so the version string cannot identify the code that made a file. The manifest therefore also records `netvoronoi_code_sha256`: one SHA-256 over the name and content of every `.py` file of the package, in name order, with Windows line endings (CRLF) turned into LF first, so that a Windows checkout and a Linux checkout of the same commit give the same value.
