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
