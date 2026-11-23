# Changelog

## Unreleased

### Documentation alignment

- Synchronize the README, mathematical design notes and visual guide with the merged adaptive rendering, territory evidence, provenance and transactional publication APIs.
- Correct the visual guide's exact network result field to `unassigned_edges` and clarify the difference between pre-replacement and post-replacement failures.
- Keep the package version at `0.2.0` until a separate versioned release.

### Adaptive surface rendering and v3 hardening (merged into `main` via PR #1)

- Convergence comparisons reject different geographic domains, different CRSs and different attachment policies. Internal comparisons reuse the stored clipping boundary without reconstructing an expensive grid union.
- Territory QA reports `territory_evidence`: `no_road_inside` and `shared_road_only` are *undetermined*, not proofs of absent land. Adaptive manifests distinguish grid-to-grid stability from exact territory completeness.
- CLI snapshots input files and directory datasets by SHA-256 before reading, checks them after reading, and checks again immediately before output replacement. It refuses to overwrite a source dataset.
- Transactional publication rejects destination WAL/journal sidecars and symlink targets, and reports a post-replacement directory-sync failure as `PublicationDurabilityError(published=True)` rather than silently ignoring it.
- Existing fixed-rendering geometry, network ownership and tie rules are unchanged.


**Compatibility:** the exact network engine and default fixed-resolution geometry are unchanged. Existing GeoPackage map layers retain their meaning; CLI outputs gain two attribute tables (`netvoronoi_manifest`, and `surface_territory_qa` for surfaces). Package metadata remains at version `0.2.0` until a new release is cut.

- **Transactional GeoPackage output.** All layers are written to a temporary file in the target's folder, read back and checked (SQLite integrity and foreign keys, layer names, row counts, column names, CRS), flushed, and then moved onto the target with one `os.replace`. Before, `--force` deleted the old file first, so a failure in a later layer lost the previous output. The flush uses a read-write handle because Windows cannot flush a read-only one.
- **`netvoronoi_manifest`** in every CLI output: a table without geometry holding one JSON text with software versions, input files (size and SHA-256), network size and CRS, tie rules and surface settings.
- **`attachment=` / `--attachment`** for the 2-D surface: `nearest_network` (default, the previous behaviour) or `nearest_reachable_network` (land attaches only to roads on components that contain a point). The exact network partition and the `edge_id` tie rule are the same under both.
- **`adaptive_surface_voronoi` / `--adaptive`**: renders finer and finer grids under a per-grid `max_cells` budget, computes the exact network partition once, compares consecutive grids by the share of land that changed owner and by per-cluster IoU, and stops with `converged`, `budget_limited` or `refinement_limit`. A too-fine first spacing is coarsened to the finest one that fits, without building a grid. The CLI writes a non-converged result only with `--accept-nonconverged`.
- **`territory_qa` and `surface_territory_qa`**: one row per cluster, separating clusters lost only by the grid (`grid_lost`, provably recoverable) from clusters whose road lies outside the boundary or is shared, and from clusters without network length.
- **CLI option checks before reading input**: options that need `--adaptive` or `--boundary` are refused first.
- `SurfaceVoronoiResult` gains `resolution` and `attachment` fields with defaults; the five existing fields are unchanged.

## 0.2.0 - 2026-09-30

Clean replacement of the previous `net-voronoi` implementation.

- Makes point cluster labels the primary sites of the Voronoi construction.
- Uses minimum network distance to any point in each cluster.
- Removes additive-tolerance/epsilon Voronoi behavior entirely.
- Removes the `spaghetti` dependency and builds topology directly from road vertices.
- Inserts snapped points as true graph nodes before shortest-path analysis.
- Adds exact lineal cluster cells, deterministic tie handling, disconnected-component output, snap-ambiguity QA, optional 2-D surface rendering, and point-containment QA.
- Validates that custom `SpatialNetwork` geometry, edge lengths, node endpoints, and adjacency agree.
- Preserves curved custom edge geometry when cutting Voronoi segments.
- Makes the surface allocation guard use exactly the grid dimensions that are later allocated.
- Adds explicit optional Parquet input support through `netvoronoi[parquet]`.
- Rejects non-finite road and point coordinates early.

### Review fixes included in 0.2.0

Errors fixed:

- A road segment shorter than the numerical tolerance (about 1e-9) was dropped while inserting points, which could disconnect the network and leave road unassigned. Every segment is now kept as at least one edge.
- Surface grid squares computed each right side as `left + resolution`, which at large coordinates differs from the neighbouring square's left side in the last bits; the dissolved cluster polygons then had thin gaps and extra parts. Squares now share one set of grid lines.
- `snap_ties` and `anchor_ties` counted equally near segments or edges, so every point next to an ordinary bend or junction was reported as ambiguous. They now count distinct network locations.
- `SpatialNetwork` validation added repeated adjacency entries together and dropped stored zeros before comparing, while SciPy's shortest-path functions read both as connections. A custom matrix could pass validation and describe a different graph. Stored entries are now compared as SciPy reads them.

Additional audit fixes:

- Tie decisions at nodes now use a local numerical tolerance. The previous network-wide tolerance let a very large distant component turn distinct short-range distances into a tie.
- Adjacency weight validation now uses an elementwise tolerance. A huge edge elsewhere can no longer hide an incorrect weight on a short edge.
- `max_snap_distance` limits the distance from each point to its nearest road location; the returned `snap_distance` can exceed that distance by at most the segment tolerance, and this is now documented.
- Custom `SpatialNetwork` IDs now obey the same string-identity rules as `from_geodataframes`: empty IDs and values that collapse to the same string are rejected.
- Invalid or non-finite polygon boundaries are rejected before surface intersection, and non-finite `max_cells` values receive a clean validation error.
- The README now states explicitly that noise labels such as DBSCAN/HDBSCAN `-1` are treated as ordinary clusters unless callers filter or relabel them.

Final-pass changes:

- The second pass had added a check of `max_snap_distance` on the final node that allowed only the tolerance of `max_snap_distance` itself. It rejected points lying exactly on a long road next to a vertex, because the builder moves such a point onto the vertex by up to the segment tolerance. The check is removed; the rule on the distance to the road stays, and the bound on the returned `snap_distance` is documented and tested.
- The local tie test is applied only at nodes where a candidate cluster is within twice the network-wide bound of the minimum, instead of at every node for every candidate. This restores the speed of inputs with ties everywhere (about 69 s back to about 30 s for 1,028,600 edges and 5,000 clusters, measured in one session).
- The tie-candidate argument in `core.py` and DESIGN.md now matches the local tie rule, with every step explained.
- Boundary validation in `surface.py` is one path for both input types, with named geometry types. A non-polygonal row of a GeoDataFrame boundary now gets the same message as a non-polygonal geometry: "boundary must be polygonal".

Other changes:

- The node winners come from one multi-source shortest-path search plus a bounded search for each cluster that can win a tie, instead of one full search per cluster. Candidate discovery uses a conservative network-wide tolerance, while final tie decisions use local tolerances.
- Network building, validation, edge cutting, surface QA and snap connectors work on whole arrays instead of Python loops.
- A snapped point's geometry is now exactly its network node, and `snap_distance` is measured to that node.
- A cluster boundary within the tolerance of an edge end is placed on that end.
- The CLI reports input errors as one `Error: ...` line with exit code 1, and reads `.geoparquet` files with `read_parquet`.
- Two validation messages have new wording: "edge u/v must contain integer node positions within the node table" and "point network_node must contain integer node positions within the node table".

## 0.1.0 - 2026-09-24

Initial public release.

- Exact lineal network Voronoi assignments using shortest-path distance.
- Exact positive-length additive-epsilon network memberships.
- Deterministic hard tie-breaking and disconnected-network handling.
- Point-to-network snapping through PySAL `spaghetti`, with snapped sites inserted as graph nodes.
- Projected undirected spatial networks with geometric edge length as impedance.
- Optional maximum snap-distance QA guard and approximate 2-D surface rendering.
- GeoPackage output, validation, Python API, and command-line interface.
