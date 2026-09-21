# Changelog

## 0.4.3

Small release-hardening revision. The network Voronoi mathematics, site
placement, surface algorithm, and 1.7.6 fast-path results are unchanged.

- guard the private ``spaghetti`` component-skipping optimization by the exact
  verified version (1.7.6). Other 1.x versions automatically use the ordinary
  ``w_components=True`` path until they are explicitly verified;
- validate the network CRS before geometric edge lengths, so a geographic CRS
  is rejected without first triggering GeoPandas geographic-length warnings;
- add regression tests for both behaviours.

## 0.4.2

Correctness and performance follow-up. Results are unchanged: the same
networks, Voronoi segments, and surfaces as 0.4.1.

- cheap checks run before expensive work again. In 0.4.1, `surface_voronoi`
  built the whole grid before `network_voronoi` checked `epsilon`, `batch_size`
  and the network; now the order is arguments, grid size, network result,
  grid. `network_voronoi` checks `epsilon` and `batch_size` before it
  validates the network;
- the CLI checks the surface grid size (`--max-cells`) before it builds the
  road network, which is the slowest step for a large road file. The check
  is the new public function `surface.check_grid_size`, and the grid shape is
  computed in one place (`_grid_shape`) for both the check and the grid;
- the too-large-grid message says "use a larger resolution value (larger
  cells)"; "increase resolution" could be read in the opposite direction;
- `DEFAULT_MAX_CELLS` names the default limit and records what it costs: about
  60 s and 0.9 GB for 950,625 candidate squares on a 4 GB test machine;
- the adapter no longer asks `spaghetti` for its connected-component
  labelling, which the package never used. On a 44,700-segment road network,
  `from_geodataframes` takes 4.4 s instead of 174.5 s, with an identical
  result. A live test builds the same network both ways;
- tests: 74 (four new ones, for the argument order in `surface_voronoi` and in
  the CLI, for `check_grid_size`, and for the adapter change above);

## 0.4.1

Small contract-and-safety revision. The network Voronoi mathematics,
site-placement rule, and vectorized algorithms are unchanged.

- require CRS metadata on the internal ``edges`` and ``sites_snapped`` layers;
  a declared ``SpatialNetwork.crs`` no longer causes unlabelled coordinates to
  be silently treated as if they used that CRS;
- validate direct-Python ``batch_size`` values as positive integers, with a
  clear error instead of a later ``range``/NumPy failure;
- add ``max_cells`` to ``surface_voronoi`` (default 1,000,000) and
  ``--max-cells`` to the CLI, checking the candidate bounding-box grid before
  the exact network computation or any large surface allocation;
- make site QA output tolerant of custom ``SpatialNetwork`` objects that carry
  ``source_x``/``source_y`` but omit the optional ``snap_distance`` column;
- correct stale test-count documentation and expand the suite to 70 tests
  (the 5 live ``spaghetti`` tests are skipped when that dependency is absent;
  with spaghetti 1.7.6 installed, all 70 pass).

## 0.4.0

Correctness, speed and readability revision. Every fix below was reproduced
first with a small runnable script against the installed `spaghetti` 1.7.6.

Corrections:

- the `spaghetti` adapter no longer compares `spaghetti`'s `arc_lengths` with
  its own geometry: `spaghetti` computes those lengths from one rounded and one
  unrounded vertex, so ordinary projected road data (full-precision UTM
  coordinates) was rejected with "spaghetti arc length disagrees with its
  geometry". Arc geometry and arc length now both come from the rounded vertex
  coordinates;
- a repeated vertex in a road no longer stops the build; `spaghetti` turns it
  into an arc from a vertex to itself, which is now skipped instead of raising
  "spaghetti produced an invalid arc length";
- `snap_tolerance` no longer removes road. Up to 0.3.0 a road piece shorter
  than `snap_tolerance` was dropped, which silently shortened the network and
  left parts of it unassigned;
- site merging no longer creates two graph nodes closer than `snap_tolerance`,
  which could leave no edge between them and disconnect the network. Sites are
  now placed by the rule documented in `_place_sites_on_arc`;
- `SpatialNetwork.validate()` now recognizes a geographic CRS given as a string
  such as `"EPSG:4326"`; up to 0.3.0 only objects with an `is_projected`
  attribute were checked, so degree-based networks passed;
- NaN is rejected wherever a number is expected: `epsilon`, `resolution`,
  `snap_tolerance`, `max_snap_distance`, and the matching CLI options. NaN
  passed every `x < 0` style check before, and produced empty or wrong output
  without an error;
- the CLI checks the output path before it reads inputs or computes anything,
  instead of doing the whole computation and then refusing to overwrite;
- NaN or infinite input coordinates are rejected with a clear message instead
  of failing inside `spaghetti`;
- the `surface_unassigned` layer no longer carries a `cell_id` column that held
  the number of the first cell only.

Speed (150 x 150 grid, 22,500 nodes, 44,700 edges, 200 sites, `epsilon = 150`):

- `network_voronoi`: 49.2 s to 1.2 s;
- `SpatialNetwork.validate()`: 3.5 s to 0.1 s;
- `surface_voronoi` over 22,201 cells: 28.8 s to 3.6 s (of which 1.2 s is the
  network result it computes first).

The gains come from replacing per-row pandas access and per-edge Python loops
with array operations, and from the closed-form interval solver.

Readability and structure:

- `core.py` states the per-edge mathematics in one function, `_epsilon_interval`,
  with the monotone-excess argument and the four endpoint cases written out;
  the kink-cutting solver and the interval merging step are gone;
- `spaghetti_backend.py` states in its module docstring what `spaghetti` does
  and the three consequences for the adapter, and keeps the site placement rule
  in one small function;
- `surface.py` is a four-step pipeline over arrays with no loop over cells;
- edge numbering follows sorted `spaghetti` arc order. (Correction added in
  0.4.2: this did not change any edge number. `spaghetti` 1.7.6 already sorts
  its arcs when it builds the network, so 0.3.0 used the same order; the
  explicit sort only removes the dependence on that `spaghetti` detail.);
- validation errors name the first offending `edge_id` or `site_id`;
- the test suite grows from 25 to 57 tests, including live `spaghetti`
  regression tests for each defect above, and comparisons against slower
  reference implementations.

## 0.3.0

Readability and input-contract revision.

- refactored the three largest routines into a small number of named domain helpers;
- kept the piecewise-linear epsilon solver self-contained and expanded its mathematical comments;
- reorganized tests by topic instead of keeping one large regression test module;
- fixed `surface_grid_debug.network_dist`, which was actually an edge offset; it is now `edge_offset`;
- added the true `nearest_site_dist` surface QA field;
- allow site layers in geographic or other valid CRSs and reproject them to the projected road CRS;
- validate finite snap distances, snap offsets, and spaghetti arc lengths more explicitly;
- made the spaghetti `unique_arcs=True` assumption explicit;
- constrained the dependency to `spaghetti>=1.7.6,<2` because the adapter targets the documented 1.x API;
- clarified which coincident edges can be preserved by the core model versus the default spaghetti adapter;
- expanded the suite to 25 tests.

## 0.2.0

Fine-tooth pre-publication revision.

- made all distance fields CRS-unit neutral rather than assuming meters;
- added stronger graph/geometry/model validation;
- made hard Voronoi tie-breaking deterministic and independent of SciPy's
  unspecified shortest-path tie resolution;
- fixed cross-component epsilon candidate pruning when distances are infinite;
- exposed unassigned network components in output;
- preserved coincident graph-edge multiplicity in collected network cells instead of unary-unioning it away;
- added `max_snap_distance` as an optional snapping QA guard;
- exploded MultiLineString road input and rejected unsupported geometry types;
- stopped silently overwriting existing GeoPackages; CLI now requires `--force`;
- added GeoPackage layer selectors to the CLI;
- made surface nearest-edge ties deterministic and exposed `anchor_ties`;
- improved numerical tolerances and interval-cut handling;
- reduced unnecessary `spaghetti` graph extraction work;
- expanded tests from 9 to 22, including randomized direct-distance checks and
  adapter conversion tests.

## 0.1.0

Initial prototype.
