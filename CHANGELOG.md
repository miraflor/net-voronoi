# Changelog

## 0.1.0

Initial public release.

- exact lineal network Voronoi assignments using shortest-path distance;
- exact positive-length additive-epsilon network memberships;
- deterministic hard tie-breaking and disconnected-network handling;
- point-to-network snapping through PySAL `spaghetti`, with snapped sites inserted as graph nodes;
- support for projected undirected spatial networks with geometric edge length as impedance;
- optional maximum snap-distance QA guard;
- optional approximate 2-D surface rendering with an explicit nearest-network attachment model;
- configurable surface grid-size guard;
- GeoPackage output, explicit layer selection, and overwrite protection;
- validation for CRS, geometry, finite parameters, coordinates, and network invariants;
- vectorized performance improvements and randomized/reference-based regression tests;
- Python API and command-line interface.

This release consolidates the pre-publication development history into the
initial public version, `v0.1.0`.
