import math

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from netvoronoi.core import _epsilon_interval, network_voronoi


def _length_by_site(gdf):
    return {str(row.site_id): float(row.geometry.length) for row in gdf.itertuples(index=False)}


def _interval(a, b, A, B, length, epsilon):
    """Call the vectorized solver for one edge; return ``(start, stop)`` or None."""
    start, stop, keep = _epsilon_interval(a, b, A, B, length, epsilon)
    return (float(start), float(stop)) if bool(keep) else None


def _reference_intervals(a, b, A, B, length, epsilon):
    """The general piecewise-linear solver used up to v0.3.0, kept as an oracle.

    It does not use the monotone-excess argument. It cuts the edge at both
    kink points, treats both distance functions as linear on each part, and
    solves ``f_s(x) - f_min(x) <= epsilon`` there. Agreement with the closed
    form in ``_epsilon_interval`` is therefore an independent check.
    """
    tol = 1e-9

    def active_line(left, right, x):
        # slope and intercept of min(left + x, right + length - x) at x
        return (1.0, left) if left + x <= right + length - x else (-1.0, right + length)

    cuts = {0.0, float(length)}
    for left, right in ((a, b), (A, B)):
        kink = (right + length - left) / 2.0
        if tol < kink < length - tol:
            cuts.add(float(kink))
    cuts = sorted(cuts)

    valid = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        midpoint = (start + stop) / 2.0
        site_slope, site_intercept = active_line(a, b, midpoint)
        min_slope, min_intercept = active_line(A, B, midpoint)
        slope, intercept = site_slope - min_slope, site_intercept - min_intercept
        if slope == 0.0:
            if intercept <= epsilon + tol:
                valid.append([start, stop])
            continue
        root = (epsilon - intercept) / slope
        if slope > 0:
            stop = min(stop, root)
        else:
            start = max(start, root)
        if stop - start > tol:
            valid.append([start, stop])

    merged = []
    for lo, hi in valid:
        if merged and lo <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged if hi - lo > tol]


def _random_consistent_case(rng):
    """Endpoint distances that real shortest paths could produce.

    Real distances satisfy |a - b| <= L, A <= a, B <= b and |A - B| <= L.
    About one case in five is given an exact tie between the site and the
    nearest distance at an endpoint.
    """
    length = float(rng.uniform(1.0, 100.0))
    a = float(rng.uniform(0.0, 200.0))
    b = float(rng.uniform(max(0.0, a - length), a + length))
    A = a if rng.random() < 0.2 else float(rng.uniform(0.0, a))
    low, high = max(0.0, A - length), min(b, A + length)
    if low > high:
        return None
    B = b if (rng.random() < 0.2 and abs(A - b) <= length) else float(rng.uniform(low, high))
    epsilon = 0.0 if rng.random() < 0.2 else float(rng.uniform(0.0, 50.0))
    return a, b, A, B, length, epsilon


def test_hard_line_partition(line_network):
    result = network_voronoi(line_network, epsilon=0)
    lengths = _length_by_site(result.hard_cells)
    assert np.isclose(lengths["A"], 500.0)
    assert np.isclose(lengths["B"], 500.0)


def test_epsilon_line_overlap(line_network):
    result = network_voronoi(line_network, epsilon=100)
    lengths = _length_by_site(result.epsilon_cells)
    assert np.isclose(lengths["A"], 550.0)
    assert np.isclose(lengths["B"], 550.0)


def test_epsilon_zero_matches_hard_lengths_without_nondegenerate_ties(line_network):
    result = network_voronoi(line_network, epsilon=0)
    assert _length_by_site(result.hard_cells) == _length_by_site(result.epsilon_cells)


def test_interval_solver_known_case():
    # Middle edge: A is 0 away at u and 600 at v; global minima are 0 at
    # both endpoints. With epsilon 100, A remains admissible through x=350.
    interval = _interval(a=0, b=600, A=0, B=0, length=600, epsilon=100)
    assert np.allclose(interval, (0.0, 350.0))


def test_interval_solver_covers_all_four_cases():
    # Admissible at both endpoints: the whole edge.
    assert np.allclose(_interval(a=10, b=10, A=10, B=10, length=50, epsilon=0), (0.0, 50.0))
    # Admissible at u only: see test_interval_solver_known_case.
    # Admissible at v only: t = (b + L - A - epsilon) / 2 = (0 + 50 - 0 - 10) / 2 = 20.
    assert np.allclose(_interval(a=50, b=0, A=0, B=0, length=50, epsilon=10), (20.0, 50.0))
    # Admissible at neither endpoint: nothing on the edge.
    assert _interval(a=30, b=30, A=0, B=0, length=50, epsilon=5) is None
    # Admissible only at the single point u: the site ties the nearest site
    # at u (a = A = 50) but is farther at every x > 0, because the nearest
    # site arrives through v (B = 0). A single point is not emitted.
    assert _interval(a=50, b=20, A=50, B=0, length=50, epsilon=0) is None


def test_closed_form_matches_reference_solver_and_dense_sampling():
    rng = np.random.default_rng(42)
    compared = 0
    while compared < 2000:
        case = _random_consistent_case(rng)
        if case is None:
            continue
        a, b, A, B, length, epsilon = case
        compared += 1
        closed = _interval(a, b, A, B, length, epsilon)
        reference = _reference_intervals(a, b, A, B, length, epsilon)

        # The admissible set is one interval (or empty) on every edge.
        assert len(reference) <= 1
        if closed is None:
            assert reference == []
        else:
            assert len(reference) == 1
            assert np.allclose(closed, reference[0], atol=1e-7)

        # Direct check at sample points away from the interval ends.
        for x in np.linspace(0.001 * length, 0.999 * length, 51):
            site_distance = min(a + x, b + length - x)
            minimum_distance = min(A + x, B + length - x)
            brute = site_distance <= minimum_distance + epsilon + 1e-7
            analytic = closed is not None and closed[0] - 1e-7 <= x <= closed[1] + 1e-7
            assert analytic == brute


def test_hard_tie_break_is_lexicographic_and_input_order_independent(make_network):
    # A and B are tied at node 2 and along the whole tail 2--3.
    network = make_network(
        [(0, 0), (0, 2), (1, 1), (3, 1)],
        [(0, 2), (1, 2), (2, 3)],
        ["B", "A"],
        [0, 1],
    )
    result = network_voronoi(network, epsilon=0, batch_size=1)
    tail = result.hard_segments.loc[result.hard_segments.edge_id == 2]
    epsilon_tail = result.epsilon_segments.loc[result.epsilon_segments.edge_id == 2]
    assert set(tail.site_id.astype(str)) == {"A"}
    assert set(epsilon_tail.site_id.astype(str)) == {"A", "B"}


def test_network_cells_preserve_coincident_graph_edges(make_network):
    # These links have identical map geometry but are distinct graph elements.
    network = make_network(
        [(0, 0), (10, 0), (0, 0), (10, 0)],
        [(0, 1), (2, 3)],
        ["A", "A2"],
        [0, 2],
    )
    result = network_voronoi(network)
    assert np.isclose(result.hard_segments.geometry.length.sum(), 20.0)
    assert np.isclose(result.hard_cells.geometry.length.sum(), 20.0)


def test_nan_epsilon_is_rejected(line_network):
    # A NaN passes an "epsilon < 0" test, because every comparison with NaN
    # is false; v0.3.0 then returned an empty epsilon layer without an error.
    with pytest.raises(ValueError, match="epsilon"):
        network_voronoi(line_network, epsilon=float("nan"))
    with pytest.raises(ValueError, match="epsilon"):
        network_voronoi(line_network, epsilon=-1.0)


def test_infinite_epsilon_makes_every_reachable_site_a_member(line_network):
    result = network_voronoi(line_network, epsilon=math.inf)
    lengths = _length_by_site(result.epsilon_cells)
    assert np.isclose(lengths["A"], 1000.0)
    assert np.isclose(lengths["B"], 1000.0)


def test_segment_geometry_matches_its_distance_bounds(line_network):
    # from_dist / to_dist are distances from endpoint u; on this straight
    # east-west line they are also x offsets from each edge's first vertex.
    result = network_voronoi(line_network, epsilon=100)
    for segments in (result.hard_segments, result.epsilon_segments):
        edge_start_x = line_network.edges.set_index("edge_id").geometry.apply(lambda g: g.coords[0][0])
        for row in segments.itertuples(index=False):
            x0 = edge_start_x[row.edge_id]
            first_x, last_x = row.geometry.coords[0][0], row.geometry.coords[-1][0]
            assert np.isclose(first_x, x0 + row.from_dist)
            assert np.isclose(last_x, x0 + row.to_dist)
            assert np.isclose(row.geometry.length, row.segment_dist)


def test_random_networks_match_direct_continuous_distance_sampling(make_network):
    rng = np.random.default_rng(123)
    for _ in range(20):
        n_nodes = 7
        xy = rng.uniform(0.0, 100.0, size=(n_nodes, 2))
        pairs = set()
        for i in range(1, n_nodes):
            pairs.add(tuple(sorted((i, int(rng.integers(0, i))))))
        while len(pairs) < n_nodes + 1:
            a, b = rng.choice(n_nodes, 2, replace=False)
            pairs.add(tuple(sorted((int(a), int(b)))))

        network = make_network(
            xy,
            sorted(pairs),
            ["C", "A", "B"],
            rng.choice(n_nodes, 3, replace=False),
        )
        epsilon = float(rng.uniform(0.0, 15.0))
        result = network_voronoi(network, epsilon=epsilon, batch_size=2)

        hard = {}
        for row in result.hard_segments.itertuples(index=False):
            hard.setdefault(int(row.edge_id), []).append(
                (row.from_dist, row.to_dist, str(row.site_id))
            )
        eps = {}
        for row in result.epsilon_segments.itertuples(index=False):
            eps.setdefault(int(row.edge_id), []).append(
                (row.from_dist, row.to_dist, str(row.site_id))
            )

        distances = dijkstra(network.adjacency, directed=False, indices=network.site_nodes)
        site_ids = [str(value) for value in network.site_ids]
        for edge in network.edges.itertuples(index=False):
            length = float(edge.length)
            for x in np.linspace(0.05 * length, 0.95 * length, 7):
                ds = np.minimum(
                    distances[:, int(edge.u)] + x,
                    distances[:, int(edge.v)] + length - x,
                )
                minimum = float(ds.min())
                expected_eps = {
                    site_ids[i]
                    for i in range(len(site_ids))
                    if ds[i] <= minimum + epsilon + 1e-7
                }
                actual_eps = {
                    site_id
                    for lo, hi, site_id in eps[int(edge.edge_id)]
                    if lo - 1e-7 <= x <= hi + 1e-7
                }
                assert actual_eps == expected_eps

                tied = [
                    site_ids[i]
                    for i in range(len(site_ids))
                    if np.isclose(ds[i], minimum, rtol=1e-12, atol=1e-9)
                ]
                expected_hard = sorted(tied)[0]
                actual_hard = {
                    site_id
                    for lo, hi, site_id in hard[int(edge.edge_id)]
                    if lo - 1e-7 <= x <= hi + 1e-7
                }
                assert expected_hard in actual_hard


@pytest.mark.parametrize("bad", [0, -1, 1.5, float("nan"), True])
def test_batch_size_must_be_a_positive_integer(line_network, bad):
    with pytest.raises(ValueError, match="batch_size"):
        network_voronoi(line_network, batch_size=bad)
