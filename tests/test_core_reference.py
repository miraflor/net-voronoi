"""Randomized comparison of ``network_voronoi`` with the slow reference in ``reference.py``.

Two families of inputs are used:

* random road networks, where exact ties are rare and roads share vertices
  only by construction;
* integer lattices with points on nodes and on edge midpoints, and points of
  several clusters at the same place, where exact ties are everywhere.

Cluster IDs such as ``"c10"`` and ``"c9"`` check that ties follow string
order, in which ``"c10"`` comes before ``"c9"``.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

import reference
from netvoronoi import from_geodataframes, network_voronoi

CRS = "EPSG:32651"
LABELS = np.asarray(["c1", "c2", "c9", "c10", "A", "b"], dtype=object)


def _random_case(seed: int):
    rng = np.random.default_rng(seed)
    offset = rng.choice([0.0, 500_000.0])  # small and UTM-sized coordinates
    pool = rng.uniform(0, 1000, (40, 2)) + offset
    lines = []
    for _ in range(25):  # lines through shared vertices: junctions
        k = rng.integers(2, 5)
        lines.append(LineString(pool[rng.choice(len(pool), k, replace=False)]))
    for _ in range(4):  # lines with their own vertices: crossings and separate components
        lines.append(LineString(rng.uniform(0, 1000, (2, 2)) + offset))
    roads = gpd.GeoDataFrame(geometry=lines, crs=CRS)
    n_points = rng.integers(3, 25)
    xy = rng.uniform(0, 1000, (n_points, 2)) + offset
    points = gpd.GeoDataFrame(
        {"cluster": rng.choice(LABELS[: rng.integers(2, len(LABELS) + 1)], n_points)},
        geometry=[Point(p) for p in xy],
        crs=CRS,
    )
    return roads, points


def _lattice_case(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    rows = [LineString([(i, j) for i in range(n)]) for j in range(n)]
    cols = [LineString([(i, j) for j in range(n)]) for i in range(n)]
    roads = gpd.GeoDataFrame(geometry=rows + cols, crs=CRS)
    n_points = int(rng.integers(2, 12))
    xy = rng.integers(0, n, (n_points, 2)).astype(float)
    half = rng.random(n_points) < 0.3  # some points on edge midpoints
    xy[half, 0] = np.minimum(xy[half, 0] + 0.5, n - 1.5)
    labels = rng.choice(LABELS[: int(rng.integers(2, len(LABELS) + 1))], n_points)
    same = rng.random(n_points) < 0.3  # some points of other clusters at an existing place
    for i in np.flatnonzero(same):
        j = int(rng.integers(0, n_points))
        xy[i] = xy[j]
    points = gpd.GeoDataFrame({"cluster": labels}, geometry=[Point(p) for p in xy], crs=CRS)
    return roads, points


def _check_against_reference(roads, points):
    net = from_geodataframes(roads, points, cluster_col="cluster")
    result = network_voronoi(net)
    D, winner, Dc, t = reference.node_partition(net)
    clusters = net.cluster_ids

    # 1. node distances and winners
    finite = np.isfinite(D)
    assert np.array_equal(np.isfinite(result.node_min_distance), finite)
    assert np.allclose(result.node_min_distance[finite], D[finite], rtol=0, atol=1e-9)
    expected = np.asarray([None if w < 0 else clusters[w] for w in winner], dtype=object)
    assert np.array_equal(result.node_cluster, expected)

    # 2. every reachable edge is covered exactly once, from 0 to its length
    edges = net.edges.set_index("edge_id")
    reachable = set(edges.index[finite[edges["u"].to_numpy()]])
    assert set(result.segments.edge_id) == reachable
    assert set(result.unassigned_edges.edge_id) == set(edges.index) - reachable
    for edge_id, pieces in result.segments.groupby("edge_id", sort=False):
        start = pieces["from_dist"].to_numpy()
        stop = pieces["to_dist"].to_numpy()
        assert start[0] == 0.0
        assert np.array_equal(stop[:-1], start[1:])
        assert stop[-1] == edges.loc[edge_id, "length"]
        assert np.all(stop > start)

    # 3. the cluster of each piece is the winner at positions inside it
    for piece in result.segments.itertuples(index=False):
        edge = edges.loc[piece.edge_id]
        u, v, L = int(edge["u"]), int(edge["v"]), float(edge["length"])
        margin = 1e-6 * max(1.0, L)
        for f in (0.02, 0.3, 0.5, 0.7, 0.98):
            s = piece.from_dist + f * (piece.to_dist - piece.from_dist)
            if min(s - piece.from_dist, piece.to_dist - s) <= margin:
                continue  # too close to a piece end: a tie by construction
            assert clusters[reference.winner_on_edge(Dc, t, u, v, L, s)] == piece.cluster_id
    return net, result


@pytest.mark.parametrize("seed", range(40))
def test_random_networks_match_reference(seed):
    _check_against_reference(*_random_case(seed))


@pytest.mark.parametrize("seed", range(60))
def test_tie_heavy_lattices_match_reference(seed):
    _check_against_reference(*_lattice_case(seed))


@pytest.mark.parametrize("first, second", [("A", "B"), ("B", "A")])
def test_tie_at_a_node_without_points_goes_to_smaller_id(first, second):
    # The node at x=5 is equally far from both points. The fast search may
    # label it with either cluster; the result must be the smaller ID.
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (5, 0)]), LineString([(5, 0), (10, 0)])], crs=CRS
    )
    points = gpd.GeoDataFrame(
        {"cluster": [first, second]}, geometry=[Point(0, 0), Point(10, 0)], crs=CRS
    )
    net, result = _check_against_reference(roads, points)
    middle = int(np.flatnonzero(np.isclose(net.nodes.geometry.x, 5.0))[0])
    assert result.node_cluster[middle] == "A"
    assert result.segments.groupby("cluster_id").segment_dist.sum().to_dict() == {"A": 5.0, "B": 5.0}


def test_large_distant_component_does_not_create_local_tie(crs):
    """Numerical equality is local: a 1e9-unit component must not blur a 0.5 mm difference elsewhere."""
    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 0), (0, 0.0005)]),
            LineString([(100, 0), (1_000_000_100, 0)]),
        ],
        crs=crs,
    )
    points = gpd.GeoDataFrame(
        {"cluster": ["B", "B", "A", "Z"]},
        geometry=[Point(0, 0), Point(1, 0), Point(0, 0.0005), Point(100, 0)],
        crs=crs,
    )
    net = from_geodataframes(roads, points, cluster_col="cluster")
    result = network_voronoi(net)

    main = net.edges[net.edges["source_row"] == 0].iloc[0]
    assert result.node_cluster[int(main.u)] == "B"
    assert result.node_cluster[int(main.v)] == "B"
    pieces = result.segments[result.segments["edge_id"] == main.edge_id]
    assert pieces[["cluster_id", "from_dist", "to_dist"]].to_dict("records") == [
        {"cluster_id": "B", "from_dist": 0.0, "to_dist": 1.0}
    ]


@pytest.mark.parametrize("seed", range(10))
def test_random_networks_match_reference_with_extreme_distant_component(seed):
    """The optimized candidate search still matches the direct method across extreme scales."""
    roads, points = _random_case(1000 + seed)
    far_y = 2_000_000_000.0 + seed * 10.0
    roads = gpd.GeoDataFrame(
        geometry=list(roads.geometry) + [LineString([(0.0, far_y), (1_000_000_000.0, far_y)])],
        crs=CRS,
    )
    points = gpd.GeoDataFrame(
        {"cluster": list(points["cluster"]) + ["Z"]},
        geometry=list(points.geometry) + [Point(0.0, far_y)],
        crs=CRS,
    )
    _check_against_reference(roads, points)


@pytest.mark.parametrize("seed", range(20))
def test_near_ties_stay_distinct_next_to_a_huge_component(seed):
    """Distances 5e-4 apart must not count as a tie because another component is 1e9 long.

    Grid points are moved 5e-4 along their row, so many node distances of
    different clusters differ by 5e-4 or 1e-3. The distant component makes
    the network-wide tolerance 1e-3, so a network-wide tie rule would merge
    them; the local rule must not.
    """
    roads, points = _lattice_case(seed)
    rng = np.random.default_rng(seed)
    xy = np.c_[points.geometry.x, points.geometry.y]
    n = int(round(roads.total_bounds[2])) + 1  # grid size: coordinates 0 .. n - 1
    shift = rng.random(len(xy)) < 0.5
    xy[shift, 0] = np.minimum(xy[shift, 0] + 5e-4, n - 1.0)
    far_y = 1_000_000.0
    roads = gpd.GeoDataFrame(
        geometry=list(roads.geometry) + [LineString([(0.0, far_y), (1_000_000_000.0, far_y)])], crs=CRS
    )
    points = gpd.GeoDataFrame(
        {"cluster": list(points["cluster"]) + ["Z"]},
        geometry=[Point(p) for p in xy] + [Point(0.0, far_y)],
        crs=CRS,
    )
    _check_against_reference(roads, points)
