"""Slow reference implementations for the tests.

These functions follow the definitions in DESIGN.md literally, without the
shortcuts used in ``netvoronoi.core``: one full shortest-path search per
cluster, and the winner taken over all clusters. The tests compare the fast
code with them on random and tie-heavy networks.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra


def cluster_distances(network) -> np.ndarray:
    """``D_c(v)`` for every cluster ``c`` (rows, in sorted ID order) and node ``v`` (columns)."""
    labels = network.points_snapped["cluster_id"].astype(str).to_numpy()
    nodes = network.points_snapped["network_node"].to_numpy(np.int64)
    rows = [
        dijkstra(network.adjacency, directed=False, indices=np.unique(nodes[labels == c]), min_only=True)
        for c in network.cluster_ids
    ]
    return np.vstack(rows)


def tie_tolerance(*values) -> np.ndarray:
    """The local numerical-equality tolerance used by the package."""
    scale = np.asarray(1.0)
    for value in values:
        a = np.abs(np.asarray(value, dtype=float))
        scale = np.maximum(scale, np.where(np.isfinite(a), a, 0.0))
    return np.maximum(1e-9, 1e-12 * scale)


def node_partition(network):
    """``(D, winner, Dc, t)``: nearest distance, winning cluster rank (-1 if unreachable), all ``D_c``, tolerance."""
    Dc = cluster_distances(network)
    D = Dc.min(axis=0)
    t = tie_tolerance(Dc, D)  # one tolerance per cluster/node comparison
    attains = Dc <= D + t
    winner = np.where(np.isfinite(D), np.argmax(attains, axis=0), -1)  # argmax: first (smallest) rank
    return D, winner, Dc, t


def winner_on_edge(Dc: np.ndarray, t, u: int, v: int, L: float, s: float) -> int:
    """Winning cluster rank at distance ``s`` from ``u`` on the edge ``(u, v)`` of length ``L``.

    The edge has no point in its interior, so every cluster reaches the
    position through one of the two end nodes.
    """
    at_s = np.minimum(Dc[:, u] + s, Dc[:, v] + (L - s))
    minimum = at_s.min()
    local_t = tie_tolerance(at_s, minimum)
    return int(np.argmax(at_s <= minimum + local_t))
