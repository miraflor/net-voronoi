import geopandas as gpd
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from shapely.geometry import LineString, Point

from netvoronoi.model import SpatialNetwork


@pytest.fixture
def make_network():
    """Factory for small geometric networks used throughout the test suite."""

    def factory(xy, edge_pairs, site_ids, site_nodes):
        xy_array = np.asarray(xy, dtype=float)
        rows = []
        rr, cc, ww = [], [], []
        for edge_id, (u, v) in enumerate(edge_pairs):
            geometry = LineString([xy_array[u], xy_array[v]])
            length = float(geometry.length)
            rows.append(
                {"edge_id": edge_id, "u": u, "v": v, "length": length, "geometry": geometry}
            )
            rr += [u, v]
            cc += [v, u]
            ww += [length, length]

        edges = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32651")
        adjacency = coo_matrix((ww, (rr, cc)), shape=(len(xy_array), len(xy_array))).tocsr()
        sites = gpd.GeoDataFrame(
            {"site_id": site_ids, "network_node": site_nodes},
            geometry=[Point(xy_array[node]) for node in site_nodes],
            crs=edges.crs,
        )
        return SpatialNetwork(
            vertices_xy=xy_array,
            edges=edges,
            adjacency=adjacency,
            site_ids=np.asarray(site_ids, dtype=object),
            site_nodes=np.asarray(site_nodes, dtype=int),
            sites_snapped=sites,
            crs=edges.crs,
            distance_unit="metre",
        )

    return factory


@pytest.fixture
def line_network(make_network):
    # 0 ----200---- A --------600-------- B ----200---- 1000
    return make_network(
        [(0.0, 0.0), (200.0, 0.0), (800.0, 0.0), (1000.0, 0.0)],
        [(0, 1), (1, 2), (2, 3)],
        ["A", "B"],
        [1, 2],
    )
