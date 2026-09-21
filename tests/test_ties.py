import geopandas as gpd
import numpy as np
from scipy.sparse import coo_matrix
from shapely.geometry import LineString, Point

from netvoronoi.core import network_voronoi
from netvoronoi.model import SpatialNetwork


def test_colocated_sites_tie_in_epsilon_but_hard_is_canonical():
    xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    edges = gpd.GeoDataFrame(
        [{"edge_id": 0, "u": 0, "v": 1, "length": 100.0, "geometry": LineString(xy)}],
        geometry="geometry",
        crs="EPSG:32651",
    )
    adjacency = coo_matrix(([100.0, 100.0], ([0, 1], [1, 0])), shape=(2, 2)).tocsr()
    sites = gpd.GeoDataFrame(
        {"site_id": ["A", "B"], "network_node": [0, 0]},
        geometry=[Point(0, 0), Point(0, 0)],
        crs=edges.crs,
    )
    net = SpatialNetwork(
        xy, edges, adjacency, np.array(["A", "B"], object), np.array([0, 0]), sites, edges.crs
    )
    result = network_voronoi(net, epsilon=0)
    assert set(result.hard_cells.site_id.astype(str)) == {"A"}
    assert set(result.epsilon_cells.site_id.astype(str)) == {"A", "B"}
    lengths = {
        str(r.site_id): r.geometry.length
        for r in result.epsilon_cells.itertuples(index=False)
    }
    assert np.isclose(lengths["A"], 100)
    assert np.isclose(lengths["B"], 100)
