import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box


@pytest.fixture
def crs():
    return "EPSG:32651"


@pytest.fixture
def line_roads(crs):
    return gpd.GeoDataFrame(
        {"name": ["main"]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs=crs,
    )


@pytest.fixture
def end_points(crs):
    return gpd.GeoDataFrame(
        {"pid": ["a", "b"], "cluster": ["A", "B"]},
        geometry=[Point(0, 0), Point(10, 0)],
        crs=crs,
    )


@pytest.fixture
def boundary(crs):
    return gpd.GeoDataFrame(geometry=[box(0, -2, 10, 2)], crs=crs)
