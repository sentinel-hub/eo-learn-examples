import sentinelhub
from sentinelhub import CRS
import math
from shapely.geometry import *

import rasterio
import warnings

import numpy as np



def MetaInfos(saving_path, N) :
    with rasterio.open(saving_path) as src0 :
        meta = src0.meta
        meta['nodata'] = np.nan
        meta['dtype'] = 'float32'

    meta.update(count=N)
    meta.update(nodata=np.nan)

    return meta


def WriteTiff(array, saving_path, meta, dim=1) :
    with rasterio.open(saving_path, 'w', **meta) as dst :
        if dim>1 :
            for id in range(dim) :
                dst.write_band(id+1, array[:, :, id].astype(np.float32))
        else :
            dst.write_band(1, array.astype(np.float32))


def check_crs(reference_file_) :
    reference_file = reference_file_.copy()
    reference_file_crs = reference_file.crs
    if reference_file_crs is None :
        reference_file.set_crs(epsg=4326, inplace=True)
        warnings.warn('Your input GeoDataFrame should have a CRS! By default, it will set to WGS84')
    elif str(reference_file_crs).split(':')[-1]!='4326' :
        reference_file.to_crs(epsg=4326, inplace=True)

    xmin, ymin, xmax, ymax = reference_file.geometry.iloc[0].bounds
    utm_crs = str(CRS.get_utm_from_wgs84(xmin, ymin))
    reference_file.to_crs(utm_crs, inplace=True)

    return reference_file


def get_bounding_box(shapefile):
    ''' Get the bounding box of a given polygon for sentinelhub request
    Inputs :
        - polygon (geometry object) : polygon of a field
    '''
    shapefile = check_crs(shapefile)
    xmin, ymin, xmax, ymax = shapefile.geometry.total_bounds
    return sentinelhub.BBox(
        bbox=[(xmin, ymin), (xmax, ymax)], crs=str(shapefile.crs)
    )


def create_polygon_bbox(longitude, latitude, distance) :


    r_earth = 6378
    dx = distance/1000
    dy = distance/1000

    ic_x = (dx/r_earth)*(180/math.pi)/math.cos(latitude*math.pi/180)
    ic_y = (dy/r_earth)*(180/math.pi)


    return box(longitude-ic_x,
               latitude-ic_y,
               longitude+ic_x,
               latitude+ic_y)





