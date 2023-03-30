import glob

import numpy as np
import pandas as pd
import rasterio
import utm
import xmltodict
from scipy.interpolate import griddata


def sentinel2_proj(path_sentinel):
    print("Sentinel 2 grid projection")
    # get path to metadata
    # path: "'path_sentinel'/GRANULE/.../MTD_TL.xml"
    metadata = glob.glob(path_sentinel + "/GRANULE/*")[0] + "/MTD_TL.xml"

    # read metadata
    # xml file
    with open(metadata, "r") as file:
        xml_data = file.read()
    xmlDict = xmltodict.parse(xml_data)

    # Projection: UTM information of satellite image
    # UTM zone
    utm_zone_proj = (
        xmlDict["n1:Level-1C_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]["HORIZONTAL_CS_NAME"]
        .split(" ")[4]
        .split("N")[0]
    )
    # UTM hemisphere (north/south)
    utm_hemi_proj = (
        xmlDict["n1:Level-1C_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]["HORIZONTAL_CS_NAME"]
        .split(" ")[4]
        .split(utm_zone_proj)[1]
    )

    # get coordinates of upper left corner of satellite image
    coo_upper_left = [
        int(xmlDict["n1:Level-1C_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]["Geoposition"][0]["ULX"]),
        int(xmlDict["n1:Level-1C_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]["Geoposition"][0]["ULY"]),
    ]

    # dimensions (number of pixels) in x and y direction
    # 10 m resolution bands
    ydim = int(xmlDict["n1:Level-1C_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]["Size"][0]["NROWS"])
    xdim = int(xmlDict["n1:Level-1C_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]["Size"][0]["NCOLS"])

    # create UTM grids (x- and y-coordinates)
    x_utm = np.array([coo_upper_left[0] + i * 10 for i in np.arange(0, xdim, 1)])
    y_utm = np.array([coo_upper_left[1] - i * 10 for i in np.arange(0, ydim, 1)])
    x_grid = np.tile(x_utm, (ydim, 1))
    y_grid = np.tile(np.vstack(y_utm), (1, xdim))

    # for simplification reasons create m x n grid with m = n
    if x_grid.shape[0] != x_grid.shape[1]:
        if x_grid.shape[0] < x_grid.shape[1]:
            x_grid = x_grid[:, 0 : x_grid.shape[0]]
            y_grid = y_grid[:, 0 : y_grid.shape[0]]
        else:
            x_grid = x_grid[0 : x_grid.shape[1], :]
            y_grid = y_grid[0 : y_grid.shape[1], :]

    print("---------------------------")

    return x_grid, y_grid, utm_zone_proj, utm_hemi_proj


# get bounding box as grid in WGS84
# projection we want at the end
def bbox_grid_wgs84(lat_grid, lon_grid, bbox_wgs84):
    print("Sentinel 2 define regular WGS84 grid")
    # get latitude and longitude in our bounding box
    lat_grid_bbox = lat_grid[(lat_grid >= bbox_wgs84[1]) & (lat_grid <= bbox_wgs84[3])]
    lon_grid_bbox = lon_grid[(lon_grid >= bbox_wgs84[0]) & (lon_grid <= bbox_wgs84[2])]

    # convert to grid
    gridLON, gridLAT = np.meshgrid(lon_grid_bbox, lat_grid_bbox)

    print("---------------------------")
    return gridLAT, gridLON


# make x_grid and y_grid smaller
def clip_satellite_scene(x_grid_big, y_grid_big, bbox_wgs84, utm_zone_proj, utm_hemi_proj):
    print("Clipping big Satellite scene to smaller size")
    ll_um = utm.from_latlon(bbox_wgs84[1] - 0.05, bbox_wgs84[0] - 0.05, force_zone_number=int(utm_zone_proj))
    ur_um = utm.from_latlon(bbox_wgs84[3] + 0.05, bbox_wgs84[2] + 0.05, force_zone_number=int(utm_zone_proj))

    idx_col = np.where((x_grid_big[0, :] >= ll_um[0]) & (x_grid_big[0, :] <= ur_um[0]))[0]
    idx_row = np.where((y_grid_big[:, 0] >= ll_um[1]) & (y_grid_big[:, 0] <= ur_um[1]))[0]

    x_grid = x_grid_big[:, idx_col]
    x_grid = x_grid[idx_row, :]

    y_grid = y_grid_big[:, idx_col]
    y_grid = y_grid[idx_row, :]

    # convert UTM to WGS84
    y_grid_wgs84, x_grid_wgs84 = utm.to_latlon(x_grid, y_grid, int(utm_zone_proj), utm_hemi_proj)

    print("---------------------------")
    return x_grid_wgs84, y_grid_wgs84


def sentinel2_bands(jp2_path, y_grid_big, x_grid_big, utm_zone_proj, bbox_wgs84):
    print("Sentinel 2 bands extraction")
    print("Sentinel 2 bands clipping to smaller size")

    ll_um = utm.from_latlon(bbox_wgs84[1] - 0.05, bbox_wgs84[0] - 0.05, force_zone_number=int(utm_zone_proj))
    ur_um = utm.from_latlon(bbox_wgs84[3] + 0.05, bbox_wgs84[2] + 0.05, force_zone_number=int(utm_zone_proj))

    idx_col = np.where((x_grid_big[0, :] >= ll_um[0]) & (x_grid_big[0, :] <= ur_um[0]))[0]
    idx_row = np.where((y_grid_big[:, 0] >= ll_um[1]) & (y_grid_big[:, 0] <= ur_um[1]))[0]

    bands_name = ["B02", "B03", "B04", "B08", "B11", "B12"]
    bands_array = []
    for j in bands_name:
        bdata = (
            rasterio.open(jp2_path[np.where([(j + ".jp2") in i for i in jp2_path])[0][0]]).read()[0, :, :].astype(float)
        )

        # check that bdata is a m x n grid with, fulfilling the condition m = n
        if bdata.shape[0] != bdata.shape[1]:
            if bdata.shape[0] < bdata.shape[1]:
                bdata = bdata[:, 0 : bdata.shape[0]]
            else:
                bdata = bdata[0 : bdata.shape[1], :]

        if bdata.shape == (y_grid_big.shape[0], x_grid_big.shape[1]):
            bdata_clip = bdata[:, idx_col]
            bdata_clip = bdata_clip[idx_row, :]
            bands_array.append(bdata_clip)
        else:
            # disaggregate 20 m bands to 10 m resolution
            bdata = np.repeat(bdata, 2, axis=1)
            bdata = np.repeat(bdata, 2, axis=0)
            bdata_clip = bdata[:, idx_col]
            bdata_clip = bdata_clip[idx_row, :]
            bands_array.append(bdata_clip)

    print("---------------------------")
    return bands_array


def sample_with_regular_wgs84_grid(y_grid_wgs84, x_grid_wgs84, bands_array, gridLON, gridLAT):
    print("Sentinel 2 sample by regular WGS84 grid")
    y_grid_wgs84_list = y_grid_wgs84.flatten()
    x_grid_wgs84_list = x_grid_wgs84.flatten()
    points = np.array(pd.DataFrame({"lon": x_grid_wgs84_list, "lat": y_grid_wgs84_list}))

    bands_array_wgs84 = []
    for i in np.arange(0, np.array(bands_array).shape[0], 1):
        values = bands_array[i].flatten()
        band_wgs84 = griddata(points, values, (gridLON, gridLAT), method="nearest")
        bands_array_wgs84.append(band_wgs84)

    lon_grid_clip_sentinel = gridLON[0, :]
    lat_grid_clip_sentinel = gridLAT[:, 0]

    print("---------------------------")
    return bands_array_wgs84, lon_grid_clip_sentinel, lat_grid_clip_sentinel


def sentinel2_predictors(bands_array):
    print("Sentinel 2 predictors calculation:")

    band_2 = bands_array[0]
    band_3 = bands_array[1]
    band_4 = bands_array[2]
    band_8 = bands_array[3]
    band_11 = bands_array[4]
    band_12 = bands_array[5]

    # normalization of bands
    band_2_norm = (band_2 - np.nanmin(band_2)) / (np.nanmax(band_2) - np.nanmin(band_2))
    band_4_norm = (band_4 - np.nanmin(band_4)) / (np.nanmax(band_4) - np.nanmin(band_4))
    band_8_norm = (band_8 - np.nanmin(band_8)) / (np.nanmax(band_8) - np.nanmin(band_8))
    band_11_norm = (band_11 - np.nanmin(band_11)) / (np.nanmax(band_11) - np.nanmin(band_11))
    band_12_norm = (band_12 - np.nanmin(band_12)) / (np.nanmax(band_12) - np.nanmin(band_12))

    # INDICES
    # source: esri
    # https://pro.arcgis.com/de/pro-app/latest/arcpy/spatial-analyst/ndbi.htm
    ndvi = (band_8 - band_4) / (band_8 + band_4)
    print("NDVI - DONE")
    ndbi = (band_11 - band_8) / (band_11 + band_8)
    print("NDBI - DONE")
    ndwi = (band_3 - band_8) / (band_3 + band_8)
    print("NDWI - DONE")
    # source: Naegeli et al. (2017)
    albedo = (
        0.356 * band_2_norm
        + 0.130 * band_4_norm
        + 0.373 * band_8_norm
        + 0.085 * band_11_norm
        + 0.072 * band_12_norm
        - 0.0018
    )
    print("Albedo - DONE")
    print("---------------------------")

    return ndvi, ndbi, ndwi, albedo
