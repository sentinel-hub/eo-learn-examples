import pandas as pd
import numpy as np
import glob
import xmltodict
import utm
import rasterio
from scipy.interpolate import griddata

# get the projection of the landsat 8 image
def landsat8_proj(path_landsat):
    
    print("Landsat 8 grid projection")
    # get path to meta data
    path_grid_meta = glob.glob(path_landsat + "/*ANG.txt")[0]
    grid_meta = pd.read_table(path_grid_meta)

    # UTM zone
    idx_utm_zone = np.where(["UTM_ZONE" in str(grid_meta.loc[i]) for i in np.arange(0, len(grid_meta), 1)])[0]
    utm_zone_proj = int(grid_meta.loc[int(idx_utm_zone)][0].split("UTM_ZONE = ")[1])

    # define pixel size (30 m Landsat Bands 10, 11)
    # we use 10 m since we want the same resolution as Sentinel 2
    pixel_size = 30

    # upper left corner
    idx_coo_upper_left = np.where(["UL_CORNER" in str(grid_meta.loc[i]) for i in np.arange(0, len(grid_meta), 1)])[0]
    coo_upper_left = [
        float(grid_meta.loc[int(idx_coo_upper_left)][0].split("UL_CORNER = ")[1].split("( ")[1].split(",")[0]),
        float(
            grid_meta.loc[int(idx_coo_upper_left)][0].split("UL_CORNER = ")[1].split("( ")[1].split(",")[1].split(")")[
                0])]

    # lower right corner
    idx_coo_lower_right = np.where(["LR_CORNER" in str(grid_meta.loc[i]) for i in np.arange(0, len(grid_meta), 1)])[0]
    coo_lower_right = [
        float(grid_meta.loc[int(idx_coo_lower_right)][0].split("LR_CORNER = ")[1].split("( ")[1].split(",")[0]),
        float(
            grid_meta.loc[int(idx_coo_lower_right)][0].split("LR_CORNER = ")[1].split("( ")[1].split(",")[1].split(")")[
                0])]

    # create UTM grids (x- and y-coordinates)
    x_utm = np.arange(coo_upper_left[0], coo_lower_right[0] + 1, pixel_size)
    y_utm = np.arange(coo_upper_left[1], coo_lower_right[1] - 1, -pixel_size)
    x_grid = np.tile(x_utm, (len(y_utm), 1))
    y_grid = np.tile(np.vstack(y_utm), (1, len(x_utm)))

    # for simplification reasons create m x n grid with m = n
    if x_grid.shape[0] != x_grid.shape[1]:
        if x_grid.shape[0] < x_grid.shape[1]:
            x_grid = x_grid[:,0:x_grid.shape[0]]
            y_grid = y_grid[:,0:y_grid.shape[0]]
        else:
            x_grid = x_grid[0:x_grid.shape[1],:]
            y_grid = y_grid[0:y_grid.shape[1],:]

    print("---------------------------")

    return x_grid, y_grid, utm_zone_proj


def landsat8_bands(tif_path, y_grid_big, x_grid_big, utm_zone_proj, bbox_wgs84):
    print("Landsat 8 bands extraction")
    print("Landsat 8 bands clipping to smaller size")

    ll_um = utm.from_latlon(bbox_wgs84[1]-0.05, bbox_wgs84[0]-0.05, force_zone_number=int(utm_zone_proj))
    ur_um = utm.from_latlon(bbox_wgs84[3]+0.05, bbox_wgs84[2]+0.05, force_zone_number=int(utm_zone_proj))

    idx_col = np.where((x_grid_big[0,:] >= ll_um[0]) & (x_grid_big[0,:] <= ur_um[0]))[0]
    idx_row = np.where((y_grid_big[:,0] >= ll_um[1]) & (y_grid_big[:,0] <= ur_um[1]))[0]

    bands_name = ["B10", "B11"]
    bands_array = []

    for j in bands_name:
        bdata = rasterio.open(tif_path[np.where([(j + ".TIF") in i for i in tif_path])[0][0]]).read()[0, :, :]

        # disaggregate gridded data to 10 x 10 m (same resolution as Sentinel 2)
        # bdata = np.repeat(bdata, 3, axis=1)
        # bdata = np.repeat(bdata, 3, axis=0)

        # check that bdata is a m x n grid with, fulfilling the condition m = n
        if (bdata.shape[0] != bdata.shape[1]):
            if bdata.shape[0] < bdata.shape[1]:
                bdata = bdata[:, 0:bdata.shape[0]]
            else:
                bdata = bdata[0:bdata.shape[1], :]

        bdata_clip = bdata[:, idx_col]
        bdata_clip = bdata_clip[idx_row, :]
        bands_array.append(bdata_clip)

    return bands_array

def sample_lst_sentinel2_grid(y_grid_wgs84, x_grid_wgs84, lst, gridLON, gridLAT):
    print("Landsat 8 sample by regular WGS84 grid")
    y_grid_wgs84_list = y_grid_wgs84.flatten()
    x_grid_wgs84_list = x_grid_wgs84.flatten()
    points = np.array(pd.DataFrame({"lon":x_grid_wgs84_list, "lat":y_grid_wgs84_list}))

    values = lst.flatten()
    lst_new = griddata(points, values, (gridLON, gridLAT), method = "nearest")
    
    print("---------------------------")
    return lst_new

# get the thermal constants of the landsat 8 image
def landsat8_thermal_constants(path_landsat):
    # sensor and sun position related meta data
    path_sensor_meta = glob.glob(path_landsat + "/*MTL.xml")[0]

    # open meta data (xml)
    xml_data = open(path_sensor_meta, 'r').read()  # Read data
    xmlDict = xmltodict.parse(xml_data)

    # sensor and sun position related constants
    rad_mb_10 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['RADIANCE_MULT_BAND_10'])
    rad_mb_11 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['RADIANCE_MULT_BAND_11'])
    rad_ab_10 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['RADIANCE_ADD_BAND_10'])
    rad_ab_11 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['RADIANCE_ADD_BAND_11'])
    k1_10 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_10'])
    k1_11 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_11'])
    k2_10 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_10'])
    k2_11 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_11'])

    return rad_mb_10, rad_mb_11, rad_ab_10, rad_ab_11, k1_10, k1_11, k2_10, k2_11


def landsat8_predictors(bands_array, path_landsat):
    
    print("Landsat 8 predictors calculation:")

    # sensor and sun position related meta data
    path_sensor_meta = glob.glob(path_landsat + "/*MTL.xml")[0]

    # open meta data (xml)
    xml_data = open(path_sensor_meta, 'r').read()  # Read data
    xmlDict = xmltodict.parse(xml_data)

    # sensor and sun position related constants
    rad_mb_10 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['RADIANCE_MULT_BAND_10'])
    rad_mb_11 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['RADIANCE_MULT_BAND_11'])
    rad_ab_10 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['RADIANCE_ADD_BAND_10'])
    rad_ab_11 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['RADIANCE_ADD_BAND_11'])
    k1_10 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_10'])
    k1_11 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_11'])
    k2_10 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_10'])
    k2_11 = float(xmlDict['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_11'])


    band_10 = bands_array[0]
    band_11 = bands_array[1]

    # top-of-atmosphere (TOA) radiance
    toa_band_10 = rad_mb_10 * band_10 + rad_ab_10
    toa_band_11 = rad_mb_11 * band_11 + rad_ab_11

    # brightness temperature (LST)
    lst_band_10 = k2_10 / (np.log((k1_10 / toa_band_10) + 1))
    lst_band_11 = k2_11 / (np.log((k1_11 / toa_band_11) + 1))

    # mean of land surface temperature of Band 10 and 11
    lst_mean = (lst_band_10 + lst_band_11) / 2

    print("LST - DONE")
    print("---------------------------")

    return lst_mean