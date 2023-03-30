import netCDF4 as nc
import numpy as np


def create_sat_netcdf_example_notebook(
    ndvi, ndbi, ndwi, albedo, lst, lat_grid_clip_sentinel, lon_grid_clip_sentinel, city
):
    print("Create output netCDF file")

    # get dimensions (latitude, longitude)
    lat_dim = lat_grid_clip_sentinel
    lon_dim = lon_grid_clip_sentinel

    # get variables
    ndvi = np.transpose(ndvi)
    ndbi = np.transpose(ndbi)
    ndwi = np.transpose(ndwi)
    albedo = np.transpose(albedo)
    lst = np.transpose(lst)

    # create netCDF file
    nc_fn = "output/" + city + "_RS_Data_10x10.nc"
    nc_ds = nc.Dataset(nc_fn, "w", format="NETCDF4")

    # define dimensions
    lat = nc_ds.createDimension("lat", len(lat_dim))
    lon = nc_ds.createDimension("lon", len(lon_dim))

    # define variables
    lat = nc_ds.createVariable("lat", "f4", ("lat",))
    lon = nc_ds.createVariable("lon", "f4", ("lon",))
    NDVI = nc_ds.createVariable(
        "NDVI",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    NDBI = nc_ds.createVariable(
        "NDBI",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    NDWI = nc_ds.createVariable(
        "NDWI",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    Albedo = nc_ds.createVariable(
        "Albedo",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    LST = nc_ds.createVariable(
        "LST",
        "f4",
        (
            "lon",
            "lat",
        ),
    )

    # define units
    NDVI.units = ""
    NDBI.units = ""
    NDWI.units = ""
    Albedo.units = ""
    LST.units = "K"

    # define axis
    lat.axis = "Y"
    lon.axis = "X"

    # define others
    lat.datum = "WGS84"
    lon.datum = "WGS84"

    # define spacing
    lat.spacing = str((lat_dim.max() - lat_dim.min()) / len(lat_dim))
    lon.spacing = str((lon_dim.max() - lon_dim.min()) / len(lon_dim))

    # assign data to variables
    lat[:] = lat_dim
    lon[:] = lon_dim
    NDVI[:] = ndvi
    NDBI[:] = ndbi
    NDWI[:] = ndwi
    Albedo[:] = albedo
    LST[:] = lst

    # close the netCDF
    nc_ds.close()
    print("---------------------------")


def create_sat_netcdf(ndvi, ndbi, ndwi, albedo, height, lst, lat_grid_clip_sentinel, lon_grid_clip_sentinel, city):
    print("Create output netCDF file")

    # get dimensions (latitude, longitude)
    lat_dim = lat_grid_clip_sentinel
    lon_dim = lon_grid_clip_sentinel

    # get variables
    ndvi = np.transpose(ndvi)
    ndbi = np.transpose(ndbi)
    ndwi = np.transpose(ndwi)
    albedo = np.transpose(albedo)
    height = np.transpose(height)
    lst = np.transpose(lst)

    # create netCDF file
    nc_fn = "output/" + city + "_RS_Data_10x10.nc"
    nc_ds = nc.Dataset(nc_fn, "w", format="NETCDF4")

    # define dimensions
    lat = nc_ds.createDimension("lat", len(lat_dim))
    lon = nc_ds.createDimension("lon", len(lon_dim))

    # define variables
    lat = nc_ds.createVariable("lat", "f4", ("lat",))
    lon = nc_ds.createVariable("lon", "f4", ("lon",))
    NDVI = nc_ds.createVariable(
        "NDVI",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    NDBI = nc_ds.createVariable(
        "NDBI",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    NDWI = nc_ds.createVariable(
        "NDWI",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    Albedo = nc_ds.createVariable(
        "Albedo",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    Height = nc_ds.createVariable(
        "Height",
        "f4",
        (
            "lon",
            "lat",
        ),
    )
    LST = nc_ds.createVariable(
        "LST",
        "f4",
        (
            "lon",
            "lat",
        ),
    )

    # define units
    NDVI.units = ""
    NDBI.units = ""
    NDWI.units = ""
    Albedo.units = ""
    Height.units = "m"
    LST.units = "K"

    # define axis
    lat.axis = "Y"
    lon.axis = "X"

    # define others
    lat.datum = "WGS84"
    lon.datum = "WGS84"

    # define spacing
    lat.spacing = str((lat_dim.max() - lat_dim.min()) / len(lat_dim))
    lon.spacing = str((lon_dim.max() - lon_dim.min()) / len(lon_dim))

    # assign data to variables
    lat[:] = lat_dim
    lon[:] = lon_dim
    NDVI[:] = ndvi
    NDBI[:] = ndbi
    NDWI[:] = ndwi
    Albedo[:] = albedo
    Height[:] = height
    LST[:] = lst

    # close the netCDF
    nc_ds.close()
    print("---------------------------")
