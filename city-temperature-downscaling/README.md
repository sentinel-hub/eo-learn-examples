# Temperature downscaling in urban areas

Responsible: Nico Bader

This demonstrates how to extract the structure of a city's temperature field by using satellite data. A slide set shows how this gets combined with weather data to generate e.g. urban heat island maps. That process will be made available via a meteoblue API that returns a heat map for the selected hour on a selected day and the notebook will be updated accordingly.

# I) Satellite pre-processing

## Getting started

**1. Clone the repository *satellite pre-processing* to your local machine.**
```
cd existing_repo
git clone https://git.meteoblue.com/metoffice/satellite-pre-processing.git
cd satellite-pre-processing
```

**2. Install and activate a new conda environment**

We will call this conda environment *sat_env* (just as an example).
```
conda create --name sat_env
conda activate sat_env
```

**3. Install all the required modules**

To run the satellite pre-processing, the following Python modules are needed:

- pip
- jupyterlab
- pandas
- zip
- xmltodict
- utm
- rasterio
- gdal
- matplotlib
- scipy
- netcdf4

Please use *conda-forge* to install the required modules.

```
conda install -c conda-forge pip jupyterlab pandas zip xmltodict utm rasterio gdal matplotlib scipy netcdf4
```

**4. Create the missing directories**

Create a output directory where the netCDF files will be stored.

```
mkdir output
```


## Run the example jupyter notebook

To understand how the satellite pre-processing works, you can use the jupyter notebook *satellite-pre-processing.ipynb*.
The satellite processing will be explained step-by-step.

Before running the example notebook, please download the example data (city: Chicago, USA) and store it into the directories
*data/satellite/*.
The example can be found under the following link:

https://meteoblue.sharepoint.com/:f:/s/External/ElIddi0pudFOo5fQLPn8W64BWpWojtj-lny0NbbgY7HLbA?e=EPtGHM


There are two files in the satellite data folder:
- *LC09_...* (Landsat 8 raw data)
- *S2A_...* (Sentinel-2 raw data)

!! Do not unzip them manually. Download them and save them into the *data/satellite/* folder.

# 2) Access High-resolution API

Will be added soon.