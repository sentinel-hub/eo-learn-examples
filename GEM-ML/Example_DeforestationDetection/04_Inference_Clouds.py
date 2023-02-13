#!/usr/bin/env python
# coding: utf-8

# # GEM ML Framework Demonstrator - Deforestation Detection
# In these notebooks, we provide an in-depth example of how the GEM ML framework can be used for segmenting deforested areas using Sentinel-2 imagery as input and the [TMF dataset](https://forobs.jrc.ec.europa.eu/TMF/) as a reference.
# The idea is to use a neural network (NN) model for the analysis.
# Thanks to the flexibility of the GEM ML framework, we can easily substitute the model in the future by adjusting only the configuration file.
# We will have a look at the following notebooks separately:
# - 00_Configuration
# - 01_DataAcquisition
# - 02_DataNormalization
# - 03_TrainingValidationTesting
# - 04_Inference_Clouds
# 
# Authors: Michael Engel (m.engel@tum.de) and Joana Reuss (joana.reuss@tum.de)
# 
# -----------------------------------------------------------------------------------
# 
# # Inference - Clouds
# This notebook shows how the GEM ML Framework can support continuous deforestation monitoring. In the chosen area, deforestation takes place for the sake of bauxite mining, and, e.g. a land surveying office is asking for an analysis.
# Reference data is not available, and several observations are cloudy. Still, we want a fast and reliable segmentation map of the area - preferably cloudless! For that purpose, a fast inference pipeline is necessary.
# For that purpose, we can use the `ModelForwardTask` method provided within the `PyTorchTasks` module.
# It enables the users to integrate an already trained PyTorch-model into their eo-learn workflows.
# The provided `ExecuteME` package does the management of GPU/CPU shifting.
# In general, we recommend doing this in standard Python scripts as Jupyter Notebooks do not support the spawn method for parallelization that PyTorch-objects ask for.

# In[1]:


import os
import sys
import platform
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import time
import natsort

import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from tensorboard import notebook

from sentinelhub import SHConfig, BBox, CRS, DataCollection, UtmZoneSplitter, DataCollection
from eolearn.core import FeatureType, EOPatch, MergeEOPatchesTask, MapFeatureTask, MergeFeatureTask, ZipFeatureTask, LoadTask, EONode, EOWorkflow, EOExecutor, OverwritePermission, SaveTask
from eolearn.io import SentinelHubDemTask, ExportToTiffTask, SentinelHubInputTask, SentinelHubEvalscriptTask, get_available_timestamps, ImportFromTiffTask
from eolearn.mask import CloudMaskTask, JoinMasksTask
from eolearn.features.feature_manipulation import SpatialResizeTask
from eolearn.features.utils import ResizeMethod, ResizeLib

import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon,Point
import folium
from folium import plugins as foliumplugins

from libs.ConfigME import Config, importME
from libs.MergeTDigests import mergeTDigests
from libs.QuantileScaler_eolearn import QuantileScaler_eolearn_tdigest
from libs.Dataset_eolearn import Dataset_eolearn
from libs import AugmentME
from libs import ExecuteME

from tasks.TDigestTask import TDigestTask
from tasks.PickIdxTask import PickIdxTask
from tasks.SaveValidTask import SaveValidTask
from tasks.PyTorchTasks import ModelForwardTask

from utils.rasterio_reproject import rasterio_reproject
from utils.transforms import batchify, predict, mover, Torchify
from utils.parse_time_interval_observations import parse_time_interval_observations

print("Working Directory:",os.getcwd())
print("Environment:",os.environ['CONDA_DEFAULT_ENV'])
print("Executable:",sys.executable)


# # Config
# First, we load our configuration file which provides all information we need throughout the script and linuxify our paths (if you are working on a Windows machine) as the eo-learn filesystem manager does not support backslashes for now.

# In[2]:


#%% load configuration file
config = Config.LOAD("config.dill")

#%% linuxify
config.linuxify()


# # Area of Interest
# Let's load the geojson of our area of interest for our use-case.

# In[3]:


#%% load geojson files
aoi_showcase = gpd.read_file(config['AOI_showcase'])

#%% find best suitable crs and transform to it
crs_showcase = aoi_showcase.estimate_utm_crs()
aoi_showcase = aoi_showcase.to_crs(crs_showcase)


# In[4]:


#%% calculate and print size
aoi_showcase_shape = aoi_showcase.geometry
aoi_showcase_width = [geom.bounds[2]-geom.bounds[0] for geom in aoi_showcase_shape]
aoi_showcase_height = [geom.bounds[3]-geom.bounds[1] for geom in aoi_showcase_shape]
print(f"Dimension of the showcase area is {np.sum(np.array(aoi_showcase_width)*np.array(aoi_showcase_height)):.2e} m2")

#%% create a splitter to obtain a list of bboxes
bbox_splitter_showcase = UtmZoneSplitter(aoi_showcase_shape, aoi_showcase.crs, config["patchpixelwidth"]*config["resolution"])

bbox_list_showcase = np.array(bbox_splitter_showcase.get_bbox_list())
info_list_showcase = np.array(bbox_splitter_showcase.get_info_list())


# # Visualization of AOI
# Please note that this part is not necessary for the analysis but we highly recommend doing these type of things!
# Again, we define our entry points for later parallelization on Windows machines.

# In[5]:


if __name__=='__main__':
    #%% determine number of coordinate reference systems
    crss = [bbox_._crs for bbox_ in bbox_list_showcase]
    crss_unique = np.array(list(dict.fromkeys(crss)))
    n_crss = len(crss_unique)

    #%% sort geometries and indices by crs and store to disk
    geometries = [[] for i in range(n_crss)]
    idxs = [[] for i in range(n_crss)]
    idxs_x = [[] for i in range(n_crss)]
    idxs_y = [[] for i in range(n_crss)]
    for i,info in enumerate(info_list_showcase):
        idx_ = np.argmax(crss_unique==bbox_list_showcase[i]._crs)

        geometries[idx_].append(Polygon(bbox_list_showcase[i].get_polygon())) # geometries sorted by crs
        idxs[idx_].append(info["index"]) # idxs sorted by crs
        idxs_x[idx_].append(info["index_x"]) # idxs_x sorted by crs
        idxs_y[idx_].append(info["index_y"]) # idxs_y sorted by crs

    tiles = []
    for i in range(n_crss):
        #%%% build dataframe of our areas of interest (and each crs)
        tiles.append(
            gpd.GeoDataFrame(
                {"index": idxs[i], "index_x": idxs_x[i], "index_y": idxs_y[i]},
                crs="EPSG:"+crss_unique[i]._value_,
                geometry=geometries[i]
            )
        )
        #%%% save dataframes to shapefiles
        tiles[-1].to_file(os.path.join(config["dir_results"],f"grid_aoi_showcase_{i}_EPSG{str(crss_unique[i]._value_)}.gpkg"), driver="GPKG")

    #%% print amount of patches
    print("Total number of tiles:",len(bbox_list_showcase))    


# In[6]:


if __name__=='__main__':
    #%% visualize using folium
    aoi_folium = aoi_showcase.to_crs("EPSG:4326")
    location = [aoi_folium.centroid.y,aoi_folium.centroid.x]

    mapwindow = folium.Map(location=location, tiles='Stamen Terrain', zoom_start=8)

    #%%% add aois
    #%%%% train
    mapwindow.add_child(
        folium.features.Choropleth(
            aoi_folium.to_json(),
            fill_color="royalblue",
            nan_fill_color="royalblue",
            fill_opacity=0,
            nan_fill_opacity=0.5,
            line_color="royalblue",
            line_weight=1,
            line_opacity=0.6,
            smooth_factor=5,
            name=f"showcase area"
        )
    )

    #%%% add grids in color
    for t_,tiles_ in enumerate(tiles):
        cp = folium.features.Choropleth(
                tiles_.to_crs("EPSG:4326").to_json(),
                fill_color="royalblue",
                nan_fill_color="black",
                fill_opacity=0,
                nan_fill_opacity=0.5,
                line_color="royalblue",
                line_weight=0.5,
                line_opacity=0.6,
                smooth_factor=5,
                name=f"showcase grid EPSG:{crss_unique[t_]._value_}"
            ).add_to(mapwindow)

        # display index next to cursor
        folium.GeoJsonTooltip(
            ['index'],
            aliases=['Index:'],
            labels=False,
            style="background-color:rgba(0,101,189,0.4); border:2px solid white; color:white;",
            ).add_to(cp.geojson)

    #%%% add some controls
    folium.LayerControl().add_to(mapwindow)
    foliumplugins.Fullscreen(force_separate_button=True).add_to(mapwindow)

    #%%% save, render and display
    mapwindow.save(os.path.join(config["dir_results"],'gridmap_showcase.html'))
    mapwindow.render()
    mapwindow


# # eo-learn worlflow
# Similar to the workflow we have defined in [01_DataAcquisition]().
# 
# It again consists of the following elements
# 
# - __EOTask__
# - __EONode__
# - __EOWorkflow__
# 
# This time, we define the following `EOTasks`:
# 
# 
# ##  1. Input data: Querying Sentinel data
# >- __*task_data*__: We take a [Sentinel-Hub-Input-Task](https://eo-learn.readthedocs.io/en/latest/eolearn.io.sentinelhub_process.html#eolearn.io.sentinelhub_process.SentinelHubInputTask) for querying **Sentinel-2 data**.
# 
# ## 2. PyTorch tasks
# >- __*task_model*__: We take a [ModelForwardTask]() for defining the model's forward function.
# 
# ## 3. Cloud removal
# >- __*task_postprocessing*__: We remove clouds from the prediction using a [MapFeatureTask](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/core/core_tasks.html#MapFeatureTask).
# 
# 
# ## 4. Exporting and Saving
# #### 4.1 Exporting
# >- __*task_tiff*__: We export the model's predictions as tif-files using an [`ExportToTiffTask`](https://eo-learn.readthedocs.io/en/latest/reference/eolearn.io.raster_io.html#eolearn.io.raster_io.ExportToTiffTask).
# >- __*task_tiff_postprocessing*__: We export the model's predictions without clouds as tif-files using an [`ExportToTiffTask`](https://eo-learn.readthedocs.io/en/latest/reference/eolearn.io.raster_io.html#eolearn.io.raster_io.ExportToTiffTask).
# 
# #### 4.2 Saving
# >- __*task_save*__: We save the created EOPatches using a __`SaveValidTask`__.

# ## 1. Input data: Querying Sentinel data
# First, we define our `EOTasks` for the input data. They are the same as for the training, validation and testing procedure. Except that we skip the checking and reference this time (as there is none).
# 
# Please note that we do not apply a cloud filtering here!

# In[7]:


#%% Sentinel-Hub-Input-Task
task_data = SentinelHubInputTask(
    data_collection = DataCollection.SENTINEL2_L1C,
    size = None,
    resolution = config["resolution"],
    bands_feature = (FeatureType.DATA, "data"),
    bands = ["B02","B03","B04","B08","B11","B12"],
    additional_data = [(FeatureType.MASK, "dataMask", "dmask_data"),(FeatureType.MASK, "CLM", "cmask_data")],
    evalscript = None,
    maxcc = 1,
    time_difference = dt.timedelta(hours=1),
    cache_folder = config["dir_cache"],
    max_threads = config["threads"],
    config = config["SHconfig"],
    bands_dtype = np.float32,
    single_scene = False,
    mosaicking_order = "mostRecent",
    aux_request_args = None
)


# ## 2. PyTorch Tasks
# As a first step, however, we need to load our Scaler as built in [02_DataNormalization]().

# In[8]:


Scaler = QuantileScaler_eolearn_tdigest.LOAD(os.path.join(config["dir_results"],config["savename_scaler"]))


# Subsequently, we load our __best model__ using the `BaseClass` of `AugmentME`.
# Further, we set it to evaluation mode and tell it to share its memory for being deployed on multiple CPUs.
# Loading the model to the `CPU` is essential as you get in trouble with the parallelization of `ExecuteME` otherwise.

# In[9]:


model = AugmentME.BaseClass(mode="torch")
model.load(os.path.join(config["dir_results"],config["model_savename_bestloss"]),device="cpu")
model.eval()
model.share_memory()


# ### ModelForwardTask
# As discussed, we want to respond to the request immediately.
# That means, we want to use our trained, validated and tested model for prediction!
# Fortunately, TUM established the `PyTorchTask` as a base class for many PyTorch related `EOTasks` like the `ModelForwardTask`, the `LayerGradCamTask` or the `GradientShapTask`, for example.
# In this notebook, we focus on the `ModelForwardTask` since we need to implement a functionality for returning the model's prediction, as the model's regular forward method only returns the logits and not the final prediction.
# This ensures that the input feature is fed to our model and the output is returned as intended.
# 
# As for all `EOTasks`, we choose the in- and output features.
# The `in_feature` is fed to the model, whereas its result is stored in the `out_feature`.
# In order to properly normalize the downloaded data, we have to insert our scaler for `in_transform`.
# Since we are interested in the predicted land cover, we have to insert our prediction transform for `out_transform`.

# In[10]:


task_model = ModelForwardTask(
    in_feature = (FeatureType.DATA,"data"),
    out_feature = (FeatureType.MASK,"model_output"),
    model = model,

    in_transform = Scaler,
    out_transform = predict,
    in_torchtype = torch.FloatTensor,
    batch_size = config["max_batch_size"],

    maxtries=3,
    timeout=22,
)


# ## 3. Cloud Removal
# Since our scenario represents a request made by a land surveying office, we preferably do not want to have clouds in our analysis while sticking to the robustness of our model using multispectral satellite data.
# Fortunately, we defined our labels accordingly: by using the maximum predicted value out of a series of predictions, we always remove a cloud (class 0) if a less cloudy observation is available, meaning a class value greater 0.
# Further, we always overwrite forest (class 2) by deforestation (class 3), since it is unlikely that a full forest follows deforestation.
# By doing so, we always catch the deforestation which took place in a certain time period which is chosen by the user within the workflow arguments.
# That post-processing is done using the [MapFeatureTask](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/core/core_tasks.html#MapFeatureTask).

# In[11]:


def maximizer(reference):
    return np.max(reference,axis=0)
task_postprocessing = MapFeatureTask(
    input_features = (FeatureType.MASK, "model_output"),
    output_features = (FeatureType.MASK_TIMELESS, "model_output_post"),
    map_function = maximizer
)


# ## 4. Exporting and saving
# #### 4.1 Exporting
# Of course, we want to export the both the model's output as a GeoTiff for others to analyze it using a common GIS software.
# 
# We can use the [`ExportToTiffTask`](https://eo-learn.readthedocs.io/en/latest/reference/eolearn.io.raster_io.html#eolearn.io.raster_io.ExportToTiffTask) in order to export both the model's output before and after removing clouds.

# In[12]:


#%% export raw model output
task_tiff = ExportToTiffTask(
    feature = (FeatureType.MASK,"model_output"),
    folder = config["dir_tiffs_showcase"],
    date_indices = None,
    band_indices = None,
    crs = None,
    fail_on_missing = True,
    compress = "deflate"
)

#%% export post-processed model output
task_tiff_postprocessing = ExportToTiffTask(
    feature = (FeatureType.MASK_TIMELESS,"model_output_post"),
    folder = config["dir_tiffs_showcase"],
    date_indices = None,
    band_indices = None,
    crs = None,
    fail_on_missing = True,
    compress = "deflate"
)


# ### 4.2 Saving
# Finally, we want to store the resulting patches.

# In[13]:


#%% save EOPatches
task_save = SaveTask(
    path = config["dir_data"],
    filesystem = None,
    config = config["SHconfig"],
    overwrite_permission = OverwritePermission.OVERWRITE_PATCH,
    compress_level = 2
)


# ## EONodes
# After we have defined all necessary __`EOTasks`__, we initialize the __`EONodes`__ which will be used in order to run through the workflow afterwards.

# In[14]:


#%% input nodes
node_data = EONode(
    task = task_data,
    inputs = [],
    name = "load Sentinel-2 data"
)

#%% inference node
node_model = EONode(
    task = task_model,
    inputs = [node_data],
    name = "predict water mask"
)

node_postprocessing = EONode(
    task = task_postprocessing,
    inputs = [node_model],
    name = "predict water mask"
)

#%% export and save
node_tiff = EONode(
    task = task_tiff,
    inputs = [node_postprocessing],
    name = "export GeoTiff of model output"
)

node_tiff_postprocessing = EONode(
    task = task_tiff_postprocessing,
    inputs = [node_tiff],
    name = "export GeoTiff of model output"
)

node_save = EONode(
    task = task_save,
    inputs = [node_tiff_postprocessing],
    name = "save EOPatch"
)


# ## Final EOWorkflow
# 
# Finally, we can define our workflow using the end node.

# In[15]:


workflow = EOWorkflow.from_endnodes(node_save)
#workflow.dependency_graph()


# ## Execution
# So far, we have defined our
# - Area of Interest
# - Input tasks
# - Model tasks
# - EOWorkflow
# 
# What is left, is the definition of our __execution (or workflow) arguments__.
# We want to execute our workflow in parallel.
# This can be done using the package `ExecuteME`.
# 
# Since PyTorch models demand the spawn start method for subprocesses, we must ensure the entry point.
# This is accomplished by setting the file that defines the subprocesses as the main file.
# We recommend doing this in standard Python scripts, as Jupyter Notebooks require clarifying the entry point in every cell (so you can easily export it as a script) and do not support the spawn method.
# That is, the parallelization does not work in Jupyter Notebooks, but you may let it run using one worker.

# ### Workflow Arguments
# First, we have to define __workflow arguments__, both temporal and spatial.
# Note that we only want to download the data that is not yet present on our device.
# Hence, we check for existence first and assign arguments afterwards.

# In[16]:


if __name__=='__main__':  
    #%% define workflow arguments
    workflow_args = []
    bbox_list_ = bbox_list_showcase
    for i in range(len(bbox_list_)):
        print(f"\rChecking workflow args {i+1}/{len(bbox_list_)}",end="\r")
        try:
            timeinterval = (config["start_showcase"],config["end_showcase"])
            timeintervalstring = f"{timeinterval[0].strftime(r'%Y-%m-%dT%H-%M-%S_%Z')}--{timeinterval[1].strftime(r'%Y-%m-%dT%H-%M-%S_%Z')}"
            dir_ = f"showcase/eopatch_{i}_{timeintervalstring}"
            if not os.path.exists(os.path.join(config["dir_data"],dir_)):### and False: ### 
                workflow_args.append(
                    {
                        node_data: {"bbox":bbox_list_[i],"time_interval":timeinterval},
                        node_tiff: {"filename": f"deforestation_raw_{i}_{timeintervalstring}"},
                        node_tiff_postprocessing: {"filename": f"deforestation_postprocessed_{i}_{timeintervalstring}"},
                        node_save: {"eopatch_folder":dir_}
                    }
                )
        except Exception as e:
            print(e)
    print()

    print(f"Number of downloads/calculations: {len(workflow_args)}")


# ### Devices
# Secondly, we need to initialize a multiprocessor queue containing our devices' names.
# This way, we can use any number of GPUs.
# Of course, it can also stay on the CPU by setting the config.device accordingly.
# 
# The `PyTorchTasks` can be used for features that contain multiple timestamps to be analyzed.
# Accordingly, the `batch_size` parameter of the `ModelForwardTask` refers to the timestamps, i.e., the first dimension of a feature array.
# If you do not have multiple timestamps, you can insert a kind of `batch_size` for a device by defining `batch_size x available_devices` devices or especially `batch_size` times the device you want to use multiple times.
# Please be careful with this as there is an additional cost per patch to initialize the model since the model is not shared between multiple `EOPatches`.
# Nevertheless, this behavior is advantageous if an analysis is to be carried out using the `ModelUncertaintyTask`, for example.
# Here it is necessary to use the model several times anyway.

# In[17]:


if __name__=='__main__':  
    devices = ExecuteME.Devices(["cuda"],multiprocessing_context="spawn")


# In the next step, we will define the multiprocessor keyword arguments that must be passed to our tasks separately. These keyword arguments must be shared between the processes because the list of available devices should be known to the different processes and, hence, be shared or provided separately.

# In[18]:


if __name__=='__main__':  
    mpkwargs = {
        node_model: {"devices":devices},
    }


# ### Run
# Now, it's time to let it run!
# Please notice that we insert a 0 for `threads` since Jupyter Notebooks do not allow for the spawn method.
# In a Python script, you may choose as many threads as you like.

# In[19]:


if __name__=='__main__':  
    start_multi = time.time()
    results_multi = ExecuteME.execute(
        fun = workflow.execute,
        kwargslist = workflow_args,
        mpkwargs = mpkwargs,
        
        kwargsmode = None,
        resultsqueue = None,
        NoReturn = True,
        
        timeout = 1,
        threads = 3,
        checkthreads = True,
        multiprocessing_context = "spawn",
        multiprocessing_mode = "std",
        bequiet = False
    )
    time_multi = time.time()-start_multi

    #%%% results
    print(f"Time Multi:\t\t{time_multi}s")


# ### Downloaded Data
# Let's have a look at how many `EOPatches` got stored to disk.

# In[20]:


if __name__=='__main__':  
    print(f"Number of showcasedownloads: {len(workflow_args)}")
    print(f"Number of stored showcase EOPatches: {len(os.listdir(config['dir_showcase']))}")


# We finally made it!
# Everything is ready for being analyzed!

# ## Analysis
# As a first analysis step, we want to merge all of our computed GeoTiffs, both for the raw and postprocessed model output.

# In[21]:


if __name__=='__main__':  
    #%% merge raw model output
    importME("../utils/RasterME_merge.raster_merge")(
        inputfiles = [
            os.path.join(config["dir_tiffs_showcase"],dir_)
            for dir_ in os.listdir(config["dir_tiffs_showcase"])
            if "deforestation" in dir_.split("_") and "raw" in dir_.split("_")
        ],
        outputfile = os.path.join(config["dir_results"],config["savename_showcase_tiff"]),
        format_option = 'COMPRESS=Deflate',
        sparse = True,
        #nmax_files = 10
    )

    #%% merge postprocessed model output
    importME("../utils/RasterME_merge.raster_merge")(
        inputfiles = [
            os.path.join(config["dir_tiffs_showcase"],dir_)
            for dir_ in os.listdir(config["dir_tiffs_showcase"])
            if "deforestation" in dir_.split("_") and "postprocessed" in dir_.split("_")
        ],
        outputfile = os.path.join(config["dir_results"],config["savename_showcase_tiff_post"]),
        format_option = 'COMPRESS=Deflate',
        sparse = True,
        #nmax_files = 10
    )


# Now you know how to use the `ModelForwardTask` and post-processing for analysis! :)
