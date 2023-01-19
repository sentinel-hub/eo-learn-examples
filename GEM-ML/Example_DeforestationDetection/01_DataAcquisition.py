#!/usr/bin/env python
# coding: utf-8

# # GEM ML Framework Demonstrator - Deforestation Detection
# In these notebooks, we will get a feeling of how the GEM ML framework can be used for the segmentation of deforested areas using Sentinel-2 imagery as input and the [TMF dataset](https://forobs.jrc.ec.europa.eu/TMF/) as a reference.
# The idea is to use a neural network (NN) model for the analysis.
# Thanks to the flexibility of the GEM ML framework, the model used can be replaced by changing the configuration only.
# We will have a look at the following notebooks separately:
# - 00_Configuration
# - 01_DataAcquisition
# - 02_DataNormalization
# - 03_TrainingValidationTesting
# - 04_Inference_Clouds
# - 04_Inference_Timeseries
# 
# by Michael Engel (m.engel@tum.de) and Joana Reuss (joana.reuss@tum.de)
# 
# -----------------------------------------------------------------------------------
# 
# # Data Acquisition
# Here, we define our `EOWorkflow` for the download of our desired data.

# In[1]:


import os
import sys
import platform
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import time
import natsort

# import torch
# import torch.multiprocessing as mp
# from tensorboardX import SummaryWriter
# from tensorboard import notebook

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
# import folium
# from folium import plugins as foliumplugins

from libs.ConfigME import Config, importME
# from libs.MergeTDigests import mergeTDigests
# from libs.QuantileScaler_eolearn import QuantileScaler_eolearn_tdigest
# from libs.Dataset_eolearn import Dataset_eolearn
# from libs import AugmentME
from libs import ExecuteME

from tasks.TDigestTask import TDigestTask
from tasks.PickIdxTask import PickIdxTask
from tasks.SaveValidTask import SaveValidTask
# from tasks.PyTorchTasks import ModelForwardTask

from utils.rasterio_reproject import rasterio_reproject
from utils.transforms import batchify, predict, mover, Torchify
from utils.parse_time_interval_observations import parse_time_interval_observations

print("Working Directory:",os.getcwd())
print("Environment:",os.environ['CONDA_DEFAULT_ENV'])
print("Executable:",sys.executable)


# # Config
# First, we load our configuration file which provides all information we need throughout the script.

# In[2]:


#%% load configuration file
config = Config.LOAD("config.dill")


# # Area of Interest
# Let's load the geojson of our area of interests for training, validation and testing, respectively.

# In[3]:


#%% load geojson files
aoi_train = gpd.read_file(config['AOI_train'])
aoi_validation = gpd.read_file(config['AOI_validation'])
aoi_test = gpd.read_file(config['AOI_test'])

#%% find best suitable crs and transform to it
crs_train = aoi_train.estimate_utm_crs()
aoi_train = aoi_train.to_crs(crs_train)
aoi_train = aoi_train.buffer(config['AOIbuffer'])

crs_validation = aoi_validation.estimate_utm_crs()
aoi_validation = aoi_validation.to_crs(crs_validation)
aoi_validation = aoi_validation.buffer(config['AOIbuffer'])

crs_test = aoi_test.estimate_utm_crs()
aoi_test = aoi_test.to_crs(crs_test)
aoi_test = aoi_test.buffer(config['AOIbuffer'])

#%% dict for query
aois = {"train":aoi_train,
        "validation":aoi_validation,
        "test":aoi_test}


# Since our **area of interests are too large**, we **split** them into a set of smaller bboxes.

# In[4]:


#%% calculate and print size
aoi_train_shape = aoi_train.geometry
aoi_train_width = [geom.bounds[2]-geom.bounds[0] for geom in aoi_train_shape]
aoi_train_height = [geom.bounds[3]-geom.bounds[1] for geom in aoi_train_shape]
print(f"Dimension of the training area is {np.sum(np.array(aoi_train_width)*np.array(aoi_train_height)):.2e} m2")
aoi_validation_shape = aoi_validation.geometry
aoi_validation_width = [geom.bounds[2]-geom.bounds[0] for geom in aoi_validation_shape]
aoi_validation_height = [geom.bounds[3]-geom.bounds[1] for geom in aoi_validation_shape]
print(f"Dimension of the validation area is {np.sum(np.array(aoi_validation_width)*np.array(aoi_validation_height)):.2e} m2")
aoi_test_shape = aoi_test.geometry
aoi_test_width = [geom.bounds[2]-geom.bounds[0] for geom in aoi_test_shape]
aoi_test_height = [geom.bounds[3]-geom.bounds[1] for geom in aoi_test_shape]
print(f"Dimension of the test area is {np.sum(np.array(aoi_test_width)*np.array(aoi_test_height)):.2e} m2")

#%% create a splitter to obtain a list of bboxes
bbox_splitter_train = UtmZoneSplitter(aoi_train_shape, aoi_train.crs, config["patchpixelwidth"]*config["resolution"])
bbox_splitter_validation = UtmZoneSplitter(aoi_validation_shape, aoi_validation.crs, config["patchpixelwidth"]*config["resolution"])
bbox_splitter_test = UtmZoneSplitter(aoi_test_shape, aoi_test.crs, config["patchpixelwidth"]*config["resolution"])

bbox_list_train = np.array(bbox_splitter_train.get_bbox_list())
info_list_train = np.array(bbox_splitter_train.get_info_list())
bbox_list_validation = np.array(bbox_splitter_validation.get_bbox_list())
info_list_validation = np.array(bbox_splitter_validation.get_info_list())
bbox_list_test = np.array(bbox_splitter_test.get_bbox_list())
info_list_test = np.array(bbox_splitter_test.get_info_list())

#%% dict for query
bbox_lists = {"train":bbox_list_train,
              "validation":bbox_list_validation,
              "test":bbox_list_test}
info_lists = {"train":info_list_train,
              "validation":info_list_validation,
              "test":info_list_test}


# The **bbox list would be sufficient** for starting the training procedure using eo-learn.
# To check if we muddled up something, however, we want to visualize it!
# Since our area of interest is rather large, we face the problem of multiple coordinate refernce systems.
# Unfortunately, **geopandas does not support multiple crs in one dataframe** as described [here](https://github.com/sentinel-hub/sentinelhub-py/issues/123).
# Hence, we have to define a set of tiles for each separately.

# In[5]:


# tiles = []
# crss_uniques = []
# for _ in ["train","validation","test"]:
#     tiles.append([])
#     #%% determine number of coordinate reference systems
#     crss = [bbox_._crs for bbox_ in bbox_lists[_]]
#     crss_unique = np.array(list(dict.fromkeys(crss)))
#     crss_uniques.append(crss_unique)
#     n_crss = len(crss_unique)

#     #%% sort geometries and indices by crs and store to disk
#     geometries = [[] for i in range(n_crss)]
#     idxs = [[] for i in range(n_crss)]
#     idxs_x = [[] for i in range(n_crss)]
#     idxs_y = [[] for i in range(n_crss)]
#     for i,info in enumerate(info_lists[_]):
#         idx_ = np.argmax(crss_unique==bbox_lists[_][i]._crs)

#         geometries[idx_].append(Polygon(bbox_lists[_][i].get_polygon())) # geometries sorted by crs
#         idxs[idx_].append(info["index"]) # idxs sorted by crs
#         idxs_x[idx_].append(info["index_x"]) # idxs_x sorted by crs
#         idxs_y[idx_].append(info["index_y"]) # idxs_y sorted by crs

#     for i in range(n_crss):
#         #%% build dataframe of our areas of interest (and each crs)
#         tiles[-1].append(
#             gpd.GeoDataFrame(
#                 {"index": idxs[i], "index_x": idxs_x[i], "index_y": idxs_y[i]},
#                 crs="EPSG:"+crss_unique[i]._value_,
#                 geometry=geometries[i]
#             )
#         )
#         #%%% save dataframes to shapefiles
#         tiles[-1][-1].to_file(os.path.join(config["dir_results"],f"grid_aoi_{_}_{i}_EPSG{str(crss_unique[i]._value_)}.gpkg"), driver="GPKG")


# We have sorted the tiles according to their corresponding crs.
# Now we want to visualize it in a nice map.
# Here, it is important to **reproject the tiles** to the crs of our **mapping application** - we do that only for this purpose, the **bbox list is not affected** by this.

# In[6]:


#%% print amount of patches
print("Total number of tiles:",[len(bbox_list) for bbox_list in bbox_lists.values()])

# #%% visualize using folium
# aoi_folium = aoi_validation.to_crs("EPSG:4326") # use validation for visualisation
# location = [np.mean(aoi_folium.centroid.y),np.mean(aoi_folium.centroid.x)]

# mapwindow = folium.Map(location=location, tiles='Stamen Terrain', zoom_start=6)

# colors = ["blue","green","red"]
# for i,_ in enumerate(["train","validation","test"]):
#     #%%% add aois
#     #%%%% train
#     mapwindow.add_child(
#         folium.features.Choropleth(
#             aois[_].to_crs("EPSG:4326").to_json(),
#             fill_color=colors[i],
#             nan_fill_color=colors[i],
#             fill_opacity=0,
#             nan_fill_opacity=0.5,
#             line_color=colors[i],
#             line_weight=1,
#             line_opacity=0.6,
#             smooth_factor=5,
#             name=f"{_} area"
#         )
#     )

#     #%%% add grids in blue color
#     for t_,tiles_ in enumerate(tiles[i]):
#         cp = folium.features.Choropleth(
#                 tiles_.to_crs("EPSG:4326").to_json(),
#                 fill_color=colors[i],
#                 nan_fill_color="black",
#                 fill_opacity=0,
#                 nan_fill_opacity=0.5,
#                 line_color=colors[i],
#                 line_weight=0.5,
#                 line_opacity=0.6,
#                 smooth_factor=5,
#                 name=f"{_} grid EPSG:{crss_uniques[i][t_]._value_}"
#             ).add_to(mapwindow)

#         # display index next to cursor
#         folium.GeoJsonTooltip(
#             ['index'],
#             aliases=['Index:'],
#             labels=False,
#             style="background-color:rgba(0,101,189,0.4); border:2px solid white; color:white;",
#             ).add_to(cp.geojson)

# #%%% add some controls
# folium.LayerControl().add_to(mapwindow)
# foliumplugins.Fullscreen(force_separate_button=True).add_to(mapwindow)

# #%%% save, render and display
# mapwindow.save(os.path.join(config["dir_results"],'gridmap.html'))
# mapwindow.render()
# mapwindow


# # Input Tasks
# Now, it is time to define some input tasks for our `eo-learn` workflows.
# As an input, we will take a [Sentinel-Hub-Input-Task](https://eo-learn.readthedocs.io/en/latest/eolearn.io.sentinelhub_process.html#eolearn.io.sentinelhub_process.SentinelHubInputTask) for querying **Sentinel-2 data**.
# As a reference, we will use a [Import-From-Tiff-Task](https://eo-learn.readthedocs.io/en/latest/reference/eolearn.io.raster_io.html#eolearn.io.raster_io.ImportFromTiffTask) to load the [TMF dataset](https://forobs.jrc.ec.europa.eu/TMF/).
# For **cloud masking, or relabeling our reference,** we use the mask calculated by S2Cloudless.
# Further, we apply a labelmapping using the [MapFeatureTask](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/core/core_tasks.html#MapFeatureTask).
# 
# The date of our reference is considered to be the 2021-12-31.
# That is, we choose the observation with `config["maxcc"]` cloud coverage closest to that date.

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
    maxcc = config["maxcc"],
    time_difference = dt.timedelta(hours=1),
    cache_folder = config["dir_cache"],
    max_threads = config["threads"],
    config = config["SHconfig"],
    bands_dtype = np.float32,
    single_scene = False,
    mosaicking_order = "mostRecent",
    aux_request_args = None
)


# In order to get the closest observation with respect to our observation date, we pick the last one only.
# Actually, we do not need that since we will query for the correct timeinterval before downloading but for the sake of safety, we use the `PickIdxTask` here.

# In[8]:


#%% Pick-Idx-Task
task_data_pick = PickIdxTask(
    in_feature = (FeatureType.DATA, "data"),
    out_feature = None, # None for replacing in_feature
    idx = [[-1],...] # -1 in brackets for keeping dimensions of numpy array
)


# The same holds true for both our data- and the cloud-mask.

# In[9]:


#%% Pick-Idx-Task data mask
task_data_pick_dmask = PickIdxTask(
    in_feature = (FeatureType.MASK, "dmask_data"),
    out_feature = None, # None for replacing in_feature
    idx = [[-1],...] # -1 in brackets for keeping dimensions of numpy array
)

#%% Pick-Idx-Task cloud mask
task_data_pick_cmask = PickIdxTask(
    in_feature = (FeatureType.MASK, "cmask_data"),
    out_feature = None, # None for replacing in_feature
    idx = [[-1],...] # -1 in brackets for keeping dimensions of numpy array
)


# For the normalization of our dataset we will use the T-Digest algorithm.
# It is designed for quantile approximation close to the tails which we need for the common linear quantile scaler in the realm of ML.

# In[10]:


#%% T-Digest-Task
task_data_tdigest = TDigestTask(
    in_feature = (FeatureType.DATA, 'data'),
    out_feature = (FeatureType.SCALAR_TIMELESS, 'tdigest_data'),
    mode = None,
    pixelwise = False
)


# # Reference Task
# The reference is calculated using some thresholded value of the NDWI.
# To enable the user to use other thresholds after downloading the patches, the raw NDWI and the corresponding bands will be stored within the `EOPatch` as well.
# Additionally, we will download the RGB bands for visualisation purposes.

# In[11]:


#%% Import-From-Tiff-Task
task_reference = ImportFromTiffTask(
    feature = (FeatureType.MASK_TIMELESS, "reference"),
    folder = config["path_reference"],
    use_vsi = False,
    timestamp_size = None
)


# We apply our labelmapping using the [MapFeatureTask](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/core/core_tasks.html#MapFeatureTask).

# In[12]:


#%% apply labelmapping
def labelmapper(reference,mapping):
    for key,value in mapping.items():
        reference[reference==key] = value
    return reference
task_reference_labelmapping = MapFeatureTask(
    input_features = (FeatureType.MASK_TIMELESS, "reference"),
    output_features = (FeatureType.MASK_TIMELESS, "reference"),
    map_function = labelmapper,
    mapping = config["labelmapping"]
)


# Since the reference data has been acquired using the Landsat missions with 30m resolution, we have to resize our reference to our chosen resolution first.

# In[13]:


task_reference_resize = SpatialResizeTask(
    features = (FeatureType.MASK_TIMELESS, "reference"),
    resize_parameters = ['new_size', [config["patchpixelwidth"]]*2],
    resize_method = ResizeMethod.NEAREST,
    resize_library = ResizeLib.PIL
)


# Further, we want our model to segment clouds as well.
# Hence, we have to apply some mapping again using the [ZipFeatureTask](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/core/core_tasks.html#ZipFeatureTask) as we combine two features to one.

# In[14]:


def mask_key_value_zipper(*arrays,key=0,value=0):
    reference = arrays[0]
    mask = arrays[1].squeeze(0) # squeeze as originally temporal feature type
    reference[mask==key] = value
    return reference
task_reference_cloudmapping = ZipFeatureTask(
    input_features = [
        (FeatureType.MASK_TIMELESS, "reference"),
        (FeatureType.MASK, "cmask_data")
    ],
    output_feature = (FeatureType.MASK_TIMELESS, "reference"),
    zip_function = mask_key_value_zipper,
    key = 1,
    value = config["class_clouds"]
)


# ## Masking
# These EOTasks define the data we want to have as an input and as a reference for our problem.
# Still, **we have areas not providing reasonable data** at all or not in a meaningful way as for indefinite land cover, for example.
# That is, we take care of our dataMasks and the indefinite data for the reference.
# 
# For the sake of simplicity we want to **filter out every sample of the input not providing the full data**.
# This filtering will be done based on the analysis of the [MapFeatureTask](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/core/core_tasks.html#MapFeatureTask) applied to the dataMask.

# In[15]:


#%% Filter out incomplete input data patches
def checker_nodata(array):
    return bool(np.all(array))
task_data_check = MapFeatureTask(
    input_features = (FeatureType.MASK, "dmask_data"),
    output_features = (FeatureType.META_INFO, "valid"),
    map_function = checker_nodata
)


# For the indefinite land cover in our reference data, we calculate a mask using the [MapFeatureTask](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/core/core_tasks.html#MapFeatureTask).

# In[16]:


#%% apply labelmapping
def calculatemask(reference,key):
    mask = np.ones(reference.shape, dtype=np.uint8)
    mask[reference==key] = 0
    return mask
task_reference_mask = MapFeatureTask(
    input_features = (FeatureType.MASK_TIMELESS, "reference"),
    output_features = (FeatureType.MASK_TIMELESS, "mask_reference"),
    map_function = calculatemask,
    key = config["class_indefinite"]
)


# ## Merging and Saving
# **Only valid EOPatches are saved** using the [Save-Valid-Task]() based on the citerion regarding the input data availability.
# Note the **compression** keyword - if not set, the memory consumption may get really large!

# In[17]:


#%% save EOPatches
task_save = SaveValidTask(
    feature_to_check = (FeatureType.META_INFO, "valid"),
    path = config["dir_data"],
    filesystem = None,
    config = config["SHconfig"],
    overwrite_permission = OverwritePermission.OVERWRITE_PATCH,
    compress_level = 2
)


# # Workflow
# Now, we can define a workflow bringing everything together
# - ### Input
# >- task_data
# >- task_data_pick
# >- task_data_pick_dmask
# >- task_data_pick_cmask
# >- task_data_tdigest
# >- task_data_check
# 
# - ### Reference
# >- task_reference
# >- task_reference_labelmapping
# >- task_reference_resize
# >- task_reference_cloudmapping
# >- task_reference_mask
# 
# - ### Merging and Saving
# >- task_save
# 
# ## Define Nodes
# Let's initialise the nodes we will use for our workflow afterwards.

# In[18]:


#%% input nodes
node_data = EONode(
    task = task_data,
    inputs = [],
    name = "load Sentinel-2 data"
)
node_data_pick = EONode(
    task = task_data_pick,
    inputs = [node_data],
    name = "pick closest observation to reference for data"
)
node_data_pick_dmask = EONode(
    task = task_data_pick_dmask,
    inputs = [node_data_pick],
    name = "pick closest observation to reference for data mask"
)
node_data_pick_cmask = EONode(
    task = task_data_pick_cmask,
    inputs = [node_data_pick_dmask],
    name = "pick closest observation to reference for cloud mask"
)
node_data_tdigest = EONode(
    task = task_data_tdigest,
    inputs = [node_data_pick_cmask],
    name = "compute T-Digest of data"
)
node_data_check = EONode(
    task = task_data_check,
    inputs = [node_data_tdigest],
    name = "check data for completeness"
)

#%% reference nodes
node_reference = EONode(
    task = task_reference,
    inputs = [node_data_check],
    name = "load reference from disk"
)
node_reference_labelmapping = EONode(
    task = task_reference_labelmapping,
    inputs = [node_reference],
    name = "apply labelmapping"
)
node_reference_resize = EONode(
    task = task_reference_resize,
    inputs = [node_reference_labelmapping],
    name = "apply labelmapping"
)
node_reference_cloudmapping = EONode(
    task = task_reference_cloudmapping,
    inputs = [node_reference_resize],
    name = "map clouds into reference"
)
node_reference_mask = EONode(
    task = task_reference_mask,
    inputs = [node_reference_cloudmapping],
    name = "mask indefinite reference data"
)

#%% merging and saving nodes
node_save = EONode(
    task = task_save,
    inputs = [node_reference_mask],
    name = "save valid EOPatch"
)


# ## Define Workflow
# Now, we finally can define a workflow based on our tasks and nodes.
# We could either put every single node in the constructor using a list or define our whole workflow by just the last node: `node_save`.

# In[19]:


workflow = EOWorkflow.from_endnodes(node_save)
#workflow.dependency_graph()


if __name__=="__main__":
    # ## Test Workflow
    # Now, we want to test our workflow with some arbitrary patch (from our training set) at some arbitrary date (not included into study).
    
    # In[20]:
    workflow.execute({
        node_data: {"bbox":bbox_list_train[596],"time_interval":("2022-10-01","2022-10-01")},
        node_save: {"eopatch_folder":"testpatch"}
    })
    eopatch = EOPatch.load(os.path.join(config["dir_data"],"testpatch"))
    eopatch
    
    
    # Let's have a look at our eopatch.
    
    # In[21]:
    
    
    RGB = eopatch["data"]["data"][0,...,np.array([2,1,0])].transpose(1,2,0)
    
    #%% plot testpatch
    plt.figure()
    plt.subplot(221)
    plt.imshow(RGB*2.5)
    plt.title("RGB")
    plt.axis("off")
    plt.subplot(222)
    plt.imshow(eopatch["mask_timeless"]["reference"],vmin=0,vmax=config["num_classes"]-1,cmap="Dark2")
    plt.title("Reference")
    plt.axis("off")
    plt.subplot(223)
    plt.imshow(eopatch["mask"]["cmask_data"][0,...],vmin=0,vmax=1,cmap="YlOrBr_r")
    plt.title("Clouds")
    plt.axis("off")
    plt.subplot(224)
    plt.imshow(eopatch["mask_timeless"]["mask_reference"],vmin=0,vmax=1,cmap="RdYlGn")
    plt.title("Mask")
    plt.axis("off")
    
    plt.savefig("testpatch.png",dpi=300)
    
    
    # # Workflow Arguments
    # Now it's time to download the data.
    # Therefore, we have to define workflow arguments, both temporal and spatially.
    # Note that we only want to download the data which does not exist on our device.
    # Hence, we check for existence first and assign arguments afterwards.
    
    # In[22]:
    
    
    workflow_args = []
    for _ in ["train","validation","test"]:
        print(_)
        bbox_list_ = bbox_lists[_]
        for i in range(len(bbox_list_)):
            print(f"\r{i+1}/{len(bbox_list_)}",end="\r")
            try:
                timestamps = parse_time_interval_observations(
                    time_interval = (config[f"start_{_}"],config[f"end_{_}"]),
                    bbox = bbox_list_[i], 
                    data_collection = DataCollection.SENTINEL2_L1C, 
                    check_timedelta = config["checktimedelta"],
                    include_borders = True,
                    time_difference = dt.timedelta(hours=1,seconds=0), 
                    maxcc = config["maxcc"], 
                    config = config["SHconfig"]
                )
    
                dir_ = f"{_}/eopatch_{i}_{timestamps[0].strftime(r'%Y-%m-%dT%H-%M-%S_%Z')}_{timestamps[1].strftime(r'%Y-%m-%dT%H-%M-%S_%Z')}"
                if not os.path.exists(os.path.join(config["dir_data"],dir_)):### and False: ### 
                    workflow_args.append(
                        {
                            node_data: {"bbox":bbox_list_[i],"time_interval":timestamps},
                            node_save: {"eopatch_folder":dir_}
                        }
                    )
            except Exception as e:
                print(e)
        print()
    
    print(f"Number of downloads: {len(workflow_args)}")
    
    
    # In[23]:
    
    
    workflow_args[-1]
    
    
    # # Executor
    # Our area of interest has been defined, our desired data has been defined, our workflow has been defined, our execution arguments have been defined, our executor...
    # This has to be done!
    
    # In[24]:
    
    
    #%% define executor
    executor = EOExecutor(workflow, workflow_args, save_logs=True)
    
    
    # Let it run!
    # That may take a while...
    
    # In[25]:
    
    
    #%% run
    print(f"Will start data acquisition using {config['threads']} threads!")
    executor.run(workers=config["threads"])
    executor.make_report()
    
    
    # # Downloaded Data
    # After a long time, our executor finished with it's work.
    # Let's **check** if there happened anything unexpected.
    
    # In[26]:
    
    
    failed_ids = executor.get_failed_executions()
    if failed_ids:
        print(
            f"Execution failed EOPatches with IDs:\n{failed_ids}\n"
            f"For more info check report at {executor.get_report_path()}"
        )
    
    
    # Let's have a look how many `EOPatches` got stored to disk.
    
    # In[27]:
    
    
    print(f"Number of stored train EOPatches: {len(os.listdir(config['dir_train']))}")
    print(f"Number of stored validation EOPatches: {len(os.listdir(config['dir_validation']))}")
    print(f"Number of stored test EOPatches: {len(os.listdir(config['dir_test']))}")
    print()
    print(f"Number of downloads: {len(workflow_args)}")
    print(f"Total number of EOPatches: {len(os.listdir(config['dir_train']))+len(os.listdir(config['dir_validation']))+len(os.listdir(config['dir_test']))}")
    
    
    # We finally made it!
    # Everything is ready for being used!
    
    # In[28]:
    
    
    print(FeatureType.DATA.ndim())
    print(FeatureType.DATA_TIMELESS.ndim())
    print(FeatureType.LABEL.ndim())
    print(FeatureType.LABEL_TIMELESS.ndim())
    print(FeatureType.MASK.ndim())
    print(FeatureType.MASK_TIMELESS.ndim())
    print(FeatureType.SCALAR.ndim())
    print(FeatureType.SCALAR_TIMELESS.ndim())
    print(FeatureType.VECTOR.ndim())
    print(FeatureType.VECTOR_TIMELESS.ndim())
    
    
    # In[ ]:
    
    
    
    
