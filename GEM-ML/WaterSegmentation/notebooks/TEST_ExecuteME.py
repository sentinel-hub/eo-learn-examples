### Michael Engel ### 2022-10-23 ### TEST_ExecuteME.py ###

import os
import sys
import platform
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import time
import natsort

import torch
print(torch.cuda.memory_allocated(0))
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from tensorboard import notebook

from sentinelhub import SHConfig, BBox, CRS, DataCollection, UtmZoneSplitter, DataCollection
from eolearn.core import FeatureType, EOPatch, MergeEOPatchesTask, MapFeatureTask, MergeFeatureTask, ZipFeatureTask, LoadTask, EONode, EOWorkflow, EOExecutor, OverwritePermission, SaveTask
from eolearn.io import SentinelHubDemTask, ExportToTiffTask, SentinelHubInputTask, SentinelHubEvalscriptTask, get_available_timestamps
from eolearn.mask import CloudMaskTask, JoinMasksTask

import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon,Point
import folium
from folium import plugins as foliumplugins

from libs.ConfigME import Config, importME
from libs.TDigestTask import TDigestTask
from libs.PickIdxTask import PickIdxTask
from libs.SaveValidTask import SaveValidTask
from libs.MergeTDigests import mergeTDigests
from libs.QuantileScaler_eolearn import QuantileScaler_eolearn_tdigest
from libs.Dataset_eolearn import Dataset_eolearn, Torchify
from libs.PyTorchTasks import ModelForwardTask,GradientShapTask,Devices
from libs.rasterio_reproject import rasterio_reproject
from libs import AugmentME
from libs.random import batchify,predict,mover

print("Working Directory:",os.getcwd())
print("Environment:",os.environ['CONDA_DEFAULT_ENV'])
print("Executable:",sys.executable)

#%% load configuration file
config = Config.LOAD("config.dill")
config.linuxify()

#%% load geojson files
# aoi_showcase = gpd.read_file(config['AOI_showcase'])
aoi_showcase = gpd.read_file(r"./misc/NigeriaFlood_large.json")

#%% find best suitable crs and transform to it
crs_showcase = aoi_showcase.estimate_utm_crs()
aoi_showcase = aoi_showcase.to_crs(crs_showcase)

#%% calculate and print size
aoi_showcase_shape = aoi_showcase.geometry.values[0]
aoi_showcase_width = aoi_showcase_shape.bounds[2]-aoi_showcase_shape.bounds[0]
aoi_showcase_height = aoi_showcase_shape.bounds[3]-aoi_showcase_shape.bounds[1]
print(f"Dimension of the showcase area is {aoi_showcase_width:.0f} x {aoi_showcase_height:.0f} m2")

#%% create a splitter to obtain a list of bboxes
bbox_splitter_showcase = UtmZoneSplitter([aoi_showcase_shape], aoi_showcase.crs, config["patchpixelwidth"]*config["resolution"])

bbox_list_showcase = np.array(bbox_splitter_showcase.get_bbox_list())
info_list_showcase = np.array(bbox_splitter_showcase.get_info_list())

#%% print amount of patches
print("Total number of tiles:",len(bbox_list_showcase))

#%% Sentinel-Hub-Input-Task
task_data = SentinelHubInputTask(
    data_collection = DataCollection.SENTINEL1_IW,
    size = None,
    resolution = config["resolution"],
    bands_feature = (FeatureType.DATA, "data"),
    bands = None,
    additional_data = (FeatureType.MASK, "dataMask", "dmask_data"),
    evalscript = None,
    maxcc = None,
    time_difference = dt.timedelta(hours=1),
    cache_folder = config["dir_cache"],
    max_threads = config["threads"],
    config = config["SHconfig"],
    bands_dtype = np.float32,
    single_scene = False,
    mosaicking_order = "mostRecent",
    aux_request_args = None
)

#%% Pick-Idx-Task
task_pick = PickIdxTask(
    in_feature = (FeatureType.DATA, "data"),
    out_feature = None, # None for replacing in_feature
    idx = [[-1],...] # -1 in brackets for keeping dimensions of numpy array
)

#%% Scaler
Scaler = QuantileScaler_eolearn_tdigest.LOAD(os.path.join(config["dir_results"],config["savename_scaler"]))
Scaler.transform = batchify

#%% Model
model = AugmentME.BaseClass(mode="torch")
model.load(os.path.join(config["dir_results"],config["model_savename_bestloss"]),device="cpu")
model.eval()
model.share_memory()

#%% ModelForwardTask
task_model = ModelForwardTask(
    in_feature = (FeatureType.DATA,"data"),
    out_feature = (FeatureType.MASK_TIMELESS,"model_output"),
    model = model,

    in_transform = Scaler,
    out_transform = predict,
    in_torchtype = torch.FloatTensor,

    maxtries=66,
    timeout=0.5,
)

#%% ExportToTiffTask
task_tiff = ExportToTiffTask(
    feature = (FeatureType.MASK_TIMELESS,"model_output"),
    folder = config["dir_tiffs_showcase"],
    date_indices = None,
    band_indices = None,
    crs = None,
    fail_on_missing = True,
    compress = "deflate"
)

#%% save EOPatches
task_save = SaveTask(
    path = config["dir_data"],
    filesystem = None,
    config = config["SHconfig"],
    overwrite_permission = OverwritePermission.OVERWRITE_PATCH,
    compress_level = 2
)

#%% input nodes
node_data = EONode(
    task = task_data,
    inputs = [],
    name = "load Sentinel-1 data"
)

node_pick = EONode(
    task = task_pick,
    inputs = [node_data],
    name = "pick closest observation to reference"
)

#%% inference node
node_model = EONode(
    task = task_model,
    inputs = [node_pick],
    name = "predict water mask"
)

#%% export and save
node_tiff = EONode(
    task = task_tiff,
    inputs = [node_model],
    name = "export GeoTiff of model output"
)

node_save = EONode(
    task = task_save,
    inputs = [node_tiff],
    name = "save EOPatch"
)

#%% workflow
workflow = EOWorkflow.from_endnodes(node_save)

#%% main
from libs.ExecuteME import execute
if __name__=='__main__':
    #%% define workflow arguments
    workflow_args = []
    bbox_list_ = bbox_list_showcase
    for i in range(len(bbox_list_)):
        print(f"\r{i+1}/{len(bbox_list_)}",end="\r")
        try:
            #%%% query available timestamps
            timestamps_ = get_available_timestamps(
                bbox = bbox_list_[i], 
                data_collection = DataCollection.SENTINEL1_IW, 
                time_interval = (config["start_showcase"],config["end_showcase"]), 
                time_difference = dt.timedelta(hours=0,seconds=0),
                maxcc = None, 
                config = config["SHconfig"]
            )
            if timestamps_:
                #print(timestamps_)
                timestamp_ = timestamps_[-1]
                dir_ = f"showcase/eopatch_{i}_{timestamp_.strftime(r'%Y-%m-%dT%H-%M-%S_%Z')}"
                if not os.path.exists(os.path.join(config["dir_data"],dir_)):### and False: ### 
                    workflow_args.append(
                        {
                            node_data: {"bbox":bbox_list_[i],"time_interval":timestamp_},
                            node_tiff: {"filename": f"water_{i}_{timestamp_.strftime(r'%Y-%m-%dT%H-%M-%S_%Z')}"},
                            node_save: {"eopatch_folder":dir_}
                        }
                    )
        except Exception as e:
            print(e,(timestamp_-config["datatimedelta"],timestamp_))
    print()
    
    print(f"Number of downloads/calculations: {len(workflow_args)}")
    
    #%% parallel execution
    print("\n### ExecuteME ###\n")
    devices = Devices(["cuda"],multiprocessing_context="spawn")
    mpkwargs = {
        node_model: {"devices":devices},
    }

    #%%% Parallel
    start_multi = time.time()
    results_multi = execute(
        fun = workflow.execute,
        kwargslist = workflow_args,
        mpkwargs = mpkwargs,
        kwargsmode = None,
        resultsqueue = None,
        NoReturn = True,
        timeout = 1,
        threads = config["threads"],
        checkthreads = True,
        multiprocessing_context = "spawn",
        multiprocessing_mode = "std",
        bequiet = False
    )
    time_multi = time.time()-start_multi

    #%%% results
    print(f"Time Multi:\t\t{time_multi}s")
    
    #%% merge outputs
    config.importME("rasterio.merge.merge")(
        datasets = [os.path.join(config["dir_tiffs_showcase"],dir_) for dir_ in os.listdir(config["dir_tiffs_showcase"]) if "water" in dir_.split("_")],
        dst_path = os.path.join(config["dir_results"],config["savename_showcase_tiff"]),
        dst_kwds = {"compress":"deflate"}
    )
    
    #%% reproject merged outputs
    rasterio_reproject(
        inputfile = os.path.join(config["dir_results"],config["savename_showcase_tiff"]),
        outputfile = os.path.join(config["dir_results"],config["savename_showcase_tiff_reproject"]),
        crs_target = 'EPSG:4326',
        compression = "deflate"
    )