import datetime

# eo-learn modules
from eolearn.core import FeatureType
from eolearn.io import SentinelHubDemTask, SentinelHubInputTask
from eolearn.visualization import PlotConfig

# Sentinelhub modules
from sentinelhub import CRS, BBox, DataCollection
from sentinelhub import SHConfig


def eo_learn_get_sentinel2_l1c(bbox_wgs84, start, end):
    # SENTINEL 2 API QUERY
    # Region of interest
    roi_bbox = BBox(bbox=bbox_wgs84, crs=CRS.WGS84)

    # query for Sentinel-2 L1C
    s2_l1c_task = SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L1C,
        bands_feature=(FeatureType.DATA, "L1C_data"),
        additional_data=[(FeatureType.MASK, "dataMask")],
        resolution=10,
        maxcc=0.8,
        time_difference=datetime.timedelta(hours=2),
        max_threads=3,
    )

    # execute query
    eopatch_s2_l1c = s2_l1c_task.execute(bbox=roi_bbox, time_interval=[start, end])

    return eopatch_s2_l1c
