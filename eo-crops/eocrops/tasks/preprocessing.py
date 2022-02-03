import numpy as np
from eolearn.geometry import VectorToRasterTask
from eolearn.core import FeatureType, EOTask
import sentinelhub
from eolearn.features.interpolation import LinearInterpolationTask, CubicInterpolationTask
import copy

import eocrops.utils.utils as utils
from eolearn.geometry.morphology import ErosionTask


class PolygonMask(EOTask) :
    """
    EOTask that performs rasterization from an input shapefile into :
    - data_timeless feature 'FIELD_ID' (0 nodata; 1,...,N for each observation of the shapefile ~ object IDs)
    - mask_timeless feature 'polygon_mask' (0 if pixels outside the polygon(s) from the shapefile, 1 otherwise

    Parameters
    ----------
    geodataframe : TYPE GeoDataFrame
        Input geodataframe read as GeoDataFrame, each observation represents a polygon (e.g. fields)
    new_feature_name : TYPE string
        Name of the new features which contains clustering_task predictions

    Returns
    -------
    EOPatch
    """

    def __init__(self, geodataframe) :
        self.geodataframe = geodataframe

    def execute(self, eopatch):

        # Check CRS and transform into UTM
        self.geodataframe = utils.check_crs(self.geodataframe)

        # Get an ID for each polygon from the input shapefile
        self.geodataframe['FIELD_ID'] = list(range(1, self.geodataframe.shape[0] + 1))

        if self.geodataframe.shape[0]>1 :
            bbox = self.geodataframe.geometry.total_bounds
            polygon_mask = sentinelhub.BBox(bbox=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], crs=self.geodataframe.crs)
            self.geodataframe['MASK'] = polygon_mask.geometry
        else :
            self.geodataframe['MASK'] = self.geodataframe['geometry']

        self.geodataframe['polygon_bool'] = True

        rasterization_task = VectorToRasterTask(self.geodataframe, (FeatureType.DATA_TIMELESS, "FIELD_ID"),
                                                values_column="FIELD_ID", raster_shape=(FeatureType.MASK, 'IS_DATA'),
                                                raster_dtype=np.uint16)
        eopatch = rasterization_task.execute(eopatch)

        rasterization_task = VectorToRasterTask(self.geodataframe,
                                                (FeatureType.MASK_TIMELESS, "MASK"),
                                                values_column="polygon_bool", raster_shape=(FeatureType.MASK, 'IS_DATA'),
                                                raster_dtype=np.uint16)
        eopatch = rasterization_task.execute(eopatch)

        eopatch.mask_timeless['MASK'] = eopatch.mask_timeless['MASK'].astype(bool)

        return eopatch


class MaskPixels(EOTask):
    def __init__(self, features, fname = 'MASK') :
        '''
        Parameters
        ----------
        feature (list): of features in data and/or data_timeless
        fname (str): name of the mask
        '''
        self.features = features
        self.fname = fname

    @staticmethod
    def _filter_array(patch, ftype, fname, mask) :
        ivs = patch[ftype][fname]

        arr0 = np.ma.array(ivs,
                           dtype=np.float32,
                           mask=(1-mask).astype(bool),
                           fill_value=np.nan)

        arr0 = arr0.filled()
        patch[ftype][fname] = arr0

        return patch

    def execute(self, patch, erosion = 0):
        copy_patch = copy.deepcopy(patch)
        times = len(patch.timestamp)
        if erosion:
            erode = ErosionTask(mask_feature=(FeatureType.MASK_TIMELESS, self.fname),
                                disk_radius=erosion)
            erode.execute(copy_patch)

        crop_mask = copy_patch["mask_timeless"][self.fname]
        # Filter the pixels of each features
        for index in self.features :
            if index in list(patch.data.keys()) :
                ftype = 'data'
                shape = patch[ftype][index].shape[-1]
                mask = crop_mask.reshape(1, crop_mask.shape[0], crop_mask.shape[1], 1)
                mask = [mask for k in range(times)]
                mask = np.concatenate(mask, axis=0)
                mask = [mask for k in range(shape)]
                mask = np.concatenate(mask, axis=-1)
            else :
                ftype = 'data_timeless'
                mask = crop_mask
            patch = self._filter_array(patch, ftype, index, mask)

        return patch



class InterpolateFeatures(EOTask):
    def __init__(self, resampled_range, algorithm = 'linear', copy_features = None,
                 features = None):
        self.resampled_range = resampled_range
        self.algorithm = algorithm
        self.features = features
        self.copy_features = copy_features
        if self.features is None :
            self.features = ['BANDS-S2-L2A', 'fapar', 'LAI', 'Cab', 'NDVI', 'EVI2', 'CVI', 'NDWI', 'GNDVI', 'GVMI',
                             'SLAVI', 'NDDI', 'VSDI', 'ECNorm']

    def _interpolate_feature(self, eopatch, feature, mask_feature):

        kwargs = dict(mask_feature=mask_feature,
                      copy_features=self.copy_features,
                      resample_range=self.resampled_range,
                      feature =  [(FeatureType.DATA, feature)],
                      bounds_error=False)

        if self.algorithm=='linear' :
            interp = LinearInterpolationTask(
                parallel=True,
                **kwargs
            )
        elif self.algorithm=='cubic' :
            interp = CubicInterpolationTask(
                **kwargs
            )
        eopatch = interp.execute(eopatch)
        return eopatch


    def execute(self, eopatch):

        '''Gap filling after data extraction, very useful if did not include it in the data extraction workflow'''

        dico = {}
        mask_feature = None
        if 'VALID_DATA' in list(eopatch.mask.keys()):
            mask_feature = (FeatureType.MASK, 'VALID_DATA')

        for feature in self.features :
            new_eopatch = copy.deepcopy(eopatch)
            new_eopatch = self._interpolate_feature(new_eopatch, feature, mask_feature)
            dico[feature] = new_eopatch.data[feature]

        eopatch['data'] = dico
        t, h, w, _ = eopatch.data[feature].shape
        eopatch.timestamp = new_eopatch.timestamp
        eopatch['mask']['IS_DATA'] = np.zeros((t, h, w, 1))+1
        eopatch['mask']['VALID_DATA'] = (np.zeros((t, h, w, 1))+1).astype(bool)
        if "CLM" in eopatch.mask.keys():
            eopatch.remove_feature(FeatureType.MASK, "CLM")

        return eopatch


def get_time_series_profile(patch, variable, mask_name, function = np.mean):
    crop_mask = patch['mask_timeless'][mask_name].squeeze()
    var = patch['data'][variable]
    shape = var.shape[-1]
    times = len(patch.timestamp)
    # Transform mask from 3D to 4D
    mask = crop_mask.reshape(1, crop_mask.shape[0], crop_mask.shape[1], 1)
    mask = [mask for k in range(times)]
    mask = np.concatenate(mask, axis=0)
    #######################
    mask = [mask for k in range(shape)]
    mask = np.concatenate(mask, axis=-1)
    ########################
    a = np.ma.array(var, mask=(1-(mask==1)).astype(bool))
    ts_mean = np.ma.apply_over_axes(function, a, [1, 2])
    ts_mean = ts_mean.reshape(ts_mean.shape[0], ts_mean.shape[-1])
    ts_mean = ts_mean.data
    return {variable: ts_mean[:, n].flatten() for n in range(shape)}


