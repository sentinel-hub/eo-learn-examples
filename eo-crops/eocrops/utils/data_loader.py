from eolearn.core import EOPatch, FeatureType, EOTask

import glob
import numpy as np

import os

from eolearn.geometry import ErosionTask
import eocrops.utils.utils as utils
import eocrops.tasks.preprocessing as preprocessing
import copy
###########################################################################################################

class EOPatchDataset:
    def __init__(self,
                 root_dir_or_list, features_data,
                 suffix='S2_L2A', resampling = None,
                 function=np.nanmedian):
        '''

        '''

        import tensorflow as tf
        import tensorflow_datasets as tfds
        global tf
        global tfds

        self.root_dir_or_list = root_dir_or_list
        self.features_data = features_data
        self.suffix = suffix
        self.mask = (FeatureType.MASK_TIMELESS, 'MASK')

        if resampling is None:
            resampling = dict(start = '-01-01', end = '-12-31', day_periods = 8)
        self.resampling = resampling
        self.function = function

        try:
            self.AUTOTUNE = tf.data.AUTOTUNE
        except:
            self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def _instance_tf_ds(self):
        file_pattern = os.path.join(self.root_dir_or_list, '*_' + self.suffix)
        files = glob.glob(file_pattern)
        if len(files) == 0:
            raise ValueError('No file in the root directory ' + self.root_dir_or_list + " ending with " + self.suffix)
        files.sort()
        self.dataset = tf.data.Dataset.from_tensor_slices(files)
        self.vector_dataset = tf.data.Dataset.from_tensor_slices(files)

    @staticmethod
    def _interpolate_feature(eopatch, feature, mask_feature,  **kwargs):
        kwargs['features'] = [feature]
        interp = preprocessing.InterpolateFeatures( **kwargs)
        eopatch = interp.execute(eopatch, mask_feature)
        return eopatch

    def _execute_gap_filling(self, eopatch,
                             resampled_range,
                             copy_features,
                             algorithm = 'linear',
                             mask_feature=None):

        '''Gap filling after data extraction, very useful if did not include it in the data extraction workflow'''
        kwargs = dict(copy_features=copy_features,
                      resampled_range=resampled_range,
                      algorithm = algorithm)

        dico = {}
        for ftype, fname, _, _, _ in self.features_data:
            new_eopatch = copy.deepcopy(eopatch)
            new_eopatch = self._interpolate_feature(new_eopatch, fname, mask_feature, **kwargs)
            dico[fname] = new_eopatch[ftype][fname]

        eopatch['data'] = dico

        return eopatch

    def _prepare_eopatch(self, patch, resampled_range, algorithm = 'linear'):

        polygon_mask = (patch.data_timeless['FIELD_ID']>0).astype(np.int32)
        patch.add_feature(self.mask[0], self.mask[1], polygon_mask.astype(bool))

        erode = ErosionTask(mask_feature=self.mask, disk_radius=1)
        erode.execute(patch)

        patch = self._execute_gap_filling(eopatch=patch,
                                          resampled_range=resampled_range,
                                          algorithm=algorithm,
                                          copy_features=[self.mask])
        return patch


    def _read_patch(self, path, algorithm = 'linear'):
        """ TF op for reading an eopatch at a given path. """
        def _func(path):
            path = path.numpy().decode('utf-8')
            # Load only relevant features
            ################################################################
            patch = EOPatch.load(path)
            year = str(patch.timestamp[0].year)
            start, end = year + self.resampling['start'], year + self.resampling['end']
            resampled_range = (start, end, self.resampling['day_periods'])
            patch = self._prepare_eopatch(patch, resampled_range, algorithm)
            #################################################################
            data = []
            for feat_type, feat_name, _, dtype, _ in self.features_data:
                arr = preprocessing.get_time_series_profile(feature_array = patch[feat_type][feat_name].astype(dtype),
                                                            crop_mask=patch[self.mask[0]][self.mask[1]],
                                                            function = self.function)
                data.append(arr)
            return data

        #################################################################
        out_types = [tf.as_dtype(data[3]) for data in self.features_data]
        data = tf.py_function(_func, [path], out_types)

        out_data = {}
        for f_data, feature in zip(self.features_data, data):
            feat_type, feat_name, out_name, dtype, _ = f_data
            feature.set_shape(feature.get_shape())
            out_data[out_name] = feature

        return out_data

    def _read_vector_data(self, path, column_path, vector_data, features_list):
        """
        TF op for reading an eopatch at a given path.
        It must have a column with the corresponding path
        """
        def _func(path) :
            path = path.numpy().decode('utf-8')
            vector_data_ = vector_data.copy()
            vector_data_ = vector_data_[vector_data_[column_path] == path]

            data = []
            for feat, dtype_ in features_list:
                data.append(np.array(vector_data_[feat].astype(dtype_)))
            return data

        data = tf.py_function(_func, [path], [feat[1] for feat in features_list])
        out_data = {}
        for fname, feature in zip(features_list, data):
            feature.set_shape(feature.get_shape())
            out_data[fname] = feature

        return out_data

    @staticmethod
    def _format_feature(out_feature):
        out_df = [np.concatenate([np.expand_dims(value, axis=1)
                                  if len(value.shape) == 1 else value
                                  for key, value in dicto.items()], axis=-1)
                  for dicto in out_feature]

        out_df = [np.expand_dims(k, axis=0) for k in out_df]
        return np.concatenate(out_df, axis=0)


    def get_eopatch_tfds(self, algorithm = 'linear'):

        self._instance_tf_ds()
        ds_numpy = self.dataset.map(lambda x : self._read_patch(path = x, algorithm=algorithm),
                                    num_parallel_calls=self.AUTOTUNE)
        out_feature = list(ds_numpy)
        return self._format_feature(out_feature)

    def get_vector_tfds(self, vector_data, features_list, column_path):

        self._instance_tf_ds()
        out_labels = list(self.vector_dataset.map(
            lambda path : self._read_vector_data(
                path, column_path, vector_data, features_list),
            num_parallel_calls=self.AUTOTUNE))

        npy_labels = self._format_feature(out_labels)
        npy_labels = npy_labels.reshape(npy_labels.shape[0], npy_labels.shape[-1])

        return npy_labels





