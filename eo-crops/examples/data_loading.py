import pandas as pd
from eolearn.core import EOPatch, FeatureType, EOTask
import numpy as np
import os
from eocrops.utils import data_loader
#################################################################################

root_dir_or_list = '/home/johann/Documents/EOPatch samples'

dict_df = pd.DataFrame(
    dict(
        labels = [0],
        path = [os.path.join(root_dir_or_list, '726B-JC7987WJFX6011GR-2021_S2_L2A')]
    )
)

features_data = [(FeatureType.DATA, 'BANDS-S2-L2A', 'BANDS-S2-L2A', 'float32', 10),
                 (FeatureType.DATA, 'NDVI', 'NDVI', 'float32', 1),
                 (FeatureType.DATA, 'NDWI', 'NDWI', 'float32', 1),
                 (FeatureType.DATA, 'GNDVI', 'GNDVI', 'float32', 1),
                 (FeatureType.DATA, 'LAI', 'LAI', 'float32', 1),
                 (FeatureType.DATA, 'fapar', 'fapar', 'float32', 1),
                 (FeatureType.DATA, 'Cab', 'Cab', 'float32', 1)]

feature_vector = [
    ('labels', 'float32'),
    ('path', 'string')
]

kwargs_S2 = dict(name_file='X_S2',
                 suffix='S2',
                 features_data=features_data)

pipeline_eopatch_tfds = data_loader.EOPatchDataset(
    root_dir_or_list, features_data,
    suffix='S2_L2A',
    resampling = None,
    function=np.nanmedian
)


npy_eopatch = pipeline_eopatch_tfds.get_eopatch_tfds(algorithm='linear')

npy_labels = pipeline_eopatch_tfds.get_vector_tfds(vector_data=dict_df,
                                                   features_list=feature_vector,
                                                   column_path='path')
