
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
from scipy.signal import savgol_filter

import os
import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import FeatureType

from eocrops.input import utils_sh as utils_sh
from eocrops.input import sentinel1 as sentinel1
from eocrops.input import sentinel2 as sentinel2
from eocrops.tasks import cmd_otb as cmd_otb
from eocrops.tasks import preprocessing as preprocessing

dir_path = os.path.dirname(os.getcwd())
print(dir_path)
# read microplot data
shapefile_input = gpd.read_file(os.path.join(dir_path, 'eo-crops/examples/layers/POLYGON.shp'))

api = ''
client_id = ''
client_secret = ''
config = utils_sh.config_sentinelhub_cred(api, client_id, client_secret)
# Provide here your planet API key
config.planet_key = ''

# %%

time_period = ('2020-02-15', '2020-08-15')
kwargs = dict(polygon=shapefile_input,
              time_stamp=time_period,
              config=config)



os.getcwd()
warnings.filterwarnings("ignore")
patch = sentinel2.workflow_instructions_S2L2A(**kwargs,
                                              path_out='/home/johann/Documents/patch',  # you can specify here a path to save the EOPatch object
                                              coverage_predicate=0.5,
                                              interpolation={'interpolate': True, 'period_length' : 8})



s1_eopatch = sentinel1.workflow_instructions_S1IW(**kwargs,
                                                  speckle_lee_window = 3,
                                                  orbit_direction = 'ASC',
                                                  backCoeff = 'SIGMA0_ELLIPSOID' )

#%%

#Compute Radar Vegetation Index
VV = s1_eopatch.data['BANDS-S1-IW'][...,0]
VH =  s1_eopatch.data['BANDS-S1-IW'][...,1]
RVI = (4*VH)/(VH+VV)


#%%
from eolearn.core import AddFeatureTask
#Add the feature to the EOPatch
add_rvi = AddFeatureTask((FeatureType.DATA, "RVI"))
add_rvi.execute(eopatch = s1_eopatch, data = RVI[..., np.newaxis])
masking = preprocessing.MaskPixels([ "RVI"])
s1_eopatch = masking.execute(s1_eopatch)

#%%

#Display RVI from SIGMA0 without any speckle filtering
s1_eopatch = masking.execute(s1_eopatch)
plt.figure(figsize=(5, 5))
plt.imshow(s1_eopatch.data['RVI'][15,].squeeze());
plt.axis(False);
plt.show()


#%%

#Can take around 30 seconds
mutlitemp = cmd_otb.MultitempSpeckleFiltering(otb_path = '/home/johann/Documents/OTB/OTB-8.0.1-Linux64/bin',
                                             window = 3)
self = mutlitemp
s1_eopatch = mutlitemp.execute(s1_eopatch)
