import os
import numpy as np
import shutil
path = '/media/DATA/johann/in_season_yield/data/Sentinel2/EOPatch_V2/training/2020/fold_2'
import shutil
files = [os.path.join(path, k) for k in os.listdir(path) if np.any([x in k for x in ['checkpoint']])]
for f in files:
    try:
        os.remove(f)
    except:
        pass