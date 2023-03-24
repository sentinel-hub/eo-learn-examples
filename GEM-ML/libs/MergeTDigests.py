### Michael Engel ### 2022-10-10 ### MergeTDigests.py ###
import numpy as np
import os
import dill as pickle
import tdigest
import uuid
from .ReduceME import reduce_BinaryTree

from typing import Iterable, Tuple

from eolearn.core import EOPatch, FeatureType


#%% define merger for TDigests
class _Merger:
    def __init__(self, feature: Tuple[FeatureType, str], mode: str = "dill", assumetmpfiles: bool = True):
        self.feature = feature
        self.mode = mode
        self.assumetmpfiles = assumetmpfiles
    
    def __call__(self,item1,item2):
        #%%% parse item1
        tdigest1 = self.itemparser(item1)
        
        #%%% parse item2
        tdigest2 = self.itemparser(item2)
        
        #%%% merge tdigests
        tdigest = tdigest1+tdigest2
        
        #%%% return
        if self.mode=="dill":
            tmp = str(uuid.uuid4())+".dill"
            with open(tmp, 'wb') as output:
                pickle.dump(tdigest, output, pickle.HIGHEST_PROTOCOL)
            return tmp
        elif self.mode=="tdigest":
            return tdigest
        else:
            raise NotImplementedError(f"{self.mode} mode not implemented yet!")
            
    def itemparser(self,item):
        #%%% EOPatches
        if os.path.isdir(item):
            tdigest = EOPatch.load(item,features=self.feature)[self.feature]
        #%%% json dicts
        elif type(item)==str:
            with open(item, 'rb') as file:
                tdigest = pickle.load(file)
            if self.assumetmpfiles:
                os.remove(item)
        #%%% TDigest objects
        else:
            tdigest = item
            
        return tdigest

#%% mergeTDigests
def mergeTDigests(
        paths: Iterable[str],
        feature: Tuple[FeatureType, str],
        threads: int = 0,
        checkthreads: bool = True,
        bequiet: bool = False) -> np.ndarray[tdigest.TDigest]:
    Merger = _Merger(feature=feature, mode="dill", assumetmpfiles=True)
    tdigestarray_merged_pickle = reduce_BinaryTree(
        samples = paths,
        reducer = Merger,
        ordered = False,
        queue = None,
        timeout = 2,
        threads = threads,
        checkthreads = checkthreads,
        bequiet = bequiet
    )
    with open(tdigestarray_merged_pickle, 'rb') as file:
        tdigestarray_merged = pickle.load(file)   
    os.remove(tdigestarray_merged_pickle)
    return tdigestarray_merged