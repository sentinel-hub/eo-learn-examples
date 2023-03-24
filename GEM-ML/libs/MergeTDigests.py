### Michael Engel ### 2022-10-10 ### MergeTDigests.py ###
import numpy as np
import os
import dill as pickle
import tdigest
import uuid
from .ReduceME import reduce_BinaryTree

from typing import Iterable, Tuple, Union

from eolearn.core import EOPatch, FeatureType


#%% define merger for TDigests
class _Merger:
    def __init__(self, feature: Tuple[FeatureType, str], mode: str = "dill", assumetmpfiles: bool = True):
        """
        Helper class to merge two TDigest objects.
        :param feature: Feature name of tdigest feature in EOPatch.
        :param mode: How to handle the merged TDigest: write to file or return object
        :param assumetmpfiles: Whether to remove files after loading into memory.
        """
        self.feature = feature
        self.mode = mode
        self.assumetmpfiles = assumetmpfiles
    
    def __call__(self, item1: Union[str, tdigest.TDigest], item2: Union[str, tdigest.TDigest]):
        """
        Performs the merging of two TDigest items. First, the items are loaded and then the merging is applied.
        :param item1: A valid TDigest item.
        :param item2: A valid TDigest item.
        :return: The merged TDigest object or a filepath to a pickled version of it.
        """
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
            
    def itemparser(self, item: Union[str, tdigest.TDigest]):
        """
        Parse items to TDigest objects.
        :param item: Either a string representing the directory of an EOPatch,
        a path to a pickled TDigest object or a TDigest object.
        :return: The parsed TDigest object
        """
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
    """
    Method to merge multiple TDigest objects.
    :param paths: An iterable containing the paths to TDigest items (see above for possible target types).
    :param feature: Feature name of tdigest feature in EOPatch.
    :param threads: Number of threads to use.
    :param checkthreads:
    :param bequiet:
    :return:
    """
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