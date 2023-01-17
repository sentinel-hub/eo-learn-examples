### Michael Engel ### 2022-09-15 ### TDigestTask.py ###
from typing import Optional
from sentinelhub import BBox, DataCollection, SHConfig
from eolearn.core import EOPatch
from eolearn.core import EOTask
import numpy as np
import tdigest as td
from itertools import product

class TDigestTask(EOTask):
    """
    A class for 
    """
    def __init__(self, in_feature, out_feature, mode=None, pixelwise=True, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)
    
        if type(in_feature)==list:
            self.in_feature = in_feature
            if type(out_feature)==list:
                assert len(in_feature)==len(out_feature)
                self.out_feature = out_feature
            else:
                self.out_feature = [out_feature]*len(self.in_feature)
        else:
            self.in_feature = [in_feature]
            if type(out_feature)==list:
                raise RuntimeError("TDigestTask: you cannot get a list of TDigests out of a single feature!")
            else:
                self.out_feature = [out_feature]
        
        if type(pixelwise)==list:
            if len(pixelwise)==len(self.out_feature):
                self.pixelwise = pixelwise
            else:
                assert len(pixelwise)==len(self.out_feature)
                pass
        else:
            self.pixelwise = [pixelwise]*len(self.out_feature)

        self.mode = mode
        pass
    
    def execute(self, eopatch,*args,**kwargs):
        """
        Execute method that computes the TDigest of the chosen features.
        
        :param eopatch: EOPatch which will be saved if suitable
        :type eopatch: EOPatch
        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        """
        
        if self.mode==None or self.mode=='standard':
            for in_feature,out_feature,pixelwise in zip(self.in_feature, self.out_feature, self.pixelwise):
                shape = np.array(eopatch[in_feature].shape)
                if pixelwise:
                    eopatch[out_feature] = np.empty(shape[1:],dtype=object)
                    if len(shape)==4:
                        for i,j,k in product(range(shape[1]),range(shape[2]),range(shape[3])):
                            eopatch[out_feature][i,j,k] = td.TDigest()
                            eopatch[out_feature][i,j,k].batch_update(eopatch[in_feature][:,i,j,k].flatten())
                    elif len(shape)==3:
                        for i,j,k in product(range(shape[0]),range(shape[1]),range(shape[2])):
                            eopatch[out_feature][i,j,k] = td.TDigest()
                            eopatch[out_feature][i,j,k].batch_update(eopatch[in_feature][i,j,k].flatten())
                    else:
                        raise RuntimeError(f"TDigestTask: {self.mode} mode only defined for DATA, DATA_TIMELESS, MASK or MASK_TIMELESS feature types!")
                else:
                    eopatch[out_feature] = np.empty(shape[-1],dtype=object)
                    for k in range(shape[-1]):
                        eopatch[out_feature][k] = td.TDigest()
                        eopatch[out_feature][k].batch_update(eopatch[in_feature][...,k].flatten())
                    
        elif self.mode=='timewise':
            for t,timestamp in enumerate(eopatch["timestamp"]):
                for in_feature,out_feature,pixelwise in zip(self.in_feature, self.out_feature, self.pixelwise):
                    out_feature_type, out_feature_name = out_feature
                    out_feature = (out_feature_type, out_feature_name+f"_{timestamp.strftime('%Y-%m-%d-%H-%M-%S')}")
                    
                    shape = np.array(eopatch[in_feature].shape)
                    if pixelwise:
                        eopatch[out_feature] = np.empty(shape[1:],dtype=object)
                        if len(shape)==4:
                            for i,j,k in product(range(shape[1]),range(shape[2]),range(shape[3])):
                                eopatch[out_feature][i,j,k] = td.TDigest()
                                eopatch[out_feature][i,j,k].batch_update(eopatch[in_feature][t,i,j,k].flatten())
                        else:
                            raise RuntimeError(f"TDigestTask: {self.mode} mode only defined for DATA or MASK feature types with time component!")
                    else:
                        eopatch[out_feature] = np.empty(shape[[0,-1]],dtype=object)
                        for t,k in product(range(shape[0]),range(shape[-1])):
                            eopatch[out_feature][t,k] = td.TDigest()
                            eopatch[out_feature][t,k].batch_update(eopatch[in_feature][t,...,k].flatten())
                    
        elif self.mode=="monthly":
            midx = []
            for m in range(12):
                midx.append(np.array([timestamp.month==m+1 for timestamp in eopatch['timestamp']]))
                
            for in_feature,out_feature,pixelwise in zip(self.in_feature, self.out_feature, self.pixelwise):                
                shape = np.array(eopatch[in_feature].shape)
                if pixelwise:
                    eopatch[out_feature] = np.empty([12,*shape[1:]],dtype=object)
                    if len(shape)==4:
                        for m,i,j,k in product(range(12),range(shape[1]),range(shape[2]),range(shape[3])):
                            eopatch[out_feature][m,i,j,k] = td.TDigest()
                            eopatch[out_feature][m,i,j,k].batch_update(eopatch[in_feature][midx[m],i,j,k].flatten())
                    else:
                        raise RuntimeError(f"TDigestTask: {self.mode} mode only defined for DATA or MASK feature types with time component!")
                else:
                    eopatch[out_feature] = np.empty([12,shape[-1]],dtype=object)
                    for m,k in product(range(12),range(shape[-1])):
                        eopatch[out_feature][m,k] = td.TDigest()
                        eopatch[out_feature][m,k].batch_update(eopatch[in_feature][midx[m],...,k].flatten())
        elif self.mode=="weekly":
            raise NotImplementedError("TDigestTask: {self.mode} mode not implemented yet!")
        elif self.mode=="daily":
            raise NotImplementedError("TDigestTask: {self.mode} mode not implemented yet!")
        elif self.mode=="idx":
            raise NotImplementedError("TDigestTask: {self.mode} mode not implemented yet!")
        elif self.mode=="total":
            for in_feature,out_feature,pixelwise in zip(self.in_feature, self.out_feature, self.pixelwise):
                if pixelwise:
                    raise NotImplementedError("TDigestTask: pixelwise for {self.mode} mode not implemented yet!")
                else:
                    eopatch[out_feature] = td.TDigest()
                    eopatch[out_feature].batch_update(eopatch[in_feature].flatten())
        else:
            raise RuntimeError("TDigestTask: mode not implemented!")
        
        return eopatch
