### 2021-05-16 ### Michael Engel ### Dataset_eolearn.py ###
from typing import Tuple, Iterable, Optional, Callable, Union

import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable as V
import torch
from eolearn.core import EOPatch, FeatureType

#%% main
class Dataset_eolearn(Dataset):
    #%%% initialize
    def __init__(self, paths: Iterable[str],
                 feature_data: Tuple[FeatureType, str] = (FeatureType.DATA, "data"),
                 feature_reference: Tuple[FeatureType, str] = (FeatureType.MASK, "reference"),
                 feature_mask: Tuple[FeatureType, str] = None,
                
                 transform_data: Optional[Union[Callable, Iterable[Callable]]] = None,
                 transform_reference: Optional[Union[Callable, Iterable[Callable]]] = None,
                 transform_mask: Optional[Union[Callable, Iterable[Callable]]] = None,
                 
                 return_idx: bool = False,
                 return_path: bool = False,

                 torchdevice: Optional[torch.device] = None,
                 torchtype_data: Union[torch.dtype, str] = torch.FloatTensor,
                 torchtype_reference: Union[torch.dtype, str] = torch.LongTensor,
                 torchtype_mask: Union[torch.dtype, str] = torch.LongTensor,
                 ):
        """
        Dataset class enabling the use of PyTorch with data obtained through eolearn.
        :param paths: An iterable of paths to the folders containing the EOPatch data.
        :param feature_data: The feature name of the data feature of the EOPatch.
        :param feature_reference: The feature name of the reference feature of the EOPatch.
        :param feature_mask: The feature name of an optional mask feature of the EOPatch.
        :param transform_data: Transforms applied to the data after loading.
        :param transform_reference: Transforms applied to the reference after loading.
        :param transform_mask: Transforms applied to the mask after loading.
        :param return_idx: Whether to return sample index together with sample data.
        :param return_path: Whether to return sample path together with sample data.
        :param torchdevice: A torch device to send the data to.
        :param torchtype_data: Torch data type to cast the data feature to after transforms.
        :param torchtype_reference: Torch data type to cast the reference feature to after transforms.
        :param torchtype_mask: Torch data type to cast the mask feature to after transforms.
        """
        super().__init__()
        
        #%%%% paths
        self.paths = paths
        
        #%%%% features
        self.feature_data = feature_data
        self.feature_reference = feature_reference
        self.feature_mask = feature_mask

        self.query = [self.feature_data]
        if self.feature_reference!=None:
            self.query.append(self.feature_reference)
        if self.feature_mask!="ones" and self.feature_mask!=None:
            self.query.append(self.feature_mask)
            
        #%%%% transformations of data and labels (e.g. scalers, label mappings)
        self.transform_data = transform_data
        self.transform_reference = transform_reference
        self.transform_mask = transform_mask
        
        #%%%% what to return besides data, reference or mask
        self.return_idx = return_idx
        self.return_path = return_path
        
        #%%%% torch stuff
        self.torchdevice = torchdevice
        self.torchtype_data = torchtype_data
        self.torchtype_reference = torchtype_reference
        self.torchtype_mask = torchtype_mask

    #%%% query length of dataset
    def __len__(self):
        return len(self.paths)

    #%%% get item or batch of dataset
    def __getitem__(self, idx):
        #%%%% determine item or batch (slice)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx)==list or type(idx)==slice:
            if type(idx)==slice:
                idx = range(idx.start if idx.start else 0, idx.stop if idx.stop else len(self.coords),idx.step if idx.step else 1)
            squeeze = False
        else:
            idx = [idx]
            squeeze = True

        #%%%% load data, reference and mask            
        data = []
        reference = []
        mask = []
        for idx_ in idx:
            eopatch = EOPatch.load(self.paths[idx_],features=self.query)
            
            #%%%%% data
            if type(self.feature_data)==list or type(self.feature_data)==np.ndarray:
                data_ = [eopatch[feature] for feature in self.feature_data]
            else:
                data_ = eopatch[self.feature_data]
            
            #%%%%% reference
            if self.feature_reference!=None:
                if type(self.feature_reference)==list or type(self.feature_reference)==np.ndarray:
                    reference_ = [eopatch[feature] for feature in self.feature_reference]
                else:
                    reference_ = eopatch[self.feature_reference]
                
            #%%%%% mask
            if self.feature_mask=='ones':
                if self.feature_reference==None:
                    raise RuntimeError("Dataset_eolearn: If you want to have an all-true mask for your reference, you need to provide such a reference!")
                else:
                    if type(self.feature_reference)==list or type(self.feature_reference)==np.ndarray:
                        mask_ = [np.ones(eopatch[feature].shape) for feature in self.feature_reference]
                    else:
                        mask_ = np.ones(eopatch[self.feature_reference].shape)
            elif self.feature_mask!=None:
                if type(self.feature_mask)==list or type(self.feature_mask)==np.ndarray:
                    mask_ = [eopatch[feature] for feature in self.feature_mask]
                else:
                    mask_ = eopatch[self.feature_mask]
            
            #%%%%% apply data transform
            if self.transform_data is not None:
                if type(data_)==list:
                    if (type(self.transform_data)==list or type(self.transform_data)==np.ndarray) and len(self.transform_data)==len(self.feature_data):
                        data.append([torch.from_numpy(transform(data_i)).type(self.torchtype_data) for transform,data_i in zip(self.transform_data,data_)])
                    else:
                        raise RuntimeError('Dataset_eolearn: transform_data has to be a list or numpy.ndarray of the same length as feature_data')    
                elif not (type(self.transform_data)==list or type(self.transform_data)==np.ndarray):
                    data.append(torch.from_numpy(self.transform_data(data_)).type(self.torchtype_data))
                else:
                    raise RuntimeError('Dataset_eolearn: either both transform_data and feature_data have to be lists or numpy.ndarrays of equal length or none of them') 
            else:
                if type(data_)==list:
                    data.append([torch.from_numpy(data_i).type(self.torchtype_data) for data_i in data_])
                else:
                    data.append(torch.from_numpy(data_).type(self.torchtype_data))

            #%%%%% apply reference transform
            if self.feature_reference!=None:
                if self.transform_reference is not None:
                    if type(reference_)==list:
                        if (type(self.transform_reference)==list or type(self.transform_reference)==np.ndarray) and len(self.transform_reference)==len(self.feature_reference):
                            reference.append([torch.from_numpy(transform(reference_i)).type(self.torchtype_reference) for transform,reference_i in zip(self.transform_reference,reference_)])
                        else:
                            raise RuntimeError('Dataset_eolearn: transform_reference has to be a list or numpy.ndarray of the same length as feature_reference')    
                    elif not (type(self.transform_reference)==list or type(self.transform_reference)==np.ndarray):
                        reference.append(torch.from_numpy(self.transform_reference(reference_)).type(self.torchtype_reference))
                    else:
                        raise RuntimeError('Dataset_eolearn: either both transform_reference and feature_reference have to be lists or numpy.ndarrays of equal length or none of them') 
                else:
                    if type(reference_)==list:
                        reference.append([torch.from_numpy(reference_i).type(self.torchtype_reference) for reference_i in reference_])
                    else:
                        reference.append(torch.from_numpy(reference_).type(self.torchtype_reference))
                    
            #%%%%% apply mask transform
            if self.feature_mask!=None:
                if self.transform_mask is not None:
                    if type(mask_)==list:
                        if (type(self.transform_mask)==list or type(self.transform_mask)==np.ndarray) and len(self.transform_mask)==len(self.feature_mask):
                            mask.append([torch.from_numpy(transform(mask_i)).type(self.torchtype_mask) for transform,mask_i in zip(self.transform_mask,mask_)])
                        else:
                            raise RuntimeError('Dataset_eolearn: transform_mask has to be a list or numpy.ndarray of the same length as feature_mask')    
                    elif not (type(self.transform_mask)==list or type(self.transform_mask)==np.ndarray):
                        mask.append(torch.from_numpy(self.transform_mask(mask_)).type(self.torchtype_mask))
                    else:
                        raise RuntimeError('Dataset_eolearn: either both transform_mask and feature_mask have to be lists or numpy.ndarrays of equal length or none of them') 
                else:
                    if type(mask_)==list:
                        mask.append([torch.from_numpy(mask_i).type(self.torchtype_mask) for mask_i in mask_])
                    else:
                        mask.append(torch.from_numpy(mask_).type(self.torchtype_mask))

        #%%%% stack to batches
        if type(data_)==list:
            data = [torch.stack([dat[i] for dat in data],dim=0) for i in range(len(data_))]
        else:
            data = torch.stack(data,dim=0)
        
        if self.feature_reference!=None:
            if type(reference_)==list:
                reference = [torch.stack([ref[i] for ref in reference],dim=0) for i in range(len(reference_))]
            else:
                reference = torch.stack(reference,dim=0)
        
        if self.feature_mask!=None:
            if type(mask_)==list:
                mask = [torch.stack([ma[i] for ma in mask],dim=0) for i in range(len(mask_))]
            else:
                mask = torch.stack(mask,dim=0)

        #%%%% squeeze
        if squeeze:
            if type(data_)==list:
                data = [dat.squeeze(0) for dat in data]
            else:
                data = data.squeeze(0)
            
            if self.feature_reference!=None:
                if type(reference_)==list:
                    reference = [ref.squeeze(0) for ref in reference]
                else:
                    reference = reference.squeeze(dim=0)
            
            if self.feature_mask!=None:
                if type(mask_)==list:
                    mask = [ma.squeeze(0) for ma in mask]
                else:
                    mask = mask.squeeze(dim=0)
        else:
            if type(data_)==list:
                data = [V(dat) for dat in data]
            else:
                data = V(data)
            
            if self.feature_reference!=None:
                if type(reference_)==list:
                    reference = [V(ref) for ref in reference]
                else:
                    reference = V(reference)
            
            if self.feature_mask!=None:
                if type(mask_)==list:
                    mask = [V(ma) for ma in mask]
                else:
                    mask = V(mask)

        #%%%% move to device
        if self.torchdevice is not None:
            if type(data_)==list:
                [dat.to(self.torchdevice) for dat in data]
            else:
                data.to(self.torchdevice)
                
            if self.feature_reference!=None:
                if type(reference_)==list:
                    [ref.to(self.torchdevice) for ref in reference]
                else:
                    reference.to(self.torchdevice)
            
            if self.feature_mask!=None:
                if type(mask_)==list:
                    [ma.to(self.torchdevice) for ma in mask]
                else:
                    mask.to(self.torchdevice)

        #%%%% return
        # print(data.device,reference.device)
        out = [data]
        if self.feature_reference!=None:
            out.append(reference)
            
        if self.feature_mask!=None:
            out.append(mask)
            
        if self.return_idx:
            out.append(idx)
            
        if self.return_path:
            out.append([self.paths[idx_] for idx_ in idx])
        
        return out

    #%%% utility for loading a feature
    def _load(self,eopatch,feature):
        if type(feature)==list or type(feature)==np.ndarray:
            stuff = [eopatch[feature_] for feature_ in feature]
        else:
            stuff = eopatch[feature]
        return stuff
    
    #%%% utility for transforming a feature
    def _transform(self,stuff,stuff_,transform,name):
        if transform is not None:
            if type(stuff_)==list:
                if (type(transform)==list or type(transform)==np.ndarray) and len(transform)==len(self.feature_data):
                    stuff.append([torch.from_numpy(transform(stuff_i)).type(self.torchtype_data) for transform,stuff_i in zip(transform,stuff_)])
                else:
                    raise RuntimeError(f'Dataset_eolearn: transform_{name} has to be a list or numpy.ndarray of the same length as feature_{name}')    
            elif not (type(transform)==list or type(transform)==np.ndarray):
                stuff.append(torch.from_numpy(transform(stuff_)).type(self.torchtype_data))
            else:
                raise RuntimeError('Dataset_eolearn: either both transform_{name} and feature_{name} have to be lists or numpy.ndarrays of equal length or none of them') 
        else:
            if type(stuff_)==list:
                stuff.append([torch.from_numpy(stuff_i).type(self.torchtype_data) for stuff_i in stuff_])
            else:
                stuff.append(torch.from_numpy(stuff_).type(self.torchtype_data))
        
        pass
            
#%% torchify
def torchify(array):
    return np.moveaxis(array, -1, 0)

class Torchify():
    def __init__(self,squeeze=False):
        self.squeeze = squeeze
    def __call__(self,array):
        array_ = torchify(array)
        
        if self.squeeze==True:
            return array_.squeeze()
        elif type(self.squeeze)==int:
            return array_.squeeze(self.squeeze)
        else:
            return array_