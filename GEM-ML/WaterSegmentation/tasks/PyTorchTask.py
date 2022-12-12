### Michael Engel ### 2022-10-16 ### PyTorchTask.py ###
import torch
from queue import Empty
import os
import numpy as np
import logging
from eolearn.core import EOTask

class PyTorchTask(EOTask):
    #%% init
    def __init__(self,
                 in_feature,
                 out_feature,
                 
                 model,
                 funkey = None,
                 funargs = (),
                 funkwargs = {},
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 
                 cudasharememory = False,
                 maxtries = 33,
                 timeout = 3,
                 *args, **kwargs):
        """
        This Task runs a chosen method of a PyTorch model for the given input feature(s) and stores the result in the output feature.
        The Task takes care of the number of devices in a thread- and processsafe manner!
        Please note that a list of output features has to match the model output.
        That is, if the model returns no list and you give a list of output features, the execute will throw an error.
        If your model returns a list of tensors, however, you have to provide a list of output features!
        """
        #%%% init super
        super().__init__(*args, **kwargs)
        
        #%%% features
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        #%%% transforms
        self.in_transform = in_transform
        self.out_transform = out_transform
        
        #%%% torchstuff
        self.in_torchtype = in_torchtype
        
        #%%% cuda stuff
        self.cudasharememory = cudasharememory
        if self.cudasharememory:
            raise NotImplementedError("PyTorchTask: Shared Cuda tensors are not supported yet! Would probably fasten things a lot so if you have ideas, let me know!")
            # [self.models.put(model.to(device)) for device in devices] ### TODO: find proper way with CUDA!!!
        
        #%%% initialise model
        self.model = model.to("cpu") # Task should always stay on CPU
        self.funkey = funkey
        self.funargs = funargs
        self.funkwargs = funkwargs
        
        #%%% some parameters for getting the model
        self.maxtries = maxtries
        self.timeout = timeout
    
    #%% execute
    def execute(self, eopatch, devices, *args,**kwargs):
        """
        Execute method that computes the output of the chosen function of the given PyTorch model.
        
        :param eopatch: EOPatch to compute the output for
        :type eopatch: EOPatch
        """
        
        #%%% get input
        if type(self.in_feature)==list:
            x = []
            for in_feature_ in self.in_feature:
                x.append(eopatch[in_feature_])
        else:
            x = eopatch[self.in_feature]
        
        #%%% transform input
        if self.in_transform is not None:
            if type(x)==list:
                if (type(self.in_transform)==list or type(self.in_transform)==np.ndarray) and len(self.in_transform)==len(x):
                    x = [torch.from_numpy(in_transform(x_i)).type(self.in_torchtype) for in_transform,x_i in zip(self.in_transform,x)]
                else:
                    raise RuntimeError('PyTorchTask.execute: in_transform has to be a list or numpy.ndarray of the same length as in_feature')    
            elif not (type(self.in_transform)==list or type(self.in_transform)==np.ndarray):
                x = torch.from_numpy(self.in_transform(x)).type(self.in_torchtype)
            else:
                raise RuntimeError('PyTorchTask.execute: either both in_transform and in_feature have to be lists or numpy.ndarrays of equal length or none of them') 
        else:
            if type(x)==list:
                x = [torch.from_numpy(x_i).type(self.in_torchtype) for x_i in x]
            else:
                x = torch.from_numpy(x).type(self.in_torchtype)
        
        #%%% query output
        counter=0
        while True:
            #%%%% get model
            try:
                dev = devices.get(timeout=self.timeout) # timeout very important (see computation not locked)
                model = self.model.to(dev)
            except Empty:
                counter = counter+1
                if counter==self.maxtries:
                    raise RuntimeError("PyTorchTask process reached timeout and maximum number of trials - will stop!")
                    break
                else:
                    continue
            try:
                #%%%% get fun
                if self.funkey==None:
                    fun = model.__call__
                else:
                    fun = getattr(model,self.funkey)
            
            
                #%%%% compute output
                if type(x)==list:
                    out = fun([x_.to(next(model.parameters()).device) for x_ in x],*self.funargs,**self.funkwargs)
                else:
                    out = fun(x.to(next(model.parameters()).device),*self.funargs,**self.funkwargs)
                    
                #%%%% move output to CPU
                if type(out)==list:
                    out_cpu = [out_.cpu().detach().numpy() for out_ in out]
                else:
                    out_cpu = out.cpu().detach().numpy()
                
                #%%%% clean cache of GPU
                del(x)
                del(out)
                del(fun)
                del(model) ### keep?
                
            except Exception as e:
                print(e)
            finally:
                #%%%% put model back and break
                self.model.to("cpu") # Task should always stay on CPU
                torch.cuda.empty_cache()
                devices.put(dev)
                break
        
        #%%% transform output on CPU
        if self.out_transform is not None:
            if type(out_cpu)==list:
                if (type(self.out_transform)==list or type(self.out_transform)==np.ndarray) and len(self.out_transform)==len(out_cpu):
                    out_cpu = [out_transform(out_i) for out_transform,out_i in zip(self.out_transform,out_cpu)]
                else:
                    raise RuntimeError('PyTorchTask.execute: out_transform has to be a list or numpy.ndarray of the same length as out_feature/model-output')    
            elif not (type(self.out_transform)==list or type(self.out_transform)==np.ndarray):
                out_cpu = self.out_transform(out_cpu)
            else:
                raise RuntimeError('PyTorchTask.execute: either both out_transform and out_feature/model-output have to be lists or numpy.ndarrays of equal length or none of them') 
        else:
            out_cpu = out_cpu

        #%%% set output
        if type(self.out_feature)==list:
            for i,out_feature_ in enumerate(self.out_feature):
                eopatch[out_feature_] = out_cpu[i]
        else:
            eopatch[self.out_feature] = out_cpu
        
        #%%% return
        return eopatch
    
    #%% call
    def __call__(self,*args,**kwargs):
        return self.execute(*args,**kwargs)