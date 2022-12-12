### Michael Engel ### 2022-10-17 ### PyTorchTasks.py ###
import torch.multiprocessing as mp
import multiprocessing as mp_orig
import torch
from .PyTorchTask import PyTorchTask
from captum.attr import GradientShap, ShapleyValueSampling, LayerGradCam, IntegratedGradients

#%% ModelForwardTask
class ModelForwardTask(PyTorchTask):
    def __init__(self,
                 in_feature,
                 out_feature,
                 model,
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 
                 cudasharememory = False,
                 maxtries = 33,
                 timeout = 3,
                 *args, **kwargs):
        super().__init__(in_feature = in_feature,
                         out_feature = out_feature,
                         model = model,
                         in_transform = in_transform,
                         out_transform = out_transform,
                         in_torchtype = in_torchtype,
                         funkey = "forward",
                         funargs = (),
                         funkwargs = {},
                         cudasharememory = cudasharememory,
                         maxtries = maxtries,
                         timeout = timeout,
                         *args, **kwargs)
  
#%% ModelPredictTask
class ModelPredictTask(PyTorchTask):
    def __init__(self,
                 in_feature,
                 out_feature,
                 model,
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 
                 cudasharememory = False,
                 maxtries = 33,
                 timeout = 3,
                 *args, **kwargs):
        super().__init__(in_feature = in_feature,
                         out_feature = out_feature,
                         model = model,
                         in_transform = in_transform,
                         out_transform = out_transform,
                         in_torchtype = in_torchtype,
                         funkey = "predict",
                         funargs = (),
                         funkwargs = {},
                         cudasharememory = cudasharememory,
                         maxtries = maxtries,
                         timeout = timeout,
                         *args, **kwargs)

#%% GradientShapTask
#%%% main task
class GradientShapTask(PyTorchTask): ########################################## NOT WORKING YET DUE TO SOME CUDA MEMORY ISSUES
    def __init__(self,
                 in_feature,
                 out_feature,
                 model,
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 
                 mode = None, # None,pix,agg
                 modeargs = (),
                 baseline = None,
                 n_samples = 5,
                 stddevs = 0.0,
                 target = 0,
                 multiply_by_inputs = False,
                 return_convergence_delta = False, # actually, you should not use this within eo-learn -> keep False
                 
                 cudasharememory = False,
                 maxtries = 33,
                 timeout = 3,
                 *args, **kwargs):
        
        if mode==None:
            self.model_ = AuxModel_None(model,*modeargs)
        elif mode=="pix":
            self.model_ = AuxModel_pix(model,*modeargs)
        elif mode=="agg":
            self.model_ = AuxModel_agg(model,*modeargs)
            
        self.modelGS = GradientShapModel(
            self.model_,
            multiply_by_inputs=False,
            baseline=None,
            n_samples=5,
            stdevs=0.0,
            targetclass=0,
            return_convergence_delta=False
        )
        
        super().__init__(in_feature = in_feature,
                         out_feature = out_feature,
                         model = self.modelGS,
                         in_transform = in_transform,
                         out_transform = out_transform,
                         in_torchtype = in_torchtype,
                         funkey = None,
                         funargs = (),
                         funkwargs = {},
                         cudasharememory = cudasharememory,
                         maxtries = maxtries,
                         timeout = timeout,
                         *args, **kwargs)

#%%% auxiliary GradientShapTask
class AuxModel_None():
    def __init__(self,model):
        self.model = model
        self.device = next(model.parameters()).device
        
    def __call__(self,x):
        return self.model(x)
    
    def to(self, device):
        self.device = device
        self.model.to(device) # here, the true model is shifted to device
        return self
    
    def parameters(self):
        return self.model.parameters()
    
class AuxModel_pix():
    def __init__(self,model,idx):
        self.model = model
        self.device = next(model.parameters()).device
        self.idx = idx
        
    def __call__(self,x):
        pass
        # out = self.model(x)
        # return out[...,*self.idx]
    
    def to(self, device):
        self.device = device
        self.model.to(device) # here, the true model is shifted to device
        return self
    
    def parameters(self):
        return self.model.parameters()

class AuxModel_agg():
    def __init__(self,model,dims=(2,3)):
        self.model = model
        self.device = next(model.parameters()).device
        self.dims = dims
        
    def __call__(self,x):
        out = self.model(x)
        return out.sum(dim=self.dims)
    
    def to(self, device):
        self.device = device
        self.model.to(device) # here, the true model is shifted to device
        return self
    
    def parameters(self):
        return self.model.parameters()

class GradientShapModel():
    def __init__(self, model, multiply_by_inputs=False,
                 baseline=None, n_samples=5, stdevs=0.0, targetclass=0, return_convergence_delta=False):
        self.model = model
        self.device = self.model.device
        
        # FA args
        self.multiply_by_inputs = multiply_by_inputs
        self.GS = GradientShap(self.model, multiply_by_inputs=multiply_by_inputs)
        
        # FA method args
        self.baseline = baseline
        self.n_samples = n_samples
        self.stdevs = stdevs
        self.targetclass = targetclass
        self.return_convergence_delta = return_convergence_delta
        
    def to(self,device):
        self.device = device
        self.model.to(device) # give device-call back to AuxModel
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def __call__(self,x): ### TODO: release cuda memory if done!!!
        # get baseline
        if self.baseline==None:
            baseline = torch.zeros(x.shape)
        else:
            baseline = self.baseline.copy()
        
        # run FA
        GS_agg_attr, GS_agg_attr_delta = self.GS.attribute(
            inputs = x.to(self.device),
            baselines = baseline.to(self.device),
            n_samples = self.n_samples,
            stdevs = self.stdevs,
            target = self.targetclass,
            return_convergence_delta = True
        )
        
        # return FA
        if self.return_convergence_delta:
            return GS_agg_attr, GS_agg_attr_delta
        else:
            return GS_agg_attr


#%% Devices
def Devices(devices,multiprocessing_context="spawn"):
    if multiprocessing_context==None:
        context = mp_orig.get_context()
    elif type(multiprocessing_context)==str:
        context = mp_orig.get_context(multiprocessing_context)
    else:
        context = multiprocessing_context
        
    queue = context.Queue()
    [queue.put(device) for device in devices]
    return queue