### Michael Engel ### 2022-10-17 ### PyTorchTasks.py ###
import torch
from .PyTorchTask import PyTorchTask
from libs.AugmentME_FAUQ import augment_FA_GradientShap, augment_FA_IntegratedGradients, augment_FA_LayerGradCam, augment_FA_ShapleyValueSampling

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
class GradientShapTask(PyTorchTask):
    def __init__(self,
                 in_feature,
                 out_feature,
                 model,
                 funkey = "GradientShap",
                 
                 multiply_by_inputs = False,
                 mode = None, # None,pix,agg
                 mode_kwargs = {},
                 
                 baselines = None,
                 n_samples = 5,
                 stdevs = 0.0,
                 target = 0,
                 return_convergence_delta = False, # actually, you should not use this within eo-learn -> keep False
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 batch_size = None,

                 cudasharememory = False,
                 maxtries = 33,
                 timeout = 3,
                 *args, **kwargs):
        # check if model provides method and attach otherwise
        if not hasattr(model,funkey):
            augment_FA_GradientShap(
                model = model,
                key = funkey,
                
                multiply_by_inputs = multiply_by_inputs,
                
                mode = mode,
                mode_kwargs = mode_kwargs
            )
        
        # additional kwargs for method
        funkwargs = {
            "baselines": baselines ,
            "n_samples": n_samples,
            "stdevs": stdevs,
            "target": target,
            "return_convergence_delta": return_convergence_delta
        }
        
        # init PyTorchTask
        super().__init__(in_feature = in_feature,
                         out_feature = out_feature,
                         model = model,
                         funkey = funkey,
                         funargs = (),
                         funkwargs = funkwargs,
                         in_transform = in_transform,
                         out_transform = out_transform,
                         in_torchtype = in_torchtype,
                         batch_size = batch_size,
                         cudasharememory = cudasharememory,
                         maxtries = maxtries,
                         timeout = timeout,
                         *args, **kwargs)
        
#%% IntegratedGradientsTask
class IntegratedGradientsTask(PyTorchTask):
    def __init__(self,
                 in_feature,
                 out_feature,
                 model,
                 funkey = "IntegratedGradients",
                 
                 multiply_by_inputs = False,
                 mode = None, # None,pix,agg
                 mode_kwargs = {},
                 
                 baselines = None,
                 target = 0,
                 additional_forward_args = None,
                 n_steps = 50,
                 method = 'gausslegendre',
                 internal_batch_size = 1,
                 return_convergence_delta = False, # actually, you should not use this within eo-learn -> keep False
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 batch_size = None,

                 cudasharememory = False,
                 maxtries = 33,
                 timeout = 3,
                 *args, **kwargs):
        # check if model provides method and attach otherwise
        if not hasattr(model,funkey):
            augment_FA_IntegratedGradients(
                model = model,
                key = funkey,
                
                multiply_by_inputs = multiply_by_inputs,
                
                mode = mode,
                mode_kwargs = mode_kwargs
            )
        
        # additional kwargs for method
        funkwargs = {
            "baselines": baselines,
            "target": target,
            "additional_forward_args": additional_forward_args,
            "n_steps": n_steps,
            "method": method,
            "internal_batch_size": internal_batch_size,            
            "return_convergence_delta": return_convergence_delta
        }
        
        # init PyTorchTask
        super().__init__(in_feature = in_feature,
                         out_feature = out_feature,
                         model = model,
                         funkey = funkey,
                         funargs = (),
                         funkwargs = funkwargs,
                         in_transform = in_transform,
                         out_transform = out_transform,
                         in_torchtype = in_torchtype,
                         batch_size = batch_size,
                         cudasharememory = cudasharememory,
                         maxtries = maxtries,
                         timeout = timeout,
                         *args, **kwargs)
        
#%% LayerGradCamTask
class LayerGradCamTask(PyTorchTask):
    def __init__(self,
                 in_feature,
                 out_feature,
                 model,
                 layer, # provide layer object
                 funkey = "LayerGradCam",
                 
                 device_ids = None,
                 mode = None, # None,pix,agg
                 mode_kwargs = {},
                 
                 target = 0,
                 additional_forward_args = None,
                 attribute_to_layer_input = False,
                 relu_attributions = False,
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 batch_size = None,

                 cudasharememory = False,
                 maxtries = 33,
                 timeout = 3,
                 *args, **kwargs):
        # check if model provides method and attach otherwise
        if not hasattr(model,funkey):
            augment_FA_LayerGradCam(
                model = model,
                layer = layer,
                key = funkey,
                
                device_ids = device_ids,
                
                mode = mode,
                mode_kwargs = mode_kwargs
            )
        
        # additional kwargs for method
        funkwargs = {
            "target": target,
            "additional_forward_args": additional_forward_args,
            "attribute_to_layer_input": attribute_to_layer_input,
            "relu_attributions": relu_attributions,
        }
        
        # init PyTorchTask
        super().__init__(in_feature = in_feature,
                         out_feature = out_feature,
                         model = model,
                         funkey = funkey,
                         funargs = (),
                         funkwargs = funkwargs,
                         in_transform = in_transform,
                         out_transform = out_transform,
                         in_torchtype = in_torchtype,
                         batch_size = batch_size,
                         cudasharememory = cudasharememory,
                         maxtries = maxtries,
                         timeout = timeout,
                         *args, **kwargs)
        
#%% ShapleyValueSamplingTask
class ShapleyValueSamplingTask(PyTorchTask):
    def __init__(self,
                 in_feature,
                 out_feature,
                 model,
                 funkey = "ShapleyValueSampling",
                 
                 mode = None, # None,pix,agg
                 mode_kwargs = {},
                 
                 baselines = None,
                 target = 0,
                 additional_forward_args = None,
                 feature_mask = None,
                 n_samples = 25,
                 perturbations_per_eval = 1,
                 show_progress = False, # actually, you should not use this except you're sure about the stdout
                 return_convergence_delta = False, # actually, you should not use this within eo-learn -> keep False
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 batch_size = None,

                 cudasharememory = False,
                 maxtries = 33,
                 timeout = 3,
                 *args, **kwargs):
        # check if model provides method and attach otherwise
        if not hasattr(model,funkey):
            augment_FA_ShapleyValueSampling(
                model = model,
                key = funkey,
                
                mode = mode,
                mode_kwargs = mode_kwargs
            )
        
        # additional kwargs for method
        funkwargs = {
            "baselines": baselines,
            "target": target,
            "additional_forward_args": additional_forward_args,
            "feature_mask": feature_mask,
            "n_samples": n_samples,
            "perturbations_per_eval": perturbations_per_eval,
            "show_progress": show_progress,
        }
        
        # init PyTorchTask
        super().__init__(in_feature = in_feature,
                         out_feature = out_feature,
                         model = model,
                         funkey = funkey,
                         funargs = (),
                         funkwargs = funkwargs,
                         in_transform = in_transform,
                         out_transform = out_transform,
                         in_torchtype = in_torchtype,
                         batch_size = batch_size,
                         cudasharememory = cudasharememory,
                         maxtries = maxtries,
                         timeout = timeout,
                         *args, **kwargs)

#%% LEGACY
from captum.attr import GradientShap, ShapleyValueSampling, LayerGradCam, IntegratedGradients
#%%% GradientShapTask
class LEGACY_GradientShapTask(PyTorchTask): ########################################## NOT WORKING YET DUE TO SOME CUDA MEMORY ISSUES -> solved?
    def __init__(self,
                 in_feature,
                 out_feature,
                 model,
                 
                 in_transform = None,
                 out_transform = None,
                 in_torchtype = torch.FloatTensor,
                 batch_size = None,
                 
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
            multiply_by_inputs=multiply_by_inputs,
            baseline=baseline,
            n_samples=n_samples,
            stdevs=stddevs,
            targetclass=target,
            return_convergence_delta=return_convergence_delta
        )
        
        super().__init__(in_feature = in_feature,
                         out_feature = out_feature,
                         model = self.modelGS,
                         in_transform = in_transform,
                         out_transform = out_transform,
                         in_torchtype = in_torchtype,
                         batch_size = batch_size,
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