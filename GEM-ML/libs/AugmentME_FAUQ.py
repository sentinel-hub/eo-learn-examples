### Michael Engel ### 2022-10-17 ### AugmentME_FAUQ.py ###
from functools import wraps
import captum.attr as capttr# GradientShap, ShapleyValueSampling, LayerGradCam, IntegratedGradients
from .AugmentME import augmentME, decorateME, attachME

#%% basic
def augment_FA(model, key="GradientShap",
               FA_method="GradientShap", FA_kwargs={},
               mode="agg", mode_kwargs={"dims":(2,3)},
               attribution_fun="attribute"):
    if mode==None:
        decorator = AuxModel_None
    elif mode=="pix":
        decorator = AuxModel_pix
    elif mode=="agg":
        decorator = AuxModel_agg
    else:
        raise RuntimeError(f"mode {mode} not defined yet!")
    
    famodel = capttr.__getattribute__(FA_method)(decorator(model,**mode_kwargs),**FA_kwargs)
    attachME(model,key,famodel.__getattribute__(attribution_fun))
    return True

#%% specific
def augment_FA_GradientShap(
    model,
    key = "GradientShap",
    
    multiply_by_inputs = False,
    
    mode = "agg",
    mode_kwargs = {"dims":(2,3)}
):
    augment_FA(model,key=key,FA_method="GradientShap",FA_kwargs={"multiply_by_inputs":multiply_by_inputs},mode=mode,mode_kwargs=mode_kwargs,attribution_fun="attribute")
    return True

def augment_FA_IntegratedGradients(
    model,
    key = "IntegratedGradients",
    
    multiply_by_inputs = False,
    
    mode = "agg",
    mode_kwargs = {"dims":(2,3)}
):
    augment_FA(model,key=key,FA_method="IntegratedGradients",FA_kwargs={"multiply_by_inputs":multiply_by_inputs},mode=mode,mode_kwargs=mode_kwargs,attribution_fun="attribute")
    return True
    
def augment_FA_LayerGradCam(
    model,
    layer,
    key = "LayerGradCam",
    
    device_ids = None,
    
    mode = "agg",
    mode_kwargs = {"dims":(2,3)}
):
    augment_FA(model,key=key,FA_method="IntegratedGradients",FA_kwargs={"layer":layer,"device_ids":device_ids},mode=mode,mode_kwargs=mode_kwargs,attribution_fun="attribute")
    return True

def augment_FA_ShapleyValueSampling(
    model,
    key = "ShapleyValueSampling",
    
    mode = "agg",
    mode_kwargs = {"dims":(2,3)}
):
    augment_FA(model,key=key,FA_method="ShapleyValueSampling",FA_kwargs={},mode=mode,mode_kwargs=mode_kwargs,attribution_fun="attribute")
    return True


#%% auxiliary
def DecoratorNone(model,**kwargs):
    return model

def DecoratorPix(model,idx):
    @wraps(model)
    def inner(*args,**kwargs):
        out = model(*args,**kwargs)
        return out[idx]
    return inner

def DecoratorAgg(model,dims=(2,3)):
    @wraps(model)
    def inner(*args,**kwargs):
        out = model(*args,**kwargs)
        return out.sum(dim=dims)
    return inner

class AuxModel_None():
    def __init__(self,model):
        self.model = model
        
    def __call__(self,x):
        return self.model(x)

    
class AuxModel_pix():
    def __init__(self,model,idx):
        self.model = model
        self.idx = idx
        
    def __call__(self,x):
        pass
        # out = self.model(x)
        # return out[...,*self.idx]

class AuxModel_agg():
    def __init__(self,model,dims=(2,3)):
        self.model = model
        self.dims = dims
        
    def __call__(self,x):
        out = self.model(x)
        return out.sum(dim=self.dims)