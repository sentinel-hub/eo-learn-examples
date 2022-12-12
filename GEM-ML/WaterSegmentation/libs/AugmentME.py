### Michael Engel ### 2022-02-22 ### AugmentME.py ###
import dill as pickle
import os
import uuid
import torch

#%% augmentME
def augmentME(model,key,method): # based on this functionality everything else works in this library
    # model.__setattr__(key,lambda *args, **kwargs: method(model,*args,**kwargs)) # add method to class and name it as key
    model.__setattr__(key,_AugmentME(model,method)) # add method to class and name it as key
    
class _AugmentME():
    def __init__(self,model,method):
        self.model = model
        self.method = method
    def __call__(self,*args,**kwargs):
        return self.method(self.model,*args,**kwargs)
        
def decorateME(model,key,decorator,args=(),kwargs={}):
    setattr(model,key,decorator(getattr(model,key),*args,**kwargs)) # decorate method key of class with decorator
    
#%%% add methods
def augment_addmethod(model,key='_addmethod'):
    augmentME(model,key=key,method=augmentME)

#%%% storing and loading
def augment_IO(model,savekey='save',loadkey='load',mode='dill'):     
    augment_save(model,key=savekey,mode=mode)
    augment_load(model,key=loadkey,mode=mode)
    return True

def augment_save(model,key='save',mode='dill'):     
    if mode=='pure' or mode=='dill' or mode==None:       
        augmentME(model,key,saveME)
    elif mode=='torch':
        augmentME(model,key,saveME_torch)
    else:
        print(f'MODE {mode} NOT IMPLEMENTED!')
        return False
    return True

def augment_load(model,key='load',mode='dill'):     
    if mode=='pure' or mode=='dill' or mode==None:       
        augmentME(model,key,loadME)
    elif mode=='torch':
        augmentME(model,key,loadME_torch)
    else:
        print(f'MODE {mode} NOT IMPLEMENTED!')
        return False
    return True

def augment_checkpoint(model,key='save_checkpoint',mode='torch'):
    if mode=='pure' or mode=='dill' or mode==None:
        print(f'MODE {mode} NOT IMPLEMENTED!')
    elif mode=='torch':
        augmentME(model,key,saveME_checkpoint)
    else:
        print(f'MODE {mode} NOT IMPLEMENTED!')
        return False
    return True

#%%% parameters
def augment_Ntheta(model,key='get_Ntheta'):
    augmentME(model,key,getME_Ntheta)
    return True

def augment_get_theta(model,key='get_theta'):
    augmentME(model,key,getME_theta)
    return True

def augment_set_theta(model,key='set_theta'):
    augmentME(model,key,setME_theta)
    return True

def augment_get_thetaindex(model,key='get_thetaindex'):
    augmentME(model,key,getME_thetaindex)
    return True

#%%% gradients
def augment_gradient(model,key='get_gradient',mode=None):
    if mode==None:
        augmentME(model,key,getME_gradient)
    elif mode=='dL_dTheta':
        augmentME(model,key,getME_gradient_dL_dTheta)
    elif mode=='dOutput_dInput':
        pass
    
#%%% feature attribution

#%% AugmentME base class for loading augmented models
class BaseClass():
    def __init__(self,mode='dill'):
        augment_IO(self,savekey='save',loadkey='load',mode=mode)
        pass

def LOAD(savename,mode="dill",device="cpu"):
    if mode=="dill":
        with open(savename, 'rb') as file_:
            model_ = pickle.load(file_)
    elif mode=="torch":
        model_ = torch.load(savename, map_location=device,pickle_module=pickle)
    else:
        raise NotImplementedError(f"{mode} mode not implemented!")
    return model_

#%% auxiliary methods for augmenting
#%%% saving and loading
#%%%% dill based
def saveME(model,savename=None):
    if hasattr(model,'savename') and savename==None:
        file = model.savename
    else:
        if savename==None:
            file = os.path.join(os.getcwd(),getME_uniquename('.dill'))
        else:
            file = savename
        
    if os.path.splitext(file)[1]==".dill":
        pass
    else:
        file = os.path.splitext(file)[0]+".dill"
        
    with open(file, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    return True

def loadME(model,savename=None):
    if hasattr(model,'savename') and savename==None:
        file = model.savename
    else:
        if savename==None:
            print('loadME: either your model has a savename or you provide one!')
            return False
        else:
            file = savename
        
    if os.path.splitext(file)[1]==".dill":
        pass
    else:
        file = os.path.splitext(file)[0]+".dill"
        
    with open(file, 'rb') as file_:
        model_ = pickle.load(file_)
    model.__class__ = model_.__class__
    model.__dict__ = model_.__dict__
    return True

#%%%% torch based
def saveME_torch(model,savename=None,mode='entirely'):
    # make path
    if hasattr(model,'savename') and savename==None:
        file = os.path.splitext(model.savename)[0]+'_'+mode
    else:
        if savename==None:
            file = os.path.join(os.getcwd(),getME_uniquename('.tar'))
        else:
            file = savename
        
    if os.path.splitext(file)[1]==".tar":
        pass
    else:
        file = os.path.splitext(file)[0]+".tar"
    
    # save model
    if mode=='entirely':
        print('saveME_torch: start saving of the entire model!')
        torch.save(model,file,pickle_module=pickle)
        
    elif mode=='inference':
        print('saveME_torch: start saving of the model for later inference only!')
        torch.save(model.state_dict(),file,pickle_module=pickle)
        
    print('saveME_torch: saved')
    return True

def saveME_checkpoint(model, savename=None, **kwargs):
    # make path
    if hasattr(model,'savename') and savename==None:
        file = os.path.splitext(model.savename)[0]+'_'+mode
    else:
        if savename==None:
            file = os.path.join(os.getcwd(),getME_uniquename('.tar'))
        else:
            file = savename
        
    if os.path.splitext(file)[1]==".tar":
        pass
    else:
        file = os.path.splitext(file)[0]+".tar"
    
    # save checkpoint
    print('saveME_checkpoint: start saving checkpoint!')
    dictionary = {'model_state_dict':model.state_dict(),
                  'dict':model.__dict__}
    dictionary.update(kwargs)
    torch.save(dictionary,file,pickle_module=pickle)
    
    print('saveME_checkpoint: saved')
    return True

def loadME_torch(model,savename=None,mode='entirely',device=None):
    # make path
    if hasattr(model,'savename') and savename==None:
        file = os.path.splitext(model.savename)[0]+'_'+mode
    else:
        if savename==None:
            print('loadME_torch: either your model has a savename or you provide one!')
            return False
        else:
            file = savename
    
    if os.path.splitext(file)[1]==".tar":
        pass
    else:
        file = os.path.splitext(file)[0]+".tar"
    
    # choose device
    if hasattr(model,'device') and device==None:
        device_ = next(model.parameters()).device
    else:
        if device==None:
            device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_ = device
    
    # load model
    if mode=='entirely':
        print('loadME_torch: start loading of the entire model!')
        ### It's bit cheeky, so currently the following is recommended: model = torch.load(file, map_location=config['device'],pickle_module=dill)
        model_ = torch.load(file, map_location=device_,pickle_module=pickle)
        model.__class__ = model_.__class__
        model.__dict__ = model_.__dict__
        
    elif mode=='inference':
        print('loadME_torch: start loading of the model for later inference only!')
        model.load_state_dict(torch.load(file, map_location=device_,pickle_module=pickle),strict=False)
        
    print('loadME_torch: loaded')
    return True

#%%% parameters of models (pytorch specific)
#%%% number of parameters
def getME_Ntheta(model,key='total',unique=False):
    if unique:
        ### Only keep unique data pointers. As far as I know, this issue may occur if self.layer4 = self.layer1 is done and the model is transferred to another device, for example.
        params = {p.data_ptr(): p for p in model.parameters()}.values()
    else:
        params = [p for p in model.parameters()]
        
    total = 0
    trainable = 0
    frozen = 0
    for p in params:
        total = total + p.numel()
        if p.requires_grad:
            trainable = trainable + p.numel()
        else:
            frozen = frozen + p.numel()
            
    params_dict = {
            'total':total,
            'trainable':trainable,
            'frozen':frozen,
            }
    
    if key==None:
        return params_dict
    else:
        return params_dict.get(key,False)

#%%% set parameters
def setME_theta(model, theta, index=None, device=None):      
    if device==None:
        device = next(model.parameters()).device
        
    oldtheta = getME_theta(model, index=None, mode='vec', device="cpu", detach=True) ### TODO: check detach behavior!
    try:
        if type(theta)==list:
            for i,p in enumerate(model.parameters()):
                if i==index or index==None:
                    if type(theta[i])==tuple:
                        p.data = theta[i][1].to(device)#.reshape(p.shape)
                    else:
                        p.data = theta[i].to(device)#.reshape(p.shape)
        elif type(theta)==torch.Tensor:
            ind0 = 0
            for i,p in enumerate(model.parameters()):
                if i==index or index==None:
                    p.data = theta[ind0:ind0+p.numel()].reshape(p.shape).to(device)
                    ind0 = ind0+p.numel()
        else:
            raise NotImplementedError('Wrong type of theta given!')
                
    except Exception as e:
        print(e)
        print(f'setME_theta: wrong theta ({theta}) given - most likely the shape is wrong! Going back to the original parameters!')
        setME_theta(model,oldtheta,index=None,device=None)
        return False
    return True

#%%% get parameters
def getME_theta(model, index=None, mode='vec', device=None, detach=False):
    if device==None and detach==False:
        if index==None:
            if mode=='vec' or mode=='sampling':
                theta = torch.concat([p.data.flatten().contiguous() for p in model.parameters()])
            elif mode=='parameters' or mode=='params':
                theta = [p.data.contiguous() for p in model.parameters()]
            elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                theta = [(name,p.data.contiguous()) for name,p in model.named_parameters()]
            else:
                print('getME_theta: wrong mode chosen!')
        elif index>=0 and index<len(list(model.parameters())):
            if mode=='vec' or mode=='sampling':
                theta = [p.data.flatten().contiguous() for p in model.parameters()][index] # what to do if slice?
            elif mode=='parameters' or mode=='params':
                theta = [p.data.contiguous() for p in model.parameters()][index]
            elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                theta = [(name,p.data.contiguous()) for name,p in model.named_parameters()][index]
            else:
                print('getME_theta: wrong mode chosen!')
        else:
            print(f'getME_theta: wrong index ({index}) chosen!')
    
    elif (device or detach):
        if device==None:
            device = next(model.parameters()).device
        
        if index==None:
            if mode=='vec' or mode=='sampling':
                theta = torch.concat([p.data.clone().detach().to(device).flatten().contiguous() for p in model.parameters()])
            elif mode=='parameters' or mode=='params':
                theta = [p.data.clone().detach().to(device).contiguous() for p in model.parameters()]
            elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                theta = [(name,p.data.clone().detach().to(device).contiguous()) for name,p in model.named_parameters()]
            else:
                print('getME_theta: wrong mode chosen!')
        elif index>=0 and index<len(list(model.parameters())):
            if mode=='vec' or mode=='sampling':
                theta = [p.data.clone().detach().to(device).flatten().contiguous() for p in model.parameters()][index] # what to do if slice?
            elif mode=='parameters' or mode=='params':
                theta = [p.data.clone().detach().to(device).contiguous() for p in model.parameters()][index]
            elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                theta = [(name,p.data.clone().detach().to(device).contiguous()) for name,p in model.named_parameters()][index]
            else:
                print('getME_theta: wrong mode chosen!')
        else:
            print(f'getME_theta: wrong index ({index}) chosen!')
    
    return theta

#%%% get parameter index
def getME_thetaindex(model,key=None):
    index = {i:name[0] for i,name in enumerate(model.named_parameters())}
    index.update({name[0]:i for i,name in enumerate(model.named_parameters())})
    
    if key==None:
        return index
    elif type(key)==list:
        return list(map(index.get,key,[False]*len(key)))
    else:
        return index.get(key,False)

#%%% gradient methods
def getME_gradient(model,mode='vec',index=None,device=None,detach=False):
    if device==None and detach==False:
        if index==None:
            if mode=='vec' or mode=='sampling':
                grads = torch.concat([p.grad.flatten().contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).flatten().contiguous() for p in model.parameters()])
            elif mode=='parameters' or mode=='params':
                grads = [p.grad.contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).contiguous() for p in model.parameters()]
            elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                grads = [(name,p.grad.contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).contiguous()) for name,p in model.named_parameters()]
            else:
                print('getME_gradient: wrong mode chosen!')
        elif index>=0 and index<len(list(model.parameters())):
            if mode=='vec' or mode=='sampling':
                grads = [p.grad.flatten().contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).flatten().contiguous() for p in model.parameters()][index] # what to do if slice?
            elif mode=='parameters' or mode=='params':
                grads = [p.grad.contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).contiguous() for p in model.parameters()][index]
            elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                grads = [(name,p.grad.contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).contiguous()) for name,p in model.named_parameters()][index]
            else:
                print('getME_gradient: wrong mode chosen!')
        elif type(index)==str:
            for name,p in model.named_parameters():
                if name==index:
                    if mode=='vec' or mode=='sampling':
                        grads = p.grad.flatten().contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).flatten().contiguous()
                    elif mode=='parameters' or mode=='params':
                        grads = p.grad.contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).contiguous()
                    elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                        grads = (name,p.grad.contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).contiguous())
                    else:
                        print('getME_gradient: wrong mode chosen!')
                    break
                else:
                    grads = False
        else:
            print(f'getME_gradient: wrong index ({index}) chosen!')
    
    elif (device or detach):
        if device==None:
            device = next(model.parameters()).device
        
        if index==None:
            if mode=='vec' or mode=='sampling':
                grads = torch.concat([p.grad.clone().detach().to(device).flatten().contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).flatten().contiguous() for p in model.parameters()])
            elif mode=='parameters' or mode=='params':
                grads = [p.grad.clone().detach().to(device).contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).contiguous() for p in model.parameters()]
            elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                grads = [(name,p.grad.clone().detach().to(device).contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).contiguous()) for name,p in model.named_parameters()]
            else:
                print('getME_gradient: wrong mode chosen!')
        elif index>=0 and index<len(list(model.parameters())):
            if mode=='vec' or mode=='sampling':
                grads = [p.grad.clone().detach().to(device).flatten().contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).flatten().contiguous() for p in model.parameters()][index] # what to do if slice?
            elif mode=='parameters' or mode=='params':
                grads = [p.grad.clone().detach().to(device).contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).contiguous() for p in model.parameters()][index]
            elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                grads = [(name,p.grad.clone().detach().to(device).contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).contiguous()) for name,p in model.named_parameters()][index]
            else:
                print('getME_gradient: wrong mode chosen!')
        elif type(index)==str:
            for name,p in model.named_parameters():
                if name==index:
                    if mode=='vec' or mode=='sampling':
                        grads = p.grad.clone().detach().to(device).flatten().contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).flatten().contiguous()
                    elif mode=='parameters' or mode=='params':
                        grads = p.grad.clone().detach().to(device).contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).contiguous()
                    elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
                        grads = (name,p.grad.clone().detach().to(device).contiguous() if p.grad.clone().detach().to(device)!=None else torch.zeros(p.data.shape).to(device).contiguous())
                    else:
                        print('getME_gradient: wrong mode chosen!')
                    break
                else:
                    grads = False
        else:
            print(f'getME_gradient: wrong index ({index}) chosen!')
        
    return grads

def freezeME(model,index=None): # freeze chosen parameters by index or name
    thetaindex = getME_thetaindex(model,key=None)
    
    pass
    
#%%%% loss
def getME_gradient_dL_dTheta(model,L,x,y,mask=None,theta=None,theta_overwrite=False,mode='vec',index=None,key='gradient'): ### TODO: check detaching behavior!!!
    if theta==None:
        pass
    else:
        oldtheta = torch.concat([p.data.flatten().contiguous() for p in model.parameters()])
        setME_theta(model,theta)
            
    if key=='list':
        return getME_gradient_dL_dTheta(model,L,x,y,theta=theta,theta_overwrite=False,mode=mode,index=index,key=['gradient','prediction','loss'])
    
    # compute prediction
    y_hat = model(x)

    # compute loss
    loss = L(y_hat, y)

    # compute gradient
    loss.backward()

    # query gradient
    if index==None:
        if mode=='vec' or mode=='sampling':
            grads = torch.concat([p.grad.flatten().contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).flatten().contiguous() for p in model.parameters()])
        elif mode=='parameters' or mode=='params':
            grads = [p.grad.contiguous() if p.grad!=None else p.grad for p in model.parameters()]
        elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
            grads = [(name,p.grad.contiguous() if p.grad!=None else p.grad) for name,p in model.named_parameters()]
        else:
            print('getME_gradient_dL_dTheta: wrong mode chosen!')
    elif index>=0 and index<len(list(model.parameters())):
        if mode=='vec' or mode=='sampling':
            grads = [p.grad.flatten().contiguous() if p.grad!=None else torch.zeros(p.data.shape).to(p.device).flatten().contiguous() for p in model.parameters()][index] # what to do if slice?
        elif mode=='parameters' or mode=='params':
            grads = [p.grad.contiguous() if p.grad!=None else p.grad for p in model.parameters()][index]
        elif mode=='named parameters' or mode=='named_parameters' or mode=='named params' or mode=='named_params':
            grads = [(name,p.grad.contiguous() if p.grad!=None else p.grad) for name,p in model.named_parameters()][index]
        else:
            print('getME_gradient_dL_dTheta: wrong mode chosen!')
    else:
        print(f'getME_gradient_dL_dTheta: wrong index ({index}) chosen!')

    # theta_overwrite
    if theta==None:
        pass
    else:
        if theta_overwrite==False:
            setME_theta(model,oldtheta)

    # return
    dict_grads = {'gradient':grads,
                  'prediction':y_hat,
                  'loss':loss,
                  }

    if key==None:
        return dict_grads
    elif type(key)==list:
        return list(map(dict_grads.get,key,[False]*len(key)))
    else:
        return dict_grads.get(key,False)

#%%%% output
def getME_gradient_dOutput_dInput(model,x,y,theta=None,theta_overwrite=False,mode='vec',index=None,key='gradient'):
    pass

#%%%% hook based (be careful as some modules do not really support that!)
def giveME_gradient(model,gradient,index=None):
    # something with backward hooks
    pass

#%%% activations
def giveME_activations(model,activations,index=None):
    # something with forward hooks
    pass

#%% auxiliary
def getME_uniquename(ending='.txt'):
    uniquename = str(uuid.uuid4())+ending
    return uniquename
    
#%% test
if __name__=='__main__':
    from torch import nn
    
    # parameters to choos
    mode = 'torch'
    
    # define some model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor([3.],requires_grad=True))
            self.b = nn.Parameter(torch.tensor([4.],requires_grad=True))
            self.c = nn.Parameter(torch.tensor([2.],requires_grad=True))
            
        def forward(self,x):
            return self.a*x**2+self.b*x+self.c
    
    # initialize model
    model = Model()
    print('\nOriginal model:\n',model.__dict__)
    
    # augment model
    augment_IO(model,savekey='save',loadkey='load',mode=mode)
    augment_Ntheta(model,key='get_Nparams')
    print('\nAugmenteded model:\n',model.__dict__)
    
    # save model
    model.save(savename='TEST_AugmentME')
    
    # load model using AugmentME base class
    model2 = BaseClass(mode=mode)
    model2.load(savename='TEST_AugmentME')
    print('\nLoaded model:\n',model2.__dict__)
    
