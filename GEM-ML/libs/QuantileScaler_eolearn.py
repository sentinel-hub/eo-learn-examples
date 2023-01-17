### Michael Engel ### 2022-10-10 ### QuantileScaler_eolearn.py ###
import numpy as np
import os
import dill as pickle
import uuid

#%% QuantileScaler_eolearn
class QuantileScaler_eolearn:
    #%%% init
    def __init__(self,minval,maxval,nanval,infval, valmin=0,valmax=1, transform=True,savename=None):
        self.minval = np.array(minval,ndmin=1)
        self.maxval = np.array(maxval,ndmin=1)
        
        self.nanval = np.array(nanval,ndmin=1)
        self.infval = np.array(infval,ndmin=1)
        
        assert np.shape(minval)==np.shape(maxval), f"Shape of given maxval does not suit the given shape of minval!\nminval:{np.shape(minval)}, maxval:{np.shape(maxval)}, nanval:{np.shape(nanval)}, infval{np.shape(infval)}"
        assert np.shape(minval)==np.shape(nanval), f"Shape of given nanval does not suit the given shape of minval!\nminval:{np.shape(minval)}, maxval:{np.shape(maxval)}, nanval:{np.shape(nanval)}, infval{np.shape(infval)}"
        assert np.shape(minval)==np.shape(infval), f"Shape of given infval does not suit the given shape of minval!\nminval:{np.shape(minval)}, maxval:{np.shape(maxval)}, nanval:{np.shape(nanval)}, infval{np.shape(infval)}"
        
        self.valmin = valmin
        self.valmax = valmax
        
        self.transform = transform
        
        if savename==None:
            self.savename = str(uuid.uuid4())
        else:
            self.savename = savename
    
    #%%% transform
    def __call__(self,array):
        shape = np.shape(array)
        assert shape[-1]==len(self.minval), f"Shape of given array does not suit the given minval(s), maxval(s), nanval(s) and infval(s)!\narray:{np.shape(array)}, scaler:{np.shape(self.minval)}"
        
        array2 = np.empty(shape)
        for k in range(shape[-1]):
            array2[...,k] = (array[...,k]-self.minval[...,k])/(self.maxval[...,k]-self.minval[...,k])*(self.valmax-self.valmin)+self.valmin
            array2[...,k][np.isnan(array2[...,k])] = self.nanval[...,k]
            array2[...,k][np.isinf(array2[...,k])] = self.infval[...,k]
        
        if self.transform==True:
            return np.moveaxis(array2,-1,0)
        elif self.transform:
            return self.transform(array2)
        else:
            return array2
    
    #%%% IO
    @classmethod
    def LOAD(cls,file):
        with open(file, 'rb') as file_:
            scaler_ = pickle.load(file_)            
        return scaler_
    
    def save(self, savename=None):
        if savename==None:
            file = self.savename
        else:
            file = savename
            
        f,ext = os.path.splitext(file)
        file = f+".dill"

        with open(file, 'wb') as outfile:
            pickle.dump(self,outfile,pickle.HIGHEST_PROTOCOL)
        return file
    
    def save_quantiles(self,savename=None):
        return (self.save_minval(savename=savename),
                self.save_maxval(savename=savename))
    
    def save_minval(self,savename=None):
        if savename==None:
            file = self.savename+"_minval"
        else:
            file = savename
            
        f,ext = os.path.splitext(file)
        file = f+".txt"
        
        np.savetxt(file,self.minval)
        return file
    
    def save_maxval(self,savename=None):
        if savename==None:
            file = self.savename+"_maxval"
        else:
            file = savename
            
        f,ext = os.path.splitext(file)
        file = f+".txt"
        
        np.savetxt(file,self.maxval)
        return file
    
#%% QuantileScaler_eolearn_tdigest
class QuantileScaler_eolearn_tdigest(QuantileScaler_eolearn):
    #%%% init
    def __init__(self,tdigestarray,minquantile,maxquantile,nanval,infval, valmin=0,valmax=1, transform=True,savename=None):
        self.minquantile = minquantile
        self.maxquantile = maxquantile
        shape = tdigestarray.shape
        minval = np.empty(shape[-1])
        maxval = np.empty(shape[-1])
        for k in range(shape[-1]):
            minval[k] = tdigestarray[k].percentile(minquantile*100)
            maxval[k] = tdigestarray[k].percentile(maxquantile*100)
        super().__init__(minval,maxval,nanval,infval,valmin=valmin,valmax=valmax,transform=transform,savename=savename)

#%% TEST
if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    #%%% define data
    x = np.stack((
        np.random.randn(512,512)*3,
        np.random.randn(512,512)*(-1)
        ),axis=-1)
    
    #%%% initialise Scaler and store to disk
    Scaler = QuantileScaler_eolearn(
        minval = [np.min(x[...,0]),np.min(x[...,1])],
        maxval = [np.max(x[...,0]),np.max(x[...,1])],
        nanval = [0,0],
        infval = [0,0],
        valmin = 0,
        valmax = 1,
        transform = True,
        savename = "TEST_QuantileScaler_eolearn"
    )
    Scaler.save()
    Scaler.save_quantiles()
    
    #%%% scale and torchify data
    x_scaled = Scaler(x)
    
    #%%% results
    #%%%% prints
    print("Minimum dim 0:",np.min(x_scaled[0,...]))
    print("Maximum dim 0:",np.max(x_scaled[0,...]))
    print("Minimum dim 1:",np.min(x_scaled[1,...]))
    print("Maximum dim 1:",np.max(x_scaled[1,...]))
    
    #%%%% visualisation
    plt.figure()
    plt.hist(x_scaled[0,...].flatten(),bins=50,alpha=0.6,label="dim 0")
    plt.hist(x_scaled[1,...].flatten(),bins=50,alpha=0.6,label="dim 1")
    plt.legend()