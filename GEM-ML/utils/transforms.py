import numpy as np

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

def batchify(array):
    return np.stack([Torchify(1)(array)]*2,axis=0)

def predict(array,dim=0):
    return np.moveaxis(np.argmax(array,axis=dim,keepdims=True),0,-1)

def mover(array):
    return np.moveaxis(array,0,-1)

def predict2(array,dim=1):
    return np.moveaxis(np.argmax(array,axis=dim),0,-1)

def torbatchify(array):
    return np.stack([Torchify(1)(array[i,...]) for i in range(len(array))],axis=0)