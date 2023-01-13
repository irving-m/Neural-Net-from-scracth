import numpy as np


def ReLu(x):
    return np.maximum(x, 0)


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis= 0)
    

def identity(x):
    return x

