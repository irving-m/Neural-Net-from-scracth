import numpy as np


def ReLu(x):
    return np.maximum(x, 0)


def softmax(x):
    max_x = np.max(x, axis= 0)
    y = x - max_x
    return np.exp(y)/np.sum(np.exp(y), axis= 0)
    

def identity(x):
    return x

