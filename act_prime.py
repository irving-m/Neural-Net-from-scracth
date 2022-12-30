import numpy as np
from activations import *

def d_ReLu(x):
    return x > 0


def d_softmax(x):
    return softmax(x)*(1 - softmax(x))
    

def d_identity(x):
    return 1




