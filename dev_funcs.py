import numpy as np
from act_funcs import *


def dstep(x):
    return 0


def dlinear(x):
    return 1

'''
def dsigmoid(x):
    return 1/(1 + np.exp(-x))
'''

def drelu(x):
    return x > 0


def dsoftmax(x):
    return softmax(x)*(1 - softmax(x))

