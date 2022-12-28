import numpy as np

from activations import ReLu
from activations import softmax

def init_params(x, y):
    k = x.shape[1]
    m = y.shape[0]

    w = np.random.rand(k, m) - 0.5
    b = np.random.rand(m) - 0.5
    return w, b


def forward(input, w, b, func):
    z = np.matmul(w, input) + b
    output = func(z)

    return output


def backward(input, func):
    pass

