import numpy as np


def d_ReLu(x):
    return x > 0


def d_softmax(x):
    return np.exp(x)/sum(np.exp(x))
    

def d_identity(x):
    return 1


if __name__ == "__main__":
    pass
    