import numpy as np


def ReLu(x):
    return np.maximum(x, 0)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))
    

def identity(x):
    return x


if __name__ == "__main__":

    x = np.array([1.3, 5.1, 2.2, 0.7, 1.1])
    print(softmax(x))
    