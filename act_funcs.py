import numpy as np

def step(x):
    return x>0


def linear(x):
    return x


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x, axis= 0)
    return np.exp(x)/np.sum(np.exp(x), axis= 0)


if __name__ == "__main__":
    a = np.array([
        [4.8, 1.21, 2.385],
        [8.9, -1.81, 0.2],
        [1.41, 1.051, 0.026]
    ])

    print(softmax(a))