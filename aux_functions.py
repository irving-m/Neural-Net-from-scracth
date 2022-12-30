import numpy as np


def one_hot(y):
    one_hot_y = np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y.astype("int")] = 1

    return one_hot_y


def get_accuracy(y, y_true):
    return np.sum(y == y_true)/y.size
