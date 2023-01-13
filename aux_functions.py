import numpy as np


def one_hot(y):
    one_hot_y = np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y.astype("int")] = 1

    return one_hot_y

def get_accuracy(y, y_true):
    return np.sum(y - y_true)/y.size


def standard_scaler(x):
    x = np.array(x)
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min)/(x_max - x_min)

if __name__ == "__main__":
    x = [2, 4, 6, 8]
    index = range(len(x))

    for i in zip(reversed(index, x)):
        print(i)