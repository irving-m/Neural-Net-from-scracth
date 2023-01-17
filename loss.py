import numpy as np


def mse(y_pred, y_true):
    return 1/2*np.mean((y_pred - y_true)**2)


def cross_entropy(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    if len(y_true.shape) == 2:
        values = np.sum(y_pred * y_true, axis= 0)

    else:
        values = y_pred[y_true, range(y_pred.shape[1])]

    return -np.log(values)

if __name__ == "__main__":
    softmax_outputs = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.5, 0.4],
        [0.02, 0.9, 0.08]
    ])

    class_targets = np.array([
        0, 1, 1
    ])
    class_targets2 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])

    #print(np.mean(cross_entropy(softmax_outputs, class_targets2)))

    #print(cross_entropy(softmax_outputs, class_targets))

    dinputs = softmax_outputs.copy()
    dinputs[range(3), class_targets] = dinputs[range(3), class_targets] - 1 

    print(softmax_outputs - class_targets2)

    print(dinputs)
