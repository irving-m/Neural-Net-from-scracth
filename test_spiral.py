import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from main import Network, FCLayer
from act_funcs import *
from dev_funcs import *
from loss import *

nnfs.init()

X, y = spiral_data(samples= 100, classes= 3)

net = Network(X, y, [
    FCLayer(64, relu, drelu),
    FCLayer(3, softmax)
], loss_func= cross_entropy, learning_rate= 1, decay_rate= 1e-3, momentum= 0.9
)

net.fit()

X_test, y_test = spiral_data(samples=100, classes=3)

preds = net.predict(X_test)
print(np.mean(np.argmax(preds, axis= 1) == y_test))
'''
net.forward_propagation()
net.back_propagation()


print(net.a_record[2].T[:5])
print(net.loss())
print("====================================================")
print(net.dw_record[0].T)
print(net.db_record[0])
print(net.dw_record[1].T)
print(net.db_record[1])

print("====================================================")
net.gradient_descent(0.1)
print(net.w[0].T)
print(net.b[0])
print(net.w[1].T)
print(net.b[1])

'''
