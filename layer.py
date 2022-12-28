import numpy as np

from activations import *

class Network():
    def __init__(self, input, layers, cost= None):
        self.input = input
        self.layers = layers
        self.cost = cost

        self.a_record = []


    def forward_prop(self):
        a = np.array(self.input).T

        for layer in self.layers:
            z = np.matmul(layer.w, a) + layer.b.reshape(-1, 1)
            a = layer.func(z)

        self.output = a

             
            


class Layer():
    def __init__(self, w, b, func):
        self.w = w
        self.b = b
        self.func = func


x = np.array([[2, -3, 0],
                [-2, 1, 5],
                [3, -1, -1],
                [1, 4, 8]])

net = Network(x, [Layer(np.array([[0.7, -0.3, -0.1],
                                [2, 1.5, -0.5]]), 
                        np.array([1.2, .3]),
                        ReLu
                        ), 
                    Layer(np.array([[-1.3, -0.6]]),
                          np.array([2.4]),
                          ReLu
                        )
                    ]
                )

net.forward_prop()
print(net.output)



