import numpy as np

from activations import *
from act_prime import *
from cost import *


class Network():
    def __init__(self, input, target, layers, cost= None):
        self.input = input
        self.target = target
        self.layers = layers
        self.cost = cost

        self.n_layers = len(layers)
        self.n = len(input)

        self.a_record = []
        self.z_record = []
        self.delta_record = []


    def forward_prop(self):
        a = np.array(self.input).T

        for layer in self.layers:

            z = np.matmul(layer.w, a) + layer.b.reshape(-1, 1)
            a = layer.func(z)

            self.z_record.append(z)
            self.a_record.append(a)

        self.output = a

    
    def back_prop(self):
        delta = np.multiply(self.output - self.target, d_ReLu(self.z_record[-1]))

        for i, (layer, a, z) in reversed(enumerate(zip(self.layers, self.a_record, self.z_record))):
            dw = np.matmul(delta, a[i - 1].T)
            db = np.matmul(delta, np.ones(self.n, 1))

            delta = np.multiply(np.matmul(layer.w.T, delta), layer.prime(a[i - 1]))


class Layer():
    def __init__(self, w, b, func, prime = None):
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
                    Layer(np.array([[1.2, -0.6]]),
                          np.array([1]),
                          ReLu
                        )
                    ]
                )

net.forward_prop()

