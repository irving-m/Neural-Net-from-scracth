import numpy as np

from activations import * #softmax, relu
from act_prime import * #d_softmax, d_relu

from aux_functions import one_hot
from aux_functions import get_accuracy

class Network():
    def __init__(self, input, target, layers, alpha = 0.1, iter= 1000):
        self.input = np.array(input).T
        self.y_true = target
        self.target = np.array(one_hot(target)).T
        self.layers = layers
        self.alpha = alpha
        self.iter = iter
        self.n_layers = len(layers)
        self.n = len(input)

        self.w = []
        self.b = []

        for i, layer in enumerate(layers):
            if i == 0:
                self.w.append(np.random.rand(layer.neurons, len(self.input)) - 0.5)
            else:
                self.w.append(np.random.rand(layer.neurons, layers[i - 1].neurons) - 0.5)

            self.b.append(np.random.rand(layer.neurons, 1) - 0.5)


        self.a_record = []
        self.z_record = []

        self.dw_record = []
        self.db_record = []


    def forward_prop(self):
        a = self.input
        self.a_record.append(a)

        for i, layer in enumerate(self.layers):

            z = np.matmul(self.w[i], a) + self.b[i].reshape(-1, 1)
            a = layer.func(z)

            self.z_record.append(z)
            self.a_record.append(a)
        
        self.output = a
        
    
    def back_prop(self):
        delta = np.multiply(self.a_record[-1] - self.target, self.layers[-1].prime(self.z_record[-1]))

        for i, layer in reversed(list(enumerate(self.layers))):
            dw = np.matmul(delta, self.a_record[i].T)
            db = np.matmul(delta, np.ones((self.n, 1)))

            self.dw_record.append(dw)
            self.db_record.append(db)

            if i > 0:
                delta = np.multiply(np.matmul(self.w[i].T, delta), layer.prime(self.z_record[i - 1]))


    def gradient_desc(self):
        w_copy = self.w
        for i, (w, dw) in enumerate(zip(w_copy, self.dw_record[::-1])):
            self.w[i] = w - self.alpha*dw

        b_copy = self.b
        for i, (b, db) in enumerate(zip(b_copy, self.db_record[::-1])):
            self.b[i] = b - self.alpha*db



    def fit(self):
        for i in range(self.iter):
            self.forward_prop()
            self.back_prop()
            self.gradient_desc()

            if i % 50 == 0:
                print(f"Iteration: {i + 1} / {self.iter}")
                print(f"Accuracy: {get_accuracy(np.argmax(self.output, axis= 0), self.y_true)}")
        

class Layer():
    def __init__(self, neurons, func, prime = None):
        self.func = func
        self.neurons = neurons
        self.prime = prime

