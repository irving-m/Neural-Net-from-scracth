import numpy as np
from layer import Network
from layer import Layer
from activations import *
from act_prime import *
from aux_functions import *

np.random.seed(13)

x = np.random.randint(-10, 10, size=(5000, 4))

y = np.matmul(np.array([[3, -4, 1, -2]]), x.T).T


mlp = Network(x, y, 
                [Layer(10, ReLu, d_ReLu),
                Layer(4, ReLu, d_ReLu),
                Layer(4, ReLu, d_ReLu),
                Layer(1, identity, d_identity)], iter= 10000, alpha= 0.4)

mlp.fit()

print(mlp.predict(standard_scaler(np.array([[1, 2, 3, 4]])).T))
