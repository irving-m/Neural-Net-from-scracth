import numpy as np
from layer import Network
from layer import Layer
from activations import *
from act_prime import *

np.random.seed(13)

x = np.random.randint(-10, 10, size=(1, 4))
            
y = 3*x[:, 0] - 4*x[:, 1] + x[:, 2] - 2*x[:, 3] 


mlp = Network(x, y, 
                [Layer(3, ReLu, d_ReLu),
                Layer(1, identity, identity)], iter= 10, alpha= 0.1)

mlp.fit()