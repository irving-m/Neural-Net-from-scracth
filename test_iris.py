import openml as oml
import pandas as pd

from main import Network, FCLayer
from loss import *
from act_funcs import *
from dev_funcs import *


mnist = oml.datasets.get_dataset(61)
X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
y = pd.get_dummies(y)


Net = Network(X, y,[
    FCLayer(10, relu, drelu),
    FCLayer(3, softmax)
], loss_func= cross_entropy, learning_rate= 0.1, decay_rate= 0.0001, momentum= 0.9
)

Net.fit()
