import openml as oml
import pandas as pd
from sklearn.model_selection import train_test_split

from main import Network, FCLayer
from loss import *
from act_funcs import *
from dev_funcs import *


mnist = oml.datasets.get_dataset(1504)
X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
y = pd.get_dummies(y)

Net = Network(X, y,[
    FCLayer(16, relu, drelu),
    FCLayer(8, relu, drelu),
    FCLayer(2, softmax)
], loss_func= cross_entropy, learning_rate= 0.1, decay_rate= 0.00, momentum= 0.9
)

Net.fit()