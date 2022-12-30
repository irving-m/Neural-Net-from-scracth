import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from layer import Network
from layer import Layer
from activations import *
from act_prime import *

dataset = openml.datasets.get_dataset(554)

x, y, c, g = dataset.get_data(dataset_format="dataframe", target= "class")
y = y.astype("int")

X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size= 0.9,
                                                    random_state=42)


mlp = Network(X_train, y_train, 
                [Layer(25, ReLu, d_ReLu),
                Layer(20, ReLu, d_ReLu),
                Layer(10, softmax, d_softmax)])


print("------------------------------------")
mlp.fit()





