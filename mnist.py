import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from layer import Network
from layer import Layer
from activations import *
from act_prime import *
from aux_functions import get_accuracy, standard_scaler


dataset = openml.datasets.get_dataset(554)

print("====================================================================")

x, y, c, g = dataset.get_data(dataset_format="dataframe", target= "class")
y = y.astype("int")

X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size= 0.2,
                                                    random_state=42)


mlp = Network(X_train, y_train, [Layer(20, ReLu, d_ReLu), Layer(10, softmax, d_softmax)], iter= 50, alpha= 0.5)

print(X_train.shape)
print(y_train.shape)

print("----------------------------------------------------------------")
mlp.fit()
print(mlp.predict(standard_scaler(np.array(X_test))[2:5, :].T))

