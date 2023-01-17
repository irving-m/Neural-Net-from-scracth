
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from main import Network, FCLayer
from loss import *
from act_funcs import *
from dev_funcs import *


# Import the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape the 3D dataset into a 2D dataset
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

Net = Network(x_train/255, y_train,[
    FCLayer(10, relu, drelu),
    FCLayer(10, softmax)
], loss_func= cross_entropy, epochs= 10001, learning_rate= 0.01, decay_rate= 0.00, momentum= 0.9
)

Net.fit()

data = pd.DataFrame()
data["iter"] = range(0, Net.epochs)
data["loss"] = Net.loss_track

sns.lineplot(data, x= "iter", y= "loss")
plt.show()
