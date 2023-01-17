import openml as oml
from main import Network, FCLayer
from loss import *
from act_funcs import *
from dev_funcs import *

np.set_printoptions(threshold=np.inf)

mnist = oml.datasets.get_dataset(554)
X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
y2 = y.astype("int")


Net = Network(X, y2,[
    FCLayer(32, relu, drelu),
    FCLayer(32, relu, drelu),
    FCLayer(10, softmax)
], loss_func= cross_entropy, learning_rate= 0.01, decay_rate= 0.0001, momentum= 0.9
)

#Net.fit()
print(Net.target.T[:10,])
print(y2[:10])
