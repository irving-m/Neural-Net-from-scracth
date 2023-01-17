import numpy as np
import nnfs


nnfs.init()

class FCLayer:
    def __init__(self, neurons, act_func= None, derivative= None):
        self.neurons = neurons
        self.act_func = act_func
        self.derivative = derivative


class Network:
    def __init__(self, input, target, layers, loss_func= None, learning_rate= 0.1, decay_rate= 0, momentum= 0) -> None:
        self.input = np.array(input).T

        if len(target.shape) == 1:
            self.target = np.eye(len(np.unique(target)))[target].T
        else:
            self.target = np.array(target).T

        self.layers = layers
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.n = len(input)

        self.w = []
        self.b = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                w = np.random.rand(len(self.input), layer.neurons) - 0.5
                self.w.append(w.T)
                
            else:
                w = np.random.rand(self.layers[i - 1].neurons, layer.neurons) - 0.5
                self.w.append(w.T)

            self.b.append(np.zeros(layer.neurons))

    def forward_propagation(self):
        self.z_record = []
        self.a_record = []

        a = self.input
        self.a_record.append(a)
        for w, b, layer in zip(self.w, self.b, self.layers):
            z = np.matmul(w, a) + b.reshape(-1, 1)
            a = layer.act_func(z)

            self.z_record.append(z)
            self.a_record.append(a)
        

    def back_propagation(self):

        self.dw_record = []
        self.db_record = []

        delta = (self.a_record[-1] - self.target)/self.n
        index = range(len(self.layers))[::-1]

        for i, layer in zip(index, reversed(self.layers)):
            dw = np.matmul(delta, self.a_record[i].T)
            db = np.matmul(delta, np.ones((self.n, 1)))

            self.dw_record.append(dw)
            self.db_record.append(db.T)

            if i > 0:
                delta = np.multiply(np.matmul(self.w[i].T, delta), self.layers[i - 1].derivative(self.z_record[i - 1]))

        self.dw_record = self.dw_record[::-1]
        self.db_record = self.db_record[::-1]

    
    def gradient_descent(self, alpha):
        learning_rate = alpha

        dw_change = [self.momentum * dwch - learning_rate * dw for dw, dwch in zip(self.dw_record, self.dw_change)]
        db_change = [self.momentum * dbch - learning_rate * db for db, dbch in zip(self.db_record, self.db_change)]

        w_new = [w + dw for w, dw in zip(self.w, dw_change)]
        b_new = [b + db for b, db in zip(self.b, db_change)]

        self.w = w_new
        self.b = b_new
        
        self.dw_change = dw_change
        self.db_change = db_change

    
    def fit(self, epochs= 10001):

        self.dw_change = [np.zeros_like(w) for w in self.w]
        self.db_change = [np.zeros_like(b) for b in self.b]

        for epoch in range(epochs):
            learning_rate = self.learning_rate/(1 + self.decay_rate * epoch)

            self.forward_propagation()
            self.back_propagation()
            self.gradient_descent(learning_rate)

            if not epoch % 100:
                print(
                    f'epoch: {epoch}, ' +
                    f'accuracy: {self.accuracy(self.a_record[-1], self.target):.3f}, ' + 
                    f'loss: {self.loss(self.a_record[-1], self.target):.5f}'
                    )

    
    def predict(self, X):
        a = X.T
        for w, b, layer in zip(self.w, self.b, self.layers):
            z = np.matmul(w, a) + b.reshape(-1, 1)
            a = layer.act_func(z)

        return a.T

    def loss(self, y_pred, y_true):
        return np.mean(self.loss_func(y_pred, y_true))


    def accuracy(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=0)
        if len(y_true.shape) == 2:
            true = np.argmax(y_true, axis= 0)
        else:
            true = y_true
        return np.mean(predictions == true)
