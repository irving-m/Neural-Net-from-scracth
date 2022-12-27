# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:11:46 2022

@author: Irving
"""

import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split

x = (np.random.rand(1000, 5) - 0.5) * 10
e = (np.random.rand(1000, 4) - 0.5) * 8
y = np.matmul(x, np.array([[2, 3, -1, 0],
                           [-2, -2, 1, 3],
                           [ 1, 1, 3, 1],
                           [-2, 1, -3, 5],
                           [ 4, 2, 0, 2]])) + [7, -13, 5, 10] + e


sns.scatterplot(x= x, y= y)

def init_params():
    w = np.random.rand(5, 4) - 0.5
    b = np.random.rand(4) - 0.5
    return w, b


def cost(y_true, y_pred):
    n = y_true.shape[0]
    
  
def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def derivative(y_true, x, w, b):
    n = y_true.shape[0]
    dcdw = 2/n * np.matmul(-x.T, y_true - np.matmul(x, w) - b)
    dcdb = 2/n * np.matmul(-np.ones((1, n)), y_true - np.matmul(x, w) - b)
    
    return dcdw, dcdb


def gradient_descent(w, b, dcdw, dcdb, alpha):
    
    w = w - alpha*dcdw
    b = b - alpha*dcdb
    return w, b

def linear_regression(y_true, x):
    
    w, b = init_params()
    
    y = np.matmul(x, w) + b
    error = [cost(y_true, y)]
    w0 = []
    b0 = []
    
    while True:
        dcdw, dcdb = derivative(y_true, x, w, b)
        w, b = gradient_descent(w, b, dcdw, dcdb, 0.01)
        
        y = np.matmul(x, w) + b
        error.append(cost(y_true, y))
        w0.append(w)
        b0.append(b)
        
        if abs(error[-2] - error[-1]) < 0.000001:
            break
    
    return error, w, b, w0, b0


x_train, x_test, y_train, y_test = train_test_split(x, y)

error, w, b, w0, b0 = linear_regression(y_train, x_train)

pderror = pd.DataFrame(error, columns= ["values"])
sns.lineplot(data= pderror, x= pderror.index, y= "values")


a = np.array([[3, 4, 5, 2],
              [2, 2, 0, 1],
              [3, 1, 0, 1]])

sum(a)
np.exp(a)/sum(np.exp(a))
a/sum(a)


softmax(a)



