#!/usr/bin/env python
# coding: utf-8


# Imports
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




# Hyper-parameters
n_epochs = 10
learn_rate = 0.1



# Variables X and y
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
print(X)
print(y)




# Parameters
W = np.zeros(3)
#W = np.random.rand(3)
print(W)



# Activation function
def activation(z):
    return 1 if z >= 0 else 0




# Prediction
def predict(W, x):
    z = W.T.dot(x)
    a = activation(z)
    return a




# Training
epoch_error = list()
for epoch in range(n_epochs):
    err = list()
    for i in range(len(X)):
        x = np.insert(X[i], 0, 1)
        y_hat = predict(W, x)
        e = y[i] - y_hat
        #print(e)
        err.append(e)
        W = W + learn_rate * e * x
    print("epoch {} -- error: {}".format(epoch, err))
    epoch_error.append(np.array(err).mean())




# Plot error
plt.plot(epoch_error)




# Print weigths
print(W)




# Make predictions and show them
y_predict = np.array([predict(W, np.insert(x, 0, 1)) for x in X])
print(y_predict)
