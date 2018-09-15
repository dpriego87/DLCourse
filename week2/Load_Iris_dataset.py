#!/usr/bin/env python
# coding: utf-8

# Este notebook es un template de inicio que puedes modificar.
# Este template importa el dataset iris, y muestra algunas de sus caracteristicas.


# Imports
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt



# Import the dataset
iris = datasets.load_iris()
X = iris.data[:100, :2]  # Features: Take just the first 2 dimensions from the first 100 elements.
y = iris.target[:100]    # Labels:   Also just the first 100 elements.
print(f"There are {X.shape[0]} elements of {X.shape[1]} features.")
print(f"And there are {y.shape} labels.")



# Print the labels
print(f"These are the labels\n {y}")
print(f"These are the first 10 elements\n {X[:10]}")



# Plot the first 100 elements as a function of their first two dimensions
fig = plt.figure(figsize=(12, 8))
plt.plot(X[y==0, 0], X[y==0, 1], 'b.', label='Setosa')
plt.plot(X[y==1, 0], X[y==1, 1], 'r.', label='Versicolour')
plt.legend()



whos
