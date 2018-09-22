#!/usr/bin/env python
# coding: utf-8

# In[53]:


#Este notebook carga el dataset MNIST, imprime algunos ejemplos, y despues extrae sólo dos clases indicadas (ejemplo: 0, 1)
#para realizar clasificación binaria.
#Tareas:
#- Diseña un MLP y entrenalo para realizar clasificación binaria.
#- Juega con distintas arquitrecturas para ver con cuál obtienes menor pérdida.
#  desde un perceptron sigmoide, hasta una red de 4 o 5 capas intermedias (con diferentes cantidades de unidades).
#- Evalua el desempeño con diferentes learning rates, tamaño de batches, y número de épocas.
#- Elige otras dos clases y repite el proceso (ejemplo, clases 3 vs 8, o 4 vs 9)


# In[1]:


# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense
from keras import backend as K


# In[2]:


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(y_train.shape)


# In[3]:


# Show some examples of the MNIST dataset (the first 10 instances)
print("Examples of the MNIST dataset (test set):")
plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# In[4]:


# Print frequencies of each class, i.e., number of instances per class
plt.figure(figsize=(12, 4))
plt.hist(y_train, np.linspace(-0.5, 9.5, 11), rwidth=0.9)
plt.title("Class distribution on the training set")
plt.xlabel("Class label (digit)")
plt.ylabel("Class frequency")
plt.grid()
# you will see that the dataset is more or less well balanced


# In[50]:


# Extract only two classes, e.g., classes 0 and 1
class0 = 0 # Try with other class by modifying this parameter
class1 = 1 # Try with other class by modifying this parameter

# Training set
training_indices = np.logical_or(y_train == class0, y_train == class1) # identify indices of selected classes
X_training = x_train[training_indices] # Copy features of identified elements
Y_training = y_train[training_indices] # Copy labels of identified elements
Y_training[Y_training == class0] = 0 # Assign value of 0 to one class
Y_training[Y_training == class1] = 1 # and value of 1 to the other one.
print("Size of training set")
print(X_training.shape)
print(Y_training.shape)

# Test set
training_indices = np.logical_or(y_test == class0, y_test == class1) # identify indices of selected classes
X_testing = x_test[training_indices] # Copy features of identified elements
Y_testing = y_test[training_indices] # Copy labels of identified elements
Y_testing[Y_testing == class0] = 0 # Assign value of 0 to one class
Y_testing[Y_testing == class1] = 1 # and value of 1 to the other one.
print("Size of test set")
print(X_testing.shape)
print(Y_testing.shape)


# In[52]:


# Build your model architecture (layers with activations), and print summary
model = Sequential()
# YOUR NETWORK HERE
# ..
model.add(Dense(units=5, input_dim=x_train.shape[1], activation=K.relu))
model.add(Dense(units=3, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(units=1, activation = 'sigmoid'))
model.summary()
Dense()


# In[44]:

from keras import optimizers
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
#model.compile(optimizer='sgd', loss='mean_squared_error')
model.compile(optimizer=sgd, loss='mean_squared_error')
# Compile your model (define optimizer and loss function)
# model.compile(optimizer='sgd', loss='mean_squared_error')


# In[45]:


# Train your model
hist = model.fit(X_training, Y_training, epochs=10, batch_size=100)


# In[46]:


# Plot training history
plt.plot(hist.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")


# In[47]:


# Make predictions for test set
y_hat = model.predict(X_testing)


# In[48]:

# Plot info about classes: test set and prediction
plt.figure(figsize=(20, 4))
ax = plt.subplot(1, 2, 1)
ax.plot(Y_testing, '.')
ax.set_title("Test dataset")
ax = plt.subplot(1, 2, 2)
ax.plot(y_hat, '.')
ax.set_title("Prediction")


# In[49]:


# Evaluate the prediction error on the test set (data not seen during training. Good to see how well our will generalize)
test_error = model.evaluate(X_testing, Y_testing)
print("Test error: {}".format(test_error))


# In[ ]:
