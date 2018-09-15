#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


# Import the Iris dataset
iris = datasets.load_iris()
X = iris.data[:100, :]  # Features: Take just the first 2 dimensions from the first 100 elements.
y = iris.target[:100]


# In[3]:


# Print info
print(X.shape)
print(y.shape)
print(X[:5])
print(y)


# In[17]:


#
from keras import Sequential
from keras.layers import Dense, Activation


# In[27]:


# Build model
model = Sequential()
model.add(Dense(units=1, input_dim=X.shape[1]))
model.add(Activation('sigmoid'))
model.summary()


# In[29]:


# Compile model
model.compile(optimizer='sgd', loss='mean_squared_error')
model.optimizer.lr


# In[30]:


model.fit(X, y, epochs=10, batch_size=32)


# In[ ]:




