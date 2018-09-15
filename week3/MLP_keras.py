#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
%matplotlib
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

model.add(Dense(units=2, input_dim=X.shape[1], activation='relu'))
model.add(Dense(units=1, activation = 'sigmoid'))

#model.add(Activation('sigmoid'))
model.summary()


# In[29]:


# Compile model
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
#model.compile(optimizer='sgd', loss='mean_squared_error')
model.compile(optimizer=sgd, loss='mean_squared_error')
model.optimizer.lr

# In[30]:

hist = model.fit(X, y, epochs=50, batch_size=5)

yhat = model.predict(X)
# In[ ]:
plt.plot(yhat)
plt.plot(y)
plt.show()

plt.plot(hist.history['loss'])
plt.show()
