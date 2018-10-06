#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[2]:


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# In[3]:


print(X_train[0])


# In[4]:


# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# In[5]:


print(X_train[0])
print(y_train[:10])


# In[6]:


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(10))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[7]:


#
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


# In[8]:


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[19]:


W = model.layers[-1].get_weights()[0]
print(W)


# In[20]:


# Save model
model.save('my_model.h5')
del model


# In[21]:


whos


# In[23]:


# Reload it
from keras.models import load_model

model = load_model('my_model.h5')
W = model.layers[-1].get_weights()[0]
print(W)


# In[24]:


# Evaluate again
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




