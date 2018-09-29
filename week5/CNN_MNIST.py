#!/usr/bin/env python
# coding: utf-8

# In[27]:


# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool1D, MaxPool2D, Flatten, BatchNormalization
from keras.regularizers import l1_l2


# In[3]:


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float64') / 255.0
x_test = x_test.astype('float64') / 255.0
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape(((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)))
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print("Training set:")
print(x_train.shape)
print(y_train.shape)
print("Test set:")
print(y_test.shape)
print(x_test.shape)


# In[4]:


# Split train into train and validation
validation_rate = 0.2
n_train_samples = round(validation_rate * len(x_train))
print("Taking {} validation samples".format(n_train_samples))
x_val = x_train[:n_train_samples]
y_val = y_train[:n_train_samples]
x_train = x_train[n_train_samples:]
y_train = y_train[n_train_samples:]
print("Training set:")
print(x_train.shape)
print(y_train.shape)
print("Validation set:")
print(x_val.shape)
print(y_val.shape)


# In[68]:


# Build your model architecture (layers with activations), and print summary
_, n_rows, n_cols, n_chans = x_train.shape
model = Sequential()
model.add(Conv2D(input_shape=(n_rows, n_cols, n_chans), filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.15))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.15))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.15))
#model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(9e-4)))
model.add(Dense(units=y_train.shape[1], activation='softmax'))
model.summary()


# In[69]:


# Compile your model (define optimizer and loss function)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


# In[70]:


# Train your model
num_epochs = 50
losses = np.zeros((num_epochs, 2))
accura = np.zeros((num_epochs, 2))
print(f"Training on {x_train.shape[0]} samples - validating on {x_val.shape[0]} samples.")
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1:3d} -- ", end="")
    model.fit(x_train, y_train, epochs=1, batch_size=256, verbose=False)
    losses[epoch, 0], accura[epoch, 0] = model.evaluate(x_train, y_train, verbose=False)
    losses[epoch, 1], accura[epoch, 1] = model.evaluate(x_val, y_val, verbose=False)
    print(f"Train loss: {losses[epoch, 0]:6.4f}, acc: {accura[epoch, 0]:6.4f} -- Val loss: {losses[epoch, 1]:6.4f}, acc: {accura[epoch, 1]:6.4f}")


# In[71]:


# Plot training history
plt.figure(figsize=(15, 10))
plt.plot(losses[:, 0], label='Loss: Training', linewidth=2)
plt.plot(losses[:, 1], label='Loss: Validation', linewidth=2)
plt.plot(accura[:, 0], label='Accu: Training', linewidth=2)
plt.plot(accura[:, 1], label='Accu: Validation', linewidth=2)
plt.legend(fontsize=18)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
#plt.ylim([0, 0.5])
plt.tick_params(labelsize=18)


# In[74]:


# Make predictions for test set and evaluate performance
y_hat = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss: {:6.4f}, acc: {:6.4f}".format(test_loss, test_acc))


# In[ ]:
