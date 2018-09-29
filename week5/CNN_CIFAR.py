#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, BatchNormalization, InputLayer, GlobalAveragePooling2D
from keras.regularizers import l1


# In[2]:


# Load dataset
from keras.datasets import cifar10
(x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()
#from keras.datasets import cifar100
#(x_train, y_train_raw), (x_test, y_test_raw) = cifar100.load_data()

# Make sure they are of type float and within [0,255]
x_train = x_train.astype('float64') / 255.0
x_test = x_test.astype('float64') / 255.0
# Convert numeric label into one-hot encodding
y_train = np_utils.to_categorical(y_train_raw)
y_test = np_utils.to_categorical(y_test_raw)
# Print info
print("Training set:")
print(x_train.shape)
print(y_train.shape)
print("Test set:")
print(x_test.shape)
print(y_test.shape)


# In[3]:


# Show some examples of the MNIST dataset (the first 10 instances)
print("Examples of the dataset (test set):")
plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# In[4]:


# Print frequencies of each class, i.e., number of instances per class
plt.figure(figsize=(12, 4))
plt.hist(y_train_raw, np.linspace(-0.5, 9.5, y_train.shape[1]+1), rwidth=0.9)
plt.title("Class distribution on the training set")
plt.xlabel("Class label")
plt.ylabel("Class frequency")
plt.grid()
# you will see that the dataset is more or less well balanced


# In[5]:


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


# In[6]:


# Build your model architecture (layers with activations), and print summary
_, n_rows, n_cols, n_chans = x_train.shape
model = Sequential()
model.add(InputLayer(input_shape=(n_rows, n_cols, n_chans)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#model.add(Conv2D(64, (3, 3), padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#model.add(Conv2D(128, (3, 3), padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#model.add(Conv2D(256, (3, 3), padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())
#model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='softmax'))
model.summary()


# In[7]:


# Compile your model (define optimizer and loss function)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


# In[8]:


# Train your model
num_epochs = 50
losses = np.zeros((num_epochs, 2))
accura = np.zeros((num_epochs, 2))
print(f"Training on {x_train.shape[0]} samples - validating on {x_val.shape[0]} samples.")
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1:3d} -- ", end="")
    model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=False)
    losses[epoch, 0], accura[epoch, 0] = model.evaluate(x_train, y_train, verbose=False)
    losses[epoch, 1], accura[epoch, 1] = model.evaluate(x_val, y_val, verbose=False)
    print(f"Train loss: {losses[epoch, 0]:6.4f}, acc: {accura[epoch, 0]:6.4f} -- Val loss: {losses[epoch, 1]:6.4f}, acc: {accura[epoch, 1]:6.4f}")


# In[9]:


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
plt.xticks(np.arange(1, len(losses)))
plt.tick_params(labelsize=18)
plt.grid()
#plt.show()


# In[10]:


# Make predictions for test set and evaluate performance
y_hat = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss: {:6.4f}, acc: {:6.4f}".format(test_loss, test_acc))


# In[ ]:




