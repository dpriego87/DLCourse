#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Crea un dataset (X, Y) de ejemplo
n_samples = 5000
n_features = 200
X0 = np.random.rand(n_samples, n_features) - 0.7
X1 = np.random.rand(n_samples, n_features) + 0.7
X = np.concatenate((X0, X1))
Y0 = np.zeros((n_samples, 1))
Y1 = np.ones((n_samples, 1))
Y = np.concatenate((Y0, Y1))
print(X.shape)
print(Y.shape)


# In[3]:


# Grafica la nube de puntos de las primeras 2 coordenadas
plt.scatter(X[:, 0], X[0:, 1])
plt.show()


# In[4]:


# Guarda las matrices en disco
np.savetxt('Features.csv', X)
np.savetxt('Labels.csv', Y)


# In[5]:


# Borra X e Y
del X, X0, X1
del Y, Y0, Y1


# In[6]:


whos


# In[7]:


# Carga los datos desde el disco
X = np.loadtxt('Features.csv')
Y = np.loadtxt('Labels.csv')
print(X.shape)
print(Y.shape)


# In[8]:


# Ejemplo cargando una imagen desde el disco
IMG = plt.imread('Faces/10comm-decarlo.jpg')
plt.imshow(IMG)
plt.show()


# In[9]:


# define una funcion que carga y escala (256x256 pixeles) varias imagenes
from os import listdir
from os.path import isfile, join
from skimage.transform import resize
def load_images(dir_path):
    file_names = [file_name for file_name in listdir(dir_path) if isfile(join(dir_path, file_name))]
    full_paths = [join(dir_path, file_name) for file_name in file_names]
    images = np.array([resize(plt.imread(file_name), (256, 256)) for file_name in full_paths])
    return images


# In[10]:


# Llama la funcion e imprime el tamaño de la lista resultante
images = load_images('Faces')
print(images.shape) # 4 imagenes de 256x256 pixeles con tres canales RGB


# In[11]:


# Muestra las imagenes
print("Images:")
plt.figure(figsize=(20, 4))
for i in range(4):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(images[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# In[ ]:




