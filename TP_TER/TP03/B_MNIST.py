

import matplotlib

from TP_TER.TP03.LogisticModel_Gradient import LogisticModel_Gradient

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import linear_model

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=2,linewidth=5000,suppress=True)



"""
Attention, on travaille avec de grosses données. Cela peut prendre un peu de temps à charger dans la mémoire.
"""



"""
Commencez par upgrader 'dask', sinon vous aurez un message d'erreur.
Si vous avez suivit le TP0 il suffit de faire :
~/tensorflow/bin/pip install -U dask
"""

"""reconnaissez vous la lettre affichez par le step ci-dessous ?
Correspond elle à l'étiquette ? A quoi sert l'option one_hot=True ?
Le mot "batch" signifie "paquet"
Les batch sont-ils aléatoires ?
"""
def step1():
    mnist = input_data.read_data_sets("../bigData/", one_hot=False)
    xs, us = mnist.train.next_batch(10)
    x=xs[5]
    y=us[5]
    x=np.reshape(x,[28,28])
    print("digit:",y)
    print(x)

    display=False
    if display:
        plt.imshow(x,cmap='gray',interpolation='nearest')
        plt.show()



""" Pourquoi maintenant ne fait-on pas l'entrainement sur un seul batch (comme nous faisions avant) ? """
def step4():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)
    nbCategories = 10
    nbDescriptors = 28*28


    model = LogisticModel_Gradient(nbDescriptors, nbCategories)
    model.verbose = True


    batchSize = 200
    for i in range(100):
        print("------ batch:",i)
        x_train, y_train = mnist.train.next_batch(batchSize)
        model.fit(x_train, y_train,100)



    """ analysons les prédiction du modèle sur les données tests"""
    x_test, y_test = mnist.test.next_batch(500)
    y_hat=model.predict(x_test)

    print("y_test:",y_test)
    print("y_har :",y_hat)




    display=False
    if display:
        b_hat, W_hat = model.b_hat, model.W_hat
        """la matrice W_hat[:,0] ci-dessous s'appelle "the evidence" de la classe (=catégorie) 0"""
        for i in range(10):
            plt.subplot(2,9,i+1)
            plt.imshow(np.reshape(W_hat[:,i],[28,28]),cmap='jet',interpolation='nearest')

        plt.show()
        """ dans votre rapport, faite l'exercice de pédagogie suivant : expliquer à un débutant, avec des mots simple,
        et à l'aide des images des evidences , comment l'ordinateur détermine si une image est un 0, un 1, un 2, ...   """



    model.close()



