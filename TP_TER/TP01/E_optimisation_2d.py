import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import tensorflow as tf
import numpy as np

from TOOL.viewer.ScatterAndFunctionViewer import ScatterAndFunctionViewer

np.set_printoptions(linewidth=500,precision=2,suppress=True)

import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def step0():
    """"""
    """ une variable de dimension 2 """
    x = tf.get_variable("x", initializer = [100.,100.])
    """ y =  x[0]**2+x[1]**2"""
    y = x[0]+x[1]**2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for k in range(5):
            """on dérive la fonction y par rapport à x.
               puis on évalue cette dérivé en x_k (au début x_0=10) """
            print('x_'+str(k)+"=", sess.run(x))
            gradient = tf.gradients(y, x)[0]
            print('gradient:', sess.run(gradient))

            """ on soustrait à x un dixième du gradient calculé précédemment   """
            sess.run(x.assign(x - gradient * 0.1 ))



def step1():
    """"""
    """ une variable de dimension 2 """
    x = tf.get_variable("x",  initializer = [0.4,0.3])
    y = tf.sin(3*x[0])*tf.cos(3*x[1])
    gradient = tf.gradients(y, x)[0]

    nb=20
    x_=np.empty([nb,2])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for k in range(nb):
            x_[k,:]=sess.run(x)
            sess.run(x.assign(x - gradient * 0.05 ))


    def fonc(x):return np.sin(3*x[:,0])*np.cos(3*x[:,1])

    y=np.ones(nb)

    viewer=ScatterAndFunctionViewer(x_,y,fonc)
    viewer.functionInterval_x0=[-2,2]
    viewer.functionInterval_x1=[-2,2]
    viewer.plot()


"""
    EXERCICE : expliquez pourquoi les méthodes de gradient donnent le plus souvent
    un minimum local (et non global).

    Refaites le petit programme en utilisant les optimiser tout fait de tensorflow, notamment :

    tf.train.GradientDescentOptimizer(0.05)
    tf.train.AdamOptimizer(0.05)
    tf.train.AdagradOptimizer(0.1)
"""



""" Pour certaines fonctions, il est très difficile de trouver le bon pas pour
la descente de gradient. Il est soit trop grand, soit trop petit.
Faites des essais (en variant 'nb' et 'training_step')
 Expliquez ce phénomène avec vos mots. Nous  retrouverons ce problème plus tard.
 """
def step2():
    """"""
    """ une variable de dimension 2 """
    x = tf.get_variable("x", initializer = [3.,3.])
    y =  50*x[0]**2 + 0.1*x[1]**2
    gradient = tf.gradients(y, x)[0]

    nb=20
    x_=np.empty([nb,2])
    training_step=0.01
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for k in range(nb):
            x_[k,:]=sess.run(x)
            sess.run(x.assign(x - gradient * training_step ))


    def fonc(x):return 50*x[:,0]**2 + 0.1*x[:,1]**2

    y=200*np.ones(nb)

    viewer=ScatterAndFunctionViewer(x_,y,fonc)
    viewer.functionInterval_x0=[-2,2]
    viewer.functionInterval_x1=[-2,2]
    viewer.plot()


step2()


