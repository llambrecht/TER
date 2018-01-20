
import matplotlib

from TP_TER.TP02.LinearModel_Gradient import LinearModel_Gradient

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv

import matplotlib.pyplot as plt
import numpy as np
from TOOL.viewer.ScatterAndFunctionViewer import ScatterAndFunctionViewer
import tensorflow as tf
import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=2,linewidth=5000,suppress=True)





"""traçons la variable de sortie en fonction de chacune des variables explicatives.
D'après vous, laquelle des deux variables explique le mieux la sortie y ? """
def step0():

    z=np.loadtxt("data/affineData2d.csv",delimiter=",")
    x=z[:,0:2].astype(np.float32)
    y=z[:,2].astype(np.float32)


    plt.subplot(2,1,1)
    plt.plot(x[:,0],y,'ro')
    plt.subplot(2,1,2)
    plt.plot(x[:,1],y,'ro')
    plt.show()





"""traçons la variable de sortie en fonction des deux variables explicatives."""
def step1():

    z=np.loadtxt("data/affineData2d.csv",delimiter=",")
    x=z[:,0:2].astype(np.float32)
    y=z[:,2].astype(np.float32)

    viewer= ScatterAndFunctionViewer(x,y)
    viewer.addLabelsOnScatter=True
    viewer.plot()



def step2():

    z=np.loadtxt("data/affineData2d.csv",delimiter=",")
    x_train=z[:,0:2].astype(np.float32)
    y_train=z[:,2].astype(np.float32)

    W = tf.get_variable("W",  initializer=tf.truncated_normal(shape=[2],stddev=0.1))
    b = tf.get_variable("b", initializer=1.)


    x_train_tf = tf.constant(x_train, dtype=tf.float32)
    y_train_tf = tf.constant(y_train, dtype=tf.float32)

    """
    y[i]= sum_j W[j] * x_train[i,j] + b  """
    y = tf.reduce_sum(W * x_train_tf,axis=1) + b
    loss = tf.reduce_mean(tf.square(y - y_train_tf))
    minimize = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    W_, b_, loss_ = None, None, None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(200):
            sess.run(minimize)
            W_, b_, loss_ = sess.run([W, b, loss])
            print("W:", W_)
            print("b:", b_)
            print("loss:", loss_)

    print("weights estimated:",W_)
    print("biais estimated:",b_)

    def hatFunction(x_test):
        return np.sum(W_ * x_test,axis=1) + b_

    viewer = ScatterAndFunctionViewer(x_train, y_train, function=hatFunction)
    viewer.functionInterval_x0 = [0, 2]
    viewer.functionInterval_x1 = [0, 2]
    viewer.plot()



""" On fait la même chose en utilisant la classe LinearModel_Gradient (du dossier Classes).
Quel sont les avantages d'avoir créer une telle classe ?
 Vous allez devoir aller observer la classe LinearModel_Gradient. Il y a des questions à l'intérieur.
 Mais juste avant lisez le fichier sur les placeholders
"""
def step3():

    z=np.loadtxt("data/affineData2d.csv",delimiter=",")
    x_train=z[:,0:2].astype(np.float32)
    y_train=z[:,2].astype(np.float32)

    model=LinearModel_Gradient(2)
    model.fit(x_train,y_train)

    print("weights estimated:",model.W_hat)
    print("biais estimated:",model.b_hat)

    def hatFunction(x_test):
        return np.sum(model.W_hat * x_test,axis=1) + model.b_hat

    viewer = ScatterAndFunctionViewer(x_train, y_train, function=hatFunction)
    viewer.functionInterval_x0 = [0, 2]
    viewer.functionInterval_x1 = [0, 2]
    viewer.plot()





