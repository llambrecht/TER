import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2,linewidth=5000,suppress=True)



"""           regression linéaire (=affine) de dimension 1.
 """


"""observons nos données"""
def step1():
    z = np.loadtxt("data/affineData.csv", delimiter=",")
    x_train = z[:, 0]
    y_train = z[:, 1]

    plt.plot(x_train,y_train,"o")
    plt.show()



"""trouvons la meilleurs droite qui passe par ces données"""
def step2():
    z=np.loadtxt("data/affineData.csv",delimiter=",")

    x_train = z[:,0].astype(np.float32)
    y_train = z[:,1].astype(np.float32)

    """
    Ci-dessus on a convertis, les tenseur-np (par défaut float64)
    en tenseur-np float32 pour la compatibilité avec tensorflow (par défaut float32).

    Autre solution : Plus loin, on aurait pu déclarer les variables-tf en float64. Ex:

                W=tf.get_variable("W",initializer=0.1,dtype=tf.float64)

    Mais en général, l'analyse de donnée ne nécessite pas une  précision de 64-bits. De plus certain modèles (les réseaux
    de neurones) nécessitent de très très gros tenseurs, alors autant rester en 32-bits.

    """

    W=tf.get_variable("W",initializer=0.1)
    b=tf.get_variable("b",initializer=1.)



    """Notre modèle : y est une fonction affine de x_train :
    y[i]= W * x_train[i] + b  """
    y= W* x_train + b
    """ on voit ci-dessus que les opérations entre tenseur-np et tenseur-tf fonctionnent bien."""


    '''on définit la fonction de coût'''
    loss = tf.reduce_mean(tf.square(y - y_train))
    '''l'oject GradientDescentOptimizer(learning_rate) représente l'algo de descente de gradient,
    attention, si l'on prend un pas (=learning_rate) trop grand (ex : 0.5)  l'algo diverge. '''
    minimize = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


    W_, b_, loss_=None,None,None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(200):
            sess.run(minimize)
            W_,b_,loss_=sess.run([W,b,loss])
            print("W:",W_)
            print("b:",b_)
            print("loss:",loss_)


    plt.plot(x_train,y_train,'ro')
    print(W_,x_train,b_)
    plt.plot(x_train,W_*x_train+b_)
    plt.show()


"""
EXO : améliorer le programme précédent pour le l'entrainement s'arête quand W,b ou loss ne varient plus beaucoup.
"""
