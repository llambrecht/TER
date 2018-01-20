import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt


np.set_printoptions(linewidth=500,precision=2,suppress=True)

import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


''' Puisque tensor flow connaît toutes les opérations formellement,
il est capable de calculer des gradients de manière formelles.
Ainsi les valeurs du gradient sont "exactes" (à la précision des float prêt).  '''


""" Je vous demande de lire très attentivement les prochains steps.   """
def step0():
    """"""
    """considérez la variable x comme une variable "formelle".
    à laquelle on assignera différentes valeur (la valeur 100 pour commencer). """
    x = tf.get_variable("x",  initializer =100.)
    """considérez y comme une fonction de x"""
    y = x**2

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


"""idem mais avec une illustration graphique"""
def step0bis():

    x = tf.get_variable("x", initializer = 100.)
    y = x**2

    x_=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for k in range(10):
            x_.append(sess.run(x))
            gradient = tf.gradients(y, x)[0]
            sess.run(x.assign(x - gradient * 0.1 ))


    xs = np.linspace(-2., 100., 50)
    ys = xs * xs

    plt.plot(xs, ys)
    plt.plot(x_, [a * a for a in x_], "ro")
    plt.show()


step0bis()




"""
Exactement la même chose, mais en utilisant un outil de tensorflow
 """
def step1():

    x = tf.get_variable("x",  initializer = 100.)
    y = x**2

    ''' l'algo de descente de gradient, dont on fixe le pas (=learning_rate)'''
    learning_rate=0.1
    minimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(y)

    x_=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for k in range(10):
            x_.append(sess.run(x))

            sess.run(minimize)
            """l'action précédente remplace exactement les deux lignes suivante:"""
            #gradient = tf.gradients(y, x)[0]
            #sess.run(x.assign(x - gradient * 0.1 ))


    xs = np.linspace(-2., 100., 50)
    ys = xs * xs

    plt.plot(xs, ys)
    plt.plot(x_, [a * a for a in x_], "ro")
    plt.show()





"""
A VOUS: faites varier le learning_rate.
Que se passe-t-il quand il est trop grand, trop petit ?
Expliquez le phénomène dans votre rapport avec des captures d'écran.
Remplacez tf.train.GradientDescentOptimizer par tf.train.AdamOptimizer
Normalement, l'algo devrait converger pour une plus grande plage de learning_rate


BONUS : quand on ne dispose pas d'un framework comme tensorflow qui calcul les gradients de manière exacte,
que peut-on quand même faire ?  N'hésitez pas à aller regarder sur wikipedia
"""








