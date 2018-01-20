
import tensorflow as tf
from datetime import time, datetime



''' découvrons maintenant les "placeholder" de tensorflow:
 une place mémoire préparée pour les entrées de données.
 C'est une sorte variable, sauf que : il faut spécifier sa valeur à chaque run !!!
 Entre 2 runs sa valeur n'est pas mémorisée.
 '''
def step0():
    """"""
    """lorsqu'on déclare un placeholder, on n'est pas obligé de le dimensionner précisément :
    Ainsi shape=[None,2] signifie que l'entrée x aura deux colonnes et un nombre indéterminé de lignes
     (en numpy on écrirait shape=[-1,2])
    """
    x=tf.placeholder(tf.float32,shape=[None,2],name="x")
    """ un placeholder pour une constante (shape=[] car tenseur d'ordre 0) """
    const=tf.placeholder(tf.float32,shape=[],name="const")
    x2 = const*x  #ou bien : tf.multiply(x,const,name="x2")
    W = tf.constant([[1., 2], [3, 4]],name="W")
    y = tf.matmul(x2, W,name="y")


    with tf.Session() as sess:
        x_data = [[0., 1],[2,4],[0,0]]
        print("jeu de donnée 1:\n",sess.run(y, feed_dict={x: x_data,const:1}))
        x_data = [[1., 1], [2, 4]]
        print("jeu de donnée 2:\n",sess.run(y, feed_dict={x: x_data,const:2}))


""" Parfois, quand on oublie de remplir un placeholder, on a un message très peu explicite:
     << Shape [-1,2] has negative dimensions >>
     Ce qui signifie que les None sont traduit par un -1 dans les couches inférieures.
"""
def step1():
    x=tf.placeholder(tf.float32,shape=[None,2],name="x")
    y=x**2
    with tf.Session() as sess:
        sess.run(y)




def step2():

    x=tf.placeholder(tf.float32,shape=[None,2],name="x")
    const=tf.placeholder(tf.float32,shape=[],name="const")
    y=x*3
    z=x*const

    with tf.Session() as sess:
        x_data = [[0., 1], [2, 4], [0, 0]]
        print(sess.run(y, feed_dict={x: x_data}))
        """QUIZ : pourquoi la ligne suivante donne-t-elle une erreur alors que la précédente n'en donne pas ?.
        Cette fois le message d'erreur est très explicite. Traduisez-le dans votre rapport"""
        print(sess.run(z, feed_dict={x: x_data}))




