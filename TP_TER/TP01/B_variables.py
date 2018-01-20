import tensorflow as tf
import numpy as np
np.set_printoptions(linewidth=500,precision=2,suppress=True)

import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



""" Les variables de tensorflow : considérez-les comme des variables "formelles" : comme le x dans une fonction mathématique f(x)=x^2.
Observez : """
def step0():
    """"""
    """ une variable scalaire qui sera initialisée à 0 (entier)"""
    x = tf.get_variable("x",initializer=0) #  ou bien x = tf.Variable(initial_value=0,name="x"), mais cela donne un problème de typage dans l'IDE

    """y est une fonction de x """
    y = x**2


    with tf.Session() as sess:
        """les variables doivent être initialisée """
        sess.run(tf.global_variables_initializer())
        for i in range(1,4):
            """changez i par i/2. Que se passe-t-il"""
            sess.run(x.assign(i))
            print("x:",sess.run(x))
            """remarquez qu'on n'affecte rien à y et pourtant, il change !"""
            print("y:",sess.run(y))




""" remarquez que tensorFlow est un langage de calcul formel. Il mémorise les liens entre les tenseurs.
 Dans un langage informatique  classique (ex: python), que donnerait ? :
 W=0
 y=W**2
 for i in range(1,4):
        W=i
        print("W",W)
        print("y=W^2",y)
 """




""" Rappelons que les constantes ne changent pas !"""
def step1():

    W=tf.constant(0)
    y=tf.square(W)

    with tf.Session() as sess:
        for i in range(1,4):
            W=tf.constant(i) #on créé ici une nouvelle constante qui n'a rien à voir avec le W précédent, donc rien à voir avec y
            print("W:", sess.run(W))
            print("y:", sess.run(y))




""" attention : à chaque appel de "sess.run(tensorA)" TOUT le graphe des calculs menant à "tensorA" est réévalué.
Observez : """
def step2():
    x=tf.random_normal(shape=[1])

    with tf.Session() as sess:
        x0 = sess.run(x)
        x1 = sess.run(x)
        print("deux runs:\n",x0,x1)
        x0,x1 = sess.run([x,x]) #un seul appel de run => une seule évaluation des tenseurs
        print("un seul run:\n",x0,x1)


""" Ainsi, lorsque l'on a de long calcul, il faut évitez les multiples sess.run()
 (on n'a pas respecté cette consigne pour l'instant ).
  Typiquement, la structure d'un programme est la suivante :"""
def step3():

    """"""
    """construction du graph des calculs"""
    a=tf.constant(2.,shape=[3,3])
    power=tf.get_variable("power",initializer=0.)
    b=a**power
    c=4*b+2.

    """ démarrage de la session de calcul"""
    with tf.Session() as sess:
        """ initialisation des variables"""
        sess.run(tf.global_variables_initializer())
        """l'arbre des calculs est parcouru 5 fois:"""
        for i in range(5):
            assignation=power.assign(i+power)
            _,b_,c_,power_=sess.run([assignation,b,c,power])
            print("étape:",i,", power:",power_)
            print(b_,"\n",c_)



"""  Remarquez que d'un run à l'autre, les variables gardent leur valeur précédente.
Par contre il faut penser à les initialiser à chaque démarrage de session.
Dans le programme précédent,  on aurait aussi pu exécuter l'assignation en premier :
       assignation=power.assign(i+power)
       sess.run(assignation)
       b_,c_,power_=sess.run([b,c,power])
(un assignation n'est pas coûteuse)
Par contre il aurait été dommage de faire 2 run pour calculer b et c
"""





