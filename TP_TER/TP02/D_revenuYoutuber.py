import matplotlib

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import numpy as np


""" On étudie le lien entre le nombre de followers et le revenu d'un youtuber.
 Observez les données train avant de créer un modèle (observez bien !)

 Établissez un modèle.
 Implantez-le (=programmer) et  ajustez-le (en franglais on dit "fitter").
 Estimez les revenus des données tests.
 Mettez-le tout sur un même graphique.

 Attention : réglez bien le training_step pour que cela converge.
 A quoi reconnaît-on un apprentissage qui diverge ?


 Remarque: le programme demandé fait 10 lignes (en utilisant la classe LinearModel).
 Il faut s'inspirer du cours.
 Si vraiment vous bloquez, demandez l'enveloppe réponse.
 """

def step1():

    z_train=np.loadtxt('data/revenus_train.csv',delimiter=',')
    x_train=z_train[:,0]
    y_train=z_train[:,1]

    plt.plot(x_train,y_train,'+',label='train')
    plt.xlabel("kilo-followers")
    plt.ylabel("kilo-euros")
    plt.show()


step1()

