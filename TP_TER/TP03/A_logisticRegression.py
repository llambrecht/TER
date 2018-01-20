import matplotlib

from TP_TER.TP03.LogisticModel_Gradient import LogisticModel_Gradient

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv


import random

import numpy as np
import tensorflow as tf

from sklearn import linear_model
from TOOL.viewer.ScatterAndFunctionViewer import ScatterAndFunctionViewer

import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=2,linewidth=5000,suppress=True)



"""fonction pour charger les données. Changer le chemin d'accès pour changer de jeu de données.
Ces données représentes 3 espèces de fleurs caractérisées par leur taille d'étamine et de pétale (en abscisse et ordonnée)
"""
def getData(path:str):
    z = np.loadtxt(path, delimiter=',')

    x = z[:, 0:2].astype(np.float32)
    """puisque les catégories servent d'indices, il faut que ce soit des entiers"""
    y = z[:, 2].astype(np.int32)
    nbData = x.shape[0]
    nbCategories = np.max(y) + 1
    nbDescriptors = x.shape[1]

    print("nbData:", nbData, " nbCategories:", nbCategories, " nbDescriptors:", nbDescriptors)

    return x,y,nbData,nbCategories,nbDescriptors





def step2():
    x_train, y_train, nbData, nbCategories, nbDescriptors=getData('data/species3_train.csv')

    """ on transforme l'étiquette y en une dirac. Ex : 2 -> [0,0,1]
    JEU (très optionnel): savez-vous faire l'opération ci-dessous sans boucle ? = en pure numpy ? Moi je ne sais pas."""
    y_dirac = np.zeros([nbData, nbCategories],dtype=np.float32)
    for i in range(nbData):
        y_dirac[i, int(y_train[i])] = 1


    """construction du modèles dans tensorflow"""
    W = tf.get_variable(name='Weight',  initializer=tf.truncated_normal(([nbDescriptors, nbCategories]), stddev=0.1))
    b = tf.get_variable(name="bias", initializer=tf.truncated_normal([nbCategories], stddev=0.1))

    y_dirac_hat = tf.nn.softmax(tf.matmul(x_train, W) + b, name='y_dirac_hat')

    loss = - tf.reduce_mean(y_dirac * tf.log(tf.clip_by_value(y_dirac_hat, 1e-5, 1)), name='loss')

    y_hat = tf.cast(tf.argmax(y_dirac_hat,axis=1),tf.int32)
    """accuracy=proportion de bien classés"""
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_train, y_hat), tf.float32))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    """entraînement"""
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for step in range(801):
            sess.run(train)
            if step % 20 == 0:
                loss_ ,accuracy_= sess.run([loss,accuracy])
                print("loss:",loss_)
                print("accuracy:",accuracy_)








"""Le petit programme précédent à été intégré dans une classe LogisticModel_Gradient que nous exploitons maintenant.
Jetez un oeil à la classe quand vous le voulez (mais faites-le, il y a des questions dedans)."""
def step3():
    x_train, y_train, nbData, nbCategories, nbDescriptors=getData('data/species2_train.csv')


    model=LogisticModel_Gradient(nbDescriptors,nbCategories)
    model.verbose=True

    model.fit(x_train,y_train)

    """décrivez ce qu'on affiche quand on met False ci-dessous"""
    categoriesVersusProba=True
    if categoriesVersusProba:
        fonc=lambda x: model.predict(x)
    else :
        fonc=lambda x: model.predict_proba(x)[:,0]

    viewer = ScatterAndFunctionViewer(
        x_scatter=x_train,
        u_scatter=y_train,
        function=fonc
    )

    viewer.scatter_edgeColor = "black"
    viewer.plot()
    model.close()





"""
Nous effectuons une classification via un modèle logistique fait avec tensorflow, ou avec le modèle
logistique de sklearn
"""
def step4():
    x_train, y_train, nbData, nbCategories, nbDescriptors=getData('data/species3_train.csv')

    """mettre à False pour voir le modèle logistic via sklearn (cf. commentaires plus bas)"""
    ourModel=True

    if ourModel:
        model = LogisticModel_Gradient(nbDescriptors, nbCategories)
        model.verbose = False
    else:
        model=linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg')

    model.fit(x_train, y_train)


    """ analysons les prédiction du modèle sur les données train (qui ont entraîné le modèle)"""
    x=x_train
    y=y_train
    y_hat=model.predict(x)

    print("\nJEU TRAIN")
    print("y    ",y)
    print("y_hat",y_hat)
    print("accuracy sur le train:",model.accuracy)



    """ analysons les prédiction du modèle sur les données tests"""
    x_test, y_test, nbData, nbCategories, nbDescriptors=getData('data/species3_test.csv')
    x=x_test
    y=y_test

    y_hat=model.predict(x)
    print("\nJEU TEST")
    print("y    ", y)
    print("y_hat", y_hat)
    print("accuracy sur le test:", model.accuracy)

    """Il y a un truc étrange sur les accuracy non ?
    cherchez à comprendre."""



    if ourModel: model.close()



"""
====================================TRAVAIL 1======================================================
Faites vous un petit outil qui calcul la matrice de confusion.
Calculez aussi le pourcentage donnée par la classification triviale : c'est la classification qui
classe tout le monde dans la catégorie la plus importante.
C'est toujours important d'avoir ce chiffre en tête.



Refaites tourner ces programmes avec le jeu de données species2.
Particularité :  les deux catégories n'ont pas le même nombre de point.
La catégorie minoritaire est écrasée par la majoritaire.
Regardez la matrice de confusion :
En temps que classifier, le modèle logistique fait à peine mieux que le classifier trivial.
Les éléments de la classe minoritaires sont très mal détectée. Imaginez que c'est une espèce malade, on n'en soignera
que la moitié. Catastrophe !


Par contre il ne faut pas perdre de vue que le modèle logistique ne fait pas que classifier : il renvoie aussi des probabilités.
Utilisez ces probabilités pour mieux détecter la catégorie minoritaire : faite du favoritisme en changeant les seuils.


Pour illustrer votre nouvelle classification vous pouvez le faire graphiquement avec
viewer = ScatterAndFunctionViewer(
        x_scatter=x_train,
        u_scatter=y_train,
        function=fonc
    )
en modifiant fonc.



BONUS (facile) : pour le jeu species2, dresser la courbe ROC en faisant varier le seuil de probabilité pour être dans la classe 0.
Pas besoin d'aller chercher dans une librairie, il s'agit d'une simple boucle à faire après la classification.

Remarque : le favoritisme détériore forcément le nombre global de "bien classé" (l'accuracy).
Mais il faut mieux soigner quelques non-malades, que de ne pas soigner des malades (quand le médicament n'est pas trop nocif).



====================================TRAVAIL 2======================================================

comparez notre classifier avec celui de sklearn. La doc se trouve ici :

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Ils sont un tout petit peu meilleurs que nous.  Mais on n'a pas trop forcer sur le nombre d'itération.
    Mais ce qui est pénible avec sklearn c'est qu'il n'est pas facile de savoir exactement quelles formules ils ont utilisées. Voici ce que j'en ai
    compris sur la signification des options

         * multi_class='multinomial' => le modèle logistique tel que nous l'avons vu. Essayer de comprendre ce que fait l'option par défaut.

         * solver='newton-cg' => la méthode de newton-Raphston, très adapté puis la fonction loss est convexe

         * C  (= 1 par défaut).   "Inverse of regularization strength".
         Donc quand il est petit la pénalisation est plus forte.
         Donc on ne sait pas comment faire pour mettre la pénalisation à 0.
         S'agit-il d'une pénalisation L1 ou L2 ?  (lissez la doc, ça c'est très clair)

     N'hésitez pas à préciser les choses dans votre rapport, et  faites des test en variant quelques paramètres.

"""

