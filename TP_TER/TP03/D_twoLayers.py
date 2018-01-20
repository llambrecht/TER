import matplotlib

from TP_TER.TP03.LogisticModel_Gradient import LogisticModel_Gradient
from TP_TER.TP03.TowLayersModel import TwoLayerModel

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt

import random

import numpy as np
import tensorflow as tf

from sklearn import linear_model

from TOOL.statProba.probaTools import drand
from TOOL.viewer.ScatterAndFunctionViewer import ScatterAndFunctionViewer

import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=2,linewidth=5000,suppress=True)


"""fonction pour charger les données. Changer le chemin d'accès pour changer de jeu de données.
on commencera par 'Abis_layers_visu/species3_train.csv'
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



""" voici des jeux de données avec de nouvelles espèces de fleurs :

   Abis_layers_visu/species2_diago_train.csv
   Abis_layers_visu/species2_central_train.csv

 Observez comment le modèle logistique se plante pour ces plantes (essayez les 2 jeux de données).
 """
def step3():


    x_train, y_train, nbData, nbCategories, nbDescriptors=getData('Abis_layers_visu/species2_diago_train.csv')



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




"""On peut améliorer cette classification en étendant les variables.
On propose 2 façons d'étendre les données.
Essayez ces 2 façons sur les deux jeux de données.
Commentez.
"""
def step4():

    x_train, y_train, nbData, nbCategories, nbDescriptors=getData('Abis_layers_visu/species2_diago_train.csv')



    def standardize(x:np.ndarray):
        si=np.std(x)
        return (x-np.mean(x))/si


    def expandX_withProd(x):
        x0=standardize(x[:, 0])
        x1=standardize(x[:, 1])
        x_prod = standardize(x0*x1)

        return np.stack([x0,x1,x_prod], axis=1)

    def expandX_withNorm(x):
        x0 = standardize(x[:, 0])
        x1 = standardize(x[:, 1])
        x_norm = standardize(x0**2+x1**2)
        return np.stack([x0, x1, x_norm], axis=1)

    withProd=False
    if withProd: expandX= expandX_withProd
    else : expandX=expandX_withNorm


    nbCategories=2
    nbDescriptors=3#car on en rajoute 1


    model=LogisticModel_Gradient(nbDescriptors,nbCategories)
    model.verbose=True


    model.fit(expandX(x_train),y_train)



    categoriesVersusProba=True
    if categoriesVersusProba:
        fonc=lambda x: model.predict(expandX(x))
    else :
        fonc=lambda x: model.predict_proba(expandX(x))[:,0]

    viewer = ScatterAndFunctionViewer(
        x_scatter=x_train,
        u_scatter=y_train,
        function=fonc
    )

    viewer.scatter_edgeColor = "black"
    viewer.plot()
    model.close()





"""
Et  maintenant, vous allez construire votre premier réseau de neurone !
Ci-dessous, je vous donne le step qui va faire tourner la classe "TwoLayerModel" que vous allez inventer.
Ce step crée le film de l'apprentissage de votre modèle. Il faut bien regarder ce film jusqu'au bout.
Vous pourrez tester cela sur les 2 jeux de donnée : vous allez voir comme ce modèle tout simple paraîtra intelligent.


Qu'est ce TwoLayerModel ?
Vous partez d'un modèle logistique, et vous lui rajouter une seconde couche de neurone.

Ainsi, si le modèle logistique s'écrit :

    Y = S (L X)
S: softmax
L: transformation affine

Le modèle TowLayer c'est

   Y = S ( L ( S'(LX) ))
Avec S' qui peut-être aussi un soft max, ou bien un autre transformation non-linéaire (ex : Relu, sigmoïde)

En cours, je re-expliquerais cela.

"""
def step5():

    x_train, y_train, nbData, nbCategories, nbDescriptors=getData('Abis_layers_visu/species2_diago_train.csv')

    nbCategories=2
    nbDescriptors=2

    model=TwoLayerModel(nbDescriptors,nbCategories)
    model.nbHidden=7
    model.verbose=True


    categoriesVersusProba=True
    if categoriesVersusProba:
        fonc=lambda x: model.predict(x)
    else :
        fonc=lambda x: model.predict_proba(x)[:,0]

    plt.ion()

    try :
        while True:
            model.fit(x_train, y_train, 1000)

            viewer = ScatterAndFunctionViewer(
                x_scatter=x_train,
                u_scatter=y_train,
                function=fonc
            )

            viewer.scatter_edgeColor = "black"
            viewer.plot()

            plt.pause(0.1)
            plt.clf()
    except KeyboardInterrupt:
        print("FIN du training")

    model.close()

