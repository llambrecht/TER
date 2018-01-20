import matplotlib

from TP_M2.Classes.LinearModel_Gradient import LinearModel_Gradient

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv

from TOOL.viewer.ScatterAndFunctionViewer import ScatterAndFunctionViewer

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=1)




"""observez les données. Quelles sont les variables qualitatives et quantitatives ? """
def step1():

    z=np.loadtxt('data/deuxQuartiers_train.csv',delimiter=',')
    x,y=z[:,0:2],z[:,2]

    viewer = ScatterAndFunctionViewer(x,y)
    viewer.xLabel="surface en m^2"
    viewer.yLabel="quartier"
    viewer.title="prix en euros selon les quartiers"
    viewer.plot()




"""deux fonctions utilitaires
Parfois le mot standardiser est synonyme de centrer-réduire.
Perso, je l'utilise pour dire qu'on donne aux quantités des valeurs ni trop grande ni trop petites.
"""
def standardizeXY(x, y):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    y /= 100000
    x[:, 0] /= 100
    return x, y


def plotOneQuartier(quartier, x, y, y_hat, symbol):
    index_test_Quart = x[:, 1] == quartier
    plt.plot(x[index_test_Quart, 0], y_hat[index_test_Quart],label="quartier"+str(quartier)+", data")
    plt.plot(x[index_test_Quart, 0], y[index_test_Quart], symbol,label="quartier"+str(quartier)+", estimations")



def step4():

    z = np.loadtxt('data/deuxQuartiers_train.csv', delimiter=',')
    x_train, y_train = z[:, 0:2], z[:, 2]

    """à quoi sert cette standardisation ?
    Que se passe si on l'enlève ?
    Comment pourrait-on faire cette standardisation automatiquement (=sans regarder les données)
    Quand les variables ont des ordres de grandeurs très différent, cela marche moins bien. Relier ce phénomène à
    un phénomène observé dans le TP sur l'optimisation.
    """
    x_train,y_train=standardizeXY(x_train, y_train)


    model=LinearModel_Gradient(2)
    model.training_step=0.1
    model.verbose=True
    model.fit(x_train,y_train)


    """deux façons  d'afficher les résultats"""
    plot_levelColor=True

    if plot_levelColor:
        def function_hat(x): return model.predict(x)
        viewer = ScatterAndFunctionViewer(x_train, y_train,function_hat)
        viewer.xLabel = "surface (/100) m^2"
        viewer.yLabel = "quartier"
        viewer.title = "prix (/10000 euros) euros"
        viewer.plot()

    else:
        """"""
        """regardons si l'on  prédis correctement les données tests,
        quartier par quartier"""
        z = np.loadtxt('data/deuxQuartiers_test.csv', delimiter=',')
        x_test, y_test = z[:, 0:2], z[:, 2]

        x_test,y_test=standardizeXY(x_test, y_test)
        y_test_hat=model.predict(x_test)


        print("biais estimé:",model.b_hat)
        print("coef estimé:",model.W_hat)

        plotOneQuartier(0, x_test, y_test, y_test_hat, '+')
        plotOneQuartier(1, x_test, y_test, y_test_hat, '.')


    plt.show()



""" Commentaire : notre modèle de prix est le suivant :

    prix = W0 * surface + W1 * quartier + biais

Avec un peu de bon sens on voit que quelque chose cloche ! Écrivez sur papier un meilleurs modèle
qui fait intervenir 3 variables W0,W1,W2
Implémentez-le.
Si vous n'arrivez pas à écrire le modèle sur papier, inutile de coder au hasard :demandez l'enveloppe réponse.


Maintenant vous allez devoir estimer le prix sur 3 quartiers. Écrivez-le modèle.
Attention : il n'y a aucune raison que les numéros de quartier correspondent à un niveau de prix.

Si vous bloquez, demandez l'enveloppe réponse (mais réfléchissez longtemps, c'est tellement plus
drôle de trouver par soi même).
 """
