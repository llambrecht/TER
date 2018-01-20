import matplotlib

from TP_TER.tool.ScatterAndFunctionViewer import ScatterAndFunctionViewer

matplotlib.use('TkAgg')  # truc bizarre à rajouter spécifique à mac+virtualenv

"""ne pas supprimer (même si cela semble ne servir à rien)"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(linewidth=500, precision=2, suppress=True)

import os

"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
NOTA BENNE: pour respecter l'usage du machine learning,
 la lettre "x" désignera les "entrées" (input) et la lettre "y" désigne les "sorties" (output).
 (et non pas abscisse-ordonnée)

Par exemple une fonction de R^2 dans R pourra être écrite
                y =  sin(x0^2+x1^2)
où x0 et x1 représente les deux coordonnées de R^2
"""

""" Évaluer la fonction (x0,x1)-> sin(x0^2+x1^2) sur les points d'une grille.
C'est une fonction de R^2 dans R que nous représenterons graphiquement dans le prochain step.
"""


def step1():
    """"""
    """On crée des vecteurs de points répartis régulièrement.
    Ils définissent les abscisses et ordonnées de notre grille."""
    nbPoints_x0 = 5
    nbPoints_x1 = 3
    x0 = np.linspace(0, 1, nbPoints_x0)
    x1 = np.linspace(0, 1, nbPoints_x1)
    """on créer des matrices où se répètent x et y"""
    X0, X1 = np.meshgrid(x0, x1)
    """ On évalue la fonction (x,y)-> x^2+y^2 sur toute notre grille.
    En fait, cela revient à faire un calcul terme à terme sur les matrices X et Y"""
    R = np.sqrt(X0 ** 2 + X1 ** 2)
    """on applique la fonction sin() à la matrice R.
    Au final, nous avons évaluer la fonction (x0,x1)->sin(x0^2+x1^2)"""
    Y = np.sin(R)

    print('x0\n', x0)
    print('x1\n', x1)
    print('X0\n', X0)
    print('X1\n', X1)
    print('R\n', R)
    print('Y\n', Y)


""" On effectue un tracé 3d de la fonction (x0,x1)->sin(x0^2+x1^2).
 Inutile de passer trop de temps sur ce step : par la suite, on
 préférera les tracer en 'niveau de couleurs', moins spectaculaires mais plus précis.
 """


def step2():
    """"""
    """on créer la fonction à représenter (cf step1)"""
    x0 = np.linspace(-5, 5, 20)
    x1 = np.linspace(-5, 5, 20)
    X0, X1 = np.meshgrid(x0, x1)
    R = np.sqrt(X0 ** 2 + X1 ** 2)
    Y = np.sin(R)

    """ VIEWER """
    # la figure
    fig = plt.figure()
    # la partie mathématique de la figure (la partie dans les axes)
    axis = fig.gca(projection='3d')
    """ Dans la suite : certain méthode se lancent à partir de l'objet 'fig' et
      d'autre à partir de l'objet 'axis'.  """

    # voici ce qu'on va tracer
    surf = axis.plot_surface(X0, X1, Y,
                             rstride=1,
                             cstride=1,
                             cmap='jet',
                             linewidth=0,
                             antialiased=False)
    """options de présentation"""
    # les limites en altitude
    axis.set_zlim(-1.01, 1.01)
    # pour préciser que l'on veut 10 graduations en Z, et préciser le format des graduations
    axis.zaxis.set_major_locator(LinearLocator(10))
    axis.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # la bare de couleur sur les côtés
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # tout est prêt : on trace
    plt.show()


""" la même fonction mathématique, mais on utilise un tracé en niveau de couleur"""


def step3():
    """ """
    """on créer la fonction à représenter (cf step1)"""
    x0_min = -5
    x0_max = 5
    x1_min = -2
    x1_max = 2
    x0 = np.linspace(x0_min, x0_max, 30)
    x1 = np.linspace(x1_min, x1_max, 30)
    X0, X1 = np.meshgrid(x0, x1)
    R = np.sqrt(X0 ** 2 + X1 ** 2)
    Y = np.sin(R)
    mini = np.min(Y)
    maxi = np.max(Y)

    im = plt.imshow(Y,
                    # aspect='auto', # décommenter pour avoir une image carrée
                    interpolation='bilinear',  # commenter, ou mettre à None ou à nearest pour supprimer le lissage
                    extent=[x0_min, x0_max, x1_min, x1_max],
                    cmap='jet',
                    norm=plt.Normalize(vmin=mini, vmax=maxi)
                    )

    plt.colorbar(im)
    plt.show()


""" un scatter-plot :  des points avec un code couleur et éventuellement des légendes  """


def step4():
    nbData = 20
    x = np.random.random([nbData, 2])
    y = np.matmul(x, [1., 2.])
    mini = np.min(y)
    maxi = np.max(y)

    print('x\n', x)
    print('y\n', y)

    plt.scatter(
        x[:, 0],
        x[:, 1],
        marker='o',
        c=y,
        cmap='jet',
        norm=plt.Normalize(vmin=mini, vmax=maxi)
    )

    """mettre à false pour supprimer les labels"""
    addLabelsOnScatter = True
    if addLabelsOnScatter:
        labels = ['{:.1f}'.format(i) for i in y]
        for label, x, y in zip(labels, x[:, 0], x[:, 1]):
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(-2, 2),
                textcoords='offset points', ha='right', va='bottom'
            )
    plt.show()


""" idem, mais en utilisant un outil fournis dans TOOL"""


def step5():
    nbData = 20
    x = np.random.random([nbData, 2])
    y = np.matmul(x, [1., 2.])

    viewer = ScatterAndFunctionViewer(x, y)
    viewer.addLabelsOnScatter = True
    viewer.labelFormat = '{:.2f}'
    viewer.plot()


""" l'outil ScatterAndFunctionViewer a été conçu pour
  superposer facilement un scatter-plot et une fonction représentée en niveau de couleur.
  Attention,
  la fonction doit prendre en argument
  un tableau numpy de shape (?,2) et doit renvoyer
  un tableau numpy de shape (?)
  cf. exemple ci-dessous
  """


def step6():
    nbData = 20
    x = np.random.random([nbData, 2]) * 2 - 1
    y = np.matmul(x, [1., 2.])

    print(x.shape)

    print(y.shape)

    def sinR(x):
        return np.sin(x[:, 0] ** 2 + x[:, 1] ** 2)

    print(sinR(x).shape)

    viewer = ScatterAndFunctionViewer(x, y, sinR)
    viewer.addLabelsOnScatter = True
    viewer.labelFormat = '{:.2f}'
    """si on ne donne pas les intervalles de définition de la fonction,
    ces intervalles seront calculés au plus juste pour englober les points du scatter-plot"""
    viewer.functionInterval_x0 = [-1, 1]
    viewer.functionInterval_x1 = [-1, 1]
    viewer.plot()


step6()

""" RÉCRÉATION  (15 min)
 Créez une répartition de points qui simule une galaxie, en respectant les règles suivantes:
 La densité d'étoiles est plus forte au centre de la galaxie.
 L'énergie des étoiles est représenter par une couleur,
 Les étoiles de forte énergie se concentrent au centre de la galaxie.

 On  choisira une fonction  d'energie déterministe (x0,x1)-> énergie  que l'on représentera en niveau de couleur.
 Mais le vrai niveau d'énergie des étoiles ne suivra pas exactement cette fonction (ajoutez un bruit aléatoire)


 Inutile de faire varier la taille des étoiles.
 Aide : np.random.normal(0,1,10), donne 10 réels répartis de manière gaussienne.
 """
