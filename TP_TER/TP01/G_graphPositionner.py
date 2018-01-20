import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

np.set_printoptions(linewidth=500,precision=2,suppress=True)

import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
  optimisation en grande dimension:
  On se donne un graph (des vertices et des edges).
  On essaye de positionner les vertices au mieux pour que la longueurs de chaque edges soit proche de 1.
  """


"""
Voici 3 objets qui permettent de créer un graph.
Lisez-les, cela fait une bonne révision du python-objet.
"""


class Vertex:

    def __init__(self,name:str):
        self.name=name
        self.edges=[]
        self.position=None

    def addEdge(self,vertex):
        edge=Edge(vertex)
        self.edges.append(edge)

    def asNeighbor(self,otherVertex):
        for edge in self.edges:
            if edge.arr==otherVertex: return True

        return False


class Edge:

    def __init__(self,arr:Vertex):
        self.arr=arr


class Graph:

    def __init__(self):
        self.vertices=[]

    def addUnorientedEdge(self,vert0:Vertex,vert1:Vertex):
        vert0.addEdge(vert1)
        vert1.addEdge(vert0)

    def size(self): return len(self.vertices)

    def plotMe(self, positions):

        if not isinstance(positions,np.ndarray): raise Exception("arg must be numpy array")
        if positions.shape!=(len(self.vertices),2): raise Exception("arg has not the good shape")

        for i in range(self.size()):
            self.vertices[i].position=positions[i,:]

        plt.axis('off')

        plt.plot(positions[:, 0], positions[:, 1], "o")

        plt.scatter(
            positions[:, 0],
            positions[:, 1],
            marker='o'
        )

        labels = []
        for vertex in self.vertices:
            labels.append(vertex.name)

        for label, x, y in zip(labels, positions[:, 0], positions[:, 1]):
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(-2, 2),
                textcoords='offset points', ha='right', va='bottom'
            )

        for vertex in self.vertices:
            for edge in vertex.edges:
                plt.plot([vertex.position[0],edge.arr.position[0]],[vertex.position[1],edge.arr.position[1]])



    def logMe(self):

        for vertex in self.vertices:
            print(vertex.name)
            for edge in vertex.edges:
                print('->',edge.arr.name)
                print('----------')


"""
Remarque : dans ce cadre très simple, la classe Edge est quasi-inutile.
Mais je trouve qu'elle rajoute de la clarté à la structure.
Dans quel cas est-ce vraiment nécessaire d'avoir une classe Edge ?
"""


"""
On crée un graph aléatoire.
On positionne les vertices aléatoirement.
"""
def step0():

    size=10
    graph=Graph()

    for i in range(size):
        vert=Vertex(str(i))
        graph.vertices.append(vert)


    for i in range(size):
        for j in range(size):
            if i<j and np.random.random(1)<0.3 :
                graph.addUnorientedEdge(graph.vertices[i],graph.vertices[j])

    randomPosition=np.random.random([graph.size(),2])

    graph.plotMe(randomPosition)
    plt.show()



"""
On effectue une petite animation : les vertices sont légèrement bougés aléatoirement.

Remarque : j'utilise une manière dépréciée de faire des animations. Mais la nouvelle
façon de procéder me semble tellement plus compliquée ...

Si quelqu'un me fait la même chose avec la nouvelle façon de coder, je suis preneur.
"""
def step1():

    size = 10
    graph = Graph()

    for i in range(size):
        vert = Vertex(str(i))
        graph.vertices.append(vert)

    for i in range(size):
        for j in range(size):
            if i < j and np.random.random(1) < 0.3:
                graph.addUnorientedEdge(graph.vertices[i], graph.vertices[j])


    randomPosition = np.random.random([graph.size(), 2])


    plt.ion()
    writeImg=True
    for t in range(100):
        randomPerturbation = 0.03*np.random.random([graph.size(), 2])
        randomPosition+=randomPerturbation
        graph.plotMe(randomPosition)
        plt.draw()
        if writeImg: plt.savefig('out/image%05d.png'%t, bbox_inches='tight')
        plt.pause(0.01)
        plt.clf()





"""
Avec l'option writeImg=True :
Les images seront stockées dans le sous dossier "out" (qu'il faut créer dans le dossier du TP10)
Les images porteront les noms : image00000.png, image00001.png, image00002.png etc.

Ensuite allez dans un terminal (=console) ; pas besoins d'aller loin, il y en a une intégrée à pycharm, dans les onglets du bas.
Si ce n'est pas déjà fait, installer ffmpeg  (=le "couteau suisse" pour les conversions d'images et de vidéos)
linux :
apt-get install ffmpeg
mac :
brew install ffmpeg

Dans le terminal aller sur le répertoire out/ puis pour créer un gif animé:
ffmpeg -i image%05d.png -vf palettegen palette.png
ffmpeg -i image%05d.png -i palette.png -r 10 -lavfi paletteuse movie.gif

explication : par défaut gif compresse énormément les couleurs. La première ligne commande permet de créer une palette
de couleur de bonne qualité.
Dans la seconde ligne de commande, les options :
-i image%05d.png signifie que nom de vos images  sera  image00000.png, image00001.png, image00002.png etc.
-r 10 signifie 10 images par secondes

Remarques :
* ffmpeg permet aussi de créer des vidéo mpeg4 à partir d'images
* si vous voulez automatisé vos programme, les lignes de commandes bash peuvent se lancé via python à l'aide du module 'subprocess'



A VOUS DE JOUER :
vous devez créer une fonction "energy" (=fonction loss) qui est minimale quand tous les edges ont une longueur de 1.
Ensuite effectuer une descente de gradient dessus.
Mémorisez toutes les positions obtenue.
Faites-en une animation.
Cette animation validera votre algorithme "visuellement".
Merci d'exporter vos animations et de les inclure dans votre rapport (ex: avec google-doc c'est insersion>image)

On pourra aussi  valider l'algo en affichant la courbe descendante de la fonction energy.


VARIANTE (importante) :  Faites partir votre algorithme avec un initialization non générique:
par exemple, placer tous les vertices sont sur la diagonale.
Faites tourner votre algo, observez, tirez-en une leçon qui sera primordiale pour les réseaux de neurones.



BONUS (qui demande de l'imagination, mais ne prend pas de temps)
Quelle modification pourriez-vous apporter à votre programme pour améliorer le rendu visuel du graph
(par exemple, pour éviter la superposition des vertices ?)









"""

def step4():

    size=10

    graph = Graph()

    for i in range(size):
        vert = Vertex(str(i))
        graph.vertices.append(vert)

    for i in range(size):
        for j in range(size):
            if i < j and np.random.random(1) < 0.3:
                graph.addUnorientedEdge(graph.vertices[i], graph.vertices[j])


    positions=tf.get_variable("positions",initializer=tf.truncated_normal(shape=[size,2],stddev=0.1))
    energy=0

    #TODO : la suite






