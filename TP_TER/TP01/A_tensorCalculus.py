import tensorflow as tf
import numpy as np

np.set_printoptions(linewidth=500,precision=2,suppress=True)

import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''  le graphe de tensor flow'''
def step0():
    x=5
    y=5*x
    z=x*x
    res=y+z
    """ programmation classique: les opérations sont effectuées au fur et à mesure"""
    print("résultat avec calcul classique",res)

    """refaisons la même chose en tensorflow"""
    x=tf.constant(5)
    """comme x est un tenseur, y,z,et res seront aussi des tenseurs"""
    y=5*x
    z = x * x
    res = y + z
    """jusqu'ici il ne c'est rien passé.
    tensorflow va analyser les calculs à faire, dresser un arbre de calcul
       (x=5)
       |    \
    (y=5*x,z=x*x)
       |    /
    (res = y + z)
    Rem : les calculs (y=5*x,z=x*x) peuvent être parallélisés
    """
    sess = tf.Session()
    print("résultat avec calcul en arbre",sess.run(res))
    """ à la fin, on ferme la session de calcul"""
    sess.close()





""" le 'with' de python crée un block de code particulier appelé Context-Manager.
Dans le cas de l'objet tf.Session() cela fait au début:
    sess=tf.Session()
         et à la fin :
    sess.close()

    Si vous voulez en savoir plus sur le 'with', lisez le dernier fichier de ce TP (facultatif)
    """
def step0bis():
    x = tf.constant(5)
    y = 5 * x
    z = x * x
    res = y + z

    with tf.Session() as sess:
        print("résultat avec calcul en arbre", sess.run(res))





'''créer un tenseur. Accéder à ses éléments. Connaître sa forme (shape)'''
def step1():

    tensor1=tf.constant([[1,2,1],[3,4,3]])
    grosTenseur=tf.constant([[[1,2,0],[3,4,0]],[[5,6,0],[7,8,0]]])

    with tf.Session() as sess:
        print("tensor1\n",sess.run(tensor1))
        print("tf.shape(tensor1): ",sess.run(tf.shape(tensor1)))
        print("tensor1[0,1]: ",sess.run(tensor1[0,1]))
        print("tensor1[0,:]: ",sess.run(tensor1[0,:]))
        print("grosTenseur\n",sess.run(grosTenseur))
        """ Attention: tf.shape(tenseur) donne la vrai forme du tenseur (au moment du run).
        Tandis que tenseur.shape donne la taille-statique
        (statique signifie: avant les calculs qui peuvent éventuellement changer la forme) """
        print("tf.shape(grosTenseur): ",sess.run(tf.shape(grosTenseur)))

    '''apprenez très vite à repérer les indices dans les sorties consoles.
    tenseur de rang 1 : sont présentés en lignes.
    tenseur de rang 2 : sont mis en matrices.  l'indice 0 (i) balaye les lignes, l'indice 1 (j) balaye les colonnes

    tenseur de rang 3 : plusieurs matrices.
     l'indice 0 (i) balaye les matrices,
     l'indice 1 (j) balaye les lignes de chaque matrice
     l'indice 2 (k) balaye les colonnes de chaque matrice
    voici comment sont les indices
    [i=0: [ j=0:[ k=0 k=1 k=2]
            j=1:[ k=0 k=1 k=2]]

    [i=1: ...

    '''



''' Les types des nombre dans les tenseurs (float32, int32, ...).
si on print un tenseur sans faire appel à sess.run, on a pas sa valeur, mais son 'name', sa 'shape'-statique et son dtype (Abis_layers_visu-type)  '''
def step2():
    x=tf.constant([5,5],name="x") # par défaut en int 32
    y=tf.constant([5.,5],name="y") # par défaut en float 32 (quelle est la différence avec la ligne d'avant ?)
    z=tf.constant([5,5.5],dtype=tf.float64,name="z") # pour travailler en double précision
    t=tf.cast(z,tf.int32,name="t")

    print("x, sans run:",x)
    print("y, sans run:",y)
    print("z, sans run:",z)
    print("t, sans run:",t)

    with tf.Session() as sess:
        print ('x',sess.run(x))
        print('y',sess.run(y))
        print('z', sess.run(z))
        print('t', sess.run(t))

    '''attention aux erreurs de types: ci-dessous une erreur que vous ferez souvent : essayer de combiner (ici additionner) deux tenseurs de types différents.
    DANS LE RAPPORT : indiquez la partie "instructive du message d'erreur" produit '''
    #err=tf.add(x,y)
    #print('err',err)




"""opération élémentaire sur un tenseur"""
def step3():
    tensor1= tf.constant(1, shape=[3, 2])
    tensor2= tf.transpose(tensor1)

    """on fait la somme selon l'indice 0 (i)  :
           = sum_i tensor1[i,j]  """
    tensor3= tf.reduce_sum(tensor1,axis=0)

    tensor4= tensor1*3
    tensor5= tensor1+tensor1

    with tf.Session() as sess:
        print("tensor1:\n",sess.run(tensor1))
        print("tensor2:\n",sess.run(tensor2 ))
        print("tensor3:\n",sess.run(tensor3 ))
        print("tensor4:\n",sess.run(tensor4 ))
        print("tensor5:\n",sess.run(tensor5 ))

        """vérifions que toutes ces opérations n'ont pas affecté le tensor1"""
        print('tensor1 pour la seconde fois',sess.run(tensor1))
        """tout comme avec numpy, la majorité des opérations ne sont pas 'in-place'. On a donc affaire à un langage 'fonctionnel'   """






"""comment utiliser des dimensions supplémentaires pour éviter des boucles (et ainsi rendre possible l'optimisation)
Attention: l'extension de dimension est difficile au début. Conseil : toujours écrire les indices comme ci-dessous.
Mais
"""
def step4():
    size=3
    data=[i for i in range(size)]


    """deux vecteurs 'lignes' """
    tensor1 = tf.constant(data)
    tensor2=tf.constant(data)
    """tensor1Exp[ij]=tensor1[j] """
    tensor1Exp=tf.expand_dims(tensor1,0)
    """tensor2Exp[ij]=tensor2[i] """
    tensor2Exp=tf.expand_dims(tensor2,1)
    """
    tensor3[ij]= tensor1Exp[ij] + tensor2Exp[ij]
              = tensor1[j]     + tensor2[i]
    """
    tensor3=tensor1Exp+tensor2Exp


    """on peut additionner,multiplier, soustraire,  des tenseurs de dimensions différentes.
    Par défaut tf (comme np) fait des expand_dims(,0). Cette extension automatique s'appelle le broadcast.
    Remarquons qu'en math, on fait naturellement des extensions, par exemple quand on effectue:
     Sum_i (a[i] - b )^2
     V[i]= Sum_i (a[ij] - b[i] )^2
     M[ij]=a[i]+b[j]
     """
    tensor4=tf.constant([[1.,2,3],[4,5,6]])
    tensor5=tf.constant([10.,10,10])
    tensor6=tf.constant(100.)
    tensor7=tensor4+tensor5
    tensor8=tensor6+tensor7

    with tf.Session() as sess:
        print("tensor1:\n",sess.run(tensor1 ))
        print("tensor2:\n",sess.run(tensor2 ))
        print("tensor1Exp:\n",sess.run(tensor1Exp ))
        print("tensor2Exp:\n",sess.run(tensor2Exp ))
        print("tf.shape(tensor1):\n",sess.run(tf.shape(tensor1) ))
        print("tf.shape(tensor1Exp):\n",sess.run(tf.shape(tensor1Exp) ))
        print("tensor3:\n",sess.run(tensor3 ))
        print("tensor7:\n",sess.run(tensor7))
        print("tensor8:\n",sess.run(tensor8) )


"""coller des tenseurs"""
def step5():
    tensor1=tf.constant([1.,2,3])
    tensor2=tf.constant([0.,7,1])

    tensor3=tf.stack([tensor1,tensor2],axis=0)
    tensor4=tf.stack([tensor1,tensor2],axis=1)
    with tf.Session() as sess:
        print("tensor1:\n", sess.run(tensor1))
        print("tensor2:\n", sess.run(tensor2))
        print("tensor3:\n", sess.run(tensor3))
        print("tensor4:\n", sess.run(tensor4))


"""coller des tenseurs bis"""
def step5Bis():
    """"""
    """comparez avec le step d'avant, jouez au jeu des 4 différences"""
    tensor1=tf.constant([[1.,2,3]])
    tensor2=tf.constant([[0.,7,1]])

    tensor3=tf.concat([tensor1,tensor2],axis=0)
    tensor4=tf.concat([tensor1,tensor2],axis=1)
    with tf.Session() as sess:
        print("tensor1:\n", sess.run(tensor1))
        print("tensor2:\n", sess.run(tensor2))
        print("tensor3:\n", sess.run(tensor3))
        print("tensor4:\n", sess.run(tensor4))

    """ Personnellement je retiens cela en me disant que :
      tf.concat([tensor1,tensor2],axis=0)  fait une 'union' suivant l'indice 0 (i)
      tf.concat([tensor1,tensor2],axis=1)  fait une 'union' suivant l'indice 1 (j)
Tout comme
      tf.reduce_sum(tensor1,axis=0)  fait une 'somme' suivant l'indice 0 (i)
      tf.reduce_sum(tensor1,axis=0)  fait une 'somme' suivant l'indice 1 (j)
      """



""" Numpy est très souvent utiliser pour les entrées et sorties de tensorflow."""
def step6():

    """"""
    """ENTRÉES """
    tensor1_np=np.zeros(shape=[3,3])
    for i in range(3):
        for j in range(3):
            tensor1_np[i,j]=i+j

    tensor2_np=np.zeros(shape=[3])
    for i in range(3): tensor2_np[i]=i

    """pré-traitement (on remplace des zéros par des -1 parce qu'on aime pas les zéros)"""
    tensor1_np[tensor1_np==0]=-1

    print("tensor1_np\n",tensor1_np)



    """on transforme les np-tenseurs en tf-tenseurs  """
    tensor1_tf=tf.constant(tensor1_np)
    tensor2_tf=tf.constant(tensor2_np)

    """on écrit le graph des calcul"""
    tensor3_tf=tf.reduce_sum(tensor1_tf*tensor2_tf,axis=0)
    """qu'obtient-on sans axis=0 ?"""


    with tf.Session() as sess:
        """ SORTIE : sess.run renvoie des np-tenseurs """
        tensor3_np=sess.run(tensor3_tf)

    """ POST-TRAITEMENT (ici, simple affichage)"""
    np.set_printoptions(precision=1)
    print("tensor3:\n",tensor3_np)


""" Exo : dans le step précédent, nous avons effectuer la multiplication matricielle tensor1 . tensor2
  Modifiez ce programme pour qu'il donne la multiplication matricielle tensor2 . tensor1

   Comme pour numpy, les multiplications matricielles peuvent se faire avec tf.matmul(A,B). Mais il faut nécessairement
   que A et B soit des matrices (donc des tenseurs d'ordre 2).
   
   On peut aussi utiliser tf.dot(A,B)  qui permet de faire une multiplication entre deux vecteurs (produit scalaire),
    deux matrices, ou bien une matrice et un vecteur. Attention, cette souplesse peut-être source d'erreur.
    

    
    Quand on travaille tenseurs d'ordre >2, je vous conseille plutôt d'utiliser les expand_dim.
    Avec les réseaux de neurone, on travaille souvent avec des tenseurs d'ordre 3 ou 4.
    
   """