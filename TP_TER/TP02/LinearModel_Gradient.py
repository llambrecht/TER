from typing import Tuple

import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import numpy as np
from TOOL.viewer.ScatterAndFunctionViewer import ScatterAndFunctionViewer
import tensorflow as tf
np.set_printoptions(precision=2,linewidth=5000)



"""
Cette classe  servira à de nombreux endroits ; elle sera donc importée par d'autre fichiers.
Une bonne tradition est de donner le même nom au fichier qu'à la classe.

A FAIRE RAPIDEMENT :
    Dans cette classe il y a deux paramètres à régler :
     le 'training_step' et le 'nbIterations'. Régler ces paramètres à chaque fois est une perte
     de temps.

     Transformer la boucle for en boucle while pour stopper l'apprentissage dès que l'écart relatif
     entre deux loss successif est inférieur à un epsilon donnée (mettez par défaut epsilon=0.001 ).

      Prenez l'habitude dès que vous écrivez une boucle while, de mettre aussi un garde fou : un paramètre
      nbIterationMax que vous pourrez régler à 10000 par exemple.






BONUS : quand vous aurez fini le reste, personnaliser cette classe à votre goût.

  Par exemple
  1/vous pouvez mettre un option qui permet de centrer-réduire les input (on en reparlera).
  Utiliser pour cela
        np.mean()
        np.std()
  Mais attention, il faudra  modifier conjointement les méthodes fit() et predict()

  2/ vous pouvez essayer de prévenir l'utilisateur contre les erreurs provoquées par :
  model=LinearModel_Gradient(2)
  print(model.W_)
  ou bien par
  model=LinearModel_Gradient(2)
  model.predict(x_test)
  --> cela crée des erreurs car l'utilisateur à oublier d'appeler la méthode .fit()


"""

class LinearModel_Gradient:

    def __init__(self, nbDescriptors:int,
                 ridgeCoef=0):
        """"""

        self.nbDescriptors=nbDescriptors

        self.training_step=0.01
        self.verbose=False


        self.buildGraph(ridgeCoef)



    def buildGraph(self, ridgeCoef):
        """"""
        """Sans cette commande: à chaque fois que l'on instancie cette classe, elle crée un nouveau graph de calcul.
        Si les graphs de calcul s'accumulent, ils encombrent la mémoire."""
        tf.reset_default_graph()

        
        self._W = tf.get_variable("W",  initializer=tf.truncated_normal(shape=[self.nbDescriptors],stddev=0.1))
        self._b = tf.get_variable("b", initializer=1.)

        self._x_train=tf.placeholder(tf.float32, shape=[None, self.nbDescriptors], name="x_train")
        self._y_train=tf.placeholder(tf.float32, shape=[None], name="y_train")


        """y[i]= sum_j W[j] * x_train[i,j] + b  """
        self._y = tf.reduce_sum(self._W * self._x_train, axis=1) + self._b

        """pour faire comme dans le cours, il faudrait mettre reduce_sum et pas reduce_mean.
            Mais quand le jeu de donnée est gros, la loss est trop grande, et l'algo du gradient échoue"""
        self._loss = tf.reduce_sum((self._y - self._y_train) ** 2) + ridgeCoef * tf.reduce_sum(self._W ** 2)
        self._loss/=tf.cast(tf.shape(self._x_train)[0],dtype=tf.float32)


        self._train = tf.train.GradientDescentOptimizer(self.training_step).minimize(self._loss)




    def _preprocessingX(self, x):
        if not isinstance(x, np.ndarray) : raise ValueError("x must be numpy array")
        """si x est un vecteur, on en fait un vecteur colonne"""
        if len(x.shape)==1 : x=x.reshape([-1,1])
        if x.shape[1]!=self.nbDescriptors: raise ValueError("x has:"+str(x.shape[1])+" columns while:"+str(self.nbDescriptors)+" is expected")
        return x


    def _preprocessingY(self, y):
        if not isinstance(y, np.ndarray) : raise ValueError("y must be numpy array")
        """si y est une matrice colonne, on en fait un vecteur"""
        if len(y.shape) == 2 :
            if y.shape[1]==1: y=y.reshape([-1])
            else: raise Exception("y must be a vector or a column matrix")
        return y


    def fit(self, x_train, y_train,nbStep=1000):

        x_train=self._preprocessingX(x_train)
        y_train=self._preprocessingY(y_train)
        if x_train.shape[0]!= y_train.shape[0]: raise ValueError("x and y must have the same number of lines")

        W_, b_ = None, None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(nbStep):
                _,W_, b_, loss_ = sess.run([self._train, self._W, self._b, self._loss], feed_dict={self._x_train:x_train, self._y_train:y_train})
                if self.verbose:
                    print("W,b,loss", W_,b_,loss_)

        """ we globalize important computations with public attributes  """
        self.W_hat=W_
        self.b_hat=b_



    def predict(self,x_test):

        x_test=self._preprocessingX(x_test)
        return np.sum(self.W_hat * x_test,axis=1) + self.b_hat




""" On génère des données et l'on teste graphiquement que la classe ci-dessus fonctionne. """
def localTest():

    def createData(nbData: int, sigma=0.1) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.random([nbData, 2]) * 2
        w = np.array([-1., 2])
        '''y[i]= sum_j x[ij] w[j]  '''
        y = np.sum(x * w, axis=1) + 5 + np.random.normal(0, sigma, size=[nbData])
        return x, y

    x_train,y_train=createData(100)

    model=LinearModel_Gradient(2)
    model.verbose=True
    model.ridgePenalisationCoef=0.1
    model.training_step=0.1
    print(x_train.shape,y_train.shape)
    model.fit(x_train,y_train)


    def hatFunction(x_test): return model.predict(x_test)

    viewer = ScatterAndFunctionViewer(x_train, y_train, function=hatFunction)
    viewer.functionInterval_x0 = [0, 2]
    viewer.functionInterval_x1 = [0, 2]
    #viewer.scatter_edgeColor="black"
    viewer.plot()

"""le test n'est lancé que lorsque l'on RUN ce fichier. """
if __name__=="__main__":
    localTest()

