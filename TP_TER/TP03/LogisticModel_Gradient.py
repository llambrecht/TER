from typing import Tuple

import matplotlib

matplotlib.use('TkAgg')  # truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import numpy as np
from TOOL.viewer.ScatterAndFunctionViewer import ScatterAndFunctionViewer
import tensorflow as tf

np.set_printoptions(precision=2, linewidth=5000)


class LogisticModel_Gradient:
    def __init__(self, nbDescriptors: int,nbCategories:int):
        """"""

        self.relativePrecisionToStop = 0.001
        self.maxNbSteps = 10000
        self.training_step = 0.5

        self.nbDescriptors = nbDescriptors
        self.nbCategories=nbCategories
        self.verbose = False
        self._wasFit = False

        self.buildGraph()


    def buildGraph(self):


        self._W = tf.get_variable(name='Weight', initializer=tf.truncated_normal(shape=[self.nbDescriptors, self.nbCategories],stddev=0.1))
        self._b = tf.get_variable(name="bias", initializer=tf.truncated_normal([self.nbCategories], stddev=0.1))


        self._x=tf.placeholder(name="x",dtype=tf.float32,shape=[None,self.nbDescriptors])
        self._y=tf.placeholder(name="y",dtype=tf.int32,shape=[None])

        self._y_dirac=tf.one_hot(self._y,self.nbCategories)

        self._y_dirac_hat = tf.nn.softmax(tf.matmul(self._x, self._W) + self._b, name='y_dirac_hat')
        self._y_hat = tf.cast(tf.argmax(self._y_dirac_hat, axis=1), tf.int32)

        self._loss = - tf.reduce_mean(self._y_dirac * tf.log(tf.clip_by_value(self._y_dirac_hat, 1e-5, 1)), name='loss')

        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._y, self._y_hat), tf.float32))
        self._train = tf.train.GradientDescentOptimizer(self.training_step).minimize(self._loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def fit(self, x_train, y_train,nbStep=1000):


        self._wasFit = True

        x_train = self._preprocessingX(x_train)
        y_train = self._preprocessingY(y_train)
        if x_train.shape[0] != y_train.shape[0]: raise ValueError("x and y must have the same number of lines")


        for step in range(nbStep):
            _, self.W_hat, self.b_hat, self.loss,self.accuracy = self.sess.run([self._train, self._W, self._b, self._loss,self._accuracy],
                                            feed_dict={self._x: x_train, self._y: y_train})

            if self.verbose and step % 20 == 0:
                print("loss:", self.loss)
                print("accuracy:", self.accuracy)



    def predict(self, x_test):

        if not self._wasFit: raise Exception("the model was not fitted")

        x_test = self._preprocessingX(x_test)

        y_=self.sess.run(self._y_hat,feed_dict={self._x:x_test})

        return y_



    def predict_proba(self, x_test):

        if not self._wasFit: raise Exception("the model was not fitted")

        x_test = self._preprocessingX(x_test)

        y_dirac_hat=self.sess.run(self._y_dirac_hat,feed_dict={self._x:x_test})

        return y_dirac_hat



    def _preprocessingX(self, x):
        if not isinstance(x, np.ndarray): raise ValueError("x must be numpy array")
        """si x est un vecteur, on en fait un vecteur colonne"""
        if len(x.shape) == 1: x = x.reshape([-1, 1])
        if x.shape[1] != self.nbDescriptors: raise ValueError(
            "x has:" + str(x.shape[1]) + " columns while:" + str(self.nbDescriptors) + " is expected")
        return x

    def _preprocessingY(self, y):
        if not isinstance(y, np.ndarray): raise ValueError("y must be numpy array")
        """si y est une matrice colonne, on en fait un vecteur"""
        if len(y.shape) == 2:
            if y.shape[1] == 1:
                y = y.reshape([-1])
            else:
                raise Exception("y must be a vector or a column matrix")
        if np.max(y) > self.nbCategories - 1: raise ValueError(
            " the value:" + str(np.max(y)) + " was found in y  while:" + str(
                self.nbCategories - 1) + " is the maximal value")
        return y.astype(np.int32)




    def close(self):
        self.sess.close()
        tf.reset_default_graph()
