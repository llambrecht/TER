import numpy as np
import matplotlib.pyplot as plt


"""pour la docstring (pas finie) je me suis basé sur : http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html"""
class ScatterAndFunctionViewer:

    """
    Args:
        x_scatter:  points to scatter. np.array of order 2, with nbPoints lines and 2 columns corresponding to x-coordinate and y-coordinate
        u_scatter:  values associated to points. np.array of order 1, with nbPoints.

        function: the function which is plot in color-level under the scatter points.
        this function must take as argument :
            a numpy array of shape (?,2)
        and must return
            a numpy array of shape (?)
        Example:
            def function(x):
                return x[:,0]**2+x[:1,0]


    Attributes:
        functionInterval_x0 (list,optional): if None, it will be ( min x_scatter[:,0],max x_scatter[:,0]  )
        functionInterval_x1 (list,optional): if None, it will be ( min x_scatter[:,1],max x_scatter[:,1]  )

    """

    def __init__(self, x_scatter, u_scatter, function=None):

        self._checkArgs(x_scatter, u_scatter, function)


        self.function = function
        self.x_scatter = x_scatter
        self.u_scatter = u_scatter
        if x_scatter is not None and self.u_scatter is None: self.u_scatter=np.ones(len(x_scatter))


        """paramètre que l'utilisateur peut changer"""
        self.resolutionOfFunctionGrid = 300
        self.markerForTrain = 'o'
        self.addLabelsOnScatter=False
        self.colorMap = 'jet'# 'Spectral'#,'viridis',
        self.labelFormat='{:.1f}'
        self.addColorBar=True
        self.xLabel=None
        self.yLabel=None
        self.title=None
        self.scatter_edgeColor=None

        self.functionInterval_x0 = None
        self.functionInterval_x1 = None


    def _checkArgs(self, x_scatter, u_scatter, function):

        if x_scatter is None and function is None: raise Exception("either the x_scatter or the function must be not none")

        if x_scatter is not None and u_scatter is not None and len(x_scatter) != len(u_scatter): raise Exception(
            "x_scatter and u_scatter must have the same nb line")

        if x_scatter is not None :
            if not isinstance(x_scatter,np.ndarray): raise ValueError("first arg must be a numpy array")
            if not len(x_scatter.shape)==2: raise ValueError("first arg must be a matrix")
            if x_scatter.shape[1]!=2: raise ValueError("first arg must have 2 column")

        if u_scatter is not None: self._checkIsNumpyVector(u_scatter)



        if function is not None:
            """ne pas mettre de vecteur constant, au cas où function se livre à un centrage des données"""
            x_bidon=np.random.random([5,2])

            #try:
            y_bidon=function(x_bidon)
            #except:
            #   raise ValueError("your function do not works on numpy matrix with two column")
            self._checkIsNumpyVector(y_bidon)



    def _checkIsNumpyVector(self, vec):

        if not isinstance(vec, np.ndarray): raise ValueError("be a numpy array")
        if not len(vec.shape) == 1: raise ValueError("must be a vector")



    def plot(self,call_plt_show=True):
        """"""
        """on calcul les min-max de u_scatter et du résultat de la fonction pour les mettre dans une même échelle de couleur"""
        if self.function is not None and self.u_scatter is not None:
            u_function = self._computeFunction(self.function)
            mini = min(np.min(u_function), np.min(self.u_scatter))
            maxi = max(np.max(u_function), np.max(self.u_scatter))
            self._draw_function(u_function, mini, maxi)
            self._draw_scatter_Plot(self.x_scatter,self.u_scatter,mini,maxi)
        elif self.function is not None :
            u_function = self._computeFunction(self.function)
            mini = np.min(u_function)
            maxi = np.max(u_function)
            self._draw_function(u_function, mini, maxi)
        elif self.u_scatter is not None:
            mini=np.min(self.u_scatter)
            maxi=np.max(self.u_scatter)
            self._draw_scatter_Plot(self.x_scatter,self.u_scatter,mini,maxi)


        if call_plt_show: plt.show()



    def _computeFunction(self,function):

        if self.x_scatter is None:
            if self.functionInterval_x0 is None or self.functionInterval_x1 is None : raise Exception("without scatter points you must give an intervalle for the function")
        if self.functionInterval_x0 is None : self.functionInterval_x0=[np.min(self.x_scatter[:, 0]), np.max(self.x_scatter[:, 0])]
        if self.functionInterval_x1 is None : self.functionInterval_x1 = [np.min(self.x_scatter[:, 1]), np.max(self.x_scatter[:, 1])]


        X0=np.linspace(self.functionInterval_x0[0], self.functionInterval_x0[1], self.resolutionOfFunctionGrid)
        X1=np.linspace(self.functionInterval_x1[0], self.functionInterval_x1[1], self.resolutionOfFunctionGrid)

        x_0, x_1 = np.meshgrid(X0, X1)
        x_0 = np.reshape(x_0, [ -1,1])
        x_1 = np.reshape(x_1, [-1,1])
        x_01 = np.concatenate([x_0, x_1], axis=1)

        u = function(x_01)
        u = np.reshape(u, [self.resolutionOfFunctionGrid, self.resolutionOfFunctionGrid])

        return u[::-1,:]



    def _draw_function(self,u_function,mini,maxi):


        """"""
        """aspect=auto pour que l'image s'adapte"""
        extend=[self.functionInterval_x0[0], self.functionInterval_x0[1] , self.functionInterval_x1[0], self.functionInterval_x1[1]]


        """rajouter  interpolation='nearest' si l'on ne veut pas lisser l'image"""
        im=plt.imshow(u_function,
                      aspect='auto',
                      extent=extend,
                      cmap=self.colorMap,
                      norm=plt.Normalize(vmin=mini, vmax=maxi)
                      )

        if self.addColorBar: plt.colorbar(im)


    def _draw_scatter_Plot(self, x_scatter, u_scatter,mini,maxi):

        #plt.subplots_adjust(bottom=0.1)
        sca=plt.scatter(
            x_scatter[:, 0],
            x_scatter[:, 1],
            marker='o',
            c= u_scatter,
            edgecolors=self.scatter_edgeColor,
            cmap=self.colorMap,
            norm=plt.Normalize(vmin=mini, vmax=maxi)
        )


        if self.addLabelsOnScatter and u_scatter is not  None:
            labels = [self.labelFormat.format(i) for i in u_scatter]

            for label, x, y in zip(labels, x_scatter[:, 0], x_scatter[:, 1]):
                plt.annotate(
                    label,
                    xy=(x, y),
                    xytext=(-2, 2),
                    textcoords='offset points', ha='right', va='bottom'
                )

        """si self.function n'est pas None, c'est le plot de la fonction qui ajoute le colorbar """
        if self.addColorBar and self.function is None: plt.colorbar(sca)

        if self.xLabel: plt.xlabel(self.xLabel)
        if self.yLabel: plt.ylabel(self.yLabel)
        if self.title: plt.title(self.title)


        #plt.xlabel("rating")
        #plt.ylabel("age")





if __name__=='__main__':

    nb_scater=10
    x_scatter=np.zeros([nb_scater, 2])
    for i in range(nb_scater) : x_scatter[i,:]=[i,i]
    u_scatter=np.linspace(0,100,nb_scater)

    def function(x): return 3*(x[:,0]+x[:,1])

    viewer=ScatterAndFunctionViewer(x_scatter=x_scatter,u_scatter=u_scatter,function=function)
    viewer.logScale=True
    viewer.plot()






#
#
# class TwoInputViewerOld:
#
#     def __init__(self,model:AbstractClassifier,x_train,u_train):
#         self.model=model
#         self.x_train=x_train
#         self.u_train=u_train
#
#         """paramètre que l'utilisateur peut changer"""
#         self.nbPointInTestGrid = 40
#         #self.markerForTest='+'
#         self.markerForTrain='o'
#
#
#     def show(self):
#         absTest = np.linspace(0, 1, self.nbPointInTestGrid)
#         x_test1, x_test2 = np.meshgrid(absTest, absTest)
#         x_test1 = np.reshape(x_test1, [1, -1])
#         x_test2 = np.reshape(x_test2, [1, -1])
#         x_test = np.concatenate([x_test1, x_test2], axis=0)
#
#         hatU = self.model.getHatU(np.transpose(x_test))
#
#
#         imgHatU=np.reshape(hatU,[self.nbPointInTestGrid,self.nbPointInTestGrid])
#
#
#         cm = plt.cm.get_cmap('RdYlBu')
#         #cm = plt.cm.get_cmap('hsv')
#
#         plt.imshow(imgHatU,extent=[0.,1,0,1])
#
#         plt.scatter(self.x_train[:, 0], self.x_train[:, 1], marker=self.markerForTrain, s=50, c=self.u_train )
#         #plt.scatter(x_test[0, :], x_test[1, :], marker=self.markerForTest, s=15, c=hatU,cmap=cm)
#
#         plt.show()
#



    # def show(self):
    #     absTest = np.linspace(0, 1, self.nbPointInTestGrid)
    #     x_test1, x_test2 = np.meshgrid(absTest, absTest)
    #     x_test1 = np.reshape(x_test1, [-1,1])
    #     x_test2 = np.reshape(x_test2, [-1,1])
    #     x_test = np.concatenate([x_test1, x_test2], axis=1)
    #
    #     hatU = self.model.getHatU(x_test)
    #     cm = plt.cm.get_cmap('RdYlBu')
    #     # cm = plt.cm.get_cmap('hsv')
    #
    #
    #     plt.scatter(self.x_train[:, 0], self.x_train[:, 1], marker=self.markerForTrain, s=50, c=self.u_train, cmap=cm)
    #     plt.scatter(x_test[0, :], x_test[1, :], marker=self.markerForTest, s=15, c=hatU, cmap=cm)
    #     plt.show()


