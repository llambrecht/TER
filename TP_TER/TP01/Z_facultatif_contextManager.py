

""" """

""" Voici un petit fichier pour s'améliorer en python.
  Le "context manager" permet de créer un objet avec le mot clef with, et de "fermer" cet objet dès que l'on
  sort du block with.

  voici tout d'abord le premier exemple qui ressort en premier dans les tutos sur internet.

  """


class File:

    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.open_file = open(self.filename, self.mode)
        return self.open_file


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.open_file.close()






def step0():

    """"""
    """on ouvre le fichier, on écrit dedans (mode=w) et on le referme"""
    with File('foo.txt', 'w') as f:
        f.write('foo1')
        f.write('foo2')
        f.write('foo3')

    """on ouvre le fichier, on lit dedans (mode:r) et on le referme"""
    with File('foo.txt', 'r') as f:
        for line in f:
            print(line)


    """en fait, cet exemple est  redondant avec le context manager qui existe déjà : """
    with open('foo.txt', 'r') as f:
        for line in f:
            print(line)








"""
Je vous laisse analyser ce second exemple.
Nous avons volontairement séparer le context Manager
de l'objet qu'il fabrique (celui renvoyer par le mot clef "as")
"""




class Homme:

    def __init__(self):
        self.personalData = []

    def addAPersonalData(self, data):
        self.personalData.append(data)

    def toutMontrer(self):
        for data in self.personalData:
            print(data)



class HommeContextManager:
    def __enter__(self):
        self.homme=Homme()
        return self.homme

    """quand l'homme meurt, on détruit toutes ces données personnelles"""
    def __exit__(self, exc_type, exc_val, exc_tb):
        for data in self.homme.personalData:
            for key in data.keys():
                data[key]=None




def step1():

    with HommeContextManager() as homme:

        data1={"banckAccount":3424563456}
        data2={"pasword":"mon petit poney"}

        homme.addAPersonalData(data1)
        homme.addAPersonalData(data2)

        homme.toutMontrer()


    print("après la mort de l'homme:")
    print(data1["banckAccount"])
    print(data2["pasword"])











