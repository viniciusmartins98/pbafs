# Classe que abstrai os morcegos virtuais utilizados no algoritmo
class VirtualBat:
    def __init__(self):
        self.x = [] # posição
        self.v = [] # velocidade
        self.f = 0 # frequencia
        self.A = 0 # intensidade
        self.r = 0 # taxa de emissão de pulso
        self.r0 = 0 # taxa de emissão de pulso inicial
        self.fit = 0 # vetor de ajuste
        self.model = None # modelo classificador construído para este morcego
        self.trainingData = None # dados de treinamento
        self.testData = None # dados de teste
        self.modelAccuracy = None # acurácia do modelo
        self.bestClassifier = None # acurácia do modelo