# Classe que abstrai os morcegos virtuais utilizados no algoritmo
class Metrics:
    def __init__(self, classification_time=[], iteration_time=[], total_time=0, scores=[]):
        self.classification_time = classification_time
        self.iteration_time = iteration_time
        self.total_time = total_time
        self.scores = []

metrics = Metrics()