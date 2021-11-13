import pbafs_utils as util
import math
import time
import numpy as np

import const
import pbafs_random as random
from classifiers import getBestClassifierAndAccuracy

# função map necessida parâmetro "t" iteração
def classificationStep(bat, iteration, X, y):

    # Remove atributos não selecionados e reduz dimensionalidade
    X_selected = util.removeFeaturesUnselected(bat.x, X)

    # Retorna melhor acurácia entre os classificadores
    best_classifier = getBestClassifierAndAccuracy(X_selected, y)

    accuracy = best_classifier.score
    best_classifier_name = best_classifier.name
    features_selected = bat.x

    print(f'Acurácia: {accuracy}')
    print(f'Classificador: {best_classifier_name}')
    print(f'Atributos: {features_selected}')
    
    rand = random.uniform(0, 1)
    if rand < bat.A and accuracy > bat.fit:
        bat.fit = accuracy
        bat.A = bat.A * const.alfa
        bat.r = bat.r0 * (1 - math.exp(-const.gama * iteration)) # 1 é o valor t
        bat.bestClassifier = best_classifier_name

    return bat


# Realiza etapa de seleção dos atributos de cada morcego com base na taxa da emissão de pulso(r) e intensidade (A)
def featureSelectionStep(bat, number_of_features, global_fit, best_features, A_avg):
    beta = random.uniform(0, 1)
    rand = random.uniform(0, 1)
    if rand > bat.r:
        for j in range(0, number_of_features):
            e = random.uniform(-1, 1)
            bat.x[j] = bat.x[j] + (e * A_avg)
            
            sigma = random.randint(0, 1)
            if (sigma < (1 / (1 + math.exp(-bat.x[j])))):
                bat.x[j] = 1
            else:
                bat.x[j] = 0
        
    rand = random.uniform(0, 1)
    if (rand < bat.A) and (bat.fit < global_fit):
        for j in range(0, number_of_features):
            bat.f = const.fmin + (const.fmax - const.fmin) * beta
            bat.v[j] += (best_features[j] - bat.x[j]) * bat.f
            bat.x[j] += bat.v[j]
            
            sigma = random.randint(0, 1)
            if (sigma < (1 / (1 + math.exp(-bat.x[j])))):
                bat.x[j] = 1
            else:
                bat.x[j] = 0
    return bat