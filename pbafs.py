import numpy as np
from spark_config import sc
from bat import VirtualBat
import pbafs_utils as util
import opfython.math.general as g
import math
import const
from map_functions import classificationStep, featureSelectionStep
import pbafs_random as random
import time
from metrics import metrics

def initializeVirtualBats(number_of_features):
    virtual_bats = []
    for _ in range (1, const.population_size+1):
        virtual_bat = VirtualBat()
        virtual_bat.x = np.array(util.randomSelectedFeatures(number_of_features)) # position
        virtual_bat.v = np.array(util.initializeBatVelocities(number_of_features)) # velocity
        virtual_bat.A = random.uniform(1, 2) # loudness
        virtual_bat.r = random.uniform(0, 1) # pulse emision rate
        virtual_bat.r0 = virtual_bat.r # initial pulse emission rate
        virtual_bat.fit = -np.inf # fitness vector
        virtual_bats.append(virtual_bat)
    
    return virtual_bats


def runPBAFS(X, y, number_of_features):
    # inicializa população de morcegos
    virtual_bats = initializeVirtualBats(number_of_features)
    rdd_virtual_bats = sc.parallelize(virtual_bats)
    global_fit = -np.inf

    for t in range(0, const.max_iterations):
        start_iteration = time.time()
        # Etapa de classificação onde são construídos modelos classificadores com os atributos selecionados e extraído acurácia obtida
        start_classification = time.time()
        rdd_virtual_bats = rdd_virtual_bats.map(lambda bat: classificationStep(bat, t, X, y)).persist()

        # Monta array atualizado com a acurácia dos morcegos
        virtual_bats = rdd_virtual_bats.collect()
        
        # Pega dados do melhor morcego
        fit_array = list(map(lambda bat: bat.fit, virtual_bats))
        max_fit = max(fit_array)
        max_fit_index = fit_array.index(max(fit_array))

        # Atualiza melhor acurácia e melhores atributos selecionados
        if max_fit > global_fit:
            global_fit = max_fit
            best_features = virtual_bats[max_fit_index].x
            best_accuracy = virtual_bats[max_fit_index].fit
            best_classifier = virtual_bats[max_fit_index].bestClassifier

        # Calcula média intensidade dos morcegos
        A_avg = sum(list(map(lambda b: b.A, virtual_bats))) / const.population_size

        end_classification = time.time()
        classification_time = end_classification - start_classification
        metrics.classification_time.append(round(classification_time * 1000))
        print (f'Tempo classificador ({t+1}): ', classification_time)
        rdd_virtual_bats = rdd_virtual_bats.map(lambda bat: featureSelectionStep(bat, number_of_features, global_fit, best_features, A_avg))
        rdd_virtual_bats.first()
        end_iteration = time.time()
        metrics.iteration_time.append(round((end_iteration - start_iteration) * 1000))
    return best_features, best_accuracy, best_classifier