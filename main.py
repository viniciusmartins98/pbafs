import os
import pandas as pd
import time
import numpy as np

import pbafs_utils as util
import const
from bat import VirtualBat
from pbafs import runPBAFS
from path import australianPath, breastPath, dnaPath, germanPath, mushPath

import classifiers

from metrics import metrics
import results

# Carrega dados CSV
#X, y = util.get_data_from_csv(australianPath)
#number_of_rows = X.shape[0]
#number_of_features = X.shape[1]
datasetArray = [mushPath]

for path in datasetArray:
    score = []
    index = 0
    for i in range (0, 5):
        # Carrega dados LIBSVM
        X, y = util.get_data_from_libsvm(path)
        number_of_rows = X.shape[0]
        number_of_features = X.shape[1]

        start_time = time.time()
        best_features, accuracy, best_classifier = runPBAFS(X, y, number_of_features)
        end_time = time.time()
        metrics.total_time = end_time - start_time
        print(f"\nTempo de execução total: {metrics.total_time} segundos")
        print(f"Tamanho da população: {const.population_size}")
        print(f"Quantidade de iterações: {const.max_iterations}")
        print(f"Vetor de atributos com maior acurácia: {best_features}")
        print(f"Qtd. atributos selecionados: {np.count_nonzero(best_features)}")
        print(f"Acurácia obtida: {accuracy}")
        print(f"Melhor classificador: {best_classifier}\n")
        results.saveResults(metrics)

        # Remove atributos selecionados
        X_selected = util.removeFeaturesUnselected(best_features, X)

        # Faz classificação
        scoreNB = classifiers.getNaiveBayesScore(X_selected, y)
        
        # Guarda acurácia
        score.append(scoreNB)

    # Salva acurácia obtida
    metrics.scores.append(sum(score)/5)

results.saveScores(metrics.scores)