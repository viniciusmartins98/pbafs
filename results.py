from metrics import Metrics
import csv

def acumulaValores(metrics=Metrics()):
    for i in range (1, len(metrics.classification_time)):
        metrics.classification_time[i] = metrics.classification_time[i-1] + metrics.classification_time[i]
    
    for i in range (1, len(metrics.iteration_time)):
        metrics.iteration_time[i] = metrics.iteration_time[i-1] + metrics.iteration_time[i]

def saveResults(metrics=Metrics()):
    acumulaValores(metrics)

    # opening the csv file in 'w+' mode 
    file = open('classification.csv', 'w+', newline ='')
    with file:     
        write = csv.writer(file) 
        write.writerows(map(lambda x: [x], metrics.classification_time)) 
        
    # opening the csv file in 'w+' mode 
    file = open('iteration.csv', 'w+', newline ='') 
    with file:     
        write = csv.writer(file) 
        write.writerows(map(lambda x: [x], metrics.iteration_time))


def saveScores(scores):
    print('SCORES: ')
    print(scores)
    
    # opening the csv file in 'w+' mode 
    file = open('scores_naive_bayes.csv', 'w+', newline ='')
    with file:     
        write = csv.writer(file) 
        write.writerows(map(lambda x: [x], scores))