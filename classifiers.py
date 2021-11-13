from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os
from pbafs_random import randint
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB

class ClassifierScore:
    def __init__(self, name="", score=0):
        self.name=name
        self.score=score


def getRandomForestScore(X, y):
    rfc = RandomForestClassifier(random_state=randint(0, 50), class_weight="balanced")
    scores = cross_val_score(rfc, X, y, cv=3)
    return scores.mean()


def getSVMScore(X, y):
    svc = SVC(random_state=randint(0, 50), max_iter=-1, class_weight="balanced")
    scores = cross_val_score(svc, X, y, cv=3)
    return scores.mean()


def getKNNScore(X, y):
    clf = KNeighborsClassifier(3, weights="uniform")
    scores = cross_val_score(clf, X, y, cv=3)
    return scores.mean()


def getNaiveBayesScore(X, y):
    clf = MultinomialNB()
    scores = cross_val_score(clf, X, y, cv=3)
    return scores.max()

def getBestClassifier(classifiers):
    classifiers.sort(key=lambda c: c.score, reverse=True)
    return classifiers[0]


def getBestClassifierAndAccuracy(X, y):
    rfc_accuracy = getRandomForestScore(X, y)
    rfc = ClassifierScore(name="Classificador Random Forest", score=rfc_accuracy)

    svm_accuracy = getSVMScore(X, y)
    svm = ClassifierScore(name="Classificador SVM", score=svm_accuracy)

    knn_accuracy = getKNNScore(X, y)
    knn = ClassifierScore(name="Classificador KNN", score=knn_accuracy)

    list_classifiers = []
    list_classifiers.append(rfc)
    list_classifiers.append(svm)
    list_classifiers.append(knn)

    best_classifier = getBestClassifier(list_classifiers)

    return best_classifier
