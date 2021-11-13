import random
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

def randomSelectedFeatures(n):
    selectedFeatures = []
    for _ in range(0, n):
        selectedFeatures.append(random.randint(0, 1))
    return selectedFeatures


def initializeBatVelocities(n):
    velocities = []
    for _ in range(0, n):
        velocities.append(0)
    return velocities


def getFeaturesSelected(selectedFeaturesBinaryArray):
    output = []
    for i in range (0, len(selectedFeaturesBinaryArray)):
        if selectedFeaturesBinaryArray[i] == 1:
            output.append(i)

    return output


def getDataFeatureSelected(data, features_selected):
    data_selected = data[:,features_selected].copy()
    return data_selected


def getRowsAndFeaturesFromDF(df):
    return df.shape[0], df.shape[1] - 1


# df: Primeira coluna deve ser a label
# test_size:
def train_test_split_from_df (df, test_size, random_state):
    X, y = split_from_df(df)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test


def split_from_df(df):
    X = df.drop([0],axis=1).values
    y = df[0].values
    return X, y


def removeFeaturesUnselected(x, X):
    features_selected = getFeaturesSelected(x) # pega atributos selecionados
    X_selected = getDataFeatureSelected(X, features_selected) # reduz dimensionalidade
    return X_selected


def get_data_from_libsvm(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def get_data_from_csv(path):
    df = pd.read_csv(path, sep=',', na_filter=False, header=None)
    return split_from_df(df)