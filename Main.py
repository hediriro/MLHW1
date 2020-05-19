import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize

df = pd.read_csv('dataset_191_wine.csv')  # create data frame
X = df.iloc[:, 1:]  # create train frame
Y = df.iloc[:, 0]  # create target vector
print('X:', X.shape, ' y:', Y.shape)

X_norm = normalize(X)  # normalizing attributes
# naive is already given
Y_ohe = pd.get_dummies(Y)  # One Hot Encoding

# Naive
X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=(1 / len(Y)), random_state=42)  # split sets & leave one out
print('X_train:', X_train.shape, ' Y_train:', Y_train.shape)
print('X_test:', X_test.shape, ' Y_test:', Y_test.shape)
# continue here ...

# One Hot Encoding
X_train_ohe, X_test_ohe, Y_train_ohe, Y_test_ohe = train_test_split(X_norm, Y_ohe, test_size=(1 / len(Y_ohe)), random_state=42)  # split sets & leave one out
print('X_train_ohe:', X_train_ohe.shape, ' Y_train_ohe:', Y_train_ohe.shape)
print('X_test_ohe:', X_test_ohe.shape, ' Y_test_ohe:', Y_test_ohe.shape)
# continue here ...