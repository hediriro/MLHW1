import mathhelp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, kernel_approximation, neighbors

df = pd.read_csv('dataset_191_wine.csv')  # create data frame
X = df.iloc[:, 1:]  # create train frame
Y = df.iloc[:, 0]  # create target vector
print('X:', X.shape, ' y:', Y.shape)

X_norm = preprocessing.normalize(X)  # normalizing attributes
# naive is already given
Y_ohe = pd.get_dummies(Y)  # One Hot Encoding

X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=(1 / len(Y)), random_state=42)  # split sets & leave one out
print('X_train:', X_train.shape, ' Y_train:', Y_train.shape)
print('X_test:', X_test.shape, ' Y_test:', Y_test.shape)
