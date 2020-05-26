import ownneighbours as on
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KernelDensity, KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.preprocessing import normalize
from math import sqrt

df = pd.read_csv('dataset_191_wine.csv')  # create data frame
df = df.sample(frac=1).reset_index(drop=True) # shuffle data frame
X = df.iloc[:, 1:]  # create train frame
y = df.iloc[:, 0]  # create target vector
X_norm = normalize(X)  # normalizing attributes
# naive is already given
y_ohe = pd.get_dummies(y)  # One Hot Encoding

# hyperparameter

k_range  = (2, int(sqrt((len(y)-1)))) # square of n
distance_metrics = [euclidean,chebyshev,minkowski]
kernel_functions = ['gaussian','epanechnikov','cosine','tophat','exponential','linear']
kernel_window =  [np.logspace(-1, 1, 20), k_range] # fixed and variable
grid_params= {
        'k': k_range,
        'distance_metric': distance_metrics
        'kernel_function': kernel_functions,
        'kernel_window': kernel_window
}

grid = GridSearchCV(on.OwnKNN(), grid_params, cv= LeaveOneOut)

KNeighborsRegressor() # own: k, metric, kernel, window(fixed,k) | extend from BaseEstimator then GridSearchCV is supported
# implement fit and predict


# Naive

# hyperparameter tuning

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=(1 / len(y)), random_state=42)  # split sets & leave one out
print('X_train:', X_train.shape, ' y_train:', y_train.shape)
print('X_test:', X_test.shape, ' y_test:', y_test.shape)

# One Hot Encoding

# hyperparameter tuning

X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_norm, y_ohe, test_size=(1 / len(y_ohe)), random_state=42)  # split sets & leave one out
print('X_train_ohe:', X_train_ohe.shape, ' y_train_ohe:', y_train_ohe.shape)
print('X_test_ohe:', X_test_ohe.shape, ' y_test_ohe:', y_test_ohe.shape)