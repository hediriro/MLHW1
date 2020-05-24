import ownneighbours as on
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.preprocessing import normalize

df = pd.read_csv('dataset_191_wine.csv')  # create data frame
df = df.sample(frac=1).reset_index(drop=True) # shuffle data frame
X = df.iloc[:, 1:]  # create train frame
Y = df.iloc[:, 0]  # create target vector
X_norm = normalize(X)  # normalizing attributes
# naive is already given
Y_ohe = pd.get_dummies(Y)  # One Hot Encoding

# hyperparameter

k_range  = (1,(len(Y)-1))
distance_metrics = ['euclidean','manhattan','chebyshev','minkowski']
weights = ['uniform','distance']
grid_params_knn= {
        'n_neighbors': k_range,
        'weights': weights,
        'metric': distance_metrics}

kernel_functions = ['gaussian','tophat','epanechnikov','exponential','linear','cosine']
kernel_bandwidths =  np.logspace(-1, 1, 20) # Or k
grid_params_kernel= {
        'kernel': kernel_functions,
        'bandwidth': kernel_bandwidths}


# Naive

# hyperparameter tuning
gs_kernel = GridSearchCV(KernelDensity(), grid_params_kernel, cv = LeaveOneOut())
gs_kernel.fit(X_norm, Y)
parameters_kernel = gs_kernel.best_params_
print(parameters_kernel)

gs_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose = 1, cv = LeaveOneOut(), n_jobs = -1, scoring ="f1_micro")
gs_knn_results = gs_knn.fit(X_norm, Y)
parameters_knn = gs_knn.best_params_
print(parameters_knn)

X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=(1 / len(Y)), random_state=42)  # split sets & leave one out
print('X_train:', X_train.shape, ' Y_train:', Y_train.shape)
print('X_test:', X_test.shape, ' Y_test:', Y_test.shape)
y_bla = on.predict(X_train, X_test, Y_train, 1)
print("Expected: ", Y_test, " / Received:", y_bla)


# One Hot Encoding

# hyperparameter tuning
gs_kernel = GridSearchCV(KernelDensity(), grid_params_kernel, cv = LeaveOneOut())
gs_kernel.fit(X_norm, Y_ohe)
parameters_kernel = gs_kernel.best_params_
print(parameters_kernel)

gs_knn = GridSearchCV(KNeighborsClassifier(), grid_params_knn, verbose = 1, cv = LeaveOneOut(), n_jobs = -1, scoring ="f1_micro")
gs_knn_results = gs_knn.fit(X_norm, Y_ohe)
parameters_knn = gs_knn.best_params_
print(parameters_knn)

X_train_ohe, X_test_ohe, Y_train_ohe, Y_test_ohe = train_test_split(X_norm, Y_ohe, test_size=(1 / len(Y_ohe)), random_state=42)  # split sets & leave one out
print('X_train_ohe:', X_train_ohe.shape, ' Y_train_ohe:', Y_train_ohe.shape)
print('X_test_ohe:', X_test_ohe.shape, ' Y_test_ohe:', Y_test_ohe.shape)