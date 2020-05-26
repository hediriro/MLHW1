import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial.distance import euclidean, chebyshev, minkowski
from sklearn.preprocessing import normalize
from math import sqrt
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut


# Own implementations

def gaussian(distances, window):
	weigths = np.exp(-(distances**2)/window)
	return weigths

class OwnKNN(BaseEstimator):

	def __init__(self, k, distance_metric, kernel_function, kernel_window):
		self.k = k
		self.distance_metric = distance_metric
		self.kernel_function = kernel_function
		self.kernel_window = kernel_window

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		predicted_labels = [self.predict_one(x) for x in X]
		return np.array(predicted_labels)

	def predict_one(self, x):
		distances = [self.distance_metric(x_i,x) for x_i in self.X_train] # compute distances
		k_indices = np.argsort(distances)[:self.k] # get indices to get labels
		for i in k_indices:
			k_nearest_labels = [self.y_train[i]] # get k nearest samples, labels
			weights = [self.kernel_function(distances[i],self.kernel_window)] # compute weights according to kernel
		most_common = Counter(k_nearest_labels).most_common(1) # majority vote, most common class labels
		return most_common[0][0]




# Main processing

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
kernel_functions = [gaussian,'epanechnikov','cosine','tophat','exponential','linear']
kernel_window =  [np.logspace(-1, 1, 20), k_range] # fixed and variable
grid_params= {
        'k': k_range,
        'distance_metric': distance_metrics,
        'kernel_function': kernel_functions,
        'kernel_window': kernel_window
}

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size= (1 / len(y)), random_state=42)  # split sets & leave one out
print('X_train:', X_train.shape, ' y_train:', y_train.shape)
print('X_test:', X_test.shape, ' y_test:', y_test.shape)

clf = OwnKNN(7,euclidean,gaussian,7)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
right_guesses = np.sum(prediction == y_test)
acc = right_guesses/len(y_test)
actually = y_test.tolist()
print("Correct guesses: ",right_guesses," / Accuracy: ",acc)
print("Predcited: ",prediction," / Actually: ",actually)
