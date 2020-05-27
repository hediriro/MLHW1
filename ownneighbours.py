import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial.distance import euclidean, chebyshev, minkowski
from sklearn.preprocessing import normalize
from math import sqrt
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score

# Own implementations

def gaussian(distances, window): # calculation from sklearn.org (neighbors, kernel.density)
	weigths = np.exp(-((distances**2)/(2*(window**2))))
	return weigths

def epanechnikov(distances, window): # calculation from sklearn.org (neighbors, kernel.density)
	weigths = 1 - ((distances**2)/(window**2))
	return weigths

def exponential(distances, window): # calculation from sklearn.org (neighbors, kernel.density)
	weigths = np.exp(-distances/window)
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
		all_classifications = []
		classification_count = []
		for i in k_indices:
			k_nearest_labels = [self.y_train[i]] # get k nearest samples, labels
			weights = [self.kernel_function(distances[i], self.kernel_window)] # compute weights according to kernel
			if self.y_train[i] not in all_classifications: # preparation for weighted counting
				all_classifications = [self.y_train[i]]
				classification_count = [0]
		for i in range(0, len(k_indices)-1): # weighted counting
			for j in range(0, len(all_classifications)-1):
				if k_nearest_labels[i] == all_classifications[j]:
					classification_count[j] += weights[i]
		max_value = np.argmax(classification_count)
		return all_classifications[max_value]

"""	def score(self, X, y, sample_weight=None):
		y_pred = self.predict(X)
		return r2_score(y, y_pred, sample_weight=sample_weight)"""

# Main processing

df = pd.read_csv('dataset_191_wine.csv')  # create data frame
df = df.sample(frac=1).reset_index(drop=True) # shuffle data frame
X = df.iloc[:, 1:]  # create train frame
y = df.iloc[:, 0]  # create target vector
X_norm = normalize(X)  # normalizing attributes
# naive is already given
y_ohe = pd.get_dummies(y)  # One Hot Encoding

# hyperparameter

k_range  = range(2, int(sqrt((len(y)-1)))) # square of n
distance_metrics = [euclidean, chebyshev, minkowski]
kernel_functions = [gaussian,epanechnikov,exponential]
kernel_window =  np.logspace(-1, 1, 20) # fixed, variable must be added
grid_params= {
        'k': k_range,
        'distance_metric': distance_metrics,
        'kernel_function': kernel_functions,
        'kernel_window': kernel_window
}

grid = GridSearchCV(OwnKNN, grid_params, cv=LeaveOneOut) # scoring has to be added
#best = grid.fit(X_norm,y)

"""for i in range(1, 20):
	X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=(1 / len(y)), random_state=42)  # split sets & leave one out
	for k in k_range:
		for distance in distance_metrics:
			for kernel in kernel_functions:
				for window in kernel_window:
					tune = OwnKNN(k, distance, kernel, window)
					tune.fit(X_train, y_train)
					prediction = tune.predict(X_test)
					attempts += 1
					right_guesses += np.sum(prediction == y_test)
					acc = right_guesses / len(y_test)
					actually = y_test.tolist()
					print("Hyperparameters: k = ",k," / Distance: ",distance, " / Kernel: ",kernel, " / Window: ",window)
					print("Correct guesses: ", right_guesses, " / Accuracy: ", acc)
					print("Predcited: ", prediction, " / Actually: ", actually)"""


X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size= (1 / len(y)), random_state=42)  # split sets & leave one out
print('X_train:', X_train.shape, ' y_train:', y_train.shape)
print('X_test:', X_test.shape, ' y_test:', y_test.shape)

k = k_range[3]
distance = distance_metrics[1]
kernel = kernel_functions[0]
window = kernel_window[2]

clf = OwnKNN(k, distance, kernel, window)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
right_guesses = np.sum(prediction == y_test)
acc = right_guesses/len(y_test)
actually = y_test.tolist()
print("Hyperparameters: k = ",k," / Distance: ",distance, " / Kernel: ",kernel, " / Window: ",window)
print("Correct guesses: ",right_guesses," / Accuracy: ",acc)
print("Predcited: ",prediction," / Actually: ",actually)