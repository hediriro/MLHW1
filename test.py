import numpy as np
from scipy.spatial.distance import minkowski, euclidean, chebyshev

def gaussian_kernel(distances, window):
	weigths = np.exp(-(distances**2)/window)
	return weigths

distance_metrics = [euclidean,chebyshev,minkowski]

for metric in distance_metrics:
    dist = metric
    X_1 = [[1,4,5,6,3],[4,2,7,5,9]]
    X_2 = [9,0,3,4,1]
    res = dist(X_1[1], X_2)
    print(metric, res)
    print(gaussian_kernel,gaussian_kernel(res,100))