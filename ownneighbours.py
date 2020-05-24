import operator

def manhattan_distance(x, y):
	return sum(abs(a - b) for a, b in zip(x, y))

def getNeighbors(trainingSet, testInstance, k): # returns list of index
	distances = []
	for x in range(len(trainingSet)):
		dist = manhattan_distance(trainingSet[x], testInstance)
		distances.append((trainingSet[x], dist, x))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][2])
	return neighbors

# muss angepasst werden
def predict(X_train, X_test, Y_train, k):
	neighbors = getNeighbors(X_train, X_test, k)
	ouput = []
	for row in neighbors:
		index = neighbors[row]
		ouput.append(Y_train[index])
	prediction = max(set(output), key=output.count)
	return prediction