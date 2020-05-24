import operator

def manhattan_distance(x, y):
	return sum(abs(a - b) for a, b in zip(x, y))

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	for x in range(len(trainingSet)):
		dist = manhattan_distance(trainingSet[x], testInstance)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

# muss angepasst werden
def predict(trainingSet, testInstance, k):
	neighbors = getNeighbors(trainingSet, testInstance, k)
	output_values = [row[0] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction