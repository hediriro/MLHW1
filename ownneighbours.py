from math import sqrt
import operator

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
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