import math
import numpy as np

def round_as_naive_approach(result): # the naive way
    if result result < 1.5:
        return 1
    if result >= 1.5 && result < 2.5:
        return 2
    if result >= 2.5:
        return 3

def normalize(v): # normalizing vectors
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def euklid_dist(v1, v2): # calculate eukilidan distance
    d = 0
    i = 0
    if len(v1) != len(v2):
        print("Vectors do not have the same length!")
        return 0
    else:
        while i < len(v1):
            h = (v1[i] - v2[i]) * (v1[i] - v2[i])
            d = d + h
            i = i + 1
    d = math.sqrt(d)
    return d

