import math
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def euklid_dist(v1, v2):
    d = 0
    i = 0
    if len(v1) != len(v2):
        print("Vektoren ungleicher Länge")
        return 0
    else:
        while i < len(v1):
            h = (v1[i] - v2[i]) * (v1[i] - v2[i])
            d = d + h
            i = i + 1
    d = math.sqrt(d)
    return d

