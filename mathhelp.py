import math

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

