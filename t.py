import os, sys, time, datetime, random, cv2, argparse
import numpy as np

from itertools import combinations, permutations


lenina_lines = {
    "top": [np.array([760, 325]), np.array([40, 305])],
    "left": [np.array([665,410]), np.array([715,550])],
    "bottom": [np.array([1200,625]), np.array([1680,505])],
    "right": [np.array([1355,345]), np.array([1625,425])]
}

COUNTER_MAP = {}
for a, b in permutations(['top', 'left', 'bottom', 'right'], 2):
    COUNTER_MAP['{}-{}'.format(a, b)] = 0

def distance(p, lines, ignore=''):
    distances = {}
    for name, l in lines.items():
        if name != ignore:
            dist = np.linalg.norm(np.cross(l[1]-l[0], l[0]-p))/np.linalg.norm(l[1]-l[0])
            distances[name] = dist
    return min(distances, key=distances.get)

a = distance([767, 504], lenina_lines)
b = distance([1345, 545], lenina_lines, a)

COUNTER_MAP['{}-{}'.format(a, b)] += 1
print(COUNTER_MAP)
