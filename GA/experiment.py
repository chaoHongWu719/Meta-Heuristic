from GA_HC import GA 
import numpy as np
import time


# ==== PARAMETER for the HC problem ===
edges = [
    (1, 2), (2,  3), (3, 4), (4, 5), (5, 1),
    (5, 6), (4, 8), (3, 10), (2, 12), (1,14),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 6),
    (15, 16), (7, 17), (9, 18),(11,19), (13, 20),
    (16, 17), (17, 18), (18, 19), (19, 20), (20, 16)
]

NUM_CITY = 20
M = 100
d = np.full((NUM_CITY, NUM_CITY), M)
np.fill_diagonal(d, 0)  # fill the diagonal with 0 (self to self)
for (i, j) in edges:  # transform to 0-based index!
    d[i-1, j-1] = 1  # fill the distance between two connected cities with 1
    d[j-1, i-1] = 1  # undirected graph, so d[i-1][j-1] = d[j-1][i-1]


# ==== PARAMETER for the GA ===
best = []
average = []

NUM_ITERATION = 300
NUM_CHROME = 200
NUM_PARENT = NUM_CHROME
Pc = 0.5
Pm = 0.02
wheelSelection = False  # by binary tournament
rankEvaluation = False  # by the original fitness (- total cost)
for wheelSelection in [False, True]:
    for rankEvaluation in [True]:
        bestObj = -float("inf")
        avgObj = list()
        totalTime = 0
        for _ in range(10):
            # run the GA and output the resulting obj
            start = time.perf_counter()
            _, objList = GA(d, NUM_CITY, NUM_ITERATION, NUM_CHROME, NUM_PARENT, Pc, Pm, wheelSelection, rankEvaluation)
            end = time.perf_counter()
            totalTime += (end - start)
            obj = objList[-1]
            avgObj.append(obj)
            bestObj = max(bestObj, obj)
        
        print(wheelSelection, rankEvaluation, "avgObj", sum(avgObj) / 10, "best", bestObj, "time", totalTime / 10)