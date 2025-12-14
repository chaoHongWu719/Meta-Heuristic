import numpy as np
import math
np.random.seed(97)


# 1. parameter for simulated annealing
DWELL = 20      # max number of iteration in the same temperature
K = 1           # Boltzmann rate
tHigh = 1000.0  # highest tempperature
tScale = 0.9    # coll down rate
tLow = 1.0      # lowest temperature


# 2. paramter for TSP
cityCnt = 10
randMatrix = np.random.randint(1, 101, size=(cityCnt, cityCnt))
cost = (randMatrix + randMatrix.T) // 2  # let cost be symmetric
np.fill_diagonal(cost, 0)


# 3. the objective function
def calCost(x):
    tempCost = 0
    for i in range(cityCnt - 1):
        tempCost += cost[x[i]][x[i+1]]
    return tempCost + cost[x[cityCnt - 1]][x[0]]  # back to the starting point


# 4. main function
x = np.random.permutation(range(cityCnt))  # e.g. [2 3 1 0]
y = calCost(x)
xBest, yBest = x, y
t = tHigh

yList = [y]
yBestList = [yBest]

while t > tLow:  # the outer loop
    for _ in range(DWELL):  # the inner loop

        xNew = x.copy()

        # swap two nodes randomly
        idxs = np.random.choice(cityCnt, 2)
        xNew[idxs[0]], xNew[idxs[1]] = x[idxs[1]], x[idxs[0]]
        yNew = calCost(xNew)

        if yNew < y:  # accept this better sol
            x = xNew.copy()
            y = yNew

            if yNew < yBest:  # renew the incumbent
                xBest = xNew.copy()
                yBest = yNew

        elif np.random.uniform(0.0, 1.0) < math.exp( - (yNew - y) / (K * t)):  # accept this worse sol
            x = xNew.copy()
            y = yNew

        yList.append(y)
        yBestList.append(yBest)
    
    t *= tScale


import matplotlib.pyplot as plt
plt.plot(yBestList, label = "yBest", color = "red", linewidth = 2)
plt.plot(yList, label = "y", color = "gray", alpha = 0.6, linewidth = 1)
plt.xlabel("iteration")
plt.ylabel("fitness")
plt.legend()
plt.show()