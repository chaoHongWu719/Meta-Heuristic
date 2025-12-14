import numpy as np
import math

# parameter
NUM_ITERATION = 1000  # number of iteration
NUM_CHROME = 50      # number of chromosome (population size)
NUM_BIT = 2  # x and y
SIGMA = 0.2  # standard deviation used in reproduction
np.random.seed(0)


# subFunction for GA
def initPop():  # initialize the population
	return np.random.uniform(low = (-100), high = 100, size = (NUM_CHROME, NUM_BIT)) 


def fitFunc(x):  # fitness function
    return 0.5 + ((math.sin(math.hypot(x[0], x[1])))**2 - 0.5) / (1.0 + 0.001 * ( x[0]**2 + x[1]**2 ) )**2


def evaluatePop(p):  # return the fitness of the given population
    return [fitFunc(p[i]) for i in range(len(p))]


def selection(p):  # randomly choose two parents
    [i, j] = np.random.choice(NUM_CHROME, 2, replace = False)  # choose two "different" parents
    return [p[i], p[j]]


def reproduction(p):  # kid = (random(Xmom, Xdad) + uniform(0, sigma), random(Ymom, Ydad) + uniform(0, sigma))
    kid = []
    for j in range(NUM_BIT):
        min_x = np.min([p[0][j], p[1][j]])
        max_x = np.max([p[0][j], p[1][j]])
        kid.append(np.random.uniform(min_x, max_x) + np.random.uniform(low = 0, high = SIGMA))
    return kid


def replace(p, p_fit, k, k_fit):
    worstIndex = np.argmin(p_fit)  # use k, k_fit to replace worstIndex, p_fit[worstIndex]
    p[worstIndex] = k
    p_fit[worstIndex] = k_fit
    return p, p_fit


# main function
pop = initPop()
pop_fit = evaluatePop(pop)

best_history = []
avg_history = []

for iter in range(NUM_ITERATION) :
    parent = selection(pop)
    kid = reproduction(parent)
    kid_fit = fitFunc(kid)
    pop, pop_fit = replace(pop, pop_fit, kid, kid_fit)
    bestIndex = np.argmax(pop_fit) 
    best_history.append(pop_fit[bestIndex])
    avg_history.append(np.mean(pop_fit))


import matplotlib.pyplot as plt
plt.plot(best_history, label = "Best fitness", color = "red")
plt.plot(avg_history, label = "Average fitness", color = "blue")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("GA Evolution")
plt.legend()
plt.show()