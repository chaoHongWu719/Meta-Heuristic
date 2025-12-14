import numpy as np


# ==== PARAMETER for the TSP problem===
NUM_CITY = 4
d = [
	[  0, 12,  1,  8 ],
    [ 12,  0,  2,  3 ],
    [  1,  2,  0, 10 ],
    [  8,  3, 10,  0 ]
]


# ==== PARAMETER for GA algorithm ====
NUM_ITERATION = 20		# number of generations (iterations)
NUM_CHROME = 20			# numbeer of solutions (chromosomes)
NUM_BIT = NUM_CITY - 1	# number of cities (genes) excluding the starting city 0
Pc = 0.5    			# cross-over rate (cross-over : Pc * NUM_CHROME / 2 times)
Pm = 0.01   			# mutation rate (mutation : Pm * NUM_CHROME * NUM_BIT times)

NUM_PARENT = NUM_CHROME                         # number of parents
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # times of cross-over
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # times of mutation

np.random.seed(0)


# ==== SUB-FUNCTION for GA algorithm ====
def initPop():             # population initialization
    p = []
    for _ in range(NUM_CHROME) :
        p.append(np.random.permutation(range(1, NUM_BIT+1))) # permutation of cities 1, 2, ..., NUM_CITY-1 (NUM_BIT)     
    return p


def fitFunc(x):            # a transformed fitness function
    cost = d[0][x[0]]
    for i in range(NUM_BIT-1) :
        cost += d[x[i]][x[i+1]]
    cost += d[x[NUM_BIT-1]][0]
    return -cost  # since this is a minimization problem


def evaluatePop(p):        # evaluate the whole population by fitness
    return [fitFunc(p[i]) for i in range(len(p))]


def selection(p, p_fit):   # use binary tournament selection to select parents
	a = []
	for _ in range(NUM_PARENT):
		[j, k] = np.random.choice(NUM_CHROME, 2, replace = False)  # randomly choose two different indices
		if p_fit[j] > p_fit[k] :
			a.append(p[j].copy())
		else:
			a.append(p[k].copy())
	return a


def crossover_uniform(p):  # uniform crossover
	a = []

	for _ in range(NUM_CROSSOVER) :
		mask = np.random.randint(2, size = NUM_BIT)  # randomly generate a 0 / 1 mask
		[j, k] = np.random.choice(NUM_PARENT, 2, replace = False)  # randomly choose two different parents
       
		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # cities which are not in the children yet
       
		for m in range(NUM_BIT):
			if mask[m] == 1 :
				remain2.remove(child1[m])   # remain this bit, so remove it from remain2
				remain1.remove(child2[m])   # remain this bit, so remove it from remain1
		
		t = 0
		for m in range(NUM_BIT):
			if mask[m] == 0 :
				child1[m] = remain2[t]
				child2[m] = remain1[t]
				t += 1
		
		a.append(child1)
		a.append(child2)

	return a


def mutation(p):
	for _ in range(NUM_MUTATION) :
		row = np.random.randint(NUM_CROSSOVER * 2)   # randomly choose a chromosome
		[j, k] = np.random.choice(NUM_BIT, 2)        # randomly choose two different genes
		p[row][j], p[row][k] = p[row][k], p[row][j]  # swap them


def sortChrome(a, a_fit):  # sort a according to a_fit (accending order)
	a_index = range(len(a))
	a_fit, a_index = zip(*sorted(zip(a_fit, a_index), reverse = True))
	return [a[i] for i in a_index], a_fit


def replace(p, p_fit, a, a_fit):
	b = np.concatenate((p,a), axis = 0)   # merge two generations's chromosomes
	b_fit = p_fit + a_fit                 # merge two generations's fitness value
	b, b_fit = sortChrome(b, b_fit)
	return b[:NUM_CHROME], list(b_fit[:NUM_CHROME])  # select the best NUM_CHROME ones


# ==== MAIN-FUNCTION ====
pop = initPop()
pop_fit = evaluatePop(pop)

best_outputs = []
best_outputs.append(np.max(pop_fit))
mean_outputs = []
mean_outputs.append(np.average(pop_fit))


for i in range(NUM_ITERATION) :
	parent = selection(pop, pop_fit)            # select parents
	offspring = crossover_uniform(parent)       # uniform crossover
	mutation(offspring)                         # mutate
	offspring_fit = evaluatePop(offspring)      # calculate fit of offspring
	pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # replace
	
	print(f"iteration {i+1}: best = {pop[0]}, best_fit = {-pop_fit[0]}")
	best_outputs.append(np.max(pop_fit))        # incumbent best solution
	mean_outputs.append(np.average(pop_fit))    # average of this generation


# draw the figure
import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.plot(mean_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()