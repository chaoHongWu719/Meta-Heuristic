import numpy as np

# parameter
NUM_ITERATION = 20
NUM_CHROME = 20	  # even number
NUM_BIT = 6  # first: 0 negative, 1 positive ; second ~ last: 2進位制 so -31~+31
Pc = 0.5   # crossover rate
Pm = 0.01  # mutation range
NUM_PARENT = NUM_CHROME
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)
NUM_CROSSOVER_2 = NUM_CROSSOVER * 2
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)


# subFunction for GA
def initPop():
	return np.random.randint(2, size = (NUM_CHROME, NUM_BIT))  # binary * 6, total = NUM_CHROME


def fitFunc(x):
    x = int("".join(str(i) for i in x[1: NUM_BIT]), 2)  # the first bit is sign bit, remaining: 2進位轉10進位
    return 1024 - x * x


def evaluatePop(p): 
    return [fitFunc(p[i]) for i in range(len(p))]


def selection(p, p_fit):   # binary tournament
	parents = []

	for _ in range(NUM_PARENT):
		[j, k] = np.random.choice(NUM_CHROME, 2, replace = False)  # randomly choose two "different" parents
		if p_fit[j] > p_fit[k] :
			parents.append(p[j])
		else:
			parents.append(p[k])

	return parents


def crossover(p):
	children = []

	for _ in range(NUM_CROSSOVER) :
		c = np.random.randint(1, NUM_BIT)  # one-point cross over (index = 0~5, cutting point = 1~5)
		# if c = 3 -> left: index = 0, 1, 2; right: index = 3, 4, 5
		[j, k] = np.random.choice(NUM_PARENT, 2, replace = False)  # parent index
       
		children.append(np.concatenate((p[j][0: c], p[k][c: NUM_BIT]), axis = 0))
		children.append(np.concatenate((p[k][0: c], p[j][c: NUM_BIT]), axis = 0))

	return children


def mutation(p):
	for _ in range(NUM_MUTATION) :
		row = np.random.randint(NUM_CROSSOVER_2)  # randomly choose a chromosome (sol)
		col = np.random.randint(NUM_BIT)          # randomly choose a gene (bit)
		p[row][col] = 1 - p[row][col]             # bit: 0 <-> 1
        

def sortChrome(p, p_fit):
    p_index = range(len(p))
    p_fit, p_index = zip(*sorted(zip(p_fit, p_index), reverse = True))  # large to small (by p_fit)
    return [p[i] for i in p_index], p_fit


def replace(pop, pop_fit, offSpring, offSpring_fit):  # s.t. remaining number of population = NUM_CHROME
    newPop = np.concatenate((pop, offSpring), axis = 0)
    newFit = pop_fit + offSpring_fit
    newPop, newFit = sortChrome(newPop, newFit)
    return newPop[:NUM_CHROME], list(newFit[:NUM_CHROME])


# main function
pop = initPop()
pop_fit = evaluatePop(pop)
best_outputs = []
best_outputs.append(np.max(pop_fit))
mean_outputs = []
mean_outputs.append(np.average(pop_fit)) 

for iter in range(NUM_ITERATION) :
    parent = selection(pop, pop_fit)
    offspring = crossover(parent) 
    mutation(offspring)
    offspring_fit = evaluatePop(offspring)
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)
    
    best_outputs.append(np.max(pop_fit))
    mean_outputs.append(np.average(pop_fit))


import matplotlib.pyplot as plt
plt.plot(best_outputs, label = "Best fitness", color = "red")
plt.plot(mean_outputs, label = "Average fitness", color = "blue")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("GA Evolution")
plt.show()