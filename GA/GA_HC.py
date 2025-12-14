import numpy as np


# ==== MAIN-GA ====
def GA(d, NUM_CITY, NUM_ITERATION, NUM_CHROME, NUM_PARENT, Pc, Pm, wheelSelection = True, rankEvaluation = False):
	'''
	this is to facilitate the experiments of parameter tuning and various selection method
	d : cost matrix (1 or M)
	NUM_CITY : number of city
	NUM_ITERATION : stopping criterion
	NUM_CHROME : number of chromes (population size)
	NUM_PARENT : number of parents
	Pc : cross-over rate
	Pm : mutation rate
	wheelSelection = True; else binary-tournament
	rankEvaluation = True; else by (- total cost)
	plot : whether to plot the mean and best sols for each iteration
	'''

	NUM_BIT = NUM_CITY - 1	# number of cities (genes) excluding the starting city 0
	NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # times of cross-over
	NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # times of mutation


	# ==== SUB-FUNCTION for GA algorithm ====
	def initPop():             # population initialization
		p = []
		for _ in range(NUM_CHROME) :
			p.append(np.random.permutation(range(1, NUM_BIT+1))) # permutation of cities 1, 2, ..., NUM_CITY-1 (NUM_BIT)     
		return p


	def fitFunc(x):        # a transformed fitness function
		cost = d[0][x[0]]  # city 0 is the starting point (WLOS)
		for i in range(NUM_BIT-1) :
			cost += d[x[i]][x[i+1]]
		cost += d[x[NUM_BIT-1]][0]
		return -cost  # since this is a minimization problem


	def evaluatePop(p):        # evaluate the whole population by fitness
		return [fitFunc(p[i]) for i in range(len(p))]


	def selection_binary_tournament(p, p_fit):   # use binary tournament selection to select parents
		a = []
		for _ in range(NUM_PARENT):
			[j, k] = np.random.choice(NUM_CHROME, 2, replace = False)  # randomly choose two different indices
			if p_fit[j] > p_fit[k] :
				a.append(p[j].copy())
			else:
				a.append(p[k].copy())
		return a


	def selection_wheel(p, p_fit):   # roulette wheel selection
		a = []
		
		# 1. deal with negative fitness values by shifting min to 1 
		# (not 0, since 0 means no chance of being selected)
		min_fit = min(p_fit)
		adjusted_fit = [fit - min_fit + 1 for fit in p_fit]
		total_fit = sum(adjusted_fit)

		# 2. calculate selection probabilities and select parents
		probabilities = [fit / total_fit for fit in adjusted_fit]
		selected_indices = np.random.choice(NUM_CHROME, size = NUM_PARENT, replace = True, p = probabilities)
		a = [p[i].copy() for i in selected_indices]
		return a


	def calculate_fit_rank(pop_fit):  # rank-based fitness assignment
		sorted_indices = sorted(range(NUM_CHROME), key = lambda i: pop_fit[i], reverse = False)  # ascending order
		pop_fit_rank = [0] * NUM_CHROME
		for rank, idx in enumerate(sorted_indices):
			pop_fit_rank[idx] = rank + 1  # ranks start from 1 to NUM_CHROME
		return pop_fit_rank


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


	def mutation(p):  # only applied on offspring, whose number is NUM_CROSSOVER * 2
		for _ in range(NUM_MUTATION) :
			row = np.random.randint(NUM_CROSSOVER * 2)   # randomly choose a chromosome
			[j, k] = np.random.choice(NUM_BIT, 2, replace = False)        # randomly choose two different genes
			p[row][j], p[row][k] = p[row][k], p[row][j]  # swap them


	def sortChrome(a, a_fit):  # sort a according to a_fit (descending order, absolute value ascending)
		a_index = range(len(a))
		a_fit, a_index = zip(*sorted(zip(a_fit, a_index), reverse = True))
		return [a[i] for i in a_index], a_fit


	def replace(p, p_fit, a, a_fit):
		b = np.concatenate((p,a), axis = 0)   # merge two generations's chromosomes
		b_fit = p_fit + a_fit                 # merge two generations's fitness value
		b, b_fit = sortChrome(b, b_fit)
		return b[:NUM_CHROME], list(b_fit[:NUM_CHROME])  # select the best NUM_CHROME ones

	# === main function for GA ===
	pop = initPop()
	pop_fit = evaluatePop(pop)

	best_outputs = []
	best_outputs.append(np.max(pop_fit))
	mean_outputs = []
	mean_outputs.append(np.average(pop_fit))


	for iter in range(NUM_ITERATION) :

		pop_fit_for_selection = pop_fit
		if rankEvaluation:
			pop_fit_for_selection = calculate_fit_rank(pop_fit)  # rank-based fitness assignment

		# select parents
		if wheelSelection:
			parent = selection_wheel(pop, pop_fit_for_selection)  
		else:
			parent = selection_binary_tournament(pop, pop_fit_for_selection)

		offspring = crossover_uniform(parent)       # uniform crossover
		mutation(offspring)                         # mutate
		offspring_fit = evaluatePop(offspring)      # calculate fitness of offspring
		pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # replace
		
		best_outputs.append(np.max(pop_fit))        # incumbent best solution
		mean_outputs.append(np.average(pop_fit))    # average of this generation

		print(f"iteration {iter+1}: best = {pop[0]}, best_fit = {-pop_fit[0]}")


	return mean_outputs, best_outputs


if __name__ == "__main__":
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


	# ==== (1.c. most proper) PARAMETER for the GA ===
	NUM_ITERATION = 300
	NUM_CHROME = 200
	NUM_PARENT = NUM_CHROME
	Pc = 0.5
	Pm = 0.02
	wheelSelection = False  # by binary tournament
	rankEvaluation = False  # by the original fitness (- total cost)
	mean_outputs, best_outputs = GA(d, NUM_CITY, NUM_ITERATION, NUM_CHROME, NUM_PARENT, Pc, Pm, wheelSelection, rankEvaluation)
	print("we use 0-based index rather than 1-based index, and the starting node 0 is not printed in the route")
	print(f"output obj (M = {M}) = {abs(best_outputs[-1])}")

	# draw the figure
	import matplotlib.pyplot as plt
	plt.plot(best_outputs)
	plt.plot(mean_outputs)
	plt.xlabel("Iteration")
	plt.ylabel("Fitness")
	plt.legend(["Best", "Mean"])
	plt.show()