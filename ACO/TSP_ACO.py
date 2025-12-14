import numpy as np
import random


# randomly generate city coordinates
numCities = 20
cityCoordinates = np.random.rand(numCities, 2) * 100  # 20 cities in 100x100 area

# compute distance matrix
distMatrix = np.zeros((numCities, numCities))
for i in range(numCities):
    for j in range(numCities):
        distMatrix[i, j] = np.linalg.norm(cityCoordinates[i] - cityCoordinates[j])


# objective function
def compute_objective_value(cityIDs):
    total_distance = 0
    for i in range(len(cityIDs)):
        city1 = cityIDs[i]
        city2 = cityIDs[i+1] if i < len(cityIDs)-1 else cityIDs[0]
        total_distance += distMatrix[city1, city2]
    return total_distance

#%% Ant System
class AntSystem:

    def __init__(self, pop_size, dist_matrix, pheromone_drop_amount, evaporate_rate,
                 pheromone_factor, heuristic_factor, Q = 100):
        
        self.num_ants = pop_size
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.pheromone_drop_amount = pheromone_drop_amount
        self.evaporate_rate = evaporate_rate
        self.pheromone_factor = pheromone_factor
        self.visibility_factor = heuristic_factor
        self.Q = Q


    def initialize(self):
        self.solutions = np.zeros((self.num_ants, self.num_cities), dtype = int)
        self.one_solution = np.arange(self.num_cities)  # temp storage for one ant
        self.objective_value = np.zeros(self.num_ants)  # to store objective value of each ant
        self.best_solution = np.zeros(self.num_cities, dtype = int)
        self.best_objective_value = float("inf")
        
        # visibility: 1 / dist
        self.visibility = np.zeros((self.num_cities, self.num_cities))
        self.pheromone_map = np.ones((self.num_cities, self.num_cities))  # initial pheromone = 1
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.visibility[i, j] = 1 / self.dist_matrix[i, j]
    
    # roulette wheel
    def do_roulette_wheel_selection(self, fitness_list):
        prob_list = np.array(fitness_list) / sum(fitness_list)
        return np.random.choice(len(fitness_list), p = prob_list)
    

    # construct one ant solution
    def _an_ant_construct_its_solution(self):
        candidates = list(range(self.num_cities))  # cities to be visited
        current_city = random.choice(candidates)
        self.one_solution[0] = current_city
        candidates.remove(current_city)
        
        for t in range(1, self.num_cities - 1):
            fitness_list = []
            for city_id in candidates:
                fitness = (self.pheromone_map[current_city, city_id] ** self.pheromone_factor) * \
                    (self.visibility[current_city, city_id] ** self.visibility_factor)
                fitness_list.append(fitness)

            next_city = candidates[self.do_roulette_wheel_selection(fitness_list)]
            self.one_solution[t] = next_city
            candidates.remove(next_city)
            current_city = next_city
        
        self.one_solution[-1] = candidates[0]  # last city
    

    # each ant construct solution
    def each_ant_construct_its_solution(self):
        for i in range(self.num_ants):
            self._an_ant_construct_its_solution()
            self.solutions[i] = self.one_solution.copy()
            self.objective_value[i] = compute_objective_value(self.solutions[i])
    

    def update_pheromone(self, Q=100):
        # 1. global update: evaporate pheromone
        self.pheromone_map *= (1 - self.evaporate_rate)
        
        # 2. local update: add drop_amount for each ant's path
        for sol in self.solutions:
            for j in range(self.num_cities):
                city1 = sol[j]
                city2 = sol[j+1] if j < self.num_cities-1 else sol[0]
                self.pheromone_map[city1, city2] += self.pheromone_drop_amount
        
        # 3. global update: add pheromone for best solution
        delta = self.Q / self.best_objective_value
        for i in range(self.num_cities):
            city1 = self.best_solution[i]
            city2 = self.best_solution[i+1] if i < self.num_cities-1 else self.best_solution[0]
            self.pheromone_map[city1, city2] += delta


    # update best solution
    def update_best_solution(self):
        for i, val in enumerate(self.objective_value):
            if val < self.best_objective_value:
                self.best_solution = self.solutions[i].copy()
                self.best_objective_value = val

# parameters for ACO
pop_size = 10
pheromone_drop_amount = 0.1
evaporate_rate = 0.1  # rho
pheromone_factor = 1  # alpha
heuristic_factor = 2  # beta


from TSP_opt import Opt_TSP_MTZ
status, obj, route, time = Opt_TSP_MTZ(distMatrix, distMatrix.shape[0])

solver = AntSystem(pop_size, distMatrix, pheromone_drop_amount, evaporate_rate,
    pheromone_factor, heuristic_factor)
solver.initialize()
num_iterations = 50

bestDistances = []
for iteration in range(num_iterations):
    solver.each_ant_construct_its_solution()
    solver.update_pheromone()
    solver.update_best_solution()
    
    print(f"Iteration {iteration+1}:")
    print("Best tour:", solver.best_solution)
    print("Best distance:", solver.best_objective_value)
    print("-"*40)
    bestDistances.append(solver.best_objective_value)


print("Gurobi", obj)
print("ACO", solver.best_objective_value)

# plot convergence curve
import matplotlib.pyplot as plt
plt.plot(range(1, num_iterations + 1), bestDistances, color = "blue", label = "ACO")
plt.plot([1, num_iterations + 1], [obj, obj], color = "red", label = "Gurobi")
plt.title("Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Distance")
plt.legend(loc = "upper right")
plt.show()