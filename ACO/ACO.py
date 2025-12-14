# Problem : Max y = x_1^2 + x_2^2 + x_3^3 + x_4^4 where x in [1,30]
# Solution: x_1 = x_2 = x_3 = x_4 = 30

import numpy as np
import matplotlib.pyplot as plt

# parameter for ACO
rou = 0.8      # pheromone evaporation coefficient
Q = 1          # pheromone quantity (used in global trail update : Q / L)

NGEN = 100     # number of iterations
popsize = 100  # number of ants

low = [1, 1, 1, 1]      # lower bound for x
up = [30, 30, 30, 30]   # upper bound for x

 
class ACO:

    def __init__(self, parameters):  # parameter is a list = [NGEN, pop_size, var_num_min, var_num_max]
        self.NGEN = parameters[0]           # number of iterations
        self.pop_size = parameters[1]       # number of ants
        self.var_num = len(parameters[2])   # number of variables
        self.bound = []                     # variable bounds = [lower_bound, upper_bound]
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
 
        self.pop_x = np.zeros((self.pop_size, self.var_num))  # positions of ants
        self.g_best = np.zeros((1, self.var_num))  # global best position
 
        # initialize the positions of each ant
        temp = -1  # or (-float("inf")) : global best fitness
        for i in range(self.pop_size):  # for each ant
            for j in range(self.var_num):  # for each variable
                self.pop_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
            fit = self.fitness(self.pop_x[i])
            if fit > temp:
                self.g_best = self.pop_x[i]
                temp = fit
 

    def fitness(self, ind_var):
        x1 = ind_var[0]
        x2 = ind_var[1]
        x3 = ind_var[2]
        x4 = ind_var[3]
        y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
        return y
 

    def update_operator(self, gen, t, t_max):
        lamda = 1 / gen
        pi = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            for j in range(self.var_num):
                pi[i] = (t_max - t[i]) / t_max
                # 更新位置
                if pi[i] < np.random.uniform(0, 1):
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * lamda
                else:
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * (
                                self.bound[1][j] - self.bound[0][j]) / 2
                
                # < lower bound or > upper bound: set to the bound value
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            
            # 更新t值
            t[i] = (1 - rou) * t[i] + Q * self.fitness(self.pop_x[i])
            # 更新全域最優值
            if self.fitness(self.pop_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]
        t_max = np.max(t)
        return t_max, t
 

    def main(self):
        popobj = []
        best = np.zeros((1, self.var_num))[0]
        for gen in range(1, self.NGEN + 1):
            if gen == 1:
                tmax, t = self.update_operator(gen, np.array(list(map(self.fitness, self.pop_x))),
                    np.max(np.array(list(map(self.fitness, self.pop_x)))))
            else:
               tmax, t = self.update_operator(gen, t, tmax)
            popobj.append(self.fitness(self.g_best))
            if self.fitness(self.g_best) > self.fitness(best):
                best = self.g_best.copy()
 
        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size = 14)
        plt.ylabel("fitness", size = 14)
        t = [t for t in range(1, self.NGEN + 1)]
        plt.plot(t, popobj, color = "b", linewidth = 2)
        plt.show()


parameters = [NGEN, popsize, low, up]
aco = ACO(parameters)
aco.main()