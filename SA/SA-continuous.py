# min f(x) = 3 x^4 - 8 x^3 - 6 x^2 + 24 x
# global opt sol = (-1, -19), local opt sol = (2, 8)

import numpy as np
import math

K  = 1.0		 # Boltzmann rate
DWELL = 20		 # number of iteration in equilibrium (the same temperature)
T_high = 1000.0	 # the highest temperature (= initial temperature)
T_scale = 0.9	 # cool-down rate t0 -> t0*r -> t0*r*r -> ...
T_low = 1.0		 # the lowest temperature

# cost function
def SAfunc(x):
	return (((3 * x - 8) * x - 6) * x + 24) * x

#  ==== MAIN FUNCTION ====
np.random.seed(0)

# initial sol as the incumbent
xbest = x = np.random.uniform(-3.0, 3.0)  # random initial solution
ybest = y = SAfunc(x)

iter = 0    # current number of iteration
t = T_high  # current temperature

while t > T_low:  # the outer loop: annealing

    for i in range(DWELL) :  # the inner loop: preturb
        # generate a neighbor solution
        xnew = x + np.random.uniform(-0.1, 0.1)   # a random real number between -0.1 and 0.1
        ynew = SAfunc(xnew)
        
        # accept (xnew, ynew) if (1) better (2) uniform(0, 1) < threshold
        if ynew < y or np.random.uniform(0.0, 1.0) < math.exp( - (ynew - y) / (K * t) ):
            x, y = xnew, ynew

        # renew the incumbent
        if ynew < ybest:
            xbest, ybest = xnew, ynew

    print(f"temp = {t:.5f}, current sol: f({x:.5f}) = {y:.5f}, incumbent: f({xbest:.5f}) = {ybest:.5f}")
    t *= T_scale