import numpy as np
import scipy as sp
import winsound
import time

from allFunctions_SimAndProc import *
from findMaxDev import *


def meanSquaredDiff(params, dataTrajs):
# params = [N, std, r]

    N = params[0] # population size
    std = params[1] # standard deviation of noise
    r = params[2] # geometric distribution parameter
    
    trajectories = simulation(N, 60, 10000, np.random.uniform(0.4, 0.9, 10000), autoRegP, c = r, d = 1000, noiseSTD = std)
    
    simulatedDevs = findMaxDevs(trajectories, trajectories[0,:], 10000)
    dataDevs = findMaxDevs(dataTrajs, dataTrajs[0,:], np.shape(dataTrajs)[1])
    
    dataHist = np.histogram(dataDevs, bins=100, density=True)
    simHist = np.histogram(simulatedDevs, bins=dataHist[1], density=True) # forcing simulated distribution to use the same bins as the data
    
    MSE = np.sum((simHist[0] - dataHist[0])**2)/10000

    return MSE

# produce a "data" set (known parameters)
dataTrajs = simulation(5000, 60, 10000, np.random.uniform(0.4,0.9,10000), autoRegP, c = 0.6, d = 1000, noiseSTD = 0.01)

# initial guesses for N, noiseSTD, and r
init = [3000, 0.01, 0.2]

winsound.Beep(500,800)
t1 = time.time()

res = sp.optimize.minimize(meanSquaredDiff, init, args=dataTrajs, method='nelder-mead', options={'disp': True})

t2 = time.time()
winsound.Beep(500,800)

print(t2-t1)