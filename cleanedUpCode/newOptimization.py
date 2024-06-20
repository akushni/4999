import numpy as np
import scipy as sp
import winsound
import time

from allFunctions_SimAndProc import *
from findMaxDev import *

def meanSquaredDiff(params, dataDiff, fixedBins):
# params = [N, std, r]

    N = params[0] # population size
    std = params[1] # standard deviation of noise
    r = params[2] # geometric distribution parameter
    
    trajectories = simulation(N, 60, 10000, np.random.uniform(0.4, 0.9, 10000), autoRegP, c = r, d = 10, noiseSTD = std)
    
    # filter out fixed trajectories
    tempArray = trajectories[:,trajectories[-1,:] != 1]
    tempArray = tempArray[:, tempArray[-1,:] != 0]
    trajectories = tempArray
    numTrajectories = trajectories.shape[1]
    
    # transformation
    FisherTrajs, newTrajs, reverseTransform = procWithFT(trajectories, numTrajectories, t)
    
    simOrigDevs = findMaxDevs(trajectories, trajectories[0,:], trajectories.shape[1])
    simTransformedDevs = findMaxDevs(reverseTransform, reverseTransform[0,:], reverseTransform.shape[1])

    simOrig = np.histogram(simOrigDevs, bins = fixedBins, density = True)
    simTransformed = np.histogram(simTransformedDevs, bins = fixedBins, density = True)
    simDiff = simTransformed[0] - simOrig[0]
    
    MSE = np.sum( (simDiff - dataDiff)**2 )/(len(simDiff))
    
    return MSE

def callFunc(res):
    print(res)
    return


runfile('C:/Users/abiga/Documents/GitHub/4999/cleanedUpCode/procForSim.py')

dataOrigDevs = origDevs
dataTransformedDevs = transformedDevs

fixedBins = np.linspace(0, 0.4, 100, endpoint=True)

dataOrig = np.histogram(dataOrigDevs, bins = fixedBins, density = True)
dataTransformed = np.histogram(dataTransformedDevs, bins = fixedBins, density = True)
dataDiff = dataTransformed[0] - dataOrig[0]

# initial guesses for N, noiseSTD, and r
init = [3000, 0.01, 0.4]

winsound.Beep(500,800)
t1 = time.time()

res = sp.optimize.minimize(meanSquaredDiff, init, args=(dataDiff, fixedBins), callback = callFunc, method='nelder-mead', options={'disp': True})

t2 = time.time()
winsound.Beep(500,800)
