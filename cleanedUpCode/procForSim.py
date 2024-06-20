import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import winsound

from allFunctions_SimAndProc import *

###################################################################

N = 5000 # population size
t = 60 # number of generations
numTrajectories = 10000 #3*(10**6) # number of trajectories
p0 = np.random.uniform(0.4, 0.9, numTrajectories) # initial frequencies

sType = autoRegP # neutral, autoRegP or stepFunc

stepStrength = 0.01 # strength of selection if using step function
autoRegPDistParam = 0.6 # parameter for geometric distribution if using auto reg. order p process (or single coeff. is using AR1)
autoRegPDegree = 10 # degree for auto reg. process

noiseSTD = 0.01 # standard deviation for noise when using an autoregressive process

transformationType = fishersAngular # fishersAngular or variance

ksTestType = continuous # discrete or continuous

coarseGrain = False # True or False

filterOutFixed = True # True or False

###################################################################

# simulate
if sType == neutral:    
    trajectories = simulation(N, t, numTrajectories, p0, sType)
elif sType == stepFunc:
    trajectories = simulation(N, t, numTrajectories, p0, sType, c = stepStrength)
elif sType == autoRegP:
    trajectories = simulation(N, t, numTrajectories, p0, sType, c = autoRegPDistParam, d = autoRegPDegree, noiseSTD = noiseSTD)
    
# coarse graining
if coarseGrain:
    tempArray = np.zeros((int(t/10)+1, numTrajectories))
    tempArray[0,:] = trajectories[0,:]
    for i in range(1,int(t/10)+1):
        tempArray[i,:] = trajectories[i*10,:]
    trajectories = tempArray
    t = int(t/10) 
    
# filter out trajectories that fix
if filterOutFixed:
    tempArray = trajectories[:,trajectories[-1,:] != 1]
    tempArray = tempArray[:, tempArray[-1,:] != 0]
    trajectories = tempArray
    numTrajectories = trajectories.shape[1]

# transform and procedure
if transformationType == fishersAngular:
    FisherTrajs, newTrajs, reverseTransform = procWithFT(trajectories, numTrajectories, t)
elif transformationType == variance:
    newTrajs, reverseTransform = procWithVarT(trajectories, numTrajectories, t)

# deviations
origDevs = findMaxDevs(trajectories, trajectories[0,:], numTrajectories)
transformedDevs = findMaxDevs(reverseTransform, reverseTransform[0,:], numTrajectories)

# KS test
if ksTestType == discrete:
    pVal = discreteKStest(origDevs, transformedDevs)
elif ksTestType == continuous:
    pVal = sp.stats.kstest(origDevs, transformedDevs).pvalue

# os.system('say finished')
winsound.Beep(500,800)
