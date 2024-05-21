import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os

from allFunctions_SimAndProc import *

###################################################################

N = 1000 # population size
t = 60 # number of generations
numTrajectories = 3*(10**6) # number of trajectories
p0 = np.random.uniform(0.4, 0.9, numTrajectories) # initial frequencies

sType = autoRegP # neutral, autoReg, autoRegP or stepFunc
stepStrength = 0.01 # strength of selection if using step function
autoRegConst = 0.6 # proportion of previous value maintained if using auto regressive process

transformationType = fishersAngular # fishersAngular or variance

ksTestType = continuous # discrete or continuous

coarseGrain = True # True or False

filterOutFixed = True # True or False

###################################################################

# simulate
if sType == neutral:    
    trajectories = simulation(N, t, numTrajectories, p0, sType)
elif sType == stepFunc:
    trajectories = simulation(N, t, numTrajectories, p0, sType, stepStrength)
elif sType == autoReg:
    trajectories = simulation(N, t, numTrajectories, p0, sType, autoRegConst)
elif sType == autoRegP:
    trajectories = simulation(N, t, numTrajectories, p0, sType)
    
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

os.system('say finished')
