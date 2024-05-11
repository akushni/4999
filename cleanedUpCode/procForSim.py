import numpy as np
import matplotlib.pyplot as plt

from allFunctions_SimAndProc import *

###################################################################

N = 10000 # population size
t = 200 # number of generations
numTrajectories = 500 # number of trajectories
p0 = np.random.uniform(0.3, 0.7, numTrajectories) # initial frequencies

sType = autoReg
stepStrength = 0.01 # strength of selection if using step function
autoRegConst = 0.3 # proportion of previous value maintained if using auto regressive process

transformationType = fishersAngular

###################################################################

# simulate
if sType == neutral:    
    trajectories = simulation(N, t, numTrajectories, p0, sType)
elif sType == stepFunc:
    trajectories = simulation(N, t, numTrajectories, p0, sType, stepStrength)
elif sType == autoReg:
    trajectories = simulation(N, t, numTrajectories, p0, sType, autoRegConst)

# transform
if transformationType == fishersAngular:
    FisherTrajs, newTrajs, reverseTransform = procWithFT(trajectories, numTrajectories, t)
elif transformationType == variance:
    newTrajs, reverseTransform = procWithVarT(trajectories, numTrajectories, t)

# deviations
origDevs = findMaxDevs(trajectories, trajectories[0,:], numTrajectories)
transformedDevs = findMaxDevs(reverseTransform, reverseTransform[0,:], numTrajectories)

# KS test
pVal = discreteKStest(origDevs, transformedDevs)

