import numpy as np
import scipy as sp
import queue

from discreteKStest import *
from findMaxDev import *

# implemented selection processes
neutral = 0
stepFunc = 1
autoReg = 2
autoRegP = 3

# implemented transformations
fishersAngular = 0
variance = 1

# KS tests
discrete = 0
continuous = 1

# step function selection structure
def s_stepFunction(t, swapPoint, c):
    if t <= swapPoint:
        return c
    else:
        return - c
    
# auto regressive (degree 1) selection structure
def s_autoReg(c,s):
    return c * s + np.random.normal(0,0.01,len(s))

# degree p auto regressive selection structure
def s_autoRegP(sVals, p, t):
    
    coeffs = np.zeros(p)
    for i in range(p): # draw weights from geometric distribution with p = 0.? and centered at 0
        coeffs[i] = sp.stats.geom.pmf(i, 0.8, -1) 
    
    recentSVals = np.zeros((p, sVals.shape[1]))
    
    if t < p: # if less than p generations have passed
        for i in range(1, t+1):
            recentSVals[i-1,:] = sVals[t-i,:]
    else:
        for i in range(1, p+1):
            recentSVals[i-1] = sVals[t-i,:]
        
    newSVals = np.zeros(sVals.shape[1])
    for i in range(sVals.shape[1]):
        newSVals[i] = np.sum(coeffs * recentSVals[:, i]) + np.random.normal(0,0.01)
        
    return newSVals

# simulate trajectories
def simulation(N, t, numTrajs, p0, sType, c = None):
    
    trajectories = np.zeros((t + 1, numTrajs)) # array to hold frequencies
    
    if sType == neutral:

        trajectories[0,:] = p0 # save first set of frequencies

        for i in range(1, t+1): # for each generation
            n = np.random.binomial(N, trajectories[i-1]) # draw from binomial
            p = n/N # calculate new frequency
            trajectories[i,:] = p


    elif sType == stepFunc:
    
        trajectories[0,:] = p0 + c * p0 * (1 - p0) # save first set of frequencies (adjusted for selection constant)

        for i in range(1, t+1): # for each generation
            n = np.random.binomial(N, trajectories[i-1]) # draw from binomial
            p = n/N

            s = s_stepFunction(i, t/2, c)

            trajectories[i,:] = p + s * p * (1 - p) # save new frequencies (adjusted for selection)


    elif sType == autoReg:

        s = np.random.normal(0, 0.01, numTrajs)
        trajectories[0,:] = p0 + s * p0 * (1 - p0)

        for i in range(1, t+1):
            n = np.random.binomial(N, trajectories[i-1]) # draw from binomial
            p = n/N

            s = s_autoReg(c,s)

            trajectories[i,:] = p + s * p * (1 - p) # save new frequencies (adjusted for selection)


    elif sType == autoRegP:
        
        sVals = np.zeros((t + 1, numTrajs)) # initialize array to store s values
        sVals[0,:] = np.random.normal(0, 0.01, numTrajs)
        trajectories[0, :] = p0 + sVals[0,:] * p0 * (1 - p0) # save first set of frequencies (adjusted for selection)
        
        for i in range(1, t+1):
            n = np.random.binomial(N, trajectories[i-1]) # draw from binomial
            p = n/N
        
            s = s_autoRegP(sVals, 10, i)
           
            trajectories[i,:] = p + s * p * (1 - p) # save new frequencies (adjusted for selection)
            sVals[i] = s # save s values
        
    return trajectories

# apply Fisher's Angular transform and permute
def procWithFT(trajectories, numTrajs, t):
    
    FisherTrajs = np.arccos(1 - 2 * trajectories) # the transform

    increments = np.zeros((t, numTrajs))  # setting up matrix to hold the t increments for each traj.
    FisherIncrements = np.zeros((t, numTrajs))
    for i in range(1,t + 1):  # for each generation
        increments[i-1,:] = trajectories[i,:] - trajectories[i-1,:]  # calculate change in freq.
        FisherIncrements[i-1,:] = FisherTrajs[i,:] - FisherTrajs[i-1,:]

    FisherIncsShuffled = np.zeros((t, numTrajs))  # setting up matrix to hold shuffled increments
    shuffledIncs = np.zeros((t, numTrajs))
    for i in range(numTrajs):  # for each traj.
        shuffledIncs[:,i] = np.random.permutation(increments[:,i])  # shuffle the increments
        FisherIncsShuffled[:,i] = np.random.permutation(FisherIncrements[:,i])

    newTrajs = np.zeros((t + 1 , numTrajs))  # setting up matrix to hold new freq. after shuffling
    newFisherTrajs = np.zeros((t + 1, numTrajs))
    newTrajs[0,:] = trajectories[0,:]  # initial frequencies remain the same
    newFisherTrajs[0,:] = FisherTrajs[0,:]
    for i in range(1, t + 1):  # for each generation
        newTrajs[i,:] = newTrajs[i-1,:] + shuffledIncs[i-1,:]  # add corresponding increment
        newFisherTrajs[i,:] = newFisherTrajs[i-1,:] + FisherIncsShuffled[i-1,:]

    reverseTransform = -0.5 * (np.cos(newFisherTrajs) - 1) # inverse of the transformation

    return FisherTrajs, newTrajs, reverseTransform

def procWithVarT(trajectories, numTrajs, t):

    increments = np.zeros((t, numTrajs))  # setting up matrix to hold the t increments for each traj.
    for i in range(1, t + 1):  # for each generation
        increments[i - 1, :] = trajectories[i, :] - trajectories[i - 1, :]

    transformedIncs = np.zeros((t, numTrajs))  # setting up matrix to hold transformed increments
    for i in range(0, t):  # for each generation
        transformedIncs[i, :] = increments[i, :] / np.sqrt(trajectories[i, :] * (1 - trajectories[i, :]))

    transformedIncsShuffled = np.zeros((t, numTrajs))  # setting up matrix to hold shuffled increments
    shuffledIncs = np.zeros((t, numTrajs))
    for i in range(numTrajs):  # for each traj.
        shuffledIncs[:, i] = np.random.permutation(increments[:, i])  # shuffle the increments
        transformedIncsShuffled[:, i] = np.random.permutation(transformedIncs[:, i])
    
    newTrajs = np.zeros((t + 1, numTrajs))  # setting up matrix to hold new freq. after shuffling
    newTransformedTrajs = np.zeros((t + 1, numTrajs))

    newTrajs[0, :] = trajectories[0, :]  # initial frequencies remain the same
    newTransformedTrajs[0, :] = trajectories[0, :]

    for i in range(1, t + 1):  # for each generation
        newTrajs[i, :] = newTrajs[i - 1, :] + shuffledIncs[i - 1, :]
        newTransformedTrajs[i, :] = newTransformedTrajs[i - 1, :] + transformedIncsShuffled[i - 1, :] * (
            np.sqrt(newTransformedTrajs[i - 1, :] * (1 - newTransformedTrajs[i - 1, :])))

    return newTrajs, newTransformedTrajs