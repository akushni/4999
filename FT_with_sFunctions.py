import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def s_stepFunction(t,swapPoint):
    
    if t <= swapPoint:
        return 0.01
    else:
        return - 0.01
    
def s_autoReg(c,s):
    return c * s + np.random.normal(0,0.0001,len(s))


N = 10000  # population size
p0 = 0.35 # initial frequency
t = 500 # number of generations
numTrajectories = 10**4  # number of trajectories

s = 0.01 # using step function, s is initially 0.01
# s = np.random.normal(0,0.01,numTrajectories); # for autoReg, initially drawing random s values for each traj

p_s = p0 + s * p0 * (1 - p0)  # adjust frequency to incorporate s
trajectories = np.zeros((t + 1, numTrajectories))  # setting up array to hold all trajectories throughout time
trajectories[0,:] = p_s  # saving first set of frequencies

for i in range(1, t + 1):  # for each generation
    n = np.random.binomial(N, trajectories[i-1])  # draw value from binomial distribution using previous frequencies
    p = n/N  # calculate frequency using number of alleles from above
    
    s = s_stepFunction(i,t/2)
    # s = s_autoReg(0.3,s)
    
    p_s = p + s * p * (1 - p)  # adjust frequency to incorporate s
    trajectories[i,:] = p_s  # store list of frequencies

FisherTrajs = np.arccos(1 - 2 * trajectories)  # Fisher's angular transform: arccos(1-2x)

increments = np.zeros((t, numTrajectories))  # setting up matrix to hold the t increments for each traj.
FisherIncrements = np.zeros((t, numTrajectories))
for i in range(1,t + 1):  # for each generation
    increments[i-1,:] = trajectories[i,:] - trajectories[i-1,:]  # calculate change in freq.
    FisherIncrements[i-1,:] = FisherTrajs[i,:] - FisherTrajs[i-1,:]

FisherIncsShuffled = np.zeros((t, numTrajectories))  # setting up matrix to hold shuffled increments
shuffledIncs = np.zeros((t, numTrajectories))
for i in range(numTrajectories):  # for each traj.
    ## USING RANDOM PERMUTAION
    shuffledIncs[:,i] = np.random.permutation(increments[:,i])  # shuffle the increments
    FisherIncsShuffled[:,i] = np.random.permutation(FisherIncrements[:,i])
    ## USING PAIRWISE PERMUTATIONS
    # permutation = np.random.permutation(t)
    # shuffledIncs[:,i] = increments[:,i][permutation]
    # FisherIncsShuffled[:,i] = FisherIncrements[:,i][permutation]

newTrajs = np.zeros((t + 1 , numTrajectories))  # setting up matrix to hold new freq. after shuffling
newFisherTrajs = np.zeros((t + 1, numTrajectories))
newTrajs[0,:] = trajectories[0,:]  # initial frequencies remain the same
newFisherTrajs[0,:] = FisherTrajs[0,:]
for i in range(1, t + 1):  # for each generation
    newTrajs[i,:] = newTrajs[i-1,:] + shuffledIncs[i-1,:]  # add corresponding increment
    newFisherTrajs[i,:] = newFisherTrajs[i-1,:] + FisherIncsShuffled[i-1,:]

reverseTransform = -0.5 * (np.cos(newFisherTrajs) - 1) # inverse of the transformation
