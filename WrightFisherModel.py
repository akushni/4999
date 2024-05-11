import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from discreteKStest import *
from findMaxDev import *

N = 10000  # population size
p0 = np.random.uniform(0.3, 0.7, 500) #0.35 # initial frequency
t = 200 # number of generations
numTrajectories = 500 #10**4  # number of trajectories

s = 0 #np.random.normal(0,0.01,numTrajectories); # drawing a normally distributed s value for each trajectory

p_s = p0 + s * p0 * (1 - p0)  # adjust frequency to incorporate s
trajectories = np.zeros((t + 1, numTrajectories))  # setting up array to hold all trajectories throughout time
trajectories[0,:] = p_s  # saving first set of frequencies

for i in range(1, t + 1):  # for each generation
    n = np.random.binomial(N, trajectories[i-1])  # draw value from binomial distribution using previous frequencies
    p = n/N  # calculate frequency using number of alleles from above
    
    #s = np.random.normal(0,0.01,numTrajectories); # drawing a normally distributed s value for each trajectory each generation
    
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

origDevs = findMaxDevs(trajectories,trajectories[0,:],numTrajectories)
transformedDevs = findMaxDevs(reverseTransform,reverseTransform[0,:],numTrajectories)
print('Finished')
#
# np.savetxt('/Users/abigailkushnir/Desktop/4999/FinalPresentation/trajs', trajectories)
# np.savetxt('/Users/abigailkushnir/Desktop/4999/FinalPresentation/transformedDevs', transformedDevs)


#print(discreteKStest(origDevs, transformedDevs))

# for i in range(100,275,25):
#     np.savetxt('trajectories'+ str(i),trajectories[i])
#     np.savetxt('fisherTrajs' + str(i), FisherTrajs[i])
#     np.savetxt('shuffledTrajs' + str(i), newTrajs[i])
#     np.savetxt('shuffledFisher' + str(i), newFisherTrajs[i])
#     np.savetxt('reversedTransform' + str(i), reverseTransform[i])

# np.savetxt('trajectories',trajectories[int(t/2)])
# np.savetxt('fisherTrajs',FisherTrajs[int(t/2)])
# np.savetxt('shuffledTrajs',newTrajs[int(t/2)])
# np.savetxt('shuffledFisher',newFisherTrajs[int(t/2)])
# np.savetxt('reversedTransform',reverseTransform[int(t/2)])

# origEDF = sp.stats.ecdf(trajectories[int(t/2)])
# shuffledEDF = sp.stats.ecdf(newTrajs[int(t/2)])
# transformedEDF = sp.stats.ecdf(reverseTransform[int(t/2)])

# ax = plt.subplot()
# origEDF.cdf.plot(ax)
# shuffledEDF.cdf.plot(ax)
# transformedEDF.cdf.plot(ax)
# plt.show()


