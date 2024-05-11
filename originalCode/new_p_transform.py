import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# import statsmodels.api as sm

N = 10000  # population size
p0 = 0.35 # initial frequenc
t = 500 # number of generations
numTrajectories = 10**4  # number of trajectories

s = 0 #np.random.normal(0,0.001,numTrajectories); # drawing a normally distributed s value for each trajectory

p_s = p0 + s * p0 * (1 - p0) # adjust frequency to incorporate s
trajectories = np.zeros((t + 1, numTrajectories))  # setting up array to hold all trajectories throughout time
trajectories[0,:] = p_s  # saving first set of frequencies

for i in range(1, t + 1):  # for each generation
    n = np.random.binomial(N, trajectories[i-1])  # draw value from binomial distribution using previous frequencies
    p = n/N  # calculate frequency using number of alleles from above
    p_s = p + s * p * (1 - p)  # adjust frequency to incorporate s
    trajectories[i,:] = p_s  # store list of frequencies
    

increments = np.zeros((t, numTrajectories))  # setting up matrix to hold the t increments for each traj.
for i in range(1,t + 1):  # for each generation
    increments[i-1,:] = trajectories[i,:] - trajectories[i-1,:]  # calculate change in freq.

transformedIncs = np.zeros((t, numTrajectories))  # setting up matrix to hold transformed increments
for i in range(0,t): # for each generation
    transformedIncs[i,:] = increments[i,:] / np.sqrt(trajectories[i,:] * (1 - trajectories[i,:]))

transformedIncsShuffled = np.zeros((t, numTrajectories))  # setting up matrix to hold shuffled increments
shuffledIncs = np.zeros((t, numTrajectories))
for i in range(numTrajectories):  # for each traj.
    ## USING RANDOM PERMUTAION
    shuffledIncs[:,i] = np.random.permutation(increments[:,i])  # shuffle the increments
    transformedIncsShuffled[:,i] = np.random.permutation(transformedIncs[:,i])
    ## USING PAIRWISE PERMUTATIONS
    # permutation = np.random.permutation(t)
    # shuffledIncs[:,i] = increments[:,i][permutation]
    # transformedIncsShuffled[:,i] = transformedIncs[:,i][permutation]


newTrajs = np.zeros((t + 1, numTrajectories))  # setting up matrix to hold new freq. after shuffling
newTransformedTrajs = np.zeros((t + 1, numTrajectories))

newTrajs[0,:] = trajectories[0,:]  # initial frequencies remain the same
newTransformedTrajs[0,:] = trajectories[0,:]

for i in range(1, t + 1):  # for each generation
    newTrajs[i,:] = newTrajs[i - 1,:] + shuffledIncs[i - 1,:]
    newTransformedTrajs[i,:] = newTransformedTrajs[i - 1,:] + transformedIncsShuffled[i - 1,:] * (np.sqrt(newTransformedTrajs[i - 1,:] * (1 - newTransformedTrajs[i - 1,:])))

np.savetxt("/Users/abigailkushnir/Desktop/originalFreq", trajectories[250,:])
np.savetxt("/Users/abigailkushnir/Desktop/tranformedFreq", newTransformedTrajs[250,:])

# for k in range(0,1000):
#     print(k);
#     runfile('/Users/abigailkushnir/Desktop/4999/Code/new_p_transform.py');
#     origVshuff = sp.stats.kstest(trajectories[5,:],newTrajs[5,:]);
#     origVSshuff_pVals[k] = origVshuff[1];
#     origVtrans = sp.stats.kstest(trajectories[5,:],reverseTransform[5,:]);
#     origVStrans_pVals[k] = origVtrans[1];
#     shuffVtrans = sp.stats.kstest(newTrajs[5,:],reverseTransform[5,:]);
#     shuffVStrans_pVals[k] = shuffVtrans[1];
