import numpy as np

N = 500  # population size
p0 = 0.5  # initial frequency
s = -0.01  # s value
t = 100  # number of generations
numTrajectories = 1000  # number of trajectories

p_s = p0 + s * p0 * (1 - p0)  # adjust frequency to incorporate s
trajectories = np.zeros((t + 1, numTrajectories))  # setting up array to hold all trajectories throughout time
trajectories[0][:] = p_s  # saving first set of frequencies

for i in range(1, t + 1):  # for each generation
    n = np.random.binomial(N, trajectories[i-1])  # draw value from binomial distribution using previous frequencies
    p = n/N  # calculate frequency using number of alleles from above
    p_s = p + s * p * (1 - p)  # adjust frequency to incorporate s
    trajectories[i][:] = p_s  # store list of frequencies


# with open("trajectories",'w') as f: # for saving
    # np.savetxt(f, trajectories)

