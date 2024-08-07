from multiprocessing import Pool
import numpy as np
import scipy as sp
import itertools
import time


# degree p auto regressive selection structure
def s_autoRegP(sVals, p, t, coeffs, noiseSTD):
    
    recentSVals = np.zeros((p, sVals.shape[1]))
    
    if t < p: # if less than p generations have passed
        for i in range(1, t+1):
            recentSVals[i-1,:] = sVals[t-i,:]
    else:
        for i in range(1, p+1):
            recentSVals[i-1] = sVals[t-i,:]
        
    newSVals = np.zeros(sVals.shape[1])
    for i in range(sVals.shape[1]):
        newSVals[i] = np.sum(coeffs * recentSVals[:, i]) + np.random.normal(0,noiseSTD)
        
    return newSVals

def keepInBounds(x):
    if (x>1):
        return 1
    elif (x<0):
        return 0
    else:
        return x

# simulate trajectories
def simulation(N, t, numTrajs, p0, sType, c = None, d = None, noiseSTD = None):
    
    trajectories = np.zeros((t + 1, numTrajs)) # array to hold frequencies
    
    if sType == 0:

        trajectories[0,:] = p0 # save first set of frequencies

        for i in range(1, t+1): # for each generation
            n = np.random.binomial(N, trajectories[i-1]) # draw from binomial
            p = n/N # calculate new frequency
            trajectories[i,:] = p


    elif sType == 1:
    
        trajectories[0,:] = p0 + c * p0 * (1 - p0) # save first set of frequencies (adjusted for selection constant)

        for i in range(1, t+1): # for each generation
            n = np.random.binomial(N, trajectories[i-1]) # draw from binomial
            p = n/N

            s = s_stepFunction(i, t/2, c)

            trajectories[i,:] = p + s * p * (1 - p) # save new frequencies (adjusted for selection)


    elif sType == 2:
        
        sVals = np.zeros((t + 1, numTrajs)) # initialize array to store s values
        sVals[0,:] = np.random.normal(0, 0.01, numTrajs)
        trajectories[0, :] = p0 + sVals[0,:] * p0 * (1 - p0) # save first set of frequencies (adjusted for selection)
        
        degree = d
        
        coeffs = np.zeros(degree)
        for i in range(degree): # draw weights from geometric distribution with p = 0.? and centered at 0
            coeffs[i] = ((-1)**(i+1))*sp.stats.geom.pmf(i, c, -1) 
        
        for i in range(1, t+1):
            n = np.random.binomial(N, trajectories[i-1]) # draw from binomial
            p = n/N
        
            s = s_autoRegP(sVals, degree, i, coeffs, noiseSTD)
           
            trajectories[i,:] = p + s * p * (1 - p) # save new frequencies (adjusted for selection)
            sVals[i] = s # save s values
        
            trajectories[i] = np.array(list(map(keepInBounds, trajectories[i]))) # make sure trajectories are in bounds
        
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



def meanSquaredDiff(params):
# params = [N, std, r]

    N = params[0][0] # population size
    std = params[0][1] # standard deviation of noise
    r = params[0][2] # geometric distribution parameter
    
    dataDiff = params[1]
    fixedBins = params[2]
    
    # drawing p0s from a normal distribution, numTrajs = 100000
    p0 = np.random.normal(0.6, 0.1, 100000)
    p0 = p0[p0 < 1]
    numTrajectories = len(p0)

    trajectories = simulation(N, 6, numTrajectories, p0, 2, c = r, d = 1, noiseSTD = std)
    
    # filter out fixed trajectories
    tempArray = trajectories[:,trajectories[-1,:] != 1]
    tempArray = tempArray[:, tempArray[-1,:] != 0]
    trajectories = tempArray
    discarded = (numTrajectories - trajectories.shape[1]) / numTrajectories
    numTrajectories = trajectories.shape[1]
    
    # transformation
    FisherTrajs, newTrajs, reverseTransform = procWithFT(trajectories, numTrajectories, 6)
    
    simOrigDevs = findMaxDevs(trajectories, trajectories[0,:], trajectories.shape[1])
    simTransformedDevs = findMaxDevs(reverseTransform, reverseTransform[0,:], reverseTransform.shape[1])

    simOrig = np.histogram(simOrigDevs, bins = fixedBins, density = True)
    simTransformed = np.histogram(simTransformedDevs, bins = fixedBins, density = True)
    simDiff = simTransformed[0] - simOrig[0]
    
    MSE = np.sum( (simDiff - dataDiff)**2 )/(len(fixedBins))
    
    return MSE, discarded



def findMaxDevs(trajMat, p0, numTrajs):
    
    maxDevs = np.zeros(numTrajs)

    for i in range(numTrajs):

        initFreq = p0[i]

        # absolute max deviation in either direction
        dev = max(abs(trajMat[:,i] - initFreq))
        maxDevs[i] = dev
    
    return maxDevs

#######################################################################

N = 5000 # population size
t = 60 # number of generations
numTrajectories = 100 # number of trajectories
p0 = np.random.normal(0.6, 0.1, numTrajectories)#np.random.uniform(0.4, 0.9, numTrajectories) # initial frequencies
p0 = p0[p0 < 1]
numTrajectories = len(p0)

sType = 2 # neutral, stepFunc, or autoRegP 

autoRegPDistParam = 0.6 # parameter for geometric distribution if using auto reg. order p process (or single coeff. is using AR1)
autoRegPDegree = 1 # degree for auto reg. process

noiseSTD = 0.01 # standard deviation for noise when using an autoregressive process

fixedBins = np.linspace(0,0.3,100,endpoint=True)

########################################################################
start = time.time()

# MAKE FAKE DATA #
# trajectories = simulation(N, t, numTrajectories, p0, sType, c = autoRegPDistParam, d = autoRegPDegree, noiseSTD = noiseSTD)

# # filter out fixed/lost
# tempArray = trajectories[:,trajectories[-1,:] != 1]
# tempArray = tempArray[:, tempArray[-1,:] != 0]
# trajectories = tempArray
# numTrajectories = trajectories.shape[1]

# FisherTrajs, newTrajs, reverseTransform = procWithFT(trajectories, numTrajectories, t)

# dataOrigDevs = findMaxDevs(trajectories, trajectories[0,:], numTrajectories)
# dataTransDevs = findMaxDevs(reverseTransform, reverseTransform[0,:], numTrajectories)

# dataDiff = np.histogram(dataTransDevs, bins=fixedBins, density=True)[0] - np.histogram(dataOrigDevs, bins=fixedBins, density=True)[0]



# USE REAL DATA
# dataFreq = np.loadtxt('/Users/abigailkushnir/Desktop/data p0 normal/dataR1FrequenciesFiltered.csv');
# dataFreq = np.transpose(dataFreq)
# data100k = dataFreq[:,np.random.permutation(100000)]

# FisherTrajs, newTrajs, reverseTransform = procWithFT(data100k, data100k.shape[1], data100k.shape[0]-1)

# dataOrigDevs = findMaxDevs(data100k, data100k[0,:], data100k.shape[1])
# dataTransDevs = findMaxDevs(reverseTransform, reverseTransform[0,:], reverseTransform.shape[1])

# dataDiff = np.histogram(dataTransDevs, bins=fixedBins, density=True)[0] - np.histogram(dataOrigDevs, bins=fixedBins, density=True)[0]

########################################################################
Nvals = np.linspace(100,2000,20,endpoint=True)
stdVals = np.linspace(0, 0.05, 20, endpoint=True)
rVals = np.linspace(0.1,0.7,20,endpoint=True)

combos = list(itertools.product(Nvals, stdVals, rVals))

# for i in range(len(combos)):
#     combos[i] = [combos[i], dataDiff, fixedBins]
    
    
# if __name__ == '__main__':
#     with Pool(1) as p:
#         mse = p.map(meanSquaredDiff, combos)
#         mseVals = np.array(list(zip(*mse))[0])
#         props = np.array(list(zip(*mse))[1])

# np.savetxt('./dataMSEVals.csv', mseVals)
# np.savetxt('./dataPropsDiscarded.csv', props)
end = time.time()

# with open('./time.txt','w') as f:
#     f.write('Time: '+str(end-start))

