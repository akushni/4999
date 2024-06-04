import numpy as np

def findMaxDevs(trajMat, p0, numTrajs):
    
    maxDevs = np.zeros(numTrajs)

    for i in range(numTrajs):

        initFreq = p0[i]

        # absolute max deviation in either direction
        dev = max(abs(trajMat[:,i] - initFreq))
        maxDevs[i] = dev
        
        # max deviation with direction
        # devs = [min(trajMat[:,i] - p0), max(trajMat[:,i] - p0)]
        # if abs(devs[0]) > abs(devs[1]):
        #     maxDevs[i] = devs[0]
        # else:
        #     maxDevs[i] = devs[1]
    
    return maxDevs
