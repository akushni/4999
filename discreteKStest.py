import numpy as np
import scipy as sp


# From pages 188 - 189 of Schroer and Trenkler paper

def recursiveStep(i,j,A):
    
    if i < 0 or j < 0:
        A = 0
        
    else:
        A = A[i, j]
    
    return A



def testStat(i, k, n1, n2, c, N, z):
    # Eq. 9
    
    if ((k - i) / n2) - (i / n1) < c:# or (k < N and z[k] == z[k + 1]):
        D = 1
    
    else:
        D = 0
        
    return D



def discreteKStest(dist1, dist2, c):
    
    n1 = len(dist1)
    n2 = len(dist2)
    N = n1 + n2
    
    orderedCombStat = np.sort(np.append(dist1, dist2))
    
    A = np.zeros((n1 + 1, n2 + 1))
    A[0,0] = 1
    
    for k in range(1, N + 1):
        for i in range(max(0,k - n2), min(k, n1) + 1):
            # Eq. 5
            A[i, k - i] = testStat(i, k, n1, n2, c, N, orderedCombStat) * (recursiveStep(i, k - i - 1, A) + recursiveStep(i - 1, k - i, A))
            
    return 1 - A[n1,n2] / sp.special.comb(N, n1)
