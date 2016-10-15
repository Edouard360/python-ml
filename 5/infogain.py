# Feature selection with the Information Gain measure

from numpy import *
from math import log

def entropy(z):
    cl = unique(z)
    hz = 0
    for i in range(len(cl)):
        c = cl[i]
        pz = float(sum(z == c)) / len(z)
        hz = hz + pz * log(pz, 2)
    hz = -hz
    return hz


def infogain(x, y):
    '''
        x: features (data)
        y: output (classes)
    '''
    info_gains = ones(x.shape[1]) # features of x
    nrows = x.shape[0]
    # calculate entropy of the data *hy* with regards to class y

    hy = entropy(y)
    info_gains *= hy

    # ====================== YOUR CODE HERE ================================
    # Instructions: calculate the information gain for each column (feature)

    for i in range(len(info_gains)):
        xi_unique = unique(x[:, i])
        #xi_unique = unique(array(x[:,i])[:,0])
        for j in range(len(xi_unique)):
            indexi = (xi_unique[j] == x[:, i])
            #indexi = (xi_unique[j] == array(x[:,i])[:,0])
            info_gains[i] -= entropy(y[indexi])*(sum(indexi)/nrows)

    return info_gains


def testinfogain():
    X = matrix([[1,1],[1,3],[2,3],[2,3]])
    # X could also be an array...
    Y = array([1,2,3,2])
    S = infogain(X,Y)
    print(S)
    return
