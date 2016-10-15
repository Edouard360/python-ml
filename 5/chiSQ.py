# Feature selection with the Chi^2 measure

from numpy import *

def chiSQ(x, y):
    '''
        x: features (data)
        y: output (classes)
    '''
    cl = unique(y) # unique number of classes
    rows = x.shape[0]
    dim = x.shape[1]
    chisq = zeros(dim) # initialize array (vector) for the chi^2 values
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: calculate the importance for each feature

    numberOfInstancesInClass = zeros(len(cl))
    for m in range(len(cl)):
        numberOfInstancesInClass[m] = sum(cl[m] == y)

    for i in range(dim):
        # We got lg and cl and we iterate over those two
        lg = unique(x[:,i])

        numberOfInstancesWithValue = zeros(len(lg))
        for m in range(len(lg)):
            numberOfInstancesWithValue[m] = sum(lg[m]==x[:,i])

        Eij = zeros((len(lg),len(cl)))
        for m in range(len(lg)):
            for p in range(len(cl)):
                Eij[m,p] = numberOfInstancesWithValue[m] * numberOfInstancesInClass[p]/len(lg)

        for m in range(len(lg)):
            for p in range(len(cl)):
                indexIntersection = [all(t) for t in zip(lg[m] == x[:, i], cl[p] == y)]
                Oij = sum(indexIntersection)/len(lg)
                chisq[i] += pow(Oij-Eij[m,p],2)/Eij[m,p]

    return chisq

def testchiSQ():
    X = array([[1,1,1,1],[1,1,2,2],[1,2,2,3],[2,2,2,4]])
    #1,1,1,2
    #1,1,2,2
    #1,2,2,2
    #1,2,3,4
    # X = array([[1,1],[1,2],[2,3],[2,4]])
    Y = array([1,1,1,1])
    S = chiSQ(X, Y)
    print(S)
    return

