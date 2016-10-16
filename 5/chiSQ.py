# Feature selection with the Chi^2 measure

from numpy import *

def chiSQ(x, y):
    '''
        x: features (data)
        y: output (classes)
    '''
    cl = unique(y)  # unique number of classes
    rows = x.shape[0]
    dim = x.shape[1]
    chisq = zeros(dim)  # initialize array (vector) for the chi^2 values

    # ====================== YOUR CODE HERE ======================
    # Instructions: calculate the importance for each feature

    numberOfInstancesInClass = zeros(len(cl))
    for n in range(len(cl)):
        numberOfInstancesInClass[n] = sum(cl[n] == y)

    for i in range(dim):
        # We got lg and cl and we iterate over those two
        lg = unique(x[:, i])

        numberOfInstancesWithValue = zeros(len(lg))
        for m in range(len(lg)):
            numberOfInstancesWithValue[m] = sum(lg[m] == x[:, i])

        table = zeros((len(lg), len(cl)))
        for j in range(rows):
            table[lg==x[j,i],cl == y[j]]+=1

        for m in range(len(lg)):
            for n in range(len(cl)):
                Emn = numberOfInstancesWithValue[m]*numberOfInstancesInClass[n]/len(lg)
                chisq[i] += pow(table[m,n] - Emn, 2) / Emn

    return chisq

def testchiSQ():
    X = array([[1,1,1,1],[1,1,2,2],[1,2,2,3],[2,2,2,4]])
    #X = array([[1, 1], [1, 1], [1, 2], [2, 2]])
    #1,1,1,2
    #1,1,2,2
    #1,2,2,2
    #1,2,3,4
    # X = array([[1,1],[1,2],[2,3],[2,4]])
    Y = array([1,1,2,2])
    S = chiSQ2(X, Y)
    print(S)
    return

# DEPRECATED CHI
# def chiSQ(x, y):
#     '''
#         x: features (data)
#         y: output (classes)
#     '''
#     cl = unique(y)  # unique number of classes
#     rows = x.shape[0]
#     dim = x.shape[1]
#     chisq = zeros(dim)  # initialize array (vector) for the chi^2 values
#
#     # ====================== YOUR CODE HERE ======================
#     # Instructions: calculate the importance for each feature
#
#     numberOfInstancesInClass = zeros(len(cl))
#     for m in range(len(cl)):
#         numberOfInstancesInClass[m] = sum(cl[m] == y)
#
#     for i in range(dim):
#         # We got lg and cl and we iterate over those two
#         lg = unique(x[:, i])
#
#         numberOfInstancesWithValue = zeros(len(lg))
#         for m in range(len(lg)):
#             numberOfInstancesWithValue[m] = sum(lg[m] == x[:, i])
#
#         Eij = zeros((len(lg), len(cl)))
#         for m in range(len(lg)):
#             for p in range(len(cl)):
#                 Eij[m, p] = numberOfInstancesWithValue[m] * numberOfInstancesInClass[p] / len(lg)
#
#         for m in range(len(lg)):
#             local_index = (lg[m] == x[:, i])
#             for p in range(len(cl)):
#                 indexIntersection = (local_index & (cl[p] == y))
#                 Oij = sum(indexIntersection) / len(lg)
#                 chisq[i] += pow(Oij - Eij[m, p], 2) / Eij[m, p]
#
#     return chisq
