import numpy as np
import scipy as sp
import scipy.linalg as linalg


def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """

    classLabels = np.unique(Y) # different class labels on the dataset
    classNum = len(classLabels)
    datanum, dim = X.shape # dimensions of the dataset
    totalMean = np.mean(X,0) # total mean of the data

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.
    #print(np.argsort(Y).index())
    #print(Y[np.argsort(Y)].index(1))
    #print(np.where(Y==1))


    meanX = np.zeros((classNum,dim))
    for i in range(1,classNum+1):
        distance = 0
        index = np.where(Y==i)
        meanX[i-1,:] = np.mean(X[index],axis = 0)

    B = np.zeros((dim, dim))
    W = np.zeros((dim, dim))
    for i in range(datanum):
        L = X[i,:] - meanX[int(Y[i])-1,:]
        B = B + np.dot(np.transpose([L]),[L])

    for i in range(datanum):
        L = X[i,:] - totalMean
        W = W + np.dot(np.transpose([L]),[L])

    values,vectors = linalg.eig(np.dot(linalg.inv(W),B))
    # np.argsort(values)[0:classNum-1]

    M = vectors[values.argsort()[::-1][0:classNum-1]].transpose()
    X_lda = X.dot(M)

    meanX_lda = np.zeros((classNum,classNum-1))
    for i in range(0,classNum):
        meanX_lda[i,:] = np.dot([meanX[i,:]],M)

    return M,meanX_lda,X_lda


    # =============================================================

    #return W, projected_centroid, X_lda
