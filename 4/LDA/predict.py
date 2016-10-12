import numpy as np
import scipy as sp
import scipy.linalg as linalg
import pdb

def predict(X, projected_centroid, W):

    """Apply the trained LDA classifier on the test data
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """


    # Project test data onto the LDA space defined by W
    projected_data  = np.dot(X, W)
    l , c = projected_data.shape
    numClass = len(projected_centroid)
    label = np.zeros(l)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in the code to implement the classification
    # part of LDA. Follow the steps given in the assigment.

    # projected_dataprojected_centroid
    for j in range(l):
        tmp = linalg.norm(projected_data[j,:] - projected_centroid[0,:])
        jclass = 0
        for i in range(1,numClass):
            tmp1 = linalg.norm(projected_data[j,:] - projected_centroid[i,:])
            if tmp1<tmp :
                tmp=tmp1
                jclass = i
        label[j]=jclass

    # =============================================================

    # Return the predicted labels of the test data X
    return label
