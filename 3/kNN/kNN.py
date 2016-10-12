from numpy import *
from euclideanDistance import euclideanDistance
import pdb

def kNN(k, X, labels, x):
    # Assigns to the test instance the label of the majority of the labels of the k closest
	# training examples using the kNN with euclidean distance.
    #
    # Input: k: number of nearest neighbors
    #        X: training data
    #        labels: class labels of training data
    #        x: test instance

    height,width = X.shape #784
    distance = ones(height)
    for i in range(height):
        distance[i] = euclideanDistance(X[i,:],x)

    closestLabels = labels[distance.argsort()[0:k]]

    countLabel = zeros(max(labels)+1)
    for i in range(k):
        countLabel[closestLabels[i]]+=1

    label = countLabel.argmax()

    # ====================== ADD YOUR CODE HERE =============================
    # Instructions: Run the kNN algorithm to predict the class of
    #               x. Rows of X correspond to observations, columns
    #               to features. The 'labels' vector contains the
    #               class to which each observation of the training
    #               data X belongs. Calculate the distance between x and each
    #               row of X, find  the k closest observations and give x
    #               the class of the majority of them.
    #
    # Note: To compute the distance between two vectors A and B use
    #       use the euclideanDistance(A,B) function.
    #


    # return the label of the test data
    return label
