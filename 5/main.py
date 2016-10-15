################ Feature selection for classification ######################
# This program computes the infornation gain and chisquare measures for the 
# data set "data.csv".
# Then, for a specificed number of features trains the classifier (in this 
# case logistic regression). Finally, it computes and visualizes precision/
# recall curves. 
############################################################################

import numpy as np
from math import pow
import timeit
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from chiSQ import chiSQ
from infogain import infogain
from logisticRegression import logisticRegression
from tools import computePrerec
from scipy import stats

cmap = get_cmap('nipy_spectral')
print("\nRunning ...")

# Load the data set 
data = np.loadtxt('data.csv', delimiter=',')
#load 1st column 
Y = data[:,0:1]
# load columns 2 - end
X = data[:,1:data.shape[1]]

# Load test data and split data from class labels
data = np.loadtxt('test.csv', delimiter=',')
rY = data[:, 0] # The real class labels
test = np.transpose(data[:, 1:data.shape[1]])
 
# Enables or disables feature selection
## TODO: Set 'True' or 'False' this variable
featureSelection = True

if featureSelection == True:
    # For each feature we get its feature selection value (x^2 or IG)
    ## TODO: uncommnent chiSQ(X,Y) to compute chi^2 measure
    gain_1 = infogain(X,Y)
    #gain_2 = chiSQ(X,Y)

    index_1 = np.argsort(gain_1)[::-1]
    #index_2 = np.argsort(gain_2)[::-1]

    ########## ADD YOUR CODE HERE #######################################
    # Compute Kendall tau correlation
    ## TODO: Compute the Kendall tau correlation of the features' lists produced
    # by the two feature selection measures

    #stats.kendalltau(gain_1, gain_2)

    #####################################################################

    # Number of features to be considered in the classification
    # should be changed to compare performance for different number
    # of features
    ## TODO: Set the 'num_feat' variable to the number of features

    # Select the top num_feat features

dimensions = [20,40,60,80,100,150]

for i in dimensions:
    Xtmp = X[:, index_1[:i]]
    Xtmp = np.transpose(Xtmp)
    # Start measuring execution time
    start = timeit.default_timer()

    # Train the classifier
    w = logisticRegression(Xtmp,Y)

    # Print logistic regression learning execution time
    stop = timeit.default_timer()
    print ('Running Time with '+ str(i)+' dimensions: ' + str(stop-start))
    testtmp = test[index_1[:i],:]
    # Perform predictions of the test data
    pY = 1 / (1 + np.exp(-np.dot(np.transpose(testtmp), w)))  # predictions for the class
    prerec = computePrerec(pY, rY)
    # Draw the precision recall curve
    plt.plot(prerec[:, 0], prerec[:, 1], marker='+', c=cmap(i/X.shape[1]))
    print('Trapezoidal rule Area: ' + str(np.trapz(prerec[:, 0], dx=0.01))+'\n')

if featureSelection == True:
     plt.title('Logistic regression with ' + str(dimensions) + ' features')
else:
    plt.title('Logistic regression with all features')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
        
          
