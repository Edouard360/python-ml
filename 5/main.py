################ Feature selection for classification ######################
# This program computes the infornation gain and chisquare measures for the 
# data set "data.csv".
# Then, for a specificed number of features trains the classifier (in this 
# case logistic regression). Finally, it computes and visualizes precision/
# recall curves. 
############################################################################

from featureSelection import FeatureSelection
from matplotlib.pyplot import *

print("\nRunning ...")

# Load the data set 
traindata = np.loadtxt('data.csv', delimiter=',')
#load 1st column 
Y = traindata[:,0:1]
# load columns 2 - end
X = traindata[:,1:traindata.shape[1]]

# Load test data and split data from class labels
testdata = np.loadtxt('test.csv', delimiter=',')
rY = testdata[:, 0] # The real class labels
test = np.transpose(testdata[:, 1:testdata.shape[1]])

f = FeatureSelection(traindata,testdata)
f.computeFeatureImportance()
infogain20 = f.selectFeatures(20,"infogain")
infogain50 = f.selectFeatures(50,"infogain")
infogain100 = f.selectFeatures(100,"infogain")

chiSQ20 = f.selectFeatures(20,"chiSQ")
chiSQ50 = f.selectFeatures(50,"chiSQ")
chiSQ100 = f.selectFeatures(100,"chiSQ")


figure(figsize=(11, 5))
cmap = get_cmap('nipy_spectral')

subplot(1,2,1)
plot(infogain20.prerec[:, 0], infogain20.prerec[:, 1], marker='+', c=cmap(0.2))
plot(infogain50.prerec[:, 0], infogain50.prerec[:, 1], marker='+', c=cmap(0.4))
plot(infogain100.prerec[:, 0], infogain100.prerec[:, 1], marker='+', c=cmap(0.6))
ylim([0.0, 1.05])
xlim([0.0, 1.0])
xlabel('Recall')
ylabel('Precision')
title('infogain')
subplot(1,2,2)
plot(chiSQ20.prerec[:, 0], chiSQ20.prerec[:, 1], marker='+', c=cmap(0.2))
plot(chiSQ50.prerec[:, 0], chiSQ50.prerec[:, 1], marker='+', c=cmap(0.4))
plot(chiSQ100.prerec[:, 0], chiSQ100.prerec[:, 1], marker='+', c=cmap(0.6))
ylim([0.0, 1.05])
xlim([0.0, 1.0])
xlabel('Recall')
ylabel('Precision')
title('chiSQ');

show()
