from numpy import argsort, transpose
from scipy.stats import stats

from chiSQ import chiSQ
from infogain import infogain
from regression import Regression

class FeatureSelection():
    """Class to compare two data methods"""
    def __init__(self,train,test):
        self.Y = train[:, 0:1]
        self.X = train[:, 1:train.shape[1]]
        self.Y_test = test[:, 0:1]
        self.X_test = transpose(test[:, 1:test.shape[1]])

    def computeFeatureImportance(self):
        self.infogain = argsort(infogain(self.X, self.Y))[::-1]
        self.chiSQ = argsort(chiSQ(self.X, self.Y))[::-1]
        print("Kendalltau : ")
        print(stats.kendalltau(self.infogain, self.chiSQ))
        # 1 correlated: 0 uncorrelated :-1 opposite correlation

    def selectFeatures(self,n,method):
        if method == "chiSQ":
            index = self.chiSQ[:n]
        elif method == "infogain":
            index = self.infogain[:n]
        else:
            raise NameError('No such existing method !')

        r = Regression(transpose(self.X[:,index]),self.Y)
        r.computePrerec(self.X_test[index,:],self.Y_test)
        return r
