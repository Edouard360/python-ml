import numpy as np
from logisticRegression import logisticRegression
import timeit

class Regression():
    def __init__(self,X,Y):
        self.dim = X.shape[0]
        start = timeit.default_timer()
        self.w = logisticRegression(X, Y)
        stop = timeit.default_timer()
        print('Running Time with ' + str(len(X)) + ' dimensions: ' + str(stop - start))
        self.time = stop - start

    def computePrerec(self,X_test,rY):
        oldpY = 1 / (1 + np.exp(-np.dot(np.transpose(X_test), self.w)))
        # Matrix to store the precicion recall values
        prerec = np.zeros((101, 2))

        # Here we create the Precision-Recall curve applying thresholding and using
        # several values for the threshold. To do so, we need to compute the confusion
        # matrix (true positive, false positive, false negative)
        for i in range(101):
            tp = 0  # true positive
            fp = 0  # false positive
            fn = 0  # false negative

            thres = float(i) / 100

            pY = np.copy(oldpY)

            c1 = np.where(pY >= thres)[0]  # the items classified 1
            c2 = np.where(pY < thres)[0]  # the items classified 0
            pY[c1] = 1
            pY[c2] = 0

            # Fill confusion matrix
            for j in range(pY.shape[0]):
                if pY[j] == 1:
                    if rY[j] == 1:  # the real class value for the specific data point
                        tp += 1
                    else:
                        fp += 1

                if pY[j] == 0 and rY[j] == 1:
                    fn += 1

            try:
                precisiontmp = float(tp) / (tp + fp)
            except ZeroDivisionError:
                precisiontmp = np.nan

            recalltmp = float(tp) / (tp + fn)

            # Fill precision-recall values for the precision-recall curve
            prerec[i, 0] = recalltmp
            prerec[i, 1] = precisiontmp

        self.prerec = prerec

