from numpy import *

def poly_exp(X, degree):
    N,D = X.shape
    for d in range(2,degree+1):
        X = column_stack([X,X[:,0:D]**d])
    return X

def MSE(yt,yp):
	s = 0
    for i in yt:
        s += (yt[i]-yp[i])**2
    return s

def USELES():
	return 0