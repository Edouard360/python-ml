from numpy import *
from scipy.linalg import inv

def poly_exp(X, degree):
    N, D = X.shape
    for d in range(2, degree + 1):
        X = column_stack([X, X[:, 0:D] ** d])
    return X

def MSE(yt, yp):
    assert len(yt)==len(yp),"Check vector size"
    # for i in range(len(yt)):
    #     s += (yt[i] - yp[i]) ** 2
    return sum((yt-yp)**2)/len(yt)

# return two arrays of the computed MSE
# for the train data
# for the test data
def regression(train,y,test,y_test,max_degree):
    MSE_train = zeros(max_degree)
    MSE_test = zeros(max_degree)
    for degree in range(1,max_degree+1):
        Z = poly_exp(train, degree)
        Z = column_stack([ones(len(Z)), Z])

        Z_test = poly_exp(test, degree)
        Z_test = column_stack([ones(len(Z_test)), Z_test])

        w = dot(dot(inv(dot(transpose(Z), Z)), transpose(Z)), y)
        y_train = dot(Z, w)
        y_pred = dot(Z_test, w)
        MSE_train[degree-1]=MSE(y, y_train)
        MSE_test[degree-1]=MSE(y_test, y_pred)

    return MSE_train, MSE_test
