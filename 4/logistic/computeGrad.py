from numpy import *
from sigmoid import sigmoid


def computeGrad(theta, X, y):
    # Computes the gradient of the cost with respect to
    # the parameters.

    m = X.shape[0]  # number of training examples

    grad = zeros(size(theta))  # initialize gradient

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.

    tmp = dot(X, transpose([theta]))[:, 0]
    for i in range(len(tmp)):
        tmp[i] = sigmoid(tmp[i])

    delta = diag(tmp - y).dot(X)
    tdelta = mean(delta, axis=0)
    return tdelta

