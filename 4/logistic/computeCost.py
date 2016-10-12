from numpy import *
from sigmoid import sigmoid


def computeCost(theta, X, y):
    # Computes the gradient of the cost with respect to
    # the parameters.

    m = X.shape[0]  # number of training examples

    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.

    tmp = dot(X, transpose([theta]))[:, 0]
    for i in range(len(tmp)):
        tmp[i] = sigmoid(tmp[i])

    J = -1 / m * (sum(y * log(tmp) + (1 - y) * log(1 - tmp)))

    # =============================================================

    return J
