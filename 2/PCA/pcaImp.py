from numpy import *


# data : the data matrix
# k the number of component to return
# return the new data and  the variance that was maintained AND the principal components (ALL)
def pca(data, k):
    # Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
    # Rows of A correspond to observations (wines), columns to variables.
    ## TODO: Implement PCA

    c = data - mean(data, axis=0)
    w = dot(transpose(c), c)
    values, vector = linalg.eig(w)

    sort_values = values.argsort()[::-1]
    eigvec = -vector[:, sort_values]

    perc = sum(values[sort_values[:k]])/sum(values)
    new_data = dot(c,eigvec[:,:k])

    # compute covariance matrix
    # compute eigenvalues and eigenvectors of covariance matrix
    # Sort eigenvalues
    # Sort eigenvectors according to eigenvalues
    # Project the data to the new space (k-D)
    return new_data,perc,eigvec
