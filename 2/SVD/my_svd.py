from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import pdb #pdb.set_trace();

# Load the "gatlin" image data
X = loadtxt('gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the
#               three factors in variables U, S and V
#

height, width = X.shape
U,S,V = linalg.svd(X)

#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()

#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values
#

def reconstruct(numberOfValues):
    Utmp = U[:,:numberOfValues]
    Stmp = diag(S[:numberOfValues])
    Vtmp=V[:numberOfValues,:]
    return dot(Utmp,dot(Stmp,Vtmp))

X5 = reconstruct(5)
X20 = reconstruct(20)
X50 = reconstruct(50)
X100 = reconstruct(100)
X200 = reconstruct(200)

#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
#
def printFro(matrix,numberOfValues):
    print('Error for ',numberOfValues,' is : ',linalg.norm(matrix-X)/linalg.norm(X))

#print('Error for 10'+linalg.norm(X-X10)/linalg.norm(X))
printFro(X5,5)
printFro(X20,20)
printFro(X50,50)
printFro(X100,100)
printFro(X200,200)


#=========================================================================



# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X5,cmap = cm.Greys_r)
plt.title('Best rank' + str(5) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(X20,cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X50,cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X100,cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X200,cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k

plt.figure()
plt.plot(S[0:200], 'o')
plt.xlabel('rank')
plt.ylabel('singular values')


#=========================================================================

plt.show()
