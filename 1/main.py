from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import inv
import pdb

# Load the data

data = loadtxt('data/data_train.csv', delimiter=',')
data_test = loadtxt('data/data_test.csv', delimiter=',')

# Prepare the data

X = data[:,0:-1]
y = data[:,-1]

X_test = data_test[:,0:-1]
y_test = data_test[:,-1]

# Inspect the data

figure()
hist(X[:,1], 10)

# <TASK 1>

figure()
plot(X[:,1],X[:,2], 'o')
xlabel('x1')
ylabel('x2')

# <TASK 2>

figure()
plot(X[:,0],y, 'o')

show()

# Standardization
X_mean = mean(X, axis=0) # Vector of 3 components
X_std = std(X, axis=0) # Vector of 3 components


X = (X - X_mean)/X_std
X_test = (X_test - X_mean)/X_std



# <TASK 2>

# Feature creation

from tools import poly_exp
Z = poly_exp(X,2)
Z = column_stack([ones(len(Z)), Z])

Z_test = poly_exp(X_test,2)
Z_test = column_stack([ones(len(Z_test)), Z_test])

# Building a model

# <TASK 3>

# Evaluation 

w = dot(dot(inv(dot(transpose(Z),Z)),transpose(Z)),y)
y_pred = dot(Z_test,w)

# <TASK 4>

from tools import MSE

# <TASK 5>

mse = MSE(y_test,y_pred)
#y_pred has more values than y_test
#y_test doesn't have enough values.
s1 = sum(mse)
pdb.set_trace()
mse = MSE(y_test,mean(y,axis=0)*ones(len(y))) #This formula is good
s2 = sum(mse)

# <TASK 6>
# <TASK 7>

# <TASK 8: You will need to make changes from '# Feature creation'
#          To get the exact results, you will need to reverse the second part of Task 7 (your own modifications)>
