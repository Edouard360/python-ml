from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import inv
from tools import poly_exp
from tools import MSE
from tools import regression

# Load the data
data = loadtxt('data/data_train.csv', delimiter=',')
data_test = loadtxt('data/data_test.csv', delimiter=',')

# Prepare the data
X = data[:,0:-1]
y = data[:,-1]
y[y >= 8] = 0       # <-- Task 7 data cleaning

# Prepare the test
X_test = data_test[:,0:-1]
y_test = data_test[:,-1]

# Inspect the data - Uncomment to see the plots

# figure()
# hist(X[:,0], 10)
# show()
# figure()
# hist(X[:,1], 10)
# show()
# hist(X[:,2], 10)
# show()

# <TASK 1>

# figure()
# plot(X[:,1],X[:,2], 'o')
# xlabel('x1')
# ylabel('x2')
# show()

# <TASK 2>

# figure()
# plot(X[:,0],y, 'o')
# show()

# Standardization
X_mean = mean(X, axis=0) # Vector of 3 components
X_std = std(X, axis=0) # Vector of 3 components

X = (X - X_mean)/X_std
X_test = (X_test - X_mean)/X_std

Z = poly_exp(X,2)
Z = column_stack([ones(len(Z)), Z])

Z_test = poly_exp(X_test,2)
Z_test = column_stack([ones(len(Z_test)), Z_test])

w = dot(dot(inv(dot(transpose(Z),Z)),transpose(Z)),y)
y_pred = dot(Z_test,w)

mse1 = MSE(y_test,y_pred)
mse2 = MSE(y_test,mean(y,axis=0)*ones(len(y_test)))

print("B Coefficent", w)
print("MSE on base data", mse1)
print("MSE baseline", mse2)

# Seeing the overfitting effect with high degree polynomial regression

dim = 8

MSE_train,MSE_test = regression(X,y,X_test,y_test,dim)

figure()
plot(range(1,dim+1),MSE_train, 'r-', label="Training data")
plot(range(1,dim+1),MSE_test, 'b-', label="Test data")
xlabel('degree of polynomial')
ylabel('MSE')
legend()
show()