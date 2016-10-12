from numpy import *
from random import *

# implements the unit step function
unit_step = lambda x: 0 if x < 0 else 1

def updateW(w,x,l,j,eta):
    C_x = unit_step(dot(transpose(w),x))
    y=1 if (l==j) else 0
    return w + eta*(y-C_x)*x

def training_adaline(w,n,trainingImages,trainingLabels,j,eta):

  # ====================== ADD YOUR CODE HERE =============================
  # Implement the training step of the adaline algorithm.

  for i in range(n):
      w = updateW(w,trainingImages[i,:],trainingLabels[i],j,eta)

  # return the weight vector w
  return w


def classify_adaline(testImages,w):
  # ====================== ADD YOUR CODE HERE =============================
  # Implement the classification step of the adaline algorithm.

  # return the predicted class of the test image
  return dot(w,transpose(testImages)).argmax()
