#from .NN_Tracking.code.neuralnetwork import NN
#import outputCNN as oc
import numpy as np
import sympy as sp
import time
from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot
import cvxpy as cp
from gurobipy import *
#import controller_approximation_lib as cal

#import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K



# test new approach for estimating sigmoid network's output range
eps = 0.1
NN = nn_controller_details('model_CNN',keras=True)
data = keras.datasets.fashion_mnist[0]
input_range = []
for i in range(NN.layers[0].input_dim[0]):
    input_range_row = []
    for j in range(NN.layers[0].input_dim[1]):
        input_range_row.append([data[0]-eps, data[1]+eps])
    input_range.append(input_range_row)
output_l, output_u = bp.output_range_MILP(NN, input_range, 0)


# test cvxpy
##def c(x,y):
##    return [x+y<=1]
##
##x = cp.Variable()
##y = cp.Variable()
##
##
##objective = cp.Maximize(2*x+y)
##constraints = [0 <= x, 0 <= y]
##con = c(x,y)
##constraints += con
##
##prob = cp.Problem(objective, constraints)
##
##print("Optimal value", prob.solve())
##print("Optimal var")
##print(x.value)



