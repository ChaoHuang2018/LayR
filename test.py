#from .NN_Tracking.code.neuralnetwork import NN
#import outputCNN as oc
import time
import keras

import numpy as np
import sympy as sp
import cvxpy as cp

from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot
from gurobipy import *
from analyzeCNN import output_range_MILP_CNN
#import controller_approximation_lib as cal

#import tensorflow as tf



# test new approach for estimating sigmoid network's output range
eps = 0.1
NN = nn_controller_details('model_CNN',keras=True)
# the data, split between train and test sets
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
data = x_test[0]
input_range = []
for i in range(NN.layers[0].input_dim[0]):
    input_range_row = []
    for j in range(NN.layers[0].input_dim[1]):
        input_range_row.append([data[i][j] - eps, data[i][j] + eps])
    input_range.append(input_range_row)
print(np.array(input_range).shape)
output_l, output_u = output_range_MILP_CNN(NN, np.array(input_range), 0)


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



