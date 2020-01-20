#from .NN_Tracking.code.neuralnetwork import NN
#import outputCNN as oc
import time
import keras
import os
import csv

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import numpy as np
import sympy as sp
import cvxpy as cp

from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot
from gurobipy import *
# from phased_MILP import global_robustness_analysis, function_distance_analysis, output_range_analysis
from analyzeCNN_gurobi import output_range_analysis
#import controller_approximation_lib as cal

#import tensorflow as tf
def get_tests(dataset):
    csvfile = open('data/{}_test.csv'.format(dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')
    return tests



# test new approach for estimating sigmoid network's output range
eps = 0.01
perturbation = 0.1
NN = nn_controller_details('model_CNNA', keras=True)
# NN1 = nn_controller_details('model_CNNA', keras=True)
# the data, split between train and test sets
# fashion_mnist = keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# data = x_test[0]
# data = data.reshape(data.shape[0], data.shape[1], 1)
input_range = []
tests = get_tests('mnist')
for test in tests:
    break
image= np.float64(test[1:len(test)])/np.float64(255)
data = image.reshape(28, 28, 1)
input_dim = NN.layers[0].input_dim
print(input_dim)
for i in range(input_dim[0]):
    input_range_row = []
    for j in range(input_dim[1]):
        input_range_channel = []
        for k in range(input_dim[2]):
            input_range_channel.append([data[i][j][k] - eps, data[i][j][k] + eps])
        input_range_row.append(input_range_channel)
    input_range.append(input_range_row)
print(np.array(input_range).shape)
# print("actual output of NN: {}".format(NN.controller(data)))
start_time = time.time()
output_l, output_u = output_range_analysis(NN, np.array(input_range), 9)
# output_l, output_u = global_robustness_analysis(NN, np.array(input_range), perturbation, 9)
print("lower bound: {}; upper bound: {}".format(output_l, output_u))
# print("actual output of NN: {}".format(NN.keras_model(data.reshape(1, data.shape[0], data.shape[1], 1))))
print("--- %s seconds ---" % (time.time() - start_time))
# print("actual output: {}".format(NN.keras_model_pre_softmax(data.reshape(1, data.shape[0], data.shape[1]))))

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



