#from .NN_Tracking.code.neuralnetwork import NN
#import outputCNN as oc
import time
import keras
import os
import csv
import copy
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import numpy as np
import cvxpy as cp

from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot
from gurobipy import *
from reachnn import ReachNN
from nn_range_refiner import NNRangeRefiner

#import tensorflow as tf
def get_tests(dataset):
    csvfile = open('data/{}_test.csv'.format(dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')
    return tests


# test new approach for estimating sigmoid network's output range
eps = 0.01
NN = nn_controller_details('model_MNIST_CNN_5L_sigmoid', keras=True)
print(NN.mean)
print(NN.std)

# Normalize
mean = NN.mean
std = NN.std
input_range = []
tests = get_tests('mnist')
for test in tests:
    break
image= np.float64(test[1:len(test)])/np.float64(255)
data = image.reshape(28, 28, 1)
input_dim = NN.layers[0].input_dim
print(input_dim)
for k in range(input_dim[2]):
    input_range_channel = []
    for i in range(input_dim[0]):
        input_range_row = []
        for j in range(input_dim[1]):
            input_range_row.append([max(0, data[i][j][k] - eps), min(1, data[i][j][k] + eps)])
            # input_range_row.append(
            #     [(max(0, data[i][j][k] - eps) - NN.mean) / NN.std, (min(1, data[i][j][k] + eps)- NN.mean) / NN.std])
        input_range_channel.append(input_range_row)
    input_range.append(input_range_channel)
print(np.array(input_range).shape)
start_time = time.time()
nn_analyzer = ReachNN(NN, np.array(input_range), 3, 'ERAN', global_robustness_type='L-INFINITY', perturbation_bound=0.01)
new_output_range = nn_analyzer.output_range_analysis('METRIC', 9, iteration=5, per=[0.005, 0.2])
# nn_refiner = NNRangeRefiner(NN, np.array(input_range), 'ERAN', traceback=2)
# test_range = nn_refiner.update_neuron_input_range(0, 6, 9)
print("--- %s seconds ---" % (time.time() - start_time))




