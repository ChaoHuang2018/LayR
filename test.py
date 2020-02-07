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
NN = nn_controller_details('model_CIFAR_CNN_Medium', keras=True)

w = [[[ 0.09480116, -0.04812197, -0.09353992],
  [-0.02359495, -0.01279663,  0.03123438],
  [-0.0504902,  -0.06165261, -0.05582498],
  [ 0.0008492,   0.09251167,  0.01976186]],
 [[ 0.00713766, -0.2349602,  -0.27480617],
  [-0.01125166, -0.23150535, -0.05289953],
  [ 0.02868317, -0.00037204,  0.19999257],
  [-0.03179998,  0.13388725,  0.24728076]],
 [[ 0.03181054, -0.24173619, -0.30916935],
  [ 0.04295825, -0.23104002,  0.01876903],
  [ 0.21704498,  0.06316286,  0.23970185],
  [-0.16174626,  0.21232843,  0.09777785]],
 [[ 0.05890926, -0.15512267, -0.17352128],
  [ 0.00675091,  0.07449771,  0.02546142],
  [ 0.04035579,  0.20861667, 0.09841397],
  [ 0.01244394,  0.06305319,  0.0609398]]]

b = -0.025138568

input_range = []

tests = get_tests('cifar10')
for test in tests:
    break
image= np.float64(test[1:len(test)])/np.float64(255)
data = image.reshape(32, 32, 3)
data0 = image.reshape(1, 32, 32, 3)
for k in range(3):
    input_range_channel = []
    for i in range(4):
        input_range_row = []
        for j in range(4):
            # input_range_row.append([data[i][j][k], data[i][j][k]])
            input_range_row.append([max(0, data[i][j][k] - eps), min(1, data[i][j][k] + eps)])
            # input_range_row.append(
            #     [(max(0, data[i][j][k] - eps) - NN.mean) / NN.std, (min(1, data[i][j][k] + eps)- NN.mean) / NN.std])
        input_range_channel.append(input_range_row)
    input_range.append(input_range_channel)

min_test = 0
max_test = 0
k = 0
for s in range(3):
    for p in range(4):
        for q in range(4):
            if w[p][q][s] >= 0:
                min_test += input_range[s][p][q][0] * w[p][q][s]
                max_test += input_range[s][p][q][1] * w[p][q][s]
            else:
                min_test += input_range[s][p][q][1] * w[p][q][s]
                max_test += input_range[s][p][q][0] * w[p][q][s]
min_test = min_test + b
max_test = max_test + b
print(min_test)
print(max_test)
