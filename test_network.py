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

from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot

#import tensorflow as tf
def get_tests(dataset):
    csvfile = open('data/{}_test.csv'.format(dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')
    return tests


# test new approach for estimating sigmoid network's output range
NN = nn_controller_details('model_MNIST_CNN_Small', keras=True)

# Normalize
mean = NN.mean
std = NN.std
input_range = []
# for mnist dataset
tests = get_tests('mnist')
i = 0
for test in tests:
    image= np.float64(test[1:len(test)])/np.float64(255)
    data = image.reshape(28, 28, 1)
    data0 = image.reshape(1, 28, 28, 1)
    print(NN.keras_model(data0)[0])
    i = i + 1
    if i == 10:
        break



