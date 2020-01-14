import gurobipy as gp
from gurobipy import GRB
from nn_range_refiner import NNRangeRefiner
from nn_activation import Activation

import numpy as np
import tensorflow as tf
import itertools
import math
import random
import time
import copy


def refine_by_heuristic(nn_refiner, strategy_name, output_index, number, check_output=False):
    old_input_range = copy.deepcopy(nn_refiner.input_range_all[-1][output_index])
    print('Initial input range of the interested neuron is: ' + str(old_input_range))
    print('After LP relaxation: ' + str(nn_refiner.update_neuron_input_range(nn_refiner.NN.num_of_hidden_layers - 1, output_index)))
    print('-------' + strategy_name + ' refinement begins.----------')
    for i in range(number):
        print('Start to process neuron ' + str(i))
        if strategy_name == 'RANDOM':
            layer_index, neuron_index = strategy_purely_random(nn_refiner)
        if strategy_name == 'VOLUME_FIRST':
            layer_index, neuron_index = strategy_volume_first(nn_refiner)
        nn_refiner.refine_neuron(layer_index, neuron_index)
        if check_output == True:
            new_range = nn_refiner.update_neuron_input_range(nn_refiner.NN.num_of_hidden_layers - 1, output_index)
            print('Output range updates: ' + str(new_range))
    new_input_range = nn_refiner.update_neuron_input_range(nn_refiner.NN.num_of_hidden_layers - 1, output_index)
    print('Refinement finishes.')
    print('New range after refinement process is: ' + str(new_input_range))
    return new_input_range

def strategy_purely_random(nn_refiner):
    NN = nn_refiner.NN
    # randomly choose a neuron
    if_refinable = False
    while if_refinable == False:
        layer_index = np.random.randint(max(NN.num_of_hidden_layers-nn_refiner.traceback, 0), NN.num_of_hidden_layers)
        if NN.layers[layer_index].type == 'Activation' or NN.layers[layer_index].type == 'Fully_connected':
            if_refinable = True
    if len(NN.layers[layer_index].input_dim) == 1:
        neuron_index = round(np.random.rand() * (NN.layers[layer_index].input_dim[0] - 1))
    if len(NN.layers[layer_index].input_dim) == 3:
        neuron_index_0 = np.random.randint(0, NN.layers[layer_index].input_dim[0])
        neuron_index_1 = np.random.randint(0, NN.layers[layer_index].input_dim[1])
        neuron_index_2 = np.random.randint(0, NN.layers[layer_index].input_dim[2])
        neuron_index = [neuron_index_0, neuron_index_1, neuron_index_2]
    return layer_index, neuron_index

def strategy_volume_first(nn_refiner):
    NN = nn_refiner.NN
    input_range_all = nn_refiner.input_range_all
    # construct a dictionary to store the volume of each neuron's input
    volume_all = {}
    for layer_index in range(NN.num_of_hidden_layers):
        if len(NN.layers[layer_index].input_dim) == 3:
            for s in range(NN.layers[layer_index].input_dim[2]):
                for i in range(NN.layers[layer_index].input_dim[0]):
                    for j in range(NN.layers[layer_index].input_dim[1]):
                        if NN.layers[layer_index].type == 'Activation':
                            volume_all[layer_index, i, j, s] = input_range_all[layer_index][s][i][j][1] - \
                                                               input_range_all[layer_index][s][i][j][0]
                        else:
                            volume_all[layer_index, i, j, s] = 0
        else:
            for i in range(NN.layers[layer_index].input_dim[0]):
                volume_all[layer_index, i] = input_range_all[layer_index][i][1] - input_range_all[layer_index][i][0]
    sorted_volume_all = sorted(volume_all.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    neuron_layer_index = list(sorted_volume_all[0])[0]
    layer_index = neuron_layer_index[0]
    neuron_index = list(neuron_layer_index[1:])
    if len(neuron_index) == 1:
        neuron_index = neuron_index[0]
    return layer_index, neuron_index