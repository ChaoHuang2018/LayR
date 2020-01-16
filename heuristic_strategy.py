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

class HeuristicSeachingStrategy(object):
    def __init__(
        self,
        strategy_name,
        iteration_number,
        if_check_output=False
    ):
        self.strategy_name = strategy_name
        self.iteration_number = iteration_number
        self.if_check_output = if_check_output
        self.select_num_dic = {}
        self.priority_all = {}

    def refine_by_heuristic(self, nn_refiner, output_index):
        strategy_name = self.strategy_name
        number = self.iteration_number
        if_check_output = self.if_check_output

        old_input_range = copy.deepcopy(nn_refiner.input_range_all[-1][output_index])
        print('Initial input range of the interested neuron is: ' + str(old_input_range))
        print('After LP relaxation: ' + str(nn_refiner.refine_neuron(nn_refiner.NN.num_of_hidden_layers - 1, output_index, approach='UPDATE_RANGE')))
        print('-------' + strategy_name + ' refinement begins.----------')
        for i in range(number):
            print('Start to process neuron ' + str(i))
            if strategy_name == 'RANDOM':
                self.strategy_purely_random(nn_refiner)
            if strategy_name == 'VOLUME_FIRST':
                self.strategy_volume_first(nn_refiner)
            if strategy_name == 'METRIC':
                self.strategy_metric_ranking(nn_refiner)
            layer_index, neuron_index = self.pop_neuron_by_priority()
            self.increase_selected_number(layer_index, neuron_index)
            nn_refiner.refine_neuron(layer_index, neuron_index, approach='BOTH')
            if if_check_output:
                new_range = nn_refiner.refine_neuron(nn_refiner.NN.num_of_hidden_layers - 1, output_index, approach='UPDATE_RANGE')
                print('Output range updates: ' + str(new_range))
        new_input_range = nn_refiner.refine_neuron(nn_refiner.NN.num_of_hidden_layers - 1, output_index, approach='UPDATE_RANGE')
        print('Refinement finishes.')
        print('New range after refinement process is: ' + str(new_input_range))
        return new_input_range

    def increase_selected_number(self, layer_index, neuron_index):
        if isinstance(neuron_index, list):
            key = (layer_index, neuron_index[0], neuron_index[1], neuron_index[2])
        else:
            key = (layer_index, neuron_index)

        if key in self.select_num_dic.keys():
            self.select_num_dic[key] += 1
        else:
            self.select_num_dic[key] = 1

    def pop_neuron_by_priority(self):
        priority_all = sorted(self.priority_all.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        neuron_layer_index = list(priority_all[0])[0]
        layer_index = neuron_layer_index[0]
        neuron_index = list(neuron_layer_index[1:])
        if len(neuron_index) == 1:
            neuron_index = neuron_index[0]
        return layer_index, neuron_index

    def strategy_purely_random(self, nn_refiner):
        NN = nn_refiner.NN
        # randomly set priorities for all the neuron
        for layer_index in range(NN.num_of_hidden_layers):
            if len(NN.layers[layer_index].input_dim) == 3:
                for s in range(NN.layers[layer_index].input_dim[2]):
                    for i in range(NN.layers[layer_index].input_dim[0]):
                        for j in range(NN.layers[layer_index].input_dim[1]):
                            if NN.layers[layer_index].type == 'Activation':
                                self.priority_all[layer_index, i, j, s] = np.random.rand()
                            else:
                                self.priority_all[layer_index, i, j, s] = 0
            else:
                for i in range(NN.layers[layer_index].input_dim[0]):
                    self.priority_all[layer_index, i] = np.random.rand()


    def strategy_volume_first(self, nn_refiner):
        NN = nn_refiner.NN
        input_range_all = nn_refiner.input_range_all
        for layer_index in range(NN.num_of_hidden_layers):
            if len(NN.layers[layer_index].input_dim) == 3:
                for s in range(NN.layers[layer_index].input_dim[2]):
                    for i in range(NN.layers[layer_index].input_dim[0]):
                        for j in range(NN.layers[layer_index].input_dim[1]):
                            if NN.layers[layer_index].type == 'Activation':
                                self.priority_all[layer_index, i, j, s] = input_range_all[layer_index][s][i][j][1] - \
                                                                   input_range_all[layer_index][s][i][j][0]
                            else:
                                self.priority_all[layer_index, i, j, s] = 0
            else:
                for i in range(NN.layers[layer_index].input_dim[0]):
                    self.priority_all[layer_index, i] = input_range_all[layer_index][i][1] - input_range_all[layer_index][i][0]

    def strategy_metric_ranking(self, nn_refiner):
        NN = nn_refiner.NN
        input_range_all = nn_refiner.input_range_all
        refinement_degree_all = nn_refiner.refinement_degree_all
        for layer_index in range(NN.num_of_hidden_layers):
            if len(NN.layers[layer_index].input_dim) == 3:
                for s in range(NN.layers[layer_index].input_dim[2]):
                    for i in range(NN.layers[layer_index].input_dim[0]):
                        for j in range(NN.layers[layer_index].input_dim[1]):
                            if NN.layers[layer_index].type == 'Activation':
                                volume = input_range_all[layer_index][s][i][j][1] - input_range_all[layer_index][s][i][j][0]
                                refinement = refinement_degree_all[layer_index][s][i][j]
                                if (layer_index, i, j, s) in self.select_num_dic.keys():
                                    select_number = self.select_num_dic[layer_index, i, j, s]
                                else:
                                    self.select_num_dic[(layer_index, i, j, s)] = 0
                                    select_number = 0
                                output_weight = 1
                                self.priority_all[layer_index, i, j, s] = (1 / (select_number+1)) * (
                                            volume / refinement) * output_weight
                            else:
                                self.priority_all[layer_index, i, j, s] = 0
            else:
                for i in range(NN.layers[layer_index].input_dim[0]):
                    volume = input_range_all[layer_index][i][1] - input_range_all[layer_index][i][0]
                    refinement = refinement_degree_all[layer_index][i]
                    if (layer_index, i) in self.select_num_dic.keys():
                        select_number = self.select_num_dic[layer_index, i]
                    else:
                        self.select_num_dic[(layer_index, i)] = 0
                        select_number = 0
                    output_weight = np.linalg.norm(NN.layers[layer_index].weight[i, :])
                    self.priority_all[layer_index, i] = (1 / (select_number+1)) * (volume / refinement) * output_weight