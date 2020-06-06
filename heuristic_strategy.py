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
import os

class HeuristicSeachingStrategy(object):
    def __init__(
        self,
        strategy_name,
        iteration,
        refinement_per,
        if_check_output=False
    ):
        self.strategy_name = strategy_name
        self.iteration = iteration
        self.refinement_per_cnn = refinement_per[0]
        self.refinement_per_fc = refinement_per[1]
        self.if_check_output = if_check_output
        self.select_num_dic = []
        self.priority_all = []

    def refine_by_heuristic(
        self, nn_refiner, output_index, robustness=False, approach='BOTH'
    ):
        strategy_name = self.strategy_name
        if_check_output = self.if_check_output

        for i in range(nn_refiner.NN.num_of_hidden_layers):
            self.select_num_dic.append({})
            self.priority_all.append({})

        old_input_range = copy.deepcopy(nn_refiner.input_range_all[-1][output_index])
        print('Initial input range of the interested neuron is: ' + str(old_input_range))
        # nn_refiner.update_neuron_input_range(-1, 1, [8,16,28], outputFlag=1)
        print('After LP relaxation: ' + str(
            nn_refiner.update_neuron_input_range(-1, nn_refiner.NN.num_of_hidden_layers - 1, output_index, outputFlag=1,
                                                 presolve=1)))

        new_range_dict = {}
        time_dict = {}
        for i in range(self.iteration):
            start_time = time.time()
            print('Iteration ' + str(i) + ' begins.')
            for layer_index in range(1, nn_refiner.NN.num_of_hidden_layers-1):
                if nn_refiner.NN.layers[layer_index].type not in ('Activation', 'Fully_connected'):
                    continue
                print('Start to process layer ' + str(layer_index))
                if strategy_name == 'METRIC':
                    range_neuron = self.strategy_metric_ranking_layer(nn_refiner, layer_index)
                if strategy_name == 'RANDOM':
                    range_neuron = self.strategy_random_ranking_layer(nn_refiner, layer_index)

                if nn_refiner.NN.layers[layer_index].type == 'Fully_connected':
                    neuron_list = self.pop_neurons_by_priority(layer_index, self.refinement_per_fc)
                elif nn_refiner.NN.layers[layer_index].type == 'Activation':
                    neuron_list = self.pop_neurons_by_priority(layer_index, self.refinement_per_cnn)
                print(neuron_list)
                for neuron_index in neuron_list:
                    nn_refiner.refine_neuron(layer_index, neuron_index, approach=approach, presolve=0)
                    self.increase_selected_number(layer_index, neuron_index)
                # print('Integer variable of this layer: ' + str(sum(nn_refiner.refinement_degree_all[layer_index])))
            if if_check_output:
                new_range = nn_refiner.refine_neuron(nn_refiner.NN.num_of_hidden_layers - 1, output_index, approach='UPDATE_RANGE', outputFlag=1)
                end_time = time.time() - start_time
                new_range_dict['it' + str(i)] = new_range
                time_dict['it' + str(i)] = end_time
                print('Output range updates: ' + str(new_range))


        # self.select_num_dic = {}
        # self.priority_all = {}
        # for i in range(self.iteration):
        #     if strategy_name == 'RANDOM':
        #         self.strategy_purely_random(nn_refiner)
        #     if strategy_name == 'VOLUME_FIRST':
        #         self.strategy_volume_first(nn_refiner)
        #     if strategy_name == 'METRIC':
        #         self.strategy_metric_ranking(nn_refiner)
        #     layer_index, neuron_index = self.pop_neuron_by_priority()
        #     self.increase_selected_number_test(layer_index, neuron_index)
        #     print('test: update ' + str([layer_index, neuron_index]))
        #     nn_refiner.refine_neuron(layer_index, neuron_index, approach='BOTH')

        new_input_range = nn_refiner.refine_neuron(nn_refiner.NN.num_of_hidden_layers - 1, output_index, approach='UPDATE_RANGE')
        if robustness:
            for idx in range(10):
                if idx != output_index:
                    new_range = nn_refiner.refine_neuron(
                        nn_refiner.NN.num_of_hidden_layers - 1,
                        idx, approach='UPDATE_RANGE'
                    )
                    new_range_dict['label'+str(idx)] = new_range
        new_range_dict['final_range'] = new_input_range
        new_range_dict['time'] = time_dict
        print('Refinement finishes.')
        print('New range after refinement process is: ' + str(new_input_range))
        return old_input_range, new_range_dict

    def increase_selected_number_test(self, layer_index, neuron_index):
        if isinstance(neuron_index, list):
            key = (layer_index, neuron_index[0], neuron_index[1], neuron_index[2])
        else:
            key = (layer_index, neuron_index)

        if key in self.select_num_dic.keys():
            self.select_num_dic[key] += 1
        else:
            self.select_num_dic[key] = 1

    def increase_selected_number(self, layer_index, neuron_index):
        if isinstance(neuron_index, list):
            key = (neuron_index[0], neuron_index[1], neuron_index[2])
        else:
            key = (neuron_index)

        if key in self.select_num_dic[layer_index].keys():
            self.select_num_dic[layer_index][key] += 1
        else:
            self.select_num_dic[layer_index][key] = 1

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
        for layer_index in range(NN.num_of_hidden_layers-1):
            if len(NN.layers[layer_index].input_dim) == 3:
                for s in range(NN.layers[layer_index].input_dim[2]):
                    for i in range(NN.layers[layer_index].input_dim[0]):
                        for j in range(NN.layers[layer_index].input_dim[1]):
                            if NN.layers[layer_index].type == 'Activation':
                                self.priority_all[layer_index, i, j, s] = 0# np.random.rand()
                            else:
                                self.priority_all[layer_index, i, j, s] = 0
            else:
                for i in range(NN.layers[layer_index].input_dim[0]):
                    self.priority_all[layer_index, i] = np.random.rand()


    def strategy_volume_first(self, nn_refiner):
        NN = nn_refiner.NN
        input_range_all = nn_refiner.input_range_all
        for layer_index in range(NN.num_of_hidden_layers-1):
            if len(NN.layers[layer_index].input_dim) == 3:
                for s in range(NN.layers[layer_index].input_dim[2]):
                    for i in range(NN.layers[layer_index].input_dim[0]):
                        for j in range(NN.layers[layer_index].input_dim[1]):
                            if NN.layers[layer_index].type == 'Activation':
                                self.priority_all[layer_index, i, j, s] = 0 # input_range_all[layer_index][s][i][j][1] - input_range_all[layer_index][s][i][j][0]
                            else:
                                self.priority_all[layer_index, i, j, s] = 0
            else:
                for i in range(NN.layers[layer_index].input_dim[0]):
                    self.priority_all[layer_index, i] = input_range_all[layer_index][i][1] - input_range_all[layer_index][i][0]

    def strategy_metric_ranking(self, nn_refiner):
        NN = nn_refiner.NN
        input_range_all = nn_refiner.input_range_all
        refinement_degree_all = nn_refiner.refinement_degree_all
        for layer_index in range(NN.num_of_hidden_layers-1):
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
                                self.priority_all[layer_index, i, j, s] = 0 # (1 / (select_number+1)) * (volume / refinement) * output_weight
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

    def pop_neurons_by_priority(self, layer_index, percentage):
        priority_layer = sorted(self.priority_all[layer_index].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        update_neuron_number = round(len(priority_layer) * percentage)
        neurons = list(priority_layer[0:update_neuron_number])
        neuron_list = []
        for neuron in neurons:
            if isinstance(neuron[0], tuple):
                neuron_list.append(list(neuron[0]))
            elif isinstance(neuron[0], int):
                neuron_list.append(neuron[0])
        return neuron_list

    def strategy_random_ranking_layer(self, nn_refiner, layer_index):
        NN = nn_refiner.NN
        input_range_all = nn_refiner.input_range_all
        refinement_degree_all = nn_refiner.refinement_degree_all
        if len(NN.layers[layer_index].input_dim) == 3:
            for s in range(NN.layers[layer_index].input_dim[2]):
                for i in range(NN.layers[layer_index].input_dim[0]):
                    for j in range(NN.layers[layer_index].input_dim[1]):
                        if NN.layers[layer_index].type == 'Activation':
                            self.priority_all[layer_index][i, j, s] = np.random.rand()
                        else:
                            self.priority_all[layer_index][i, j, s] = np.random.rand()
        else:
            for i in range(NN.layers[layer_index].input_dim[0]):
                self.priority_all[layer_index][i] = np.random.rand()

    def strategy_metric_ranking_layer(self, nn_refiner, layer_index):
        NN = nn_refiner.NN
        input_range_all = nn_refiner.input_range_all
        refinement_degree_all = nn_refiner.refinement_degree_all
        if len(NN.layers[layer_index].input_dim) == 3:
            for s in range(NN.layers[layer_index].input_dim[2]):
                for i in range(NN.layers[layer_index].input_dim[0]):
                    for j in range(NN.layers[layer_index].input_dim[1]):
                        if NN.layers[layer_index].type == 'Activation':
                            # for relu neuron with zero output, no need for refinement
                            if NN.layers[layer_index].activation == 'ReLU' and input_range_all[layer_index][s][i][j][1] <= 0:
                                coef = 0
                            else:
                                coef = 1
                            volume = input_range_all[layer_index][s][i][j][1] - input_range_all[layer_index][s][i][j][0]
                            refinement = refinement_degree_all[layer_index][s][i][j]
                            if (i, j, s) in self.select_num_dic[layer_index].keys():
                                select_number = self.select_num_dic[layer_index][i, j, s]
                            else:
                                self.select_num_dic[layer_index][i, j, s] = 0
                                select_number = 0
                            output_weight = 1
                            self.priority_all[layer_index][
                                i, j, s] = (1 / (select_number+1)) * (volume / refinement) * output_weight * coef
                        else:
                            self.priority_all[layer_index][i, j, s] = 0
        else:
            for i in range(NN.layers[layer_index].input_dim[0]):
                # for relu neuron with zero output, no need for refinement
                if NN.layers[layer_index].activation == 'ReLU' and input_range_all[layer_index][i][1] <= 0:
                    coef = 0
                else:
                    coef = 1
                volume = input_range_all[layer_index][i][1] - input_range_all[layer_index][i][0]
                refinement = refinement_degree_all[layer_index][i]
                if i in self.select_num_dic[layer_index].keys():
                    select_number = self.select_num_dic[layer_index][i]
                else:
                    self.select_num_dic[layer_index][i] = 0
                    select_number = 0
                output_weight = np.linalg.norm(NN.layers[layer_index].weight[i, :])
                self.priority_all[layer_index][i] = (1 / (select_number + 1)) * (volume / refinement) * output_weight * coef
