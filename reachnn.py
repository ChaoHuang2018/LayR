import gurobipy as gp
from gurobipy import GRB
from nn_range_refiner import NNRangeRefiner
from nn_robustness_evaluator import NNRobusnessEvaluator
from heuristic_strategy import HeuristicSeachingStrategy

import numpy as np
import tensorflow as tf
import itertools
import math
import random
import time
import copy

class ReachNN(object):
    """
    NN properties
     NN.type = {'Convolutional', 'Fully_connected'}
     NN.layers: list of layers

    Layer properties:
     layer.type = {'Convolutional', 'Pooling',
                   'Fully_connected', 'Flatten', 'Activation'}
     layer.weight = weight if type = 'Fully_connected', None otherwise
     layer.bias = bias if type = 'Fully_connected' or 'Convolutional'
     layer.kernal: only for type = 'Convolutional'
     layer.stride: only for type = 'Convolutional'
     layer.activation = {'ReLU', 'tanh', 'sigmoid'} if type = 'Fully_connected',
                         'Convolutional' if type = 'Convolutional',
                        {'max', 'average'} if type = 'Pooling'
     layer.filter_size: only for type = 'Pooling'
     layer.input_dim = [m, n], n==1 for fully-connected layer
     layer.output_dim = [m, n]

    Keep the original properties:
     num_of_hidden_layers: integer
    """
    def __init__(
        self,
        NN1,
        type,
        data,
        network_input_box,
        traceback,
        initialize_approach,
        global_robustness_type=None,
        perturbation_bound=None,
        NN2=None,
        approach='BOTH'
    ):
        # neural networks
        self.NN1 = NN1
        self.NN2 = NN2

        self.type = type

        # testing input
        self.data = data

        # input space
        self.network_input_box = network_input_box
        self.perturbation_bound = perturbation_bound

        self.initialize_approach = initialize_approach

        self.global_robustness_type = global_robustness_type

        self.perturbation_bound = perturbation_bound

        # traceback
        self.traceback = traceback

        # refienment
        self.approach = approach

    def output_range_analysis(self, strategy_name, output_index, iteration, per, is_test=False, robustness=False):
        if is_test:
            low = 100
            upp = -100
            for i in range(1000):
                random_per = np.random.rand(self.data.shape[0], self.data.shape[1], self.data.shape[2],
                                            self.data.shape[3]) * self.perturbation_bound
                output = self.NN1.keras_model(random_per + self.data)[0][output_index]
                if output > upp:
                    upp = output
                if output < low:
                    low = output
            print('Bound by random sampling is: ' + str([low, upp]))
        nn_refiner = NNRangeRefiner(self.NN1, self.type, self.network_input_box, self.initialize_approach, traceback=self.traceback)
        heuristic_search = HeuristicSeachingStrategy(strategy_name, iteration, per, if_check_output=True)
        old_output_range, new_output_range = heuristic_search.refine_by_heuristic(nn_refiner, output_index, approach=self.approach)
        # new_output_range = nn_refiner.update_neuron_input_range(self.NN1.num_of_hidden_layers - 1, output_index)
        return old_output_range, new_output_range

    def global_robustness_analysis(self, strategy_name, output_index, number=40):
        nn_robustness = NNRobusnessEvaluator(self.NN1, self.network_input_box, self.initialize_approach,
                                             self.perturbation_bound, type='L-INFINITY', traceback=self.traceback)
        nn_robustness.evaluate_global_robustness(output_index)
        heuristic_search = HeuristicSeachingStrategy(strategy_name, number, if_check_output=False)
        heuristic_search.refine_by_heuristic(nn_robustness, output_index)
        distance_range = nn_robustness.evaluate_global_robustness(output_index)
        return distance_range
