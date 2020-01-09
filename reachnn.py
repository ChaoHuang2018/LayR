import gurobipy as gp
from gurobipy import GRB
from nn_range_refiner import NNRangeRefiner
from heuristic_strategy import refine_by_heuristic

import numpy as np
import sympy as sp
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
        network_input_box1,
        traceback1,
        NN2=None,
        network_input_box2=None,
        traceback2=None,
        perturbation_bound=None
    ):
        # neural networks
        self.NN1 = NN1
        self.NN2 = NN2

        # input space
        self.network_input_box1 = network_input_box1
        self.network_input_box2 = network_input_box2
        self.perturbation_bound = perturbation_bound

        # traceback
        self.traceback1 = traceback1
        self.traceback2 = traceback2

    def output_range_analysis(self, strategy_name, output_index, number=40):
        nn_refiner = NNRangeRefiner(self.NN1, self.network_input_box1, self.traceback1)
        #new_output_range = refine_by_heuristic(nn_refiner, strategy_name, output_index, number, check_output=False)
        new_output_range, min_input = nn_refiner.update_neuron_input_range(self.NN1.num_of_hidden_layers - 1, output_index)
        return new_output_range, min_input
