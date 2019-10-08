from gurobipy import Model, GRB
from MILP import MILP

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
        NN2=None,
        network_input_box1=None,
        network_input_box2=None,
        perturbation_bound=None,
        output_index=0,
    ):
        # neural networks
        self.NN1 = NN1
        self.NN2 = NN2

        # input space
        self.network_input_box1 = network_input_box1
        self.network_input_box2 = network_input_box2
        self.perturbation_bound = perturbation_bound

        # output index
        self.output_index = output_index

        # input dim
        self.input_dim = self.NN1.layers[0].input_dim

    def global_robustness_analysis(self):
        # handle perturbation
        if self.network_input_box2 is None:
            self.network_input_box2 = self.network_input_box1.copy()
            perturbation_set = np.ones(self.network_input_box1.shape)

            # perturbed input set
            self.network_input_box2[:, :, :, 0] -= self.perturbation_bound
            self.network_input_box2[:, :, :, 1] += self.perturbation_bound

            # perturbation set
            perturbation_set[:, :, :, 0] *= -self.perturbation_bound
            perturbation_set[:, :, :, 1] *= self.perturbation_bound

        # naive range for the original input set
        intput_range_all_NN1 = construct_naive_input_range(
            self.NN1, self.network_input_box1, self.output_index
        )
        # naive range for the perturbed input set
        input_range_all_NN2 = construct_naive_input_range(
            self.NN1, self.network_input_box2, self.output_index
        )

        print('-------MILP global robustness analysis begins.----------')

        # refinement
        refinement_degrees_NN1 = self.initialize_refinement_degree(self.NN1)
        refinement_degrees_NN2 = self.initialize_refinement_degree(self.NN1)

        # MILP formulation
        model = Model('Global_robustness_certification')
        MILP_NN1 = MILP(
            model,
            self.NN1,
            refinement_degrees_NN1,
            self.NN1.num_of_hidden_layers,
            self.output_index
        )
        MILP_NN2 = MILP(
            model,
            self.NN1,
            refinement_degrees_NN2,
            self.NN1.num_of_hidden_layers,
            self.output_index
        )


    def initialize_refinement_degree(self, NN):
        refinement_degrees = []
        for idxLayer in range(NN.num_of_hidden_layers):
            output_dim = NN.layers[idxLayer].output_dim
            if len(output_dim) == 3:
                refinement_degrees.append(np.zeros(output_dim))
            elif len(output_dim) == 1:
                refinement_degrees.append(np.zeros(output_dim))
        return refinement_degrees
