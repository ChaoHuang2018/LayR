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


class NNRobusnessEvaluator(NNRangeRefiner):

    def __init__(
        self,
        NN,
        network_input_box,
        initialize_approach,
        perturbation,
        type='L-INFINITY',
        traceback=None
    ):
        NNRangeRefiner.__init__(self, NN, network_input_box, initialize_approach, traceback)
        self.gre_model = None
        self.all_variables2 = {}

    def evaluate_global_robustness(self, output_index):
        self.model = gp.Model('global_robustness_update')
        v_name = 'NN'
        v2_name = 'NN2'

        print('Update global robustness result of the ' + str(output_index) + '-th output.')
        # declare variables for two inputs NN and NN2. For each input variables, we need construct the LP/MILP relaxation
        # seperately with the same setting
        all_variables_NN = self._declare_variables(model, v_name, self.NN.num_of_hidden_layers - 1)
        all_variables_NN2 = self._declare_variables(model, v2_name, self.NN.num_of_hidden_layers - 1)

        # add perturbation constraint
        self._add_LInfinity_perturbedt_constraints(model, all_variables_NN, all_variables_NN2)

    # add perturbed input constraints
    def _add_LInfinity_perturbedt_constraints(self, model, all_variables_NN1, all_variables_NN2, v_name, v2_name):
        NN = self.NN
        network_in_NN1 = all_variables_NN1[0]
        network_in_NN2 = all_variables_NN2[0]
        if len(NN.layers[0].input_dim) == 3:
            for s in range(NN.layers[0].input_dim[2]):
                for i in range(NN.layers[0].input_dim[0]):
                    for j in range(NN.layers[0].input_dim[1]):
                        model.addConstr(network_in_NN1[s][i, j] - network_in_NN2[s][i, j] <= perturbation)
                        model.addConstr(network_in_NN1[s][i, j] - network_in_NN2[s][i, j] >= -perturbation)
        if len(NN.layers[0].input_dim) == 1:
            for i in range(NN.layers[0].output_dim[0]):
                model.addConstr(network_in_NN1[i] - network_in_NN2[i] <= perturbation)
                model.addConstr(network_in_NN1[i] - network_in_NN2[i] >= -perturbation)
