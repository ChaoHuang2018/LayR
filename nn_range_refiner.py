import gurobipy as gp
from gurobipy import GRB
from nn_range import NNRange
from nn_activation import Activation

import numpy as np
import sympy as sp
import tensorflow as tf
import itertools
import math
import random
import time
import copy


class NNRangeRefiner(NNRange):

    def __init__(
            self,
            NN,
            network_input_box,
            traceback=None
    ):
        NNRange.__init__(self, NN, network_input_box)
        self.traceback = traceback
        self.model = gp.Model('Input_range_update')
        self.all_variables = {}

    def refine_neuron(self, layer_index, neuron_index):
        print('Start to refine neuron with index:')
        print('layer_index: ' + str(layer_index) + ', neuron_index: ' + str(neuron_index))
        self.add_neuron_slack_integer(layer_index, neuron_index)
        new_range = self.update_neuron_input_range(layer_index, neuron_index)
        return new_range

    def add_neuron_slack_integer(self, layer_index, neuron_index):
        print('Add integer variable of neuron :' + 'layer_index: ' + str(layer_index) + ', neuron_index: ' + str(
            neuron_index))
        # need to carefully handle ReLU
        if type(neuron_index) == list:
            if self.NN.layers[layer_index].activation == 'ReLU':
                if (self.input_range_all[layer_index][neuron_index[0]][neuron_index[1]][neuron_index[2]][0] > 0 or
                        self.input_range_all[layer_index][neuron_index[0]][neuron_index[1]][neuron_index[2]][1] < 0):
                    print('No need to add more slack binary variables for this neuron!')
                elif self.refinement_degree_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]] == 2:
                    print('No need to add more slack binary variables for this neuron!')
                else:
                    self.refinement_degree_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]] += 1
                    print('Add a slack integer variables.')
            else:
                self.refinement_degree_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]] += 1
                print('Add a slack integer variables.')
        else:
            if self.NN.layers[layer_index].activation == 'ReLU':
                if (self.input_range_all[layer_index][neuron_index][0] > 0 or
                        self.input_range_all[layer_index][neuron_index][1] < 0):
                    print('No need to add more slack binary variables for this neuron!')
                elif self.refinement_degree_all[layer_index][neuron_index] == 2:
                    print('No need to add more slack binary variables for this neuron!')
                else:
                    self.refinement_degree_all[layer_index][neuron_index] += 1
                    print('Add a slack integer variables.')
            else:
                self.refinement_degree_all[layer_index][neuron_index] += 1
                print('Add a slack integer variables.')

    def update_neuron_input_range(self, layer_index, neuron_index):
        NN = self.NN
        input_range_all = self.input_range_all
        traceback = min(self.traceback, layer_index + 1)
        v_name = 'NN'

        print('Update range of neuron :' + 'layer_index: ' + str(layer_index) + ', neuron_index: ' + str(neuron_index))
        self._declare_variables(v_name, layer_index)
        for k in range(layer_index - 1, layer_index - 1 - traceback, -1):
            if k >= 0:
                self._add_innerlayer_constraint(v_name, k)
            if k >= layer_index - 1 - traceback + 2:
                self._add_interlayers_constraint(v_name, k)
            if k == -1:
                self._add_input_constraint(v_name)

        x_in = self.all_variables[v_name + '_1']
        if type(neuron_index) == list:
            x_in_neuron = x_in[layer_index][neuron_index[0]][neuron_index[1]][neuron_index[2]]
        else:
            x_in_neuron = x_in[layer_index][neuron_index]

        self.model.setObjective(x_in_neuron, GRB.MINIMIZE)
        neuron_min = self._optimize_model(1)
        self.model.setObjective(x_in_neuron, GRB.MAXIMIZE)
        neuron_max = self._optimize_model(1)

        if type(neuron_index) == list:
            old_range = input_range_all[layer_index][neuron_index[0]][neuron_index[1]][neuron_index[2]]
        else:
            old_range = input_range_all[layer_index][neuron_index]
        print('Old input range: {}'.format(old_range))
        new_range = [neuron_min, neuron_max]
        if NN.layers[layer_index].type == 'Fully_connected':
            input_range_all[layer_index][neuron_index] = new_range
        else:
            input_range_all[layer_index][neuron_index[0]][neuron_index[1]][neuron_index[2]] = new_range
        print('Finish Updating.')
        print('New input range: {}'.format(new_range))
        return new_range

    def _declare_variables(self, v_name, layer_index):
        # variables in the input layer
        traceback = min(self.traceback, layer_index + 1)
        NN = self.NN
        refinement_degree_all = self.refinement_degree_all
        model = self.model

        if layer_index - traceback == -1:
            if NN.type == 'Convolutional' or NN.type == 'Flatten':
                network_in = []
                for s in range(NN.layers[0].input_dim[2]):
                    network_in.append(
                        model.addVars(
                            NN.layers[0].input_dim[0],
                            NN.layers[0].input_dim[1],
                            lb=-GRB.INFINITY,
                            ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=v_name + '_inputs_' + str(s)
                        )
                    )
            else:
                network_in = model.addVars(
                    NN.layers[0].input_dim[0],
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=v_name + '_inputs'
                )
        else:
            network_in = []

        # variables of layers
        x_in = []
        x_out = []
        z = []

        for k in range(layer_index + 1):
            if k < layer_index - traceback:
                x_in.append([])
                x_out.append([])
                z.append([])
                continue
            if NN.layers[k].type == 'Convolutional':
                x_in_layer = []
                for s in range(NN.layers[k].input_dim[2]):
                    x_in_layer.append(
                        model.addVars(
                            NN.layers[k].input_dim[0],
                            NN.layers[k].input_dim[1],
                            lb=-GRB.INFINITY,
                            ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=v_name + '_in_layer_' + str(k) + '_channel_' + str(s)
                        )
                    )
                x_in.append(x_in_layer)
                x_out_layer = []
                for s in range(NN.layers[k].output_dim[2]):
                    x_out_layer.append(
                        model.addVars(
                            NN.layers[k].output_dim[0],
                            NN.layers[k].output_dim[1],
                            lb=-GRB.INFINITY,
                            ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=v_name + '_out_layer_' + str(k) + '_channel_' + str(s)
                        )
                    )
                x_out.append(x_out_layer)
                z.append([])

            if NN.layers[k].type == 'Activation':
                # define input and output variables
                x_in_layer = []
                for s in range(NN.layers[k].input_dim[2]):
                    x_in_layer.append(
                        model.addVars(
                            NN.layers[k].input_dim[0],
                            NN.layers[k].input_dim[1],
                            lb=-GRB.INFINITY,
                            ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=v_name + '_in_layer_' + str(k) + '_channel_' + str(s)
                        )
                    )
                x_in.append(x_in_layer)
                x_out_layer = []
                for s in range(NN.layers[k].output_dim[2]):
                    x_out_layer.append(
                        model.addVars(
                            NN.layers[k].output_dim[0],
                            NN.layers[k].output_dim[1],
                            lb=-GRB.INFINITY,
                            ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=v_name + '_out_layer_' + str(k) + '_channel_' + str(s)
                        )
                    )
                x_out.append(x_out_layer)
                # define slack binary variables
                z_layer = []
                for s in range(NN.layers[k].input_dim[2]):
                    z_channel = []
                    for i in range(NN.layers[k].input_dim[0]):
                        z_row = []
                        for j in range(NN.layers[k].input_dim[1]):
                            if NN.layers[k].activation == 'ReLU' and refinement_degree_all[k][s][i][j] > 2:
                                section = 2
                            else:
                                section = refinement_degree_all[k][s][i][j]
                            z_row.append(
                                model.addVars(
                                    section,
                                    vtype=GRB.BINARY
                                )
                            )

                        z_channel.append(z_row)
                    z_layer.append(z_channel)
                z.append(z_layer)

            if NN.layers[k].type == 'Pooling':
                # define input and output variables
                x_in_layer = []
                for s in range(NN.layers[k].input_dim[2]):
                    x_in_layer.append(
                        model.addVars(
                            NN.layers[k].input_dim[0],
                            NN.layers[k].input_dim[1],
                            lb=-GRB.INFINITY,
                            ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=v_name + '_in_layer_' + str(k) + '_channel_' + str(s)
                        )
                    )
                x_in.append(x_in_layer)
                x_out_layer = []
                for s in range(NN.layers[k].output_dim[2]):
                    x_out_layer.append(
                        model.addVars(
                            NN.layers[k].output_dim[0],
                            NN.layers[k].output_dim[1],
                            lb=-GRB.INFINITY,
                            ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=v_name + '_out_layer_' + str(k) + '_channel_' + str(s)
                        )
                    )
                x_out.append(x_out_layer)
                z.append([])

            if NN.layers[k].type == 'Flatten':
                # define input and output variables
                x_in_layer = []
                for s in range(NN.layers[k].input_dim[2]):
                    x_in_layer.append(
                        model.addVars(
                            NN.layers[k].input_dim[0],
                            NN.layers[k].input_dim[1],
                            lb=-GRB.INFINITY,
                            ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=v_name + '_in_layer_' + str(k) + '_channel_' + str(s))
                    )
                x_in.append(x_in_layer)
                x_out_layer = model.addVars(
                    NN.layers[k].output_dim[0],
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=v_name + '_out_layer_' + str(k))
                x_out.append(x_out_layer)
                z.append([])

            if NN.layers[k].type == 'Fully_connected':
                # Notice that here the dimension of x_in should be the same as the
                # one of x_out, which is not the one of the output of the previous
                # layer
                x_in_layer = model.addVars(
                    NN.layers[k].input_dim[0],
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=v_name + '_in_layer_' + str(k)
                )
                x_in.append(x_in_layer)
                x_out_layer = model.addVars(
                    NN.layers[k].output_dim[0],
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=v_name + '_out_layer_' + str(k)
                )
                x_out.append(x_out_layer)
                # define slack binary variables
                z_layer = []
                for i in range(NN.layers[k].output_dim[0]):
                    if NN.layers[k].activation == 'ReLU' and refinement_degree_all[k][i] > 2:
                        section = 2
                    else:
                        section = refinement_degree_all[k][i]
                    z_layer.append(
                        model.addVars(
                            section,
                            vtype=GRB.BINARY
                        )
                    )
                z.append(z_layer)

        self.all_variables[v_name + '_0'] = network_in
        self.all_variables[v_name + '_1'] = x_in
        self.all_variables[v_name + '_2'] = x_out
        self.all_variables[v_name + '_3'] = z

    #############################################################################################
    # add constraints for the input layer
    def _add_input_constraint(self, v_name):
        NN = self.NN
        network_input_box = self.network_input_box
        model = self.model
        network_in = self.all_variables[v_name + '_0']

        if network_in:
            if NN.type == 'Convolutional' or NN.type == 'Flatten':
                for s in range(NN.layers[0].input_dim[2]):
                    for i in range(NN.layers[0].input_dim[0]):
                        for j in range(NN.layers[0].input_dim[1]):
                            network_in[s][i, j].setAttr(
                                GRB.Attr.LB,
                                network_input_box[i][j][s][0]
                            )
                            network_in[s][i, j].setAttr(
                                GRB.Attr.UB,
                                network_input_box[i][j][s][1]
                            )
            else:
                for i in range(NN.layers[0].input_dim[0]):
                    network_in[s][i].setAttr(GRB.Attr.LB, network_input_box[i][0])
                    network_in[s][i].setAttr(GRB.Attr.UB, network_input_box[i][1])

    # add interlayer constraints between layer_index-1 and layer_index
    def _add_interlayers_constraint(self, v_name, layer_index):
        NN = self.NN
        model = self.model
        network_in = self.all_variables[v_name + '_0']
        x_in = self.all_variables[v_name + '_1']
        x_out = self.all_variables[v_name + '_2']

        if layer_index == 0:
            if NN.layers[layer_index].type == 'Fully_connected':
                for i in range(NN.layers[layer_index].output_dim[0]):
                    weight = np.reshape(NN.layers[layer_index].weight[:, i], (-1, 1))
                    weight_dic = {}
                    for j in range(weight.shape[0]):
                        weight_dic[j] = weight[j][0]
                    model.addConstr(
                        network_in.prod(weight_dic) + NN.layers[layer_index].bias[i] == x_in[layer_index][i])
            else:
                for s in range(NN.layers[layer_index].input_dim[2]):
                    for i in range(NN.layers[layer_index].input_dim[0]):
                        for j in range(NN.layers[layer_index].input_dim[1]):
                            model.addConstr(network_in[s][i, j] == x_in[layer_index][s][i, j])
        else:
            if (NN.layers[layer_index].type == 'Convolutional' or NN.layers[layer_index].type == 'Activation' or
                    NN.layers[layer_index].type == 'Pooling' or NN.layers[layer_index].type == 'Flatten'):
                for s in range(NN.layers[layer_index].input_dim[2]):
                    for i in range(NN.layers[layer_index].input_dim[0]):
                        for j in range(NN.layers[layer_index].input_dim[1]):
                            model.addConstr(x_out[layer_index - 1][s][i, j] == x_in[layer_index][s][i, j])
            else:
                # add constraint for linear transformation between layers
                for i in range(NN.layers[layer_index].output_dim[0]):
                    weight = np.reshape(NN.layers[layer_index].weight[:, i], (-1, 1))
                    weight_dic = {}
                    for j in range(weight.shape[0]):
                        weight_dic[j] = weight[j][0]
                    model.addConstr(
                        x_out[layer_index - 1].prod(weight_dic) + NN.layers[layer_index].bias[i] == x_in[layer_index][
                            i])

    # add innerlayer constraints for layer_index
    def _add_innerlayer_constraint(self, v_name, layer_index):
        NN = self.NN
        if NN.layers[layer_index].type == 'Convolutional':
            self._relaxation_convolutional_layer(v_name, layer_index)
        if NN.layers[layer_index].type == 'Activation':
            self._relaxation_activation_layer(v_name, layer_index)
        if NN.layers[layer_index].type == 'Pooling':
            self._relaxation_pooling_layer(v_name, layer_index)
        if NN.layers[layer_index].type == 'Flatten':
            self._relaxation_flatten_layer(v_name, layer_index)
        if NN.layers[layer_index].type == 'Fully_connected':
            self._relaxation_activation_layer(v_name, layer_index)

    #############################################################################################
    # convolutional layer
    # x_in should be 3-dimensional, x_out should be 3-dimensional
    def _relaxation_convolutional_layer(self, v_name, layer_index):
        NN = self.NN
        layer = NN.layers[layer_index]
        input_range_layer = self.input_range_all[layer_index]
        model = self.model
        x_in = self.all_variables[v_name + '_1'][layer_index]
        x_out = self.all_variables[v_name + '_2'][layer_index]
        kernel = layer.kernel
        bias = layer.bias
        stride = layer.stride
        # assign bound of each neuron input
        for i in range(layer.input_dim[0]):
            for j in range(layer.input_dim[1]):
                for s in range(layer.input_dim[2]):
                    x_in[s][i, j].setAttr(
                        GRB.Attr.LB,
                        input_range_layer[i][j][s][0]
                    )
                    x_in[s][i, j].setAttr(
                        GRB.Attr.UB,
                        input_range_layer[i][j][s][1]
                    )
        # assign the relation between input and output
        for i in range(layer.output_dim[0]):
            for j in range(layer.output_dim[1]):
                for k in range(layer.output_dim[2]):
                    sum_expr = 0
                    for s in range(layer.input_dim[2]):
                        for p in range(kernel.shape[0]):
                            for q in range(kernel.shape[1]):
                                sum_expr = sum_expr + x_in[s][i * stride[0] + p, j * stride[1] + q] * kernel[p, q, s, k]
                        sum_expr = sum_expr + bias[k]
                    model.addConstr(sum_expr == x_out[k][i, j], name='layer_' + str(layer_index) + '_' + str(
                        [i, j, s]) + '_convolutional_layer_linear')

    # pooling layer
    # x_in should be 3-dimensional, x_out should be 3-dimensional
    def _relaxation_pooling_layer(self, v_name, layer_index):
        NN = self.NN
        layer = NN.layers[layer_index]
        input_range_layer = self.input_range_all[layer_index]
        model = self.model
        x_in = self.all_variables[v_name + '_1'][layer_index]
        x_out = self.all_variables[v_name + '_2'][layer_index]
        filter_size = layer.filter_size
        pooling_type = layer.activation
        stride = layer.stride
        # assign bound of each neuron input
        for i in range(layer.input_dim[0]):
            for j in range(layer.input_dim[1]):
                for s in range(layer.input_dim[2]):
                    x_in[s][i, j].setAttr(
                        GRB.Attr.LB,
                        input_range_layer[i][j][s][0]
                    )
                    x_in[s][i, j].setAttr(
                        GRB.Attr.UB,
                        input_range_layer[i][j][s][1]
                    )
        # assign the relation between input and output
        if pooling_type == 'max':
            for s in range(layer.input_dim[2]):
                for i in range(layer.output_dim[0]):
                    for j in range(layer.output_dim[1]):
                        temp_list = []
                        for p in range(filter_size[0]):
                            for q in range(filter_size[1]):
                                temp_list.append(x_in[s][i * stride[0] + p, j * stride[1] + q])
                        model.addConstr(max_(temp_list) == x_out[s][i, j])
        if pooling_type == 'average':
            for s in range(layer.input_dim[2]):
                for i in range(layer.output_dim[0]):
                    for j in range(layer.output_dim[1]):
                        temp_list = []
                        temp_sum = 0
                        for p in range(filter_size[0]):
                            for q in range(filter_size[1]):
                                temp_list.append(x_in[s][i * stride[0] + p, j * stride[1] + q])
                                temp_sum = temp_sum + x_in[s][i * stride[0] + p, j * stride[1] + q]
                        model.addConstr(temp_sum / (filter_size[0] * filter_size[1]) == x_out[s][i, j],
                                        name='layer_' + str(layer_index) + '_' + str(
                                            [i, j, s]) + '_pooling_layer_linear')

    # flatten layer
    # x_in should be 3-dimensional, x_out should be 1-dimensional
    def _relaxation_flatten_layer(self, v_name, layer_index):
        NN = self.NN
        layer = NN.layers[layer_index]
        input_range_layer = self.input_range_all[layer_index]
        model = self.model
        x_in = self.all_variables[v_name + '_1'][layer_index]
        x_out = self.all_variables[v_name + '_2'][layer_index]
        # assign bound of each neuron input
        for i in range(layer.input_dim[0]):
            for j in range(layer.input_dim[1]):
                for s in range(layer.input_dim[2]):
                    x_in[s][i, j].setAttr(
                        GRB.Attr.LB,
                        input_range_layer[i][j][s][0]
                    )
                    x_in[s][i, j].setAttr(
                        GRB.Attr.UB,
                        input_range_layer[i][j][s][1]
                    )
        # assign the relation between input and output
        k = 0
        for s in range(layer.input_dim[2]):
            for i in range(layer.input_dim[0]):
                for j in range(layer.input_dim[1]):
                    model.addConstr(x_in[s][i, j] == x_out[k],
                                    name='layer_' + str(layer_index) + '_' + str([i, j, s]).replace(" ",
                                                                                                    "_") + '_flatten_layer_linear')
                    k = k + 1

    # Relu/tanh/sigmoid activation layer
    # Note the difference between the activation layer following the convolutional layer and the one in fully-connected layer
    def _relaxation_activation_layer(self, v_name, layer_index):
        NN = self.NN
        layer = NN.layers[layer_index]
        input_range_layer = self.input_range_all[layer_index]
        refinement_degree_layer = self.refinement_degree_all[layer_index]
        model = self.model
        x_in = self.all_variables[v_name + '_1'][layer_index]
        x_out = self.all_variables[v_name + '_2'][layer_index]
        z = self.all_variables[v_name + '_3'][layer_index]
        activation = layer.activation

        if layer.type == 'Activation':
            # if x_in is three-dimensional, which means this is a convolutional layer
            for s in range(layer.input_dim[2]):
                for i in range(layer.input_dim[0]):
                    for j in range(layer.input_dim[1]):
                        low = input_range_layer[i][j][s][0]
                        upp = input_range_layer[i][j][s][1]
                        # Stay inside one and only one region, thus sum of slack integers should be 1
                        model.addConstr(z[s][i][j].sum() == 1)
                        # construct segmentation_list with respect to refinement_degree_layer[s][i][j]
                        if activation == 'ReLU' and refinement_degree_layer[s][i][j] == 2:
                            segmentations_list = [low, 0, upp]
                        else:
                            seg_length = (upp - low) / refinement_degree_layer[s][i][j]
                            segmentations_list = [low]
                            for k in range(refinement_degree_layer[s][i][j]):
                                segmentations_list.append(low + seg_length * (k + 1))
                        for k in range(len(segmentations_list) - 1):
                            seg_left = segmentations_list[k]
                            seg_right = segmentations_list[k + 1]
                            self.segment_relaxation_basic(x_in[s][i, j], x_out[s][i, j], z[s][i][j][k], seg_left,
                                                     seg_right, activation, [i, j, s], layer_index)
        else:
            # if x_in is one-dimensional, which means this is a fc layer
            for i in range(layer.output_dim[0]):
                low = input_range_layer[i][0]
                upp = input_range_layer[i][1]
                # any neuron can only be within a region, thus sum of slack integers should be 1
                model.addConstr(z[i].sum() == 1)
                if activation == 'ReLU' and refinement_degree_layer[i] == 2:
                    segmentations_list = [low, 0, upp]
                else:
                    seg_length = (upp - low) / refinement_degree_layer[i]
                    segmentations_list = [low]
                    for k in range(refinement_degree_layer[i]):
                        segmentations_list.append(low + seg_length * (k + 1))
                for k in range(len(segmentations_list) - 1):
                    seg_left = segmentations_list[k]
                    seg_right = segmentations_list[k + 1]
                    self.segment_relaxation_basic(x_in[i], x_out[i], z[i][k], seg_left, seg_right,
                                             activation, i, layer_index)

    def segment_relaxation_basic(self, x_in_neuron, x_out_neuron, z_seg, seg_left, seg_right, activation, index,
                                 layer_index):
        model = self.model
        x_in_neuron.setAttr(GRB.Attr.LB, seg_left)
        x_in_neuron.setAttr(GRB.Attr.UB, seg_right)
        act = Activation(activation)

        M = 10e6

        temp_x_diff = M * (seg_right - seg_left)
        temp_y_diff = M * (act.activate(seg_right) - act.activate(seg_left))
        der = temp_y_diff / temp_x_diff
        if seg_left < 0 < seg_right:
            if activation == 'ReLU':
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_left(seg_right) * (
                                        x_in_neuron - seg_right) + act.activate(seg_right) <= 0))
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_right(seg_left) * (
                                        x_in_neuron - seg_left) + act.activate(seg_left) <= 0))
                model.addConstr((z_seg == 1) >>
                                (x_out_neuron - der * (x_in_neuron - seg_right) - act.activate(seg_right) <= 0))
            else:
                if der < act.activate_de_right(seg_left):
                    model.addConstr(
                        (z_seg == 1) >> (-x_out_neuron + der * (x_in_neuron - seg_left) + act.activate(seg_left) <= 0),
                        name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relaxation_A1')
                else:
                    model.addConstr((z_seg == 1) >> (-x_out_neuron + act.activate_de_right(seg_left) * (
                                x_in_neuron - seg_left) + act.activate(seg_left) <= 0),
                                    name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                                "_") + '_relaxation_A1_1')
                    neg_out = (act.activate_de_right(seg_left) * (0 - seg_left) + act.activate(seg_left))
                    model.addConstr((z_seg == 1) >> (
                                - x_out_neuron + ((act.activate(seg_right) - neg_out) / (seg_right - 0)) * (
                                    x_in_neuron - 0) + neg_out <= 0),
                                    name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                                "_") + '_relaxation_A1_2')
                if der < act.activate_de_left(seg_right):
                    model.addConstr(
                        (z_seg == 1) >> (-x_out_neuron + der * (x_in_neuron - seg_left) + act.activate(seg_left) >= 0),
                        name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relaxation_A2')
                else:
                    model.addConstr((z_seg == 1) >> (-x_out_neuron + act.activate_de_left(seg_right) * (
                                x_in_neuron - seg_right) + act.activate(seg_right) >= 0),
                                    name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                                "_") + '_relaxation_A2_1')
                    pos_out = (act.activate_de_left(seg_right) * (0 - seg_right) + act.activate(seg_right))
                    model.addConstr((z_seg == 1) >> (
                                - x_out_neuron + ((act.activate(seg_left) - pos_out) / (seg_left - 0)) * (
                                    x_in_neuron - 0) + pos_out >= 0),
                                    name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                                "_") + '_relaxation_A2_2')
        elif seg_right <= 0:
            if activation == 'ReLU':
                model.addConstr((z_seg == 1) >> (x_out_neuron == 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relu_B')
            else:
                # triangle relaxation
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_left(seg_right) * (
                                        x_in_neuron - seg_right) + act.activate(seg_right) <= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_B1')
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_right(seg_left) * (
                                        x_in_neuron - seg_left) + act.activate(seg_left) <= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_B2')
                # if np.isnan(temp_y_diff / temp_x_diff):
                #     print("x: {}, y : {}".format(temp_x_diff, temp_y_diff))
                model.addConstr((z_seg == 1) >>
                                (x_out_neuron - der * (x_in_neuron - seg_right) - act.activate(seg_right) <= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_B3')
        else:
            if activation == 'ReLU':
                model.addConstr((z_seg == 1) >> (x_out_neuron == x_in_neuron),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relu_C')
            else:
                # triangle relaxation
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_left(seg_right) * (
                                        x_in_neuron - seg_right) + act.activate(seg_right) >= 0),
                                name='layer_' + str(layer_index) + '_' + str(index) + '_relaxation_C1')
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_right(seg_left) * (
                                        x_in_neuron - seg_left) + act.activate(seg_left) >= 0),
                                name='layer_' + str(layer_index) + '_' + str(index) + '_relaxation_C2')
                # if np.isnan(temp_y_diff / temp_x_diff):
                #     print("x: {}, y : {}".format(temp_x_diff, temp_y_diff))
                model.addConstr((z_seg == 1) >>
                                (x_out_neuron - der * (x_in_neuron - seg_right) - act.activate(seg_right) >= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_C3')

    # optimize a model
    def _optimize_model(self, DETAILS_FLAG):
        self.model.setParam('OutputFlag', DETAILS_FLAG)
        self.model.setParam('BarHomogeneous', 1)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            opt = self.model.objVal
            return opt
        else:
            self.model.write("model.lp")
            print(self.model.printStats())
            self.model.computeIIS()
            if self.model.IISMinimal:
                print('IIS is minimal\n')
            else:
                print('IIS is not minimal\n')
            self.model.write("model.ilp")
            raise ValueError('Error: No solution founded!')