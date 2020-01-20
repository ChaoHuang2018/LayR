import gurobipy as gp
from gurobipy import GRB
from nn_range import NNRange
from nn_activation import Activation

import numpy as np
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
        initialize_approach,
        v_name='NN',
        traceback=None
    ):
        NNRange.__init__(self, NN, network_input_box, initialize_approach)
        self.v_name = v_name
        self.traceback = traceback

    def refine_neuron(self, layer_index, neuron_index, approach='BOTH'):
        print('Start to refine neuron with index:')
        print('layer_index: ' + str(layer_index) + ', neuron_index: ' + str(neuron_index))
        if approach == 'ADD_INTEGER':
            self.add_neuron_slack_integer(layer_index, neuron_index)
            if isinstance(neuron_index, list):
                old_range = self.input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]]
            else:
                old_range = self.input_range_all[layer_index][neuron_index]
            new_range = old_range
        elif approach == 'UPDATE_RANGE':
            start_layer = layer_index - min(self.traceback, layer_index + 1)
            new_range = self.update_neuron_input_range(start_layer, layer_index, neuron_index)
        else:
            start_layer = layer_index - min(self.traceback, layer_index + 1)
            self.add_neuron_slack_integer(layer_index, neuron_index)
            new_range = self.update_neuron_input_range(start_layer, layer_index, neuron_index)
        return new_range

    def add_neuron_slack_integer(self, layer_index, neuron_index):
        print('Add integer variable of neuron :' + 'layer_index: ' + str(layer_index) + ', neuron_index: ' + str(
            neuron_index))
        # need to carefully handle ReLU
        if type(neuron_index) == list:
            if self.NN.layers[layer_index].activation == 'identity':
                print('No need to add more slack binary variables for this neuron!')
                return
            elif self.NN.layers[layer_index].activation == 'ReLU':
                if (self.input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]][0] > 0 or
                        self.input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]][1] < 0):
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
            if self.NN.layers[layer_index].activation == 'identity':
                print('No need to add more slack binary variables for this neuron!')
                return
            elif self.NN.layers[layer_index].activation == 'ReLU':
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

    def update_neuron_input_range(self, start_layer, layer_index, neuron_index):
        model = gp.Model('Input_range_update')
        NN = self.NN
        input_range_all = self.input_range_all
        v_name = self.v_name

        print('Update range of neuron :' + 'layer_index: ' + str(layer_index) + ', neuron_index: ' + str(neuron_index))
        all_variables = self._declare_variables(model, v_name, start_layer, layer_index)
        for k in range(layer_index, start_layer - 1, -1):
            if k >= 0:
                self._add_innerlayer_constraint(model, all_variables, v_name, k)
            if k >= start_layer + 1:
                self._add_interlayers_constraint(model, all_variables, v_name, k)
            if k == -1:
                self._add_input_constraint(model, all_variables, v_name)

        x_in = all_variables[v_name + '_1']
        if type(neuron_index) == list:
            x_in_neuron = x_in[layer_index][neuron_index[2]][neuron_index[0], neuron_index[1]]
        else:
            x_in_neuron = x_in[layer_index][neuron_index]

        model.setObjective(x_in_neuron, GRB.MINIMIZE)
        neuron_min = self._optimize_model(model, 0)
        model.setObjective(x_in_neuron, GRB.MAXIMIZE)
        neuron_max = self._optimize_model(model, 0)

        print([start_layer,layer_index])
        if type(neuron_index) == list:
            old_range = input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]]
        else:
            old_range = input_range_all[layer_index][neuron_index]
        print('Old input range: {}'.format(old_range))
        new_range = [neuron_min, neuron_max]
        if NN.layers[layer_index].type == 'Fully_connected':
            input_range_all[layer_index][neuron_index] = new_range
        else:
            input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]] = new_range
        print('Finish Updating.')
        print('New input range: {}'.format(new_range))
        return new_range

    def _declare_variables(self, model, v_name, start_layer, end_layer):
        # variables in the input layer
        NN = self.NN
        refinement_degree_all = self.refinement_degree_all

        if start_layer == -1:
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

        for k in range(end_layer + 1):
            if k < start_layer:
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

        all_variables = {}
        all_variables[v_name + '_0'] = network_in
        all_variables[v_name + '_1'] = x_in
        all_variables[v_name + '_2'] = x_out
        all_variables[v_name + '_3'] = z
        return all_variables

    #############################################################################################
    # add constraints for the input layer
    def _add_input_constraint(self, model, all_variables, v_name):
        NN = self.NN
        network_input_box = self.network_input_box
        network_in = all_variables[v_name + '_0']

        if network_in:
            if NN.type == 'Convolutional' or NN.type == 'Flatten':
                for s in range(NN.layers[0].input_dim[2]):
                    for i in range(NN.layers[0].input_dim[0]):
                        for j in range(NN.layers[0].input_dim[1]):
                            network_in[s][i, j].setAttr(
                                GRB.Attr.LB,
                                network_input_box[s][i][j][0]
                            )
                            network_in[s][i, j].setAttr(
                                GRB.Attr.UB,
                                network_input_box[s][i][j][1]
                            )
            else:
                for i in range(NN.layers[0].input_dim[0]):
                    network_in[s][i].setAttr(GRB.Attr.LB, network_input_box[i][0])
                    network_in[s][i].setAttr(GRB.Attr.UB, network_input_box[i][1])

    # add interlayer constraints between layer_index-1 and layer_index
    def _add_interlayers_constraint(self, model, all_variables, v_name, layer_index):
        NN = self.NN
        network_in = all_variables[v_name + '_0']
        x_in = all_variables[v_name + '_1']
        x_out = all_variables[v_name + '_2']

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
    def _add_innerlayer_constraint(self, model, all_variables, v_name, layer_index):
        NN = self.NN
        if NN.layers[layer_index].type == 'Convolutional':
            self._relaxation_convolutional_layer(model, all_variables, v_name, layer_index)
        if NN.layers[layer_index].type == 'Activation':
            self._relaxation_activation_layer(model, all_variables, v_name, layer_index)
        if NN.layers[layer_index].type == 'Pooling':
            self._relaxation_pooling_layer(model, all_variables, v_name, layer_index)
        if NN.layers[layer_index].type == 'Flatten':
            self._relaxation_flatten_layer(model, all_variables, v_name, layer_index)
        if NN.layers[layer_index].type == 'Fully_connected':
            self._relaxation_activation_layer(model, all_variables, v_name, layer_index)

    #############################################################################################
    # convolutional layer
    # x_in should be 3-dimensional, x_out should be 3-dimensional
    def _relaxation_convolutional_layer(self, model, all_variables, v_name, layer_index):
        NN = self.NN
        layer = NN.layers[layer_index]
        input_range_layer = self.input_range_all[layer_index]
        x_in = all_variables[v_name + '_1'][layer_index]
        x_out = all_variables[v_name + '_2'][layer_index]
        kernel = layer.kernel
        bias = layer.bias
        stride = layer.stride
        # assign input bound of each neuron input
        for s in range(layer.input_dim[2]):
            for i in range(layer.input_dim[0]):
                for j in range(layer.input_dim[1]):
                    x_in[s][i, j].setAttr(
                        GRB.Attr.LB,
                        input_range_layer[s][i][j][0]
                    )
                    x_in[s][i, j].setAttr(
                        GRB.Attr.UB,
                        input_range_layer[s][i][j][1]
                    )
        # assign the relation between input and output
        for k in range(layer.output_dim[2]):
            for i in range(layer.output_dim[0]):
                for j in range(layer.output_dim[1]):
                    sum_expr = 0
                    for s in range(kernel.shape[2]):
                        for p in range(kernel.shape[0]):
                            for q in range(kernel.shape[1]):
                                sum_expr = sum_expr + x_in[s][i * stride[0] + p, j * stride[1] + q] * kernel[p, q, s, k]
                        sum_expr = sum_expr + bias[k]
                    model.addConstr(sum_expr == x_out[k][i, j], name='layer_' + str(layer_index) + '_' + str(
                        [i, j, s]) + '_convolutional_layer_linear')

    # pooling layer
    # x_in should be 3-dimensional, x_out should be 3-dimensional
    def _relaxation_pooling_layer(self, model, all_variables, v_name, layer_index):
        NN = self.NN
        layer = NN.layers[layer_index]
        refinement_degree_layer = self.refinement_degree_all[layer_index]
        input_range_layer = self.input_range_all[layer_index]
        x_in = all_variables[v_name + '_1'][layer_index]
        x_out = all_variables[v_name + '_2'][layer_index]
        filter_size = layer.filter_size
        pooling_type = layer.activation
        stride = layer.stride
        # assign bound of each neuron input
        for s in range(layer.input_dim[2]):
            for i in range(layer.input_dim[0]):
                for j in range(layer.input_dim[1]):
                    x_in[s][i, j].setAttr(
                        GRB.Attr.LB,
                        input_range_layer[s][i][j][0]
                    )
                    x_in[s][i, j].setAttr(
                        GRB.Attr.UB,
                        input_range_layer[s][i][j][1]
                    )
        # assign the relation between input and output
        if pooling_type == 'max':
            for s in range(layer.input_dim[2]):
                for i in range(layer.output_dim[0]):
                    for j in range(layer.output_dim[1]):
                        if refinement_degree_layer[s][i][j] == 1:
                            for p in range(filter_size[0]):
                                for q in range(filter_size[1]):
                                    model.addConstr(x_out[s][i, j] >= x_in[s][i * stride[0] + p, j * stride[1] + q])
                                    model.addConstr(x_out[s][i, j] <= x_in[s][i * stride[0] + p, j * stride[1] + q])
                        else:
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
    def _relaxation_flatten_layer(self, model, all_variables, v_name, layer_index):
        NN = self.NN
        layer = NN.layers[layer_index]
        input_range_layer = self.input_range_all[layer_index]
        x_in = all_variables[v_name + '_1'][layer_index]
        x_out = all_variables[v_name + '_2'][layer_index]
        # assign bound of each neuron input
        for s in range(layer.input_dim[2]):
            for i in range(layer.input_dim[0]):
                for j in range(layer.input_dim[1]):
                    x_in[s][i, j].setAttr(
                        GRB.Attr.LB,
                        input_range_layer[s][i][j][0]
                    )
                    x_in[s][i, j].setAttr(
                        GRB.Attr.UB,
                        input_range_layer[s][i][j][1]
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
    def _relaxation_activation_layer(self, model, all_variables, v_name, layer_index):
        NN = self.NN
        layer = NN.layers[layer_index]
        input_range_layer = self.input_range_all[layer_index]
        refinement_degree_layer = self.refinement_degree_all[layer_index]
        x_in = all_variables[v_name + '_1'][layer_index]
        x_out = all_variables[v_name + '_2'][layer_index]
        z = all_variables[v_name + '_3'][layer_index]
        activation = layer.activation
        act = Activation(activation)

        if layer.type == 'Activation':
            # if x_in is three-dimensional, which means this is a convolutional layer
            for s in range(layer.input_dim[2]):
                for i in range(layer.input_dim[0]):
                    for j in range(layer.input_dim[1]):
                        low = input_range_layer[s][i][j][0]
                        upp = input_range_layer[s][i][j][1]
                        # set input range of the neuron
                        x_in[s][i, j].setAttr(GRB.Attr.LB, low)
                        x_in[s][i, j].setAttr(GRB.Attr.UB, upp)
                        if act.activate(upp) - act.activate(low) <= 10e5:
                            model.addConstr(z[s][i][j].sum() == 0)
                            x_out[s][i, j].setAttr(GRB.Attr.LB, act.activate(low))
                            x_out[s][i, j].setAttr(GRB.Attr.UB, act.activate(upp))
                            continue
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
                            self.segment_relaxation_basic(model, x_in[s][i, j], x_out[s][i, j], z[s][i][j][k], seg_left,
                                                     seg_right, activation, [i, j, s], layer_index)
        else:
            # if x_in is one-dimensional, which means this is a fc layer
            for i in range(layer.output_dim[0]):
                low = input_range_layer[i][0]
                upp = input_range_layer[i][1]
                # set range of the neuron
                x_in[i].setAttr(GRB.Attr.LB, low)
                x_in[i].setAttr(GRB.Attr.UB, upp)
                if act.activate(upp) - act.activate(low) <= 10e5:
                    model.addConstr(z[i].sum() == 0)
                    x_out[i].setAttr(GRB.Attr.LB, act.activate(low))
                    x_out[i].setAttr(GRB.Attr.UB, act.activate(upp))
                    continue
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
                    self.segment_relaxation_basic(model, x_in[i], x_out[i], z[i][k], seg_left, seg_right,
                                             activation, i, layer_index)

    def segment_relaxation_basic(self, model, x_in_neuron, x_out_neuron, z_seg, seg_left, seg_right, activation, index,
                                 layer_index):
        act = Activation(activation)
        if isinstance(seg_left, np.ndarray):
            seg_left = seg_left[0]
        if isinstance(seg_right, np.ndarray):
            seg_right = seg_right[0]

        M = 10e6

        temp_x_diff = M * (seg_right - seg_left)
        temp_y_diff = M * (act.activate(seg_right) - act.activate(seg_left))
        der = temp_y_diff / temp_x_diff
        if activation == 'Affine':
            model.addConstr((z_seg == 1) >> (x_in_neuron == x_out_neuron),
                            name='layer_' + str(layer_index) + '_' + str(index).replace(" ","_") + '_identity')
        elif seg_left < 0 < seg_right:
            if activation == 'ReLU':
                model.addConstr((z_seg == 1) >> (x_in_neuron >= seg_left),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relu_A_lowbound')
                model.addConstr((z_seg == 1) >> (x_in_neuron <= seg_right),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relu_A_uppbound')
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_left(seg_right) * (
                                        x_in_neuron - seg_right) + act.activate(seg_right) <= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relu_A1')
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_right(seg_left) * (
                                        x_in_neuron - seg_left) + act.activate(seg_left) <= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relu_A2')
                model.addConstr((z_seg == 1) >>
                                (x_out_neuron - der * (x_in_neuron - seg_right) - act.activate(seg_right) <= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relu_A3')
            else:
                model.addConstr((z_seg == 1) >> (x_in_neuron >= seg_left),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_A_lowbound')
                model.addConstr((z_seg == 1) >> (x_in_neuron <= seg_right),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_A_uppbound')
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
                model.addConstr((z_seg == 1) >> (x_in_neuron >= seg_left),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relu_B_lowbound')
                model.addConstr((z_seg == 1) >> (x_in_neuron <= seg_right),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relu_B_uppbound')
                model.addConstr((z_seg == 1) >> (x_out_neuron == 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relu_B')
            else:
                # triangle relaxation
                model.addConstr((z_seg == 1) >> (x_in_neuron >= seg_left),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_B_lowbound')
                model.addConstr((z_seg == 1) >> (x_in_neuron <= seg_right),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_B_uppbound')
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
                model.addConstr((z_seg == 1) >>
                                (x_out_neuron - der * (x_in_neuron - seg_right) - act.activate(seg_right) <= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_B3')
        else:
            if activation == 'ReLU':
                model.addConstr((z_seg == 1) >> (x_in_neuron >= seg_left),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relu_C_lowbound')
                model.addConstr((z_seg == 1) >> (x_in_neuron <= seg_right),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relu_C_uppbound')
                model.addConstr((z_seg == 1) >> (x_out_neuron == x_in_neuron),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ", "_") + '_relu_C')
            else:
                # triangle relaxation
                model.addConstr((z_seg == 1) >> (x_in_neuron >= seg_left),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_C_lowbound')
                model.addConstr((z_seg == 1) >> (x_in_neuron <= seg_right),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_C_uppbound')
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_left(seg_right) * (
                                        x_in_neuron - seg_right) + act.activate(seg_right) >= 0),
                                name='layer_' + str(layer_index) + '_' + str(index) + '_relaxation_C1')
                model.addConstr((z_seg == 1) >>
                                (-x_out_neuron + act.activate_de_right(seg_left) * (
                                        x_in_neuron - seg_left) + act.activate(seg_left) >= 0),
                                name='layer_' + str(layer_index) + '_' + str(index) + '_relaxation_C2')
                model.addConstr((z_seg == 1) >>
                                (x_out_neuron - der * (x_in_neuron - seg_right) - act.activate(seg_right) >= 0),
                                name='layer_' + str(layer_index) + '_' + str(index).replace(" ",
                                                                                            "_") + '_relaxation_C3')

    # optimize a model
    def _optimize_model(self, model, DETAILS_FLAG):
        model.setParam('OutputFlag', DETAILS_FLAG)
        model.setParam('BarHomogeneous', 1)
        # model.setParam('DualReductions', 0)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            opt = model.objVal
            return opt
        else:
            model.write("model.lp")
            print(model.printStats())
            model.computeIIS()
            if model.IISMinimal:
                print('IIS is minimal\n')
            else:
                print('IIS is not minimal\n')
            model.write("model.ilp")
            raise ValueError('Error: No solution founded!')