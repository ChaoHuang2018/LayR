import gurobipy as gp
from gurobipy import GRB
from nn_activation import Activation
from eran_init import ERANModel

import numpy as np
import tensorflow as tf
import itertools
import math
import random
import time
import copy


class NNRange(object):

    def __init__(
        self,
        NN,
        network_input_box,
        initialize_approach,
        input_range_all=None,
        refinement_degree_all=None
    ):
        # neural networks
        self.NN = NN
        # input space
        self.network_input_box = network_input_box
        self.initialize_approach = initialize_approach
        # initialize input range of each neuron
        self.__initialize_input_range(self.initialize_approach)
        # initialize refinement degree
        self.__initialize_refinement_degree()

    # initialize refinement degree
    def __initialize_refinement_degree(self):
        NN = self.NN
        refinement_degree_all = []
        for k in range(NN.num_of_hidden_layers):
            refinement_degree_layer = []
            if len(NN.layers[k].input_dim) == 3:
                for s in range(NN.layers[k].input_dim[2]):
                    refinement_degree_layer_channel = []
                    for i in range(NN.layers[k].input_dim[0]):
                        refinement_degree_layer_row = []
                        for j in range(NN.layers[k].input_dim[1]):
                            refinement_degree_layer_row.append(self._set_refinement_degree(k, [i, j, s], 1))
                        refinement_degree_layer_channel.append(
                            refinement_degree_layer_row
                        )
                    refinement_degree_layer.append(
                        refinement_degree_layer_channel
                    )
                refinement_degree_all.append(
                    refinement_degree_layer
                )
            if len(NN.layers[k].input_dim) == 1:
                for i in range(NN.layers[k].output_dim[0]):
                    refinement_degree_layer.append(self._set_refinement_degree(k, i, 1))
                refinement_degree_all.append(refinement_degree_layer)
        self.refinement_degree_all = refinement_degree_all

    def _set_refinement_degree(self, layer_index, neuron_index, neuron_refinement_degree):
        # check relu
        if type(neuron_index) == list:
            if self.NN.layers[layer_index].activation == 'Affine':
                return 1
            elif self.NN.layers[layer_index].activation == 'ReLU':
                if (self.input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]][0] > 0 or
                        self.input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]][1] < 0):
                    return min(1, neuron_refinement_degree)
                else:
                    return min(2, neuron_refinement_degree)
            else:
                return neuron_refinement_degree
        else:
            if self.NN.layers[layer_index].activation == 'Affine':
                return 1
            elif self.NN.layers[layer_index].activation == 'ReLU':
                if (self.input_range_all[layer_index][neuron_index][0] > 0 or
                        self.input_range_all[layer_index][neuron_index][1] < 0):
                    return min(1, neuron_refinement_degree)
                else:
                    return min(2, neuron_refinement_degree)
            else:
                return neuron_refinement_degree

    # initialize input range of each neuron by naive layer-by-layer propagation output range analysis
    # We may need a better approach to obtain the initial solution
    def __initialize_input_range(self, method='BASIC'):
        NN = self.NN
        network_input_box = self.network_input_box

        input_range_all = []
        output_range_all = []
        output_range_layer = network_input_box
        print('-----------Start to construct the initial input range of each neuron for further analysis.------------')

        if method == 'ERAN' or method == 'BASIC':
            eran = ERANModel(NN)
            input_range_eran = eran.input_range_eran(network_input_box)

        # print('ERAN:' + str(input_range_eran[0][0][0]))
        # print('ERAN:' + str(input_range_eran[1][0][0]))
        # print('ERAN:' + str(input_range_eran[2][0:10]))
        # print('ERAN:' + str(input_range_eran[3]))

        i = 0
        j = 0
        while i < self.NN.num_of_hidden_layers:
            print('-------------layer: {}---------------'.format(i))
            print(NN.layers[i].type)

            # assign the input the range of each layer
            if NN.layers[i].type != 'Fully_connected':
                input_range_layer = output_range_layer
            else:
                input_range_layer = self.__input_range_fc_layer_naive(
                    i,
                    output_range_layer
                )

            if (NN.layers[i].type == 'Fully_connected' or NN.layers[i].type == 'Activation') and method == 'ERAN':
                input_range_layer = copy.deepcopy(self.merge_range(i, input_range_layer, input_range_eran[j]))
                # input_range_layer = copy.deepcopy(np.array(input_range_eran[j]))
                j += 1

            input_range_all.append(input_range_layer)
            if (NN.layers[i].type != 'Fully_connected'):
                print('BASIC:' + str(input_range_layer[0][0]))
            else:
                print('BASIC:' + str(input_range_layer[0:10]))

            print('The dimension of input range of layer ' + str(i) + ' :')
            print(input_range_layer.shape)
            print(NN.layers[i].input_dim)


            # Compute the output range of each layer
            if NN.layers[i].type == 'Convolutional':
                output_range_layer = self.__output_range_convolutional_layer_naive(
                    i,
                    input_range_layer
                )
            if NN.layers[i].type == 'Activation':
                print(NN.layers[i].activation)
                output_range_layer = self.__output_range_activation_layer_naive(
                    i,
                    input_range_layer
                )
            if NN.layers[i].type == 'Pooling':
                output_range_layer = self.__output_range_pooling_layer_naive(
                    i,
                    input_range_layer
                )
            if NN.layers[i].type == 'Flatten':
                output_range_layer = self.__output_range_flatten_layer_naive(
                    i,
                    input_range_layer
                )
            if NN.layers[i].type == 'Fully_connected':
                output_range_layer = self.__output_range_activation_layer_naive(
                    i,
                    input_range_layer
                )
            output_range_all.append(output_range_layer)
            i += 1

        self.input_range_all = input_range_all
        self.output_range_all = output_range_all
        print('-------Naive input range for each neuron is generated.----------')

    def __input_range_fc_layer_naive(self, layer_index, output_range_last_layer):
        weight = self.NN.layers[layer_index].weight
        bias = self.NN.layers[layer_index].bias

        # compute the input range of each neuron by solving LPs
        input_range_layer = []

        for i in range(self.NN.layers[layer_index].input_dim[0]):
            model_in_neuron = gp.Model()

            x_out = model_in_neuron.addVars(weight.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
            # define constraints: output range of the last layer
            for j in range(output_range_last_layer.shape[0]):
                x_out[j].setAttr(GRB.Attr.LB, output_range_last_layer[j, 0])
                x_out[j].setAttr(GRB.Attr.UB, output_range_last_layer[j, 1])

            x_in = model_in_neuron.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)

            weight_i = np.reshape(weight[:, i], (-1, 1))
            bias_i = bias[i]

            # define constraints: linear transformation by weight and bias
            # in gurobi, no matrix api for python, we need to mannally implement the matrix operation based on gurobi tupledict
            # transform a numpy array to a dictinary
            weight_i_dic = {}
            for j in range(weight_i.shape[0]):
                weight_i_dic[j] = weight_i[j][0]

            model_in_neuron.addConstr(x_out.prod(weight_i_dic) + bias_i == x_in, 'lt')

            # define objective: smallest output of [layer_index, neuron_index]
            # Set objective
            model_in_neuron.setObjective(x_in, GRB.MINIMIZE)

            model_in_neuron.setParam('OutputFlag', 0)
            model_in_neuron.optimize()

            if model_in_neuron.status == GRB.OPTIMAL:
                neuron_min = model_in_neuron.objVal
            else:
                print('prob_min.status: ' + str(model_in_neuron.status))
                raise ValueError('Error: No result for lower bound!')

            # define objective: biggest output of [layer_index, neuron_index]
            model_in_neuron.setObjective(x_in, GRB.MAXIMIZE)
            model_in_neuron.setParam('OutputFlag', 0)
            model_in_neuron.optimize()

            if model_in_neuron.status == GRB.OPTIMAL:
                neuron_max = model_in_neuron.objVal
            else:
                print('prob_max.status: ' + model_in_neuron.status)
                raise ValueError('Error: No result for upper bound!')

            input_range_layer.append([neuron_min, neuron_max])
        return np.array(input_range_layer)

    def __output_range_convolutional_layer_naive(self, layer_index, input_range_layer):
        layer = self.NN.layers[layer_index]
        kernel = self.NN.layers[layer_index].kernel
        bias = self.NN.layers[layer_index].bias
        stride = self.NN.layers[layer_index].stride

        output_range_layer = []
        print('The size of bias is: ' + str(bias.shape))
        print(layer.output_dim)

        for k in range(layer.output_dim[2]):
            output_range_layer_channel = []
            for i in range(layer.output_dim[0]):
                output_range_layer_row = []
                for j in range(layer.output_dim[1]):
                    solver = 'NAIVE'
                    if solver == 'GUROBI':
                        model_out_neuron = gp.Model()
                        x_in = []
                        for s in range(kernel.shape[2]):
                            x_in.append(
                                model_out_neuron.addVars(kernel.shape[0], kernel.shape[1], lb=-GRB.INFINITY,
                                                         ub=GRB.INFINITY,
                                                         vtype=GRB.CONTINUOUS))
                        x_out = model_out_neuron.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                        sum_expr = 0
                        for s in range(kernel.shape[2]):
                            for p in range(kernel.shape[0]):
                                for q in range(kernel.shape[1]):
                                    x_in[s][p, q].setAttr(GRB.Attr.LB, input_range_layer[s][
                                        i * stride[0] + p][j * stride[1] + q][0])
                                    x_in[s][p, q].setAttr(GRB.Attr.UB, input_range_layer[s][
                                        i * stride[0] + p][j * stride[1] + q][1])
                                    sum_expr = sum_expr + x_in[s][p, q] * kernel[p, q, s, k]
                            sum_expr = sum_expr + bias[k]
                        model_out_neuron.addConstr(sum_expr == x_out)

                        # define objective: smallest output
                        model_out_neuron.setObjective(x_out, GRB.MINIMIZE)
                        model_out_neuron.setParam('OutputFlag', 0)
                        model_out_neuron.optimize()

                        if model_out_neuron.status == GRB.OPTIMAL:
                            neuron_min = model_out_neuron.objVal
                        else:
                            print('prob_min.status: ' + str(model_out_neuron.status))
                            model_out_neuron.write("model_out_neuron.lp")
                            model_out_neuron.computeIIS()
                            if model_out_neuron.IISMinimal:
                                print('IIS is minimal\n')
                            else:
                                print('IIS is not minimal\n')
                            model_out_neuron.write("model_out_neuron.ilp")
                            raise ValueError("Error: No result for lower bound for " + str([i, j, s, k]))

                        # define objective: biggest output
                        model_out_neuron.setObjective(x_out, GRB.MAXIMIZE)
                        model_out_neuron.setParam('OutputFlag', 0)
                        model_out_neuron.optimize()

                        if model_out_neuron.status == GRB.OPTIMAL:
                            neuron_max = model_out_neuron.objVal
                        else:
                            print('prob_max.status: ' + str(model_out_neuron.status))
                            print('Error: No result for upper bound!')
                        output_range_layer_row.append([neuron_min, neuron_max])
                    else:
                        min_test = 0
                        max_test = 0
                        for s in range(kernel.shape[2]):
                            for p in range(kernel.shape[0]):
                                for q in range(kernel.shape[1]):
                                    if kernel[p, q, s, k] >= 0:
                                        min_test += input_range_layer[s][i * stride[0] + p][j * stride[1] + q][0] * \
                                                    kernel[
                                                        p, q, s, k]
                                        max_test += input_range_layer[s][i * stride[0] + p][j * stride[1] + q][1] * \
                                                    kernel[
                                                        p, q, s, k]
                                    else:
                                        min_test += input_range_layer[s][i * stride[0] + p][j * stride[1] + q][1] * \
                                                    kernel[
                                                        p, q, s, k]
                                        max_test += input_range_layer[s][i * stride[0] + p][j * stride[0] + q][1] * \
                                                    kernel[
                                                        p, q, s, k]
                            min_test = min_test + bias[k]
                            max_test = max_test + bias[k]
                        output_range_layer_row.append([min_test, max_test])
                output_range_layer_channel.append(output_range_layer_row)
            output_range_layer.append(output_range_layer_channel)
        return np.array(output_range_layer)

    def __output_range_activation_layer_naive(self, layer_index, input_range_layer):
        layer = self.NN.layers[layer_index]
        activation = self.NN.layers[layer_index].activation
        act = Activation(activation)

        if len(layer.input_dim) == 3:
            # for convolutional layer
            # compute the out range of each neuron by activation function
            output_range_layer = []
            for s in range(layer.input_dim[2]):
                output_range_layer_channel = []
                for i in range(layer.input_dim[0]):
                    output_range_layer_row = []
                    for j in range(layer.input_dim[1]):
                        # compute the minimal output
                        neuron_min = act.activate(input_range_layer[s][i][j][0])
                        # compute the maximal output
                        neuron_max = act.activate(input_range_layer[s][i][j][1])
                        if neuron_min > neuron_max:
                            print(input_range_layer[s][i][j][0])
                            print(input_range_layer[s][i][j][1])
                            raise ValueError("Wrong bound " + str([i, j, s]))
                        output_range_layer_row.append([neuron_min, neuron_max])
                    output_range_layer_channel.append(output_range_layer_row)
                output_range_layer.append(output_range_layer_channel)
        else:
            # for fully connected layer
            # compute the out range of each neuron by activation function
            output_range_layer = []
            for i in range(layer.input_dim[0]):
                # compute the minimal output
                neuron_min = act.activate(input_range_layer[i][0])
                # compute the maximal output
                neuron_max = act.activate(input_range_layer[i][1])
                output_range_layer.append([neuron_min, neuron_max])

        return np.array(output_range_layer)

    def __output_range_pooling_layer_naive(self, layer_index, input_range_layer):
        layer = self.NN.layers[layer_index]
        filter_size = self.NN.layers[layer_index].filter_size
        pooling_type = self.NN.layers[layer_index].activation
        stride = self.NN.layers[layer_index].stride

        output_range_layer = []

        for s in range(layer.output_dim[2]):
            output_range_layer_channel = []
            for i in range(layer.output_dim[0]):
                output_range_layer_row = []
                for j in range(layer.output_dim[1]):
                    if pooling_type == 'max':
                        # constraints += [cp.max(x_in) == x_out] # The max operation is convex but not affine, thus can be not be used in an equation constraint in DCP.
                        # Instead, we directly derive the constraint for max pooling
                        neuron_min = np.max(input_range_layer[
                                            i * stride[0]: i * stride[0] + filter_size[0],
                                            j * stride[1]: j * stride[1] + filter_size[1],
                                            s,
                                            0
                                            ])
                        neuron_max = np.max(input_range_layer[
                                            i * stride[0]: i * stride[0] + filter_size[0],
                                            j * stride[1]: j * stride[1] + filter_size[1],
                                            s,
                                            1
                                            ])
                    if pooling_type == 'average':
                        # Instead, we directly derive the constraint for max pooling
                        neuron_min = np.mean(input_range_layer[
                                             i * stride[0]: i * stride[0] + filter_size[0],
                                             j * stride[1]: j * stride[1] + filter_size[1],
                                             s,
                                             0
                                             ])
                        neuron_max = np.mean(input_range_layer[
                                             i * stride[0]: i * stride[0] + filter_size[0],
                                             j * stride[1]: j * stride[1] + filter_size[1],
                                             s,
                                             1
                                             ])
                    output_range_layer_row.append([neuron_min, neuron_max])
                output_range_layer_channel.append(output_range_layer_row)
            output_range_layer.append(output_range_layer_channel)
        return np.array(output_range_layer)

    # flatten layer: just flatten the output
    def __output_range_flatten_layer_naive(self, layer_index, input_range_layer):
        layer = self.NN.layers[layer_index]
        output_range_layer = []
        for s in range(input_range_layer.shape[0]):
            for i in range(input_range_layer.shape[1]):
                # consider the i-th row
                for j in range(input_range_layer.shape[2]):
                    # add the j-th neuron
                    output_range_layer.append(input_range_layer[s][i][j])
        return np.array(output_range_layer)

    def merge_range(self, layer_index, range_a, range_b):
        layer = self.NN.layers[layer_index]
        range_new = copy.deepcopy(range_a)
        if len(range_a) == 0:
            return range_b
        if len(range_b) == 0:
            return range_a
        if len(layer.input_dim) == 3:
            for s in range(layer.input_dim[2]):
                for i in range(layer.input_dim[0]):
                    for j in range(layer.input_dim[1]):
                        range_new[s][i][j] = [max(range_a[s][i][j][0], range_b[s][i][j][0]), min(range_a[s][i][j][1], range_b[s][i][j][1])]
        else:
            for i in range(layer.input_dim[0]):
                range_new[i] = [max(range_a[i][0], range_b[i][0]), min(range_a[i][1], range_b[i][1])]
                if range_new[i][0] > range_new[i][1]:
                    print(layer_index)
                    print(i)
                    print(range_new[i])
                    print(range_a[i])
                    print(range_b[i])
                    raise ValueError('ERROR range after merging!')
        return range_new
