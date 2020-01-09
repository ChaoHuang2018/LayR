import gurobipy as gp
from gurobipy import GRB
from nn_activation import Activation

import numpy as np
import sympy as sp
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
        input_range_all=None,
        refinement_degree_all=None
    ):
        # neural networks
        self.NN = NN

        # input space
        self.network_input_box = network_input_box

        # initialize refinement degree
        self.__initialize_refinement_degree()

        # initialize input range of each neuron
        self.__initialize_input_range()

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
                            refinement_degree_layer_row.append(1)
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
                    refinement_degree_layer.append(2)
                refinement_degree_all.append(refinement_degree_layer)
        self.refinement_degree_all = refinement_degree_all

    # initialize input range of each neuron by naive layer-by-layer propagation output range analysis
    # We may need a better approach to obtain the initial solution
    def __initialize_input_range(self):
        NN = self.NN
        network_input_box = self.network_input_box

        input_range_all = []
        output_range_layer_i = network_input_box
        print('-----------Start to construct the naive input range of each neuron for further analysis.------------')

        for i in range(NN.num_of_hidden_layers):
            print('-------------layer: {}---------------'.format(i))
            print(NN.layers[i].type)

            # assign the input the range of each layer
            if NN.layers[i].type != 'Fully_connected':
                input_range_layer_i = output_range_layer_i
            else:
                input_range_layer_i = self.__input_range_fc_layer_naive(
                    i,
                    output_range_layer_i
                )
            input_range_all.append(input_range_layer_i)

            print('The dimension of input range of layer ' + str(i) + ' :')
            print(input_range_layer_i.shape)

            # Compute the output range of each layer
            if NN.layers[i].type == 'Convolutional':
                output_range_layer_i = self.__output_range_convolutional_layer_naive(
                    i,
                    input_range_layer_i
                )
            if NN.layers[i].type == 'Activation':
                print(NN.layers[i].activation)
                output_range_layer_i = self.__output_range_activation_layer_naive(
                    i,
                    input_range_layer_i
                )
            if NN.layers[i].type == 'Pooling':
                output_range_layer_i = self.__output_range_pooling_layer_naive(
                    i,
                    input_range_layer_i
                )
            if NN.layers[i].type == 'Flatten':
                output_range_layer_i = self.__output_range_flatten_layer_naive(
                    i,
                    input_range_layer_i
                )
            if NN.layers[i].type == 'Fully_connected':
                output_range_layer_i = self.__output_range_activation_layer_naive(
                    i,
                    input_range_layer_i
                )

        self.input_range_all = input_range_all
        print('-------Naive input range for each neuron is generated.----------')

    def __input_range_fc_layer_naive(self, layer_index, output_range_last_layer):
        weight = self.NN.layers[layer_index].weight
        bias = self.NN.layers[layer_index].bias

        # compute the input range of each neuron by solving LPs
        input_range_layer = []

        for i in range(weight.shape[1]):
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

        for i in range(layer.output_dim[0]):
            output_range_layer_row = []
            for j in range(layer.output_dim[1]):
                output_range_layer_col = []
                for k in range(layer.output_dim[2]):
                    model_out_neuron = gp.Model()
                    x_in = []
                    for s in range(layer.input_dim[2]):
                        x_in.append(
                            model_out_neuron.addVars(kernel.shape[0], kernel.shape[1], lb=-GRB.INFINITY,
                                                     ub=GRB.INFINITY,
                                                     vtype=GRB.CONTINUOUS))
                    x_out = model_out_neuron.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    constraints = []
                    sum_expr = 0
                    for s in range(layer.input_dim[2]):
                        for p in range(kernel.shape[0]):
                            for q in range(kernel.shape[1]):
                                x_in[s][p, q].setAttr(GRB.Attr.LB, input_range_layer[
                                    i * stride[0] + p,
                                    j * stride[1] + q,
                                    s,
                                    0])
                                x_in[s][p, q].setAttr(GRB.Attr.UB, input_range_layer[
                                    i * stride[0] + p,
                                    j * stride[1] + q,
                                    s,
                                    1])
                                sum_expr = sum_expr + x_in[s][p, q] * kernel[p, q, s, k]
                        sum_expr = sum_expr + bias[k]
                    model_out_neuron.addConstr(sum_expr == x_out)

                    # define objective: smallest output
                    model_out_neuron.setObjective(x_out, GRB.MINIMIZE)
                    model_out_neuron.setParam('OutputFlag', 0)
                    model_out_neuron.optimize()

                    if model_out_neuron.status == GRB.OPTIMAL:
                        neuron_min = model_out_neuron.objVal
                        # print('lower bound: ' + str(l_neuron))
                        # for variable in prob_min.variables():
                        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                    else:
                        print('prob_min.status: ' + str(model_out_neuron.status))
                        raise ValueError("Error: No result for lower bound for " + str([i, j, s, k]))

                    # define objective: biggest output
                    model_out_neuron.setObjective(x_out, GRB.MAXIMIZE)
                    model_out_neuron.setParam('OutputFlag', 0)
                    model_out_neuron.optimize()

                    if model_out_neuron.status == GRB.OPTIMAL:
                        neuron_max = model_out_neuron.objVal
                        # print('lower bound: ' + str(l_neuron))
                        # for variable in prob_min.variables():
                        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                    else:
                        print('prob_max.status: ' + str(model_out_neuron.status))
                        print('Error: No result for upper bound!')
                    output_range_layer_col.append([neuron_min, neuron_max])
                output_range_layer_row.append(output_range_layer_col)
            output_range_layer.append(output_range_layer_row)

        return np.array(output_range_layer)

    def __output_range_activation_layer_naive(self, layer_index, input_range_layer):
        layer = self.NN.layers[layer_index]
        activation = self.NN.layers[layer_index].activation

        if len(layer.input_dim) == 3:
            # for convolutional layer
            # compute the out range of each neuron by activation function
            output_range_layer = []
            for i in range(input_range_layer.shape[0]):
                output_range_layer_row = []
                for j in range(input_range_layer.shape[1]):
                    output_range_layer_col = []
                    for s in range(input_range_layer.shape[2]):
                        # compute the minimal output
                        neuron_min = activate(activation, input_range_layer[i][j][s][0])
                        # compute the maximal output
                        neuron_max = activate(activation, input_range_layer[i][j][s][1])
                        output_range_layer_col.append([neuron_min, neuron_max])
                    output_range_layer_row.append(output_range_layer_col)
                output_range_layer.append(output_range_layer_row)
        else:
            # for fully connected layer
            # compute the out range of each neuron by activation function
            output_range_layer = []
            for i in range(input_range_layer.shape[0]):
                # compute the minimal output
                neuron_min = activate(activation, input_range_layer[i][0])
                # compute the maximal output
                neuron_max = activate(activation, input_range_layer[i][1])
                output_range_layer.append([neuron_min, neuron_max])

        return np.array(output_range_layer)

    def __output_range_pooling_layer_naive(self, layer_index, input_range_layer):
        layer = self.NN.layers[layer_index]
        filter_size = self.NN.layers[layer_index].filter_size
        pooling_type = self.NN.layers[layer_index].activation
        stride = self.NN.layers[layer_index].stride

        output_range_layer = []

        for i in range(layer.output_dim[0]):
            output_range_layer_row = []
            for j in range(layer.output_dim[1]):
                output_range_layer_col = []
                for s in range(layer.output_dim[2]):
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
                    output_range_layer_col.append([neuron_min, neuron_max])
                output_range_layer_row.append(output_range_layer_col)
            output_range_layer.append(output_range_layer_row)
        return np.array(output_range_layer)

    # flatten layer: just flatten the output
    def __output_range_flatten_layer_naive(self, layer_index, input_range_layer):
        layer = self.NN.layers[layer_index]
        output_range_layer = []
        for s in range(input_range_layer.shape[2]):
            for i in range(input_range_layer.shape[0]):
                # consider the i-th row
                for j in range(input_range_layer.shape[1]):
                    # add the j-th neuron
                    output_range_layer.append(input_range_layer[i, j, s, :])
        return np.array(output_range_layer)

    def __output_range_activation_layer_naive(self, layer_index, input_range_layer):
        layer = self.NN.layers[layer_index]
        activation = self.NN.layers[layer_index].activation
        act = Activation(activation)

        if len(layer.input_dim) == 3:
            # for convolutional layer
            # compute the out range of each neuron by activation function
            output_range_layer = []
            for i in range(input_range_layer.shape[0]):
                output_range_layer_row = []
                for j in range(input_range_layer.shape[
                                   1]):  # input_range_layer.shape[1]=1 for fully-connected layer, input_range_layer.shape[2]=2 for any activation layer
                    output_range_layer_col = []
                    for s in range(input_range_layer.shape[2]):
                        # compute the minimal output
                        neuron_min = act.activate(input_range_layer[i][j][s][0])
                        # compute the maximal output
                        neuron_max = act.activate(input_range_layer[i][j][s][1])
                        output_range_layer_col.append([neuron_min, neuron_max])
                    output_range_layer_row.append(output_range_layer_col)
                output_range_layer.append(output_range_layer_row)
        else:
            # for fully connected layer
            # compute the out range of each neuron by activation function
            output_range_layer = []
            for i in range(input_range_layer.shape[0]):
                # compute the minimal output
                neuron_min = act.activate(input_range_layer[i][0])
                # compute the maximal output
                neuron_max = act.activate(input_range_layer[i][1])
                output_range_layer.append([neuron_min, neuron_max])
        return np.array(output_range_layer)