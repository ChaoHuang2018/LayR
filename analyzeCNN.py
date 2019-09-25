from scipy.special import comb
from sympy import *
from numpy import linalg as LA
from numpy import pi, tanh, array, dot
from scipy.optimize import linprog
from multiprocessing import Pool
from functools import partial
from operator import itemgetter
from gurobipy import *
import cvxpy as cp

import numpy as np
import sympy as sp
import itertools
import math
import random
import time
import copy

##############################################################
# Data structure of a neural network
# NN.type = {'Convolutional', 'Fully_connected'}
# NN.layers: list of layers
"""
New properties:
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
 size_of_inputs: matrix (n*1 matrix for FC network)
 size_of_outputs: n*1 matrix
 num_of_hidden_layers: integer
 scale_factor: number
 offset: number
"""


##############################################################
# output range analysis by MILP relaxation for convolutional neural network
def output_range_MILP_CNN(NN, network_input_box, output_index):

    # initialization of the input range of all the neurons by naive method
    # input range is a list 'matrix'
    input_range_all = []
    input_range_layer_i = network_input_box
    output_range_layer_i_last = network_input_box
    for i in range(NN.num_of_hidden_layers):
        # in the output layer, only take the weight and
        # bias of the 'output_index'-th neuron
        if (
            i == NN.num_of_hidden_layers - 1 and
            NN.layers[i].type == 'Fully_connected'
        ):
            weight_i = np.reshape(NN.layers[i].weight[:, output_index], (-1, 1))
            bias_i = np.array([NN.layers[i].bias[output_index]])
        else:
            weight_i = NN.layers[i].weight
            bias_i = NN.layers[i].bias

        print('-------------layer: {}---------------'.format(i))

        if NN.layers[i].type == 'Convolutional':
            output_range_layer_i = output_range_convolutional_layer_naive_v1(
                NN.layers[i],
                input_range_layer_i,
                NN.layers[i].kernal,
                NN.layers[i].bias,
                NN.layers[i].stride
            )
        if NN.layers[i].type == 'Activation':
            output_range_layer_i = output_range_activation_layer_naive(
                NN.layers[i],
                input_range_layer_i,
                NN.layers[i].activation
            )
        if NN.layers[i].type == 'Pooling':
            output_range_layer_i = output_range_pooling_layer_naive_v1(
                NN.layers[i],
                input_range_layer_i,
                NN.layers[i].filter_size,
                NN.layers[i].activation,
                NN.layers[i].stride
            )
        if NN.layers[i].type == 'Flatten':
            output_range_layer_i = output_range_flatten_layer_naive(
                NN.layers[i],
                input_range_layer_i
            )
        if NN.layers[i].type == 'Fully_connected':
            input_range_layer_i = input_range_fc_layer_naive_v1(
                weight_i,
                bias_i,
                output_range_layer_i_last
            )
            output_range_layer_i = output_range_activation_layer_naive(
                NN.layers[i],
                input_range_layer_i,
                NN.layers[i].activation
            )

        input_range_all.append(input_range_layer_i)
        print('The dimension of input range of layer ' + str(i) + ' :')
        print(input_range_layer_i.shape)

        input_range_layer_i = output_range_layer_i
        output_range_layer_i_last = output_range_layer_i

    print('-------Naive range for each neuron is generated.----------')
    print('-------MILP based range analysis begins.----------')

    # only invoke our approach once to obtain the output range
    # with the basic refinement degree
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
                    refinement_degree_layer_channel.append(refinement_degree_layer_row)
                refinement_degree_layer.append(refinement_degree_layer_channel)
            refinement_degree_all.append(refinement_degree_layer)
        if len(NN.layers[k].input_dim) == 1:
            for i in range(NN.layers[k].output_dim[0]):
                refinement_degree_layer.append(1)
            refinement_degree_all.append(refinement_degree_layer)

    input_range_last_neuron, _ = neuron_input_range_cnn(
        NN,
        NN.num_of_hidden_layers-1,
        0,
        network_input_box,
        input_range_all,
        refinement_degree_all
    )

    lower_bound = activate(NN.layers[i].activation, input_range_last_neuron[0])
    upper_bound = activate(NN.layers[i].activation, input_range_last_neuron[1])

    return [lower_bound, upper_bound]


# define large positive number M to enable Big M method
M = 10e4
# Compute the input range for a specific neuron
# and return the updated input_range_all

# When layer_index = layers,
# this function outputs the output range of the neural network
def neuron_input_range_cnn(NN, layer_index, neuron_index, network_input_box, input_range_all, refinement_degree_all):

    layers = NN.layers


    # variables in the input layer
    if NN.type == 'Convolutional':
        network_in = []
        for s in range(NN.layers[0].input_dim[2]):
            network_in.append(cp.Variable((NN.layers[0].input_dim[0],NN.layers[0].input_dim[1])))
    else:
        network_in = cp.Variable(NN.layers[0].input_dim[0])

    # variables in previous layers
    x_in = []
    x_out = []
    z0 = []
    z1 = []
    z_pooling = []
    for k in range(layer_index):
        print(NN.layers[k].type + ', input dimension: ')
        print(NN.layers[k].input_dim)
        print(NN.layers[k].type + ', output dimension: ')
        print(NN.layers[k].output_dim)
        if NN.layers[k].type == 'Convolutional':
            x_in_layer = []
            for s in range(NN.layers[k].input_dim[2]):
                x_in_layer.append(cp.Variable((NN.layers[k].input_dim[0],NN.layers[k].input_dim[1])))
            x_in.append(x_in_layer)
            x_out_layer = []
            for s in range(NN.layers[k].output_dim[2]):
                x_out_layer.append(cp.Variable((NN.layers[k].output_dim[0], NN.layers[k].output_dim[1])))
            x_out.append(x_out_layer)
            z0.append([])
            z1.append([])
            z_pooling.append([])
        if NN.layers[k].type == 'Activation':
            # define input and output variables
            x_in_layer = []
            for s in range(NN.layers[k].input_dim[2]):
                x_in_layer.append(cp.Variable((NN.layers[k].input_dim[0],NN.layers[k].input_dim[1])))
            x_in.append(x_in_layer)
            x_out_layer = []
            for s in range(NN.layers[k].output_dim[2]):
                x_out_layer.append(cp.Variable((NN.layers[k].output_dim[0], NN.layers[k].output_dim[1])))
            x_out.append(x_out_layer)
            # define slack binary variables
            z0_layer = []
            z1_layer = []
            for s in range(NN.layers[k].input_dim[2]):
                z0_channel = []
                z1_channel = []
                for i in range(NN.layers[k].input_dim[0]):
                    z0_row = []
                    z1_row = []
                    for j in range(NN.layers[k].input_dim[1]):
                        z0_row.append(cp.Variable(refinement_degree_all[k][s][i][j],boolean=True))
                        z1_row.append(cp.Variable(refinement_degree_all[k][s][i][j],boolean=True))
                    z0_channel.append(z0_row)
                    z1_channel.append(z1_row)
                z0_layer.append(z0_channel)
                z1_layer.append(z1_channel)
            z0.append(z0_layer)
            z1.append(z1_layer)
            z_pooling.append([])
        if NN.layers[k].type == 'Pooling':
            # define input and output variables
            x_in_layer = []
            for s in range(NN.layers[k].input_dim[2]):
                x_in_layer.append(cp.Variable((NN.layers[k].input_dim[0],NN.layers[k].input_dim[1])))
            x_in.append(x_in_layer)
            x_out_layer = []
            for s in range(NN.layers[k].output_dim[2]):
                x_out_layer.append(cp.Variable((NN.layers[k].output_dim[0], NN.layers[k].output_dim[1])))
            x_out.append(x_out_layer)

            z0.append([])
            z1.append([])
            # define slack binary variables for max operation
            if NN.layers[k].activation == 'max':
                z_layer = []
                for s in range(NN.layers[k].input_dim[2]):
                    z_channel = []
                    for i in range(NN.layers[k].input_dim[0]):
                        z_row = []
                        for j in range(NN.layer[k].input_dim[1]):
                            z_row.append(cp.Variable((NN.layers[k].filter_size[0],NN.layers[k].filter_size[1]),boolean=True))
                        z_channel.append(z_row)
                    z_layer.append(z_channel)
                z_pooling.append(z_layer)
            else:
                z_pooling.append([])
        if NN.layers[k].type == 'Flatten':
            # define input and output variables
            x_in_layer = []
            for s in range(NN.layers[k].input_dim[2]):
                x_in_layer.append(cp.Variable((NN.layers[k].input_dim[0],NN.layers[k].input_dim[1])))
            x_in.append(x_in_layer)
            x_out_layer = cp.Variable(NN.layers[k].output_dim[0])
            x_out.append(x_out_layer)

            z0.append([])
            z1.append([])
            z_pooling.append([])
        if NN.layers[k].type == 'Fully_connected':
            # Notice that here the dimension of x_in should be the same as the one of x_out, which is not the one of the output of the previous layer
            x_in_layer = cp.Variable(NN.layers[k].output_dim[0])
            x_in.append(x_in_layer)
            x_out_layer = cp.Variable(NN.layers[k].output_dim[0])
            x_out.append(x_out_layer)
            # define slack binary variables
            z0_layer = []
            z1_layer = []
            for i in range(NN.layers[k].output_dim[0]):
                z0_layer.append(cp.Variable(refinement_degree_all[k][i],boolean=True))
                z1_layer.append(cp.Variable(refinement_degree_all[k][i],boolean=True))
            z0.append(z0_layer)
            z1.append(z1_layer)
            z_pooling.append([])

    # variables for the specific neuron
    x_in_neuron = cp.Variable()

    constraints = []

    # add constraints for the input layer
    if NN.type == 'Convolutional':
        for s in range(NN.layers[0].input_dim[2]):
            for i in range(NN.layers[0].input_dim[0]):
                for j in range(NN.layers[0].input_dim[1]):
                    constraints += [network_in[s][i,j] >= network_input_box[i][j][s][0]]
                    constraints += [network_in[s][i,j] <= network_input_box[i][j][s][1]]
    else:
        for i in range(NN.layers[0].input_dim[0]):
            constraints += [network_in[i] >= network_input_box[i][0]]
            constraints += [network_in[i] <= network_input_box[i][1]]


    # add constraints for the layers before the neuron
    for k in range(layer_index):

        if NN.layers[k].type == 'Convolutional':
            constraints += relaxation_convolutional_layer(NN.layers[k], x_in[k], x_out[k], NN.layers[k].kernal, NN.layers[k].bias, NN.layers[k].stride)
            if k == 0:
                for s in range(NN.layers[k].input_dim[2]):
                    constraints += [network_in[s] == x_in[k][s]]
            else:
                for s in range(NN.layers[k].input_dim[2]):
                    constraints += [x_in[k][s] == x_out[k-1][s]]
        if NN.layers[k].type == 'Activation':
            constraints += relaxation_activation_layer(NN.layers[k], x_in[k], x_out[k], z0[k], z1[k], input_range_all[k], NN.layers[k].activation, refinement_degree_all[k])
            if k == 0:
                for s in range(NN.layers[k].input_dim[2]):
                    constraints += [network_in[s] == x_in[k][s]]
            else:
                for s in range(NN.layers[k].input_dim[2]):
                    constraints += [x_in[k][s] == x_out[k-1][s]]
        if NN.layers[k].type == 'Pooling':
            constraints += relaxation_pooling_layer(NN.layers[k], x_in[k], x_out[k], z_pooling[k], NN.layers[k].filter_size, NN.layers[k].activation, NN.layers[k].stride)
            if k == 0:
                for s in range(NN.layers[k].input_dim[2]):
                    constraints += [network_in[s] == x_in[k][s]]
            else:
                for s in range(NN.layers[k].input_dim[2]):
                    constraints += [x_in[k][s] == x_out[k-1][s]]
        if NN.layers[k].type == 'Flatten':
            constraints += relaxation_flatten_layer(NN.layers[k], x_in[k], x_out[k])
            if k == 0:
                for s in range(NN.layers[k].input_dim[2]):
                    constraints += [network_in[s] == x_in[k][s]]
            else:
                for s in range(NN.layers[k].input_dim[2]):
                    constraints += [x_in[k][s] == x_out[k-1][s]]
        if NN.layers[k].type == 'Fully_connected':
            constraints += relaxation_activation_layer(NN.layers[k], x_in[k], x_out[k], z0[k], z1[k], input_range_all[k], NN.layers[k].activation, refinement_degree_all[k])
            # add constraint for linear transformation between layers
            weight_k = NN.layers[k].weight
            bias_k = NN.layers[k].bias
            if k == 0:
                constraints += [x_in[k] == weight_k.T @ network_in + bias_k]
            else:
                constraints += [x_in[k] == weight_k.T @ x_out[k-1] + bias_k]


    # add constraint for the last layer and the neuron
    # Notice that we only need to handle activation function layer. For other layers, update the input range of the neuron does not improve the result (which is equivalant in fact)
    if NN.layers[layer_index].type == 'Activation':
        constraints += [x_in_neuron == x_out[layer_index-1][neuron_index[2]][neuron_index[0],neuron_index[1]]]
    elif NN.layers[layer_index].type == 'Fully_connected':
        weight_neuron = np.reshape(NN.layers[layer_index].weight[:, neuron_index], (-1, 1))
        bias_neuron = NN.layers[layer_index].bias[neuron_index]
        print(x_in_neuron.shape)
        print(weight_neuron.shape)
        print(x_out[layer_index-1].shape)
        print(bias_neuron.shape)
        if layer_index >= 1:
            constraints += [x_in_neuron == weight_neuron.T @ x_out[layer_index-1] + bias_neuron]
        else:
            constraints += [x_in_neuron == weight_neuron.T @ network_in + bias_neuron]
    else:
        print('No need to update the input range of this neuron')
        return input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]], input_range_all


    # objective: smallest output of [layer_index, neuron_index]
    objective_min = cp.Minimize(x_in_neuron)

    prob_min = cp.Problem(objective_min, constraints)
    prob_min.solve(solver=cp.GUROBI)

    if prob_min.status == 'optimal':
        l_neuron = prob_min.value
        #print('lower bound: ' + str(l_neuron))
        #for variable in prob_min.variables():
        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
    else:
        print('prob_min.status: ' + prob_min.status)
        print('Error: No result for lower bound!')

    # objective: largest output of [layer_index, neuron_index]
    objective_max = cp.Maximize(x_in_neuron)
    prob_max = cp.Problem(objective_max, constraints)
    prob_max.solve(solver=cp.GUROBI)

    if prob_max.status == 'optimal':
        u_neuron = prob_max.value
        #print('upper bound: ' + str(u_neuron))
        #for variable in prob_max.variables():
        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
    else:
        print('prob_max.status: ' + prob_max.status)
        print('Error: No result for upper bound!')

    if NN.layers[layer_index].type == 'Activation':
        input_range_all[layer_index][neuron_index[2]][neuron_index[0]][neuron_index[1]] = [l_neuron, u_neuron]
    elif NN.layers[layer_index].type == 'Fully_connected':
        input_range_all[layer_index][neuron_index] = [l_neuron, u_neuron]



    return [l_neuron, u_neuron], input_range_all


##############################################################
# output range analysis by MILP relaxation for fully connected neural network
#  network_input_box: list of a two-element list
def output_range_MILP_FC(NN, network_input_box, output_index):


    # initialization of the input range of all the neurons by naive method
    # input range is a list 'matrix'
    input_range_all = []
    output_range_last_layer = network_input_box
    for i in range(NN.num_of_hidden_layers):
        # in the output layer, only take the weight of bias of the 'output_index'-th neuron
        if i < NN.num_of_hidden_layers - 1:
            weight_i = NN.layers[i].weight
            bias_i = NN.layers[i].bias
        else:
            weight_i = np.reshape(NN.layers[i].weight[output_index], (1, -1))
            bias_i = np.reshape(NN.layers[i].bias[output_index], (1, -1))

        input_range_layer = input_range_fc_layer_naive(weight_i, bias_i, output_range_last_layer)
        #print(input_range_layer[0][1][0])
        #print('range of layer ' + str(j) + ': ' + str(input_range_layer))
        input_range_all.append(input_range_layer)
        output_range_last_layer, _ = output_range_layer(weight_i, bias_i, output_range_last_layer, NN.layer[i].activation)
    print("intput range by naive method: " + str([input_range_layer[0][0], input_range_layer[0][1]]))
    print("Output range by naive method: " + str([(output_range_last_layer[0][0]-NN.offset)*NN.scale_factor, (output_range_last_layer[0][1]-NN.offset)*NN.scale_factor]))

    layer_index = 1
    neuron_index = 0

    #print('Output range by naive test: ' + str([input_range_all[layer_index][neuron_index]]))
    # compute by milp relaxation
    network_update_input,_ = neuron_input_range_fc(NN, layer_index, neuron_index, network_input_box, input_range_all)
    print("Output range by MILP relaxation: " + str([(sigmoid(network_update_input[0])-NN.offset)*NN.scale_factor, (sigmoid(network_update_input[1])-NN.offset)*NN.scale_factor]))

    range_update = copy.deepcopy(input_range_all)
    for i in range(NN.num_of_hidden_layers):
        for j in range(NN.layers.input_dim[0]):
            _, range_update = neuron_input_range_fc(NN, i, j, network_input_box, range_update)
    print(str(range_update[-1]))


    lower_bound = (activate(NN.layers[i].activation, range_update[-1][0][0])-NN.offset)*NN.scale_factor
    upper_bound = (activate(NN.layers[i].activation, range_update[-1][0][1])-NN.offset)*NN.scale_factor
    print(str([lower_bound,upper_bound]))

    return lower_bound, upper_bound


# Compute the input range for a specific neuron and return the updated input_range_all
# When layer_index = layers, this function outputs the output range of the neural network
def neuron_input_range_fc(NN, layer_index, neuron_index, network_input_box, input_range_all):

    layers = NN.layers

    # define large positive number M to enable Big M method
    M = 10e4
    # variables in the input layer
    network_in = cp.Variable((NN.size_of_inputs[0],NN.size_of_inputs[1]))
    # variables in previous layers
    x_in = {}
    x_out = {}
    z0 = []
    z1 = []
    for i in range(layer_index):
        x_in[i] = cp.Variable((NN.input_dim[i][0], NN.input_dim[i][1]))
        x_out[i] = cp.Variable((NN.output_dim[i][0], NN.output_dim[i][1]))
        # define slack integer variables only for activation layers and fully-connected layers
        if NN.layers[i].type == 'Fully_connected' or 'Activation':
            z0_layer = []
            z1_layer = []
            for j in range(NN.layer[i].input_dim[0]):
                z0_row = []
                z1_row = []
                for k in range(NN.layer[i].input_dim[1]):
                    z0_row.append(refinement_degree_all[i][j][k],cp.Variable(boolean=True))
                    z1_row.append(refinement_degree_all[i][j][k],cp.Variable(boolean=True))
                z0_layer.append(z0_row)
                z1_layer.append(z1_row)
            z0.append(z0_layer)
            z1.append(z1_layer)
        else:
            z0[i] = []
            z1[i] = []
    # variables for the specific neuron
    x_in_neuron = cp.Variable()

    constraints = []

    # add constraints for the input layer
    for i in range(NN.num_of_inputs):
        constraints += [network_in[i,0] >= network_input_box[i][0]]
        constraints += [network_in[i,0] <= network_input_box[i][1]]


    # add constraints for the layers before the neuron
    for i in range(layer_index):

        # add constraint for linear transformation between layers
        weight_i = NN.layers[i].weight
        bias_i = NN.layers[i].bias
        if i == 0:
            constraints += [x_in[i] == weight_i @ network_in + bias_i]
        else:
            constraints += [x_in[i] == weight_i @ x_out[i-1] + bias_i]

        # add constraint for activation function relaxation
        constraints += relaxation_activation_layer(x_in[i], x_out[i], z0[i], z1[i], input_range_all[i], activation, refinement_degree_all[i])


    # add constraint for the last layer and the neuron
    weight_neuron = np.reshape(NN.layers[layer_index].weight[neuron_index], (1, -1))
    bias_neuron = np.reshape(NN.layers[layer_index].bias[neuron_index], (1, -1))
    #print(x_in_neuron.shape)
    #print(weight_neuron.shape)
    #print(x_out[0:len(bias_all_layer[layer_index-1]),layer_index-1:layer_index].shape)
    #print(bias_neuron.shape)
    if layer_index >= 1:
        constraints += [x_in_neuron == weight_neuron @ x_out[layer_index-1] + bias_neuron]
    else:
        constraints += [x_in_neuron == weight_neuron @ network_in + bias_neuron]

    # objective: smallest output of [layer_index, neuron_index]
    objective_min = cp.Minimize(x_in_neuron)

    prob_min = cp.Problem(objective_min, constraints)
    prob_min.solve(solver=cp.GUROBI)

    if prob_min.status == 'optimal':
        l_neuron = prob_min.value
        #print('lower bound: ' + str(l_neuron))
        #for variable in prob_min.variables():
        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
    else:
        print('prob_min.status: ' + prob_min.status)
        print('Error: No result for lower bound!')

    # objective: largest output of [layer_index, neuron_index]
    objective_max = cp.Maximize(x_in_neuron)
    prob_max = cp.Problem(objective_max, constraints)
    prob_max.solve(solver=cp.GUROBI)

    if prob_max.status == 'optimal':
        u_neuron = prob_max.value
        #print('upper bound: ' + str(u_neuron))
        #for variable in prob_max.variables():
        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
    else:
        print('prob_max.status: ' + prob_max.status)
        print('Error: No result for upper bound!')

    input_range_all[layer_index][neuron_index] = [l_neuron, u_neuron]
    return [l_neuron, u_neuron], input_range_all

#############################################################################################
# Derive the input range of a fully-connected layer by
# Only in fully-connect layer, the input of the layer is different of the output of the previous layer
#
def input_range_fc_layer_naive_v1(weight, bias, output_range_last_layer):
    # compute the input range of each neuron by solving LPs
    input_range_layer = []

    for i in range(weight.shape[1]):
        x_in = cp.Variable()
        x_out = cp.Variable(weight.shape[0])

        weight_i = np.reshape(weight[:, i], (-1, 1))
        bias_i = bias[i]

        # define constraints: linear transformation by weight and bias
        constraints = [weight_i.T @ x_out + bias_i == x_in]
        # define constraints: output range of the last layer
        constraints += [x_out >= output_range_last_layer[:, 0], x_out <= output_range_last_layer[:, 1]]

        # define objective: smallest output of [layer_index, neuron_index]
        objective_min = cp.Minimize(x_in)

        prob_min = cp.Problem(objective_min, constraints)
        prob_min.solve(solver=cp.GUROBI)

        if prob_min.status == 'optimal':
            neuron_min = prob_min.value
            #print('lower bound: ' + str(l_neuron))
            #for variable in prob_min.variables():
            #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
        else:
            print('prob_min.status: ' + prob_min.status)
            print('Error: No result for lower bound!')

        # define objective: smallest output of [layer_index, neuron_index]
        objective_max = cp.Maximize(x_in)

        prob_max = cp.Problem(objective_max, constraints)
        prob_max.solve(solver=cp.GUROBI)

        if prob_max.status == 'optimal':
            neuron_max = prob_max.value
            #print('lower bound: ' + str(l_neuron))
            #for variable in prob_min.variables():
            #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
        else:
            print('prob_max.status: ' + prob_max.status)
            print('Error: No result for upper bound!')

        input_range_layer.append([neuron_min, neuron_max])
    return np.array(input_range_layer)


def input_range_fc_layer_naive(weight, bias, output_range_last_layer):
    # compute the input range of each neuron by solving LPs
    input_range_layer = []

    # define output variables of the last layer
    x_out = cp.Variable(weight.shape[0])
    # define input variables of the this layer
    x_in = cp.Variable(weight.shape[1])


    # define constraints: linear transformation by weight and bias
    constraints = [weight.T @ x_out + bias == x_in]

    # define constraints: output range of the last layer
    for i in range(weight.shape[0]):
        constraints += [x_out[i] >= output_range_last_layer[i][0], x_out[i] <= output_range_last_layer[i][1]]

    for i in range(weight.shape[1]):
        input_range_layer_i = []

        # define objective: smallest output of [layer_index, neuron_index]
        objective_min = cp.Minimize(x_in[i])

        prob_min = cp.Problem(objective_min, constraints)
        prob_min.solve(solver=cp.GUROBI)

        if prob_min.status == 'optimal':
            neuron_i_min = prob_min.value
            #print('lower bound: ' + str(l_neuron))
            #for variable in prob_min.variables():
            #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
        else:
            print('prob_min.status: ' + prob_min.status)
            print('Error: No result for lower bound!')

        # define objective: smallest output of [layer_index, neuron_index]
        objective_max = cp.Maximize(x_in[i])

        prob_max = cp.Problem(objective_max, constraints)
        prob_max.solve(solver=cp.GUROBI)

        if prob_max.status == 'optimal':
            neuron_i_max = prob_max.value
            #print('lower bound: ' + str(l_neuron))
            #for variable in prob_min.variables():
            #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
        else:
            print('prob_max.status: ' + prob_max.status)
            print('Error: No result for upper bound!')

        input_range_layer.append([neuron_i_min, neuron_i_max])
    return np.array(input_range_layer)

# abandon
def input_range_flatten_layer_naive(output_range_last_layer):
    input_range_layer = []
    for s in range(output_range_last_layer.shape[2]):
        for i in range(output_range_last_layer.shape[0]):
            # consider the i-th row
            for j in range(output_range_last_layer.shape[1]):
                # add the j-th neuron
                input_range_layer.append(output_range_last_layer[i,j,s,:])
    return np.array(input_range_layer)



#############################################################################################
## Derive the output ranges of different layers

# Convolutional layer
def output_range_convolutional_layer_naive_v1(layer, input_range_layer, kernal, bias, stride):
    output_range_layer = []

    for i in range(layer.output_dim[0]):
        output_range_layer_row = []
        for j in range(layer.output_dim[1]):
            output_range_layer_col = []
            for k in range(layer.output_dim[2]):
                x_in = []
                for s in range(layer.input_dim[2]):
                    x_in.append(cp.Variable((kernal.shape[0],kernal.shape[1])))
                x_out = cp.Variable()
                constraints = []
                sum_expr = 0
                for s in range(layer.input_dim[2]):
                    constraints += [
                        x_in[s] >=
                        input_range_layer[
                            i * stride[0] : i * stride[0] + kernal.shape[0],
                            j * stride[1] : j * stride[1] + kernal.shape[1],
                            s,
                            0],
                        x_in[s] <=
                        input_range_layer[
                            i * stride[0] : i * stride[0] + kernal.shape[0],
                            j * stride[1] : j * stride[1] + kernal.shape[1],
                            s,
                            1
                        ]
                    ]
                    temp_in = cp.vec(x_in[s][0:kernal.shape[0],0:kernal.shape[1]])
                    temp_kernal = cp.vec(kernal[:,:,s,k])
                    sum_expr = sum_expr + temp_kernal @ temp_in + bias[k]
                constraints += [sum_expr == x_out]

                objective_min = cp.Minimize(x_out)
                prob_min = cp.Problem(objective_min, constraints)
                prob_min.solve(solver=cp.GUROBI)

                if prob_min.status == 'optimal':
                    neuron_min = prob_min.value
                    #print('lower bound: ' + str(l_neuron))
                    #for variable in prob_min.variables():
                    #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                else:
                    print('prob_min.status: ' + prob_min.status)
                    print('Error: No result for lower bound!')

                # define objective: smallest output
                objective_max = cp.Maximize(x_out)
                prob_max = cp.Problem(objective_max, constraints)
                prob_max.solve(solver=cp.GUROBI)

                if prob_max.status == 'optimal':
                    neuron_max = prob_max.value
                    #print('lower bound: ' + str(l_neuron))
                    #for variable in prob_min.variables():
                    #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                else:
                    print('prob_max.status: ' + prob_max.status)
                    print('Error: No result for upper bound!')
                output_range_layer_col.append([neuron_min, neuron_max])
            output_range_layer_row.append(output_range_layer_col)
        output_range_layer.append(output_range_layer_row)

    return np.array(output_range_layer)


# Convolutional layer
# input range and output range should be 4-dimesional
def output_range_convolutional_layer_naive(layer, input_range_layer, kernal, bias, stride):


    # define input variables of this layer
    x_in = []
    for s in range(layer.input_dim[2]):
        x_in.append(cp.Variable((layer.input_dim[0],layer.input_dim[1])))
    # define out variables of the this layer
    x_out = []
    for s in range(layer.output_dim[2]):
        x_out.append(cp.Variable((layer.output_dim[0], layer.output_dim[1])))

    # define constraints
    constraints = []

    # add constraints: input range of this layer
    for s in range(layer.input_dim[2]):
        for i in range(layer.input_dim[0]):
            for j in range(layer.input_dim[1]):
                constraints += [x_in[s][i,j] >= input_range_layer[i][j][s][0], x_in[s][i,j] <= input_range_layer[i][j][s][1]]

    # add constraints: convolutional operation
    for k in range(layer.output_dim[2]):
        for i in range(0, x_in[0].shape[0]-kernal.shape[0]+1, stride):
            for j in range(0, x_in[0].shape[1]-kernal.shape[1]+1, stride):
                sum_expr = 0
                for s in range(layer.input_dim[2]):
                    temp_in = cp.vec(x_in[s][i:i+kernal.shape[0],j:j+kernal.shape[1]])
                    temp_kernal = cp.vec(kernal[:,:,s,k])
                    sum_expr = sum_expr + temp_kernal @ temp_in + bias[k]
                constraints += [sum_expr == x_out[k][i,j]]


    # compute the range of each neuron
    output_range_layer = []

    for i in range(layer.output_dim[0]):
        output_range_layer_row = []
        for j in range(layer.output_dim[1]):
            output_range_layer_col = []
            for k in range(layer.output_dim[2]):
                objective_min = cp.Minimize(x_out[k][i,j])
                prob_min = cp.Problem(objective_min, constraints)
                prob_min.solve(solver=cp.GUROBI)

                if prob_min.status == 'optimal':
                    neuron_min = prob_min.value
                    #print('lower bound: ' + str(l_neuron))
                    #for variable in prob_min.variables():
                    #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                else:
                    print('prob_min.status: ' + prob_min.status)
                    print('Error: No result for lower bound!')

                # define objective: smallest output
                objective_max = cp.Maximize(x_out[k][i,j])
                prob_max = cp.Problem(objective_max, constraints)
                prob_max.solve(solver=cp.GUROBI)

                if prob_max.status == 'optimal':
                    neuron_max = prob_max.value
                    #print('lower bound: ' + str(l_neuron))
                    #for variable in prob_min.variables():
                    #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                else:
                    print('prob_max.status: ' + prob_max.status)
                    print('Error: No result for upper bound!')
                output_range_layer_col.append([neuron_min, neuron_max])
            output_range_layer_row.append(output_range_layer_col)
        output_range_layer.append(output_range_layer_row)

    return output_range_layer

# Pooling layer
def output_range_pooling_layer_naive_v1(layer, input_range_layer, filter_size, pooling_type, stride):
    output_range_layer = []

    for i in range(layer.output_dim[0]):
        output_range_layer_row = []
        for j in range(layer.output_dim[1]):
            output_range_layer_col = []
            for s in range(layer.output_dim[2]):
                x_in = cp.Variable((filter_size[0],filter_size[1]))
                x_out = cp.Variable()
                constraints = [
                    x_in >=
                    input_range_layer[
                        i * stride[0] : i * stride[0] + filter_size[0],
                        j * stride[1] : j * stride[1] + filter_size[1],
                        s,
                        0
                    ],
                    x_in <=
                    input_range_layer[
                        i * stride[0] : i * stride[0] + filter_size[0],
                        j * stride[1] : j * stride[1] + filter_size[1],
                        s,
                        1
                    ]
                ]
                if pooling_type == 'max':
                    #constraints += [cp.max(x_in) == x_out] # The max operation is convex but not affine, thus can be not be used in an equation constraint in DCP.
                    # Instead, we directly derive the constraint for max pooling
                    neuron_min = np.max(input_range_layer[
                        i * stride[0] : i * stride[0] + filter_size[0],
                        j * stride[1] : j * stride[1] + filter_size[1],
                        s,
                        0
                    ])
                    neuron_max = np.max(input_range_layer[
                        i * stride[0] : i * stride[0] + filter_size[0],
                        j * stride[1] : j * stride[1] + filter_size[1],
                        s,
                        1
                    ])
                if pooling_type == 'average':
                    constraints += [cp.sum(x_in)/(filter_size[0]*filter_size[1]) == x_out]

                    objective_min = cp.Minimize(x_out)
                    prob_min = cp.Problem(objective_min, constraints)
                    prob_min.solve(solver=cp.GUROBI)

                    if prob_min.status == 'optimal':
                        neuron_min = prob_min.value
                        #print('lower bound: ' + str(l_neuron))
                        #for variable in prob_min.variables():
                        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                    else:
                        print('prob_min.status: ' + prob_min.status)
                        print('Error: No result for lower bound!')

                    # define objective: smallest output
                    objective_max = cp.Maximize(x_out)
                    prob_max = cp.Problem(objective_max, constraints)
                    prob_max.solve(solver=cp.GUROBI)

                    if prob_max.status == 'optimal':
                        neuron_max = prob_max.value
                        #print('lower bound: ' + str(l_neuron))
                        #for variable in prob_min.variables():
                        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                    else:
                        print('prob_max.status: ' + prob_max.status)
                        print('Error: No result for upper bound!')
                output_range_layer_col.append([neuron_min, neuron_max])
            output_range_layer_row.append(output_range_layer_col)
        output_range_layer.append(output_range_layer_row)

    return np.array(output_range_layer)

# Pooling layer
# input range and output range should be 4-dimesional
def output_range_pooling_layer_naive(layer, input_range_layer, filter_size, pooling_type):

    # define input variables of this layer
    x_in = {}
    for s in range(layer.input_dim[2]):
        x_in[s] = cp.Variable((layer.input_dim[0],layer.input_dim[1]))
    # define out variables of the this layer
    x_out = {}
    for s in range(layer.output_dim[2]):
        x_out = cp.Variable((layer.output_dim[0], layer.output_dim[1]))

    # define constraints
    constraints = []

    # add constraints: input range of this layer
    for s in range(layer.input_dim[2]):
        for i in range(layer.input_dim[0]):
            for j in range(layer.input_dim[1]):
                constraints += [x_in[s][i,j] >= input_range_layer[i][j][s][0], x_in[i,j] <= input_range_layer[i][j][s][1]]

    # add constraints: convolutional operation
    if pooling_type == 'max':
        for s in range(layer.output_dim[2]):
            for i in range(round(x_in.shape[0]/filter_size[0])):
                for j in range(round(x_in.shape[1]/filter_size[1])):
                    constraints += [cp.max(x_in[s][i*filter_size[0]:i*(filter_size[0]+1), j*filter_size[1]:j*(filter_size[1]+1)]) == x_out[s][i,j]]
    if pooling_type == 'average':
        for s in range(layer.output_dim[2]):
            for i in range(round(x_in.shape[0]/filter_size[0])):
                for j in range(round(x_in.shape[1]/filter_size[1])):
                    constraints += [cp.sum(x_in[s][i*filter_size[0]:i*(filter_size[0]+1), j*filter_size[1]:j*(filter_size[1]+1)])/(filter_size[0]*filter_size[1]) == x_out[s][i,j]]

    # compute the range of each neuron
    output_range_layer = []

    for i in range(layer.output_dim[0]):
        output_range_layer_row = []
        for j in range(layer.output_dim[1]):
            output_range_layer_col = []
            for k in range(layer.output_dim[2]):
                objective_min = cp.Minimize(x_out[k][i,j])
                prob_min = cp.Problem(objective_min, constraints)
                prob_min.solve(solver=cp.GUROBI)

                if prob_min.status == 'optimal':
                    neuron_min = prob_min.value
                    #print('lower bound: ' + str(l_neuron))
                    #for variable in prob_min.variables():
                    #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                else:
                    print('prob_min.status: ' + prob_min.status)
                    print('Error: No result for lower bound!')

                # define objective: smallest output
                objective_max = cp.Maximize(x_out[k][i,j])
                prob_max = cp.Problem(objective_max, constraints)
                prob_max.solve(solver=cp.GUROBI)

                if prob_max.status == 'optimal':
                    neuron_max = prob_max.value
                    #print('lower bound: ' + str(l_neuron))
                    #for variable in prob_min.variables():
                    #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
                else:
                    print('prob_max.status: ' + prob_max.status)
                    print('Error: No result for upper bound!')
                output_range_layer_col.append([neuron_min, neuron_max])
            output_range_layer_row.append(output_range_layer_col)
        output_range_layer.append(output_range_layer_row)
    return output_range_layer

# flatten layer: just flatten the output
def output_range_flatten_layer_naive(layer, input_range_layer):
    output_range_layer = []
    for s in range(input_range_layer.shape[2]):
        for i in range(input_range_layer.shape[0]):
            # consider the i-th row
            for j in range(input_range_layer.shape[1]):
                # add the j-th neuron
                output_range_layer.append(input_range_layer[i,j,s,:])
    return np.array(output_range_layer)

# general activation layer
def output_range_activation_layer_naive(layer, input_range_layer, activation):

    if len(layer.input_dim) == 3:
        # for convolutional layer
        # compute the out range of each neuron by activation function
        output_range_layer = []
        for i in range(input_range_layer.shape[0]):
            output_range_layer_row = []
            for j in range(input_range_layer.shape[1]): # input_range_layer.shape[1]=1 for fully-connected layer, input_range_layer.shape[2]=2 for any activation layer
                output_range_layer_col = []
                for s in range(input_range_layer.shape[2]):
                    # compute the minimal output
                    neuron_min = activate(activation, input_range_layer[i][j][s][0])
                    # compute the maximal output
                    neuron_max = activate(activation, input_range_layer[i][j][s][1])
                    output_range_layer_col.append([neuron_min,neuron_max])
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
            output_range_layer.append([neuron_min,neuron_max])

    return np.array(output_range_layer)

#############################################################################################
## Constraints of MILP relaxation for different layers
# convolutional layer
# x_in should be 3-dimensional, x_out should be 3-dimensional
def relaxation_convolutional_layer(layer, x_in, x_out, kernal, bias, stride):
    constraints = []
    for i in range(layer.output_dim[0]):
        for j in range(layer.output_dim[1]):
            for k in range(layer.output_dim[2]):
                sum_expr = 0
                for s in range(layer.input_dim[2]):
                    temp_in = cp.vec(x_in[s][i * stride[0] : i * stride[0] + kernal.shape[0],j * stride[1] : j * stride[1] + kernal.shape[1]])
                    temp_kernal = cp.vec(kernal[:,:,s,k])
                    sum_expr = sum_expr + temp_kernal @ temp_in + bias[k]
                constraints += [sum_expr == x_out[k][i,j]]
    return constraints


# pooling layer
# x_in should be 3-dimensional, x_out should be 3-dimensional
def relaxation_pooling_layer(layer, x_in, x_out, z_pooling, filter_size, pooling_type, stride):
    constraints = []
    if pooling_type == 'max':
        for s in range(layer.input_dim[2]):
            for i in range(round(layer.input_dim[0]/filter_size[0])):
                for j in range(round(layer.input_dim[1]/filter_size[1])):
                    # big-M relaxation for max operation
                    constraints += [cp.max(x_in[s][i*stride[0]:i*stride[0]+filter_size[0], j*stride[1]:j*stride[1]+filter_size[1]]) <= x_out[s][i,j]]
                    constraints += [cp.sum(z_pooling[s][i][j]) == 1]
                    for m in range(filter_size[0]):
                        for n in range(filter_size[1]):
                            constraints += [x_out[s][i,j] - x_in[s][i*stride[0]+m, j*stride[1]+n] <= M * (1 - z_pooling[s][i][j][m,n])]
    if pooling_type == 'average':
        for s in range(layer.input_dim[2]):
            for i in range(round(layer.input_dim[0]/filter_size[0])):
                for j in range(round(layer.input_dim[1]/filter_size[1])):
                    constraints += [cp.sum(x_in[s][i*filter_size[0]:i*(filter_size[0]+1), j*filter_size[1]:j*(filter_size[1]+1)])/(filter_size[0]*filter_size[1]) == x_out[s][i,j]]
    return constraints

# flatten layer
# x_in should be 3-dimensional, x_out should be 1-dimensional
def relaxation_flatten_layer(layer, x_in, x_out):
    constraints = []
    k = 0
    for s in range(layer.input_dim[2]):
        for i in range(layer.input_dim[0]):
            for j in range(layer.input_dim[1]):
                constraints += [x_in[s][i,j] == x_out[k]]
                k = k + 1
    return constraints


# Relu/tanh/sigmoid activation layer
# Note the difference between the activation layer following the convolutional layer and the one in fully-connected layer
def relaxation_activation_layer(layer, x_in, x_out, z0, z1, input_range_layer, activation, refinement_degree_layer):

    constraints = []

    if layer.type == 'Activation':
        # if x_in is three-dimensional, which means this is a convolutional layer
        for s in range(layer.input_dim[2]):
            for i in range(layer.input_dim[0]):
                for j in range(layer.input_dim[1]):
                    low = input_range_layer[i][j][s][0]
                    upp = input_range_layer[i][j][s][1]
                    # Stay inside one and only one region, thus sum of slack integers should be 1
                    constraints += [cp.sum(z0[s][i][j])+cp.sum(z1[s][i][j])==1]
                    if low < 0 and upp > 0:
                        neg_seg = abs(low)/refinement_degree_layer[s][i][j]
                        for k in range(refinement_degree_layer[s][i][j]):
                            seg_left = low + neg_seg * k
                            seg_right = low + neg_seg * (k+1)
                            constraints += segment_relaxation(x_in[s][i][j], x_out[s][i][j], z0[s][i][j][k], seg_left, seg_right, activation, 'convex')
                        pos_seg = abs(upp)/refinement_degree_layer[s][i][j]
                        for k in range(refinement_degree_layer[s][i][j]):
                            seg_left = 0 + neg_seg * k
                            seg_right = 0 + neg_seg * (k+1)
                            constraints += segment_relaxation(x_in[s][i][j], x_out[s][i][j], z1[s][i][j][k], seg_left, seg_right, activation, 'concave')
                    elif upp <= 0:
                        neg_seg = (upp-low)/refinement_degree_layer[s][i][j]
                        for k in range(refinement_degree_layer[s][i][j]):
                            seg_left = low + neg_seg * k
                            seg_right = low + neg_seg * (k+1)
                            constraints += segment_relaxation(x_in[s][i][j], x_out[s][i][j], z0[s][i][j][k], seg_left, seg_right, activation, 'convex')
                    else:
                        pos_seg = (upp-low)/refinement_degree_layer[s][i][j]
                        for k in range(refinement_degree_layer[s][i][j]):
                            seg_left = low + neg_seg * k
                            seg_right = low + neg_seg * (k+1)
                            constraints += segment_relaxation(x_in[s][i][j], x_out[s][i][j], z1[s][i][j][k], seg_left, seg_right, activation, 'concave')
    else:
        # if x_in is one-dimensional, which means this is a fc layer
        for i in range(layer.output_dim[0]):
            low = input_range_layer[i][0]
            upp = input_range_layer[i][1]
            # any neuron can only be within a region, thus sum of slack integers should be 1
            constraints += [cp.sum(z0[i])+cp.sum(z1[i])==1]
            if low < 0 and upp > 0:
                neg_seg = abs(low)/refinement_degree_layer[i]
                for k in range(refinement_degree_layer[i]):
                    seg_left = low + neg_seg * k
                    seg_right = low + neg_seg * (k+1)
                    constraints += segment_relaxation(x_in[i], x_out[i], z0[i][k], seg_left, seg_right, activation, 'convex')
                pos_seg = abs(upp)/refinement_degree_layer[i]
                for k in range(refinement_degree_layer[i]):
                    seg_left = 0 + pos_seg * k
                    seg_right = 0 + pos_seg * (k+1)
                    constraints += segment_relaxation(x_in[i], x_out[i], z1[i][k], seg_left, seg_right, activation, 'concave')
            elif upp <= 0:
                neg_seg = (upp-low)/refinement_degree_layer[i]
                for k in range(refinement_degree_layer[i]):
                    seg_left = low + neg_seg * k
                    seg_right = low + neg_seg * (k+1)
                    constraints += segment_relaxation(x_in[i], x_out[i], z0[i][k], seg_left, seg_right, activation, 'convex')
            else:
                pos_seg = (upp-low)/refinement_degree_layer[i]
                for k in range(refinement_degree_layer[i]):
                    seg_left = low + pos_seg * k
                    seg_right = low + pos_seg * (k+1)
                    constraints += segment_relaxation(x_in[i], x_out[i], z1[i][k], seg_left, seg_right, activation, 'concave')
    return constraints


# generate constraints for a neruon over a simple segment
def segment_relaxation(x_in_neuron, x_out_neuron, z_seg, seg_left, seg_right, activation, conv_type):
    constraints = []
    if conv_type == 'convex':
        # The triangle constraints of seg_right<=0 for ReLU, sigmoid, tanh
        constraints += [x_in_neuron - seg_right <= M * (1-z_seg)]
        constraints += [-x_in_neuron + seg_left <= M * (1-z_seg)]
        constraints += [-x_out_neuron + activate_de_left(activation,seg_right)*(x_in_neuron-seg_right) + activate(activation,seg_right) <= M * (1-z_seg)]
        constraints += [-x_out_neuron + activate_de_right(activation,seg_left)*(x_in_neuron-seg_left) + activate(activation,seg_left) <= M * (1-z_seg)]
        constraints += [x_out_neuron - (activate(activation,seg_left)-activate(activation,seg_right))/(seg_left-seg_right)*(x_in_neuron-seg_right) - activate(activation,seg_right) <= M * (1-z_seg)]
    if conv_type == 'concave':
        # The triangle constraints of seg_left>=0 for ReLU, sigmoid, tanh
        constraints += [x_in_neuron - seg_right <= M * (1-z_seg)]
        constraints += [-x_in_neuron + seg_left <= M * (1-z_seg)]
        constraints += [-x_out_neuron + activate_de_left(activation,seg_right)*(x_in_neuron-seg_right) + activate(activation,seg_right) >= M * (1-z_seg)]
        constraints += [-x_out_neuron + activate_de_right(activation,seg_left)*(x_in_neuron-seg_left) + activate(activation,seg_left) >= M * (1-z_seg)]
        constraints += [x_out_neuron - (activate(activation,seg_left)-activate(activation,seg_right))/(seg_left-seg_right)*(x_in_neuron-seg_right) - activate(activation,seg_right) >= M * (1-z_seg)]
    return constraints

# uniformly represent the activation function and the derivative
def activate(activation, x):
    if activation == 'ReLU':
        return relu(x)
    elif activation == 'sigmoid':
        return sigmoid(x)
    elif activation == 'tanh':
        return tanh(x)

def activate_de_left(activation, x):
    if activation == 'ReLU':
        return relu_de_left(x)
    elif activation == 'sigmoid':
        return sigmoid_de_left(x)
    elif activation == 'tanh':
        return tanh_de_left(x)

def activate_de_right(activation, x):
    if activation == 'ReLU':
        return relu_de_right(x)
    elif activation == 'sigmoid':
        return sigmoid_de_right(x)
    elif activation == 'tanh':
        return tanh_de_right(x)

# define relu activation function and its left/right derivative
def relu(x):
    if x >= 0:
        r = x
    else:
        r = 0
    return r

def relu_de_left(x):
    if x <= 0:
        de_l = 0
    else:
        de_l = 1
    return de_l

def relu_de_right(x):
    if x < 0:
        de_r = 0
    else:
        de_r = 1
    return de_r

# define tanh activation function and its left/right derivative
def tanh(x):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return t

def tanh_de_left(x):
    de_l = 1 - (tanh(x))**2
    return de_l

def tanh_de_right(x):
    de_r = tanh_de_left(x)
    return de_r

# define sigmoid activation function and its left/right derivative
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_de_left(x):
    de_l = sigmoid(x)*(1-sigmoid(x))
    return de_l

def sigmoid_de_right(x):
    de_r = sigmoid_de_left(x)
    return de_r
##############################################################
def lipschitz(NN_controller, network_input_box, output_index, activation):
    weight_all_layer = NN_controller.weights
    bias_all_layer = NN_controller.bias
    offset = NN_controller.offset
    scale_factor = NN_controller.scale_factor

    activation_all_layer = NN_controller.activations

    layers = len(bias_all_layer)
    lips = 1
    input_range_layer = network_input_box
    for j in range(layers):
        if j < layers - 1:
            weight_j = weight_all_layer[j]
        else:
            weight_j = np.reshape(weight_all_layer[j][output_index], (1, -1))
        if j < layers - 1:
            bias_j = bias_all_layer[j]
        else:
            bias_j = np.reshape(bias_all_layer[j][output_index], (1, -1))
        lipschitz_j = lipschitz_layer(weight_j, bias_j, input_range_layer, activation_all_layer[j])
        lips = lips * lipschitz_j
        input_range_layer, _ = output_range_layer(weight_j, bias_j, input_range_layer, activation_all_layer[j])
    return lips* scale_factor, 0

def lipschitz_layer(weight, bias, input_range_layer, activation):
    neuron_dim = bias.shape[0]
    output_range_box, new_weight = output_range_layer(weight, bias, input_range_layer, activation)
    if activation == 'ReLU':
        return LA.norm(new_weight, 2)
    if activation == 'sigmoid':
        max_singular = 0
        for j in range(neuron_dim):
            range_j = output_range_box[j]
            if range_j[0] > 0.5:
                singular_j = range_j[0]*(1-range_j[0])
            elif range_j[1] < 0.5:
                singular_j = range_j[1]*(1-range_j[1])
            else:
                singular_j = np.array([0.25])
            if max_singular < singular_j:
                max_singular = singular_j
        return max_singular*LA.norm(weight, 2)
    if activation == 'tanh':
        max_singular = 0
        for j in range(neuron_dim):
            range_j = output_range_box[j]
            if range_j[0] > 0:
                singular_j = 1 - range_j[0]**2
            elif range_j[1] < 0:
                singular_j = 1 - range_j[1]**2
            else:
                singular_j = np.array([1])
            if max_singular < singular_j:
                max_singular = singular_j
        return max_singular*LA.norm(weight, 2)




##############################################################
def degree_comb_lists(d, m):
    # generate the degree combination list
    degree_lists = []
    for j in range(m):
        degree_lists.append(range(d[j]+1))
    all_comb_lists = list(itertools.product(*degree_lists))
    return all_comb_lists

def p2c(py_b):
    str_b = str(py_b)
    c_b = str_b.replace("**", "^")
    return c_b


# a simple test case
def test_f(x):
    return math.sin(x[0])+math.cos(x[1])
