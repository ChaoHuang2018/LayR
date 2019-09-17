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
## layer.type = {'Convolutional', 'Pooling', 'Fully_connected', 'Flatten', 'Activation'}
## layer.weight = weight if type = 'Fully_connected', n*n indentical matrix otherwise, n is the dimension of the output of the previous layer
## layer.bias = bias if type = 'Fully_connected' or 'Convolutional'
## layer.kernal: only for type = 'Convolutional'
## layer.stride: only for type = 'Convolutional'
## layer.activation = {'ReLU', 'tanh', 'sigmoid'} if type = 'Fully_connected', 'Convolutional' if type = 'Convolutional', {'max', 'average'} if type = 'Pooling'
## layer.filter_size: only for type = 'Pooling'
## layer.input_dim = [m, n], n==1 for fully-connected layer
## layer.output_dim = [m, n]
# Keep the original properties:
## size_of_inputs: matrix (n*1 matrix for FC network)
## size_of_outputs: n*1 matrix
## num_of_hidden_layers: integer
## network_structure: matrix (n*1 matrix for FC network/layers) (change to layer.input/output_dim)
## scale_factor: number
## offset: number

##############################################################
# output range analysis by MILP relaxation for convolutional neural network
def output_range_MILP_CNN(NN, network_input_box, output_index):

    # initialization of the input range of all the neurons by naive method
    # input range is a list 'matrix'
    input_range_all = []
    input_range_layer_i = network_input_box
    output_range_layer_i_last = network_input_box
    for i in range(NN.num_of_hidden_layers):

        # in the output layer, only take the weight of bias of the 'output_index'-th neuron
        if i == NN.num_of_hidden_layers - 1 and NN.layers[i].type == 'Fully_connected':
            weight_i = np.reshape(NN.layers.weight[i][output_index], (1, -1))
            bias_i = np.reshape(NN.layers.bias[i][output_index], (1, -1))
        else:
            weight_i = NN.layers[i].weight
            bias_i = NN.layers[i].bias

        if NN.layers[i].type == 'Convolutional':
            output_range_layer_i = output_range_convolutional_layer_naive(input_range_layer_i, NN.layers[i].kernal, NN.layers[i].bias, NN.layers[i].stride)
        if NN.layers[i].type == 'Activation':
            output_range_layer_i = utput_range_activation_layer_naive(input_range_layer, NN.layers[i].activation)
        if NN.layers[i].type == 'Pooling':
            output_range_layer_i = output_range_pooling_layer_naive(input_range_layer, NN.layers[i].filter_size, NN.layers[i].activation)
        if NN.layers[i].type == 'Flatten':
            output_range_layer_i = output_range_flatten_layer_naive(input_range_layer_i)
        if NN.layers[i].type == 'Fully_connected':
            input_range_layer_i = input_range_fc_layer_naive(weight_i, bias_i, output_range_layer_i_last)
            output_range_layer_i = output_range_activation_layer_naive(input_range_layer, NN.layers[i].activation)


        input_range_all.append(input_range_layer_i)

        input_range_layer_i = output_range_layer_i
        output_range_layer_i_last = output_range_layer_i

    # only invoke our approach once to obtain the output range
    input_range_last_neuron,_ = neuron_input_range_cnn(NN, num_of_hidden_layers-1, [0,0], network_input_box, input_range_all)

    lower_bound = (activate(NN.layers[i].activation, input_range_last_neuron[0])-NN.offset)*NN.scale_factor
    upper_bound = (activate(NN.layers[i].activation, input_range_last_neuron[1])-NN.offset)*NN.scale_factor

    return [lower_bound, upper_bound]


# Compute the input range for a specific neuron and return the updated input_range_all
# When layer_index = layers, this function outputs the output range of the neural network
def neuron_input_range_cnn(NN, layer_index, neuron_index, network_input_box, input_range_all):

    layers = NN.layers

    # define large positive number M to enable Big M method
    M = 10e4
    # variables in the input layer
    network_in = cp.Variable((NN.size_of_inputs[0],NN.size_of_inputs[1]))
    # variables in previous layers
    x_in = {}
    x_out = {}
    z0 = {}
    z1 = {}
    for i in range(layer_index):
        x_in[i] = cp.Variable((NN.layer.input_dim[i][0], NN.layer.input_dim[i][1]))
        x_out[i] = cp.Variable((NN.layer.output_dim[i][0], NN.layer.output_dim[i][1]))
        # define slack integer variables only for activation layers and fully-connected layers
        if NN.layers[i].type == 'Fully_connected' or 'Activation':
            z0[i] = cp.Variable((NN.input_dim[i][0], NN.input_dim[i][1]), boolean=True)
            z1[i] = cp.Variable((NN.input_dim[i][0], NN.input_dim[i][1]), boolean=True)
        else:
            z0[i] = []
            z1[i] = []
    # variables for the specific neuron
    x_in_neuron = cp.Variable()

    constraints = []

    # add constraints for the input layer
    for i in range(NN.size_of_inputs[0]):
        for j in range(NN.size_of_inputs[1]):
        constraints += [network_in[i,j] >= network_input_box[i][j][0]]
        constraints += [network_in[i,j] <= network_input_box[i][j][1]]


    # add constraints for the layers before the neuron
    for i in range(layer_index):

        if NN.layers[i].type == 'Convolutional':
            constraints += relaxation_convolutional_layer(x_in[i], x_out[i], NN.layers[i].kernal, NN.layers[i].bias, NN.layers[i].stride)
            if i == 0:
                constraints += [network_in == x_in[i]]
            else:
                constraints += [x_in[i] == x_out[i-1]]
        if NN.layers[i].type == 'Activation':
            constraints += relaxation_activation_layer(x_in[i], x_out[i], z0[i], z1[i], input_range_all[i], NN.layers[i].activation)
            if i == 0:
                constraints += [network_in == x_in[i]]
            else:
                constraints += [x_in[i] == x_out[i-1]]
        if NN.layers[i].type == 'Pooling':
            constraints += relaxation_pooling_layer(x_in[i], x_out[i], NN.layers[i].filter_size, NN.layers[i].activation)
            if i == 0:
                constraints += [network_in == x_in[i]]
            else:
                constraints += [x_in[i] == x_out[i-1]]
        if NN.layers[i].type == 'Flatten':
            constraints += relaxation_flatten_layer(x_in[i], x_out[i])
            if i == 0:
                constraints += [network_in == x_in[i]]
            else:
                constraints += [x_in[i] == x_out[i-1]]
        if NN.layers[i].type == 'Fully_connected':
            constraints += relaxation_activation_layer(x_in[i], x_out[i], z0[i], z1[i], input_range_all[i], NN.layers[i].activation)
            # add constraint for linear transformation between layers
            weight_i = NN.layers[i].weight
            bias_i = NN.layers[i].bias
            if i == 0:
                constraints += [x_in[i] == weight_i @ network_in + bias_i]
            else:
                constraints += [x_in[i] == weight_i @ x_in[i-1] + bias_i]


    # add constraint for the last layer and the neuron
    # Notice that we only need to handle activation function layer. For other layers, update the input range of the neuron does not improve the result (which is equivalant in fact)
    if NN.layers[i].type == 'Activation':
        constraints += [x_in_neuron == x_out[layer_index-1][neuron_index[0],neuron_index[1]]]
    elif NN.layers[layer_index].type == 'Fully_connected':
        weight_neuron = np.reshape(NN.layers[layer_index].weight[neuron_index[0]], (1, -1))
        bias_neuron = np.reshape(NN.layers[layer_index].bias[neuron_index[0]], (1, -1))
        #print(x_in_neuron.shape)
        #print(weight_neuron.shape)
        #print(x_out[0:len(bias_all_layer[layer_index-1]),layer_index-1:layer_index].shape)
        #print(bias_neuron.shape)
        if layer_index >= 1:
            constraints += [x_in_neuron == weight_neuron @ x_out[layer_index-1] + bias_neuron]
        else:
            constraints += [x_in_neuron == weight_neuron @ network_in + bias_neuron]
    else:
        print('No need to update the input range of this neuron')
        return input_range_all[layer_index][neuron_index[0]][neuron_index[1]], input_range_all


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

    input_range_all[layer_index][neuron_index[0]][neuron_index[1]] = [l_neuron, u_neuron]
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
    z0 = {}
    z1 = {}
    for i in range(layer_index):
        x_in[i] = cp.Variable((NN.input_dim[i][0], NN.input_dim[i][1]))
        x_out[i] = cp.Variable((NN.output_dim[i][0], NN.output_dim[i][1]))
        # define slack integer variables only for activation layers and fully-connected layers
        if NN.layers[i].type == 'Fully_connected':
            z0[i] = cp.Variable((NN.input_dim[i][0], NN.input_dim[i][1]), boolean=True)
            z1[i] = cp.Variable((NN.input_dim[i][0], NN.input_dim[i][1]), boolean=True)
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
        constraints += relaxation_activation_layer(x_in[i], x_out[i], z0[i], z1[i], input_range_all[i], activation)


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
def input_range_fc_layer_naive(weight, bias, output_range_last_layer):
    # compute the input range of each neuron by solving LPs
    input_range_layer = []

    # define output variables of the last layer
    x_out = cp.Variable(weight.shape[1])
    # define input variables of the this layer
    x_in = cp.Variable(weight.shape[0])


    # define constraints: linear transformation by weight and bias
    constraints = [weight @ x_out + b == x_in]

    # define constraints: output range of the last layer
    for i in range(weight.shape[0]):
        constraints += [x_out[i] >= output_range_last_layer[i][0][0], x_out[i] <= output_range_last_layer[i][0][1]]

    for i in range(weight.shape[0]):
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
    return input_range_layer

#
def input_range_flatten_layer_naive(output_range_last_layer):
    input_range_layer = []
    for i in range(output_range_last_layer.shape[0]):
        # consider the i-th row
        for j in range(output_range_last_layer.shape[1]):
            # add the j-th neuron
            input_range_layer.append(output_range_last_layer[i][j])
    return input_range_layer



#############################################################################################
## Derive the output ranges of different layers

# Convolutional layer
def output_range_convolutional_layer_naive(input_range_layer, kernal, bias, stride):
    output_range_layer = []

    # define input variables of this layer, which are also output variables of the last layer
    x_in = cp.Variable((input_range_layer.shape[0],input_range_layer.shape[1]))
    # define out variables of the this layer
    x_out = cp.Variable((math.floor((input_range_layer.shape[0]-kernal.shape[0]+1)/stride), math.floor((input_range_layer.shape[1]-kernal.shape[1]+1)/stride)))

    # define constraints
    constraints = []

    # add constraints: input range of this layer
    for i in range(x_in.shape[0]):
        for j in range(x_in.shape[1]):
            constraints += [x_in[i,j] >= input_range_layer[i][j][0], x_in[i,j] <= input_range_layer[i][j][1]]

    # add constraints: convolutional operation
    for i in range(0, x_in.shape[0]-kernal.shape[0]+1, stride):
        for j in range(0, x_in.shape[1]-kernal.shape[1]+1, stride):
            temp_in = cp.vec(x_in[i:i+kernal.shape[0],j:j+kernal.shape[1]])
            temp_kernal = cp.vec(kernal)
            constraints += [temp_kernal @ temp_in + bias == x_out[i,j]]


    # compute the range of each neuron
    for i in range(0, x_in.shape[0]-kernal.shape[0]+1, stride):

        output_range_layer_i = []

        for j in range(0, x_in.shape[1]-kernal.shape[1]+1, stride):
            objective_min = cp.Minimize(x_out[i,j])
            prob_min = cp.Problem(objective_min, constraints)
            prob_min.solve(solver=cp.GUROBI)

            if prob_min.status == 'optimal':
                neuron_j_min = prob_min.value
                #print('lower bound: ' + str(l_neuron))
                #for variable in prob_min.variables():
                #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
            else:
                print('prob_min.status: ' + prob_min.status)
                print('Error: No result for lower bound!')

            # define objective: smallest output
            objective_max = cp.Maximize(x_in[i,j])
            prob_max = cp.Problem(objective_max, constraints)
            prob_max.solve(solver=cp.GUROBI)

            if prob_max.status == 'optimal':
                neuron_j_max = prob_max.value
                #print('lower bound: ' + str(l_neuron))
                #for variable in prob_min.variables():
                #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
            else:
                print('prob_max.status: ' + prob_max.status)
                print('Error: No result for upper bound!')
            output_range_layer_i.append([neuron_j_min, neuron_j_max])

        output_range_layer.append(output_range_layer_i)
    return output_range_layer

# Pooling layer
def output_range_pooling_layer_naive(input_range_layer, filter_size, pooling_type):
    output_range_layer = []

    # define input variables of this layer, which are also output variables of the last layer
    x_in = cp.Variable((input_range_layer.shape[0],input_range_layer.shape[1]))
    # define out variables of the this layer
    x_out = cp.Variable((round(x_in.shape[0]/filter_size[0]), round(x_in.shape[1]/filter_size[1])))

    # define constraints
    constraints = []

    # add constraints: input range of this layer
    for i in range(x_in.shape[0]):
        for j in range(x_in.shape[1]):
            constraints += [x_in[i,j] >= input_range_layer[i][j][0], x_in[i,j] <= input_range_layer[i][j][1]]

    # add constraints: convolutional operation
    if pooling_type == 'max':
        for i in range(round(x_in.shape[0]/filter_size[0])):
            for j in range(round(x_in.shape[1]/filter_size[1])):
            constraints += [cp.max(x_in[i*filter_size[0]:i*(filter_size[0]+1), j*filter_size[1]:j*(filter_size[1]+1)]) == x_out[i,j]]
    if pooling_type == 'average':
        for i in range(round(x_in.shape[0]/filter_size[0])):
            for j in range(round(x_in.shape[1]/filter_size[1])):
            constraints += [cp.sum(x_in[i*filter_size[0]:i*(filter_size[0]+1), j*filter_size[1]:j*(filter_size[1]+1)])/(filter_size[0]*filter_size[1]) == x_out[i,j]]

    # compute the range of each neuron
    for i in range(round(x_in.shape[0]/filter_size[0])):

        output_range_layer_i = []

        for j in range(round(x_in.shape[1]/filter_size[1])):
            objective_min = cp.Minimize(x_out[i,j])
            prob_min = cp.Problem(objective_min, constraints)
            prob_min.solve(solver=cp.GUROBI)

            if prob_min.status == 'optimal':
                neuron_j_min = prob_min.value
                #print('lower bound: ' + str(l_neuron))
                #for variable in prob_min.variables():
                #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
            else:
                print('prob_min.status: ' + prob_min.status)
                print('Error: No result for lower bound!')

            # define objective: smallest output
            objective_max = cp.Maximize(x_in[i,j])
            prob_max = cp.Problem(objective_max, constraints)
            prob_max.solve(solver=cp.GUROBI)

            if prob_max.status == 'optimal':
                neuron_j_max = prob_max.value
                #print('lower bound: ' + str(l_neuron))
                #for variable in prob_min.variables():
                #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
            else:
                print('prob_max.status: ' + prob_max.status)
                print('Error: No result for upper bound!')
            output_range_layer_i.append([neuron_j_min, neuron_j_max])

        output_range_layer.append(output_range_layer_i)
    return output_range_layer

# flatten layer: just flatten the output
def output_range_flatten_layer_naive(input_range_layer):
    output_range_layer = []
    for i in range(input_range_layer.shape[0]):
        # consider the i-th row
        for j in range(input_range_layer.shape[1]):
            # add the j-th neuron
            output_range_layer.append(input_range_layer[i][j])
    return output_range_layer

# general activation layer
def output_range_activation_layer_naive(input_range_layer, activation):

    # compute the out range of each neuron by solving LPs
    output_range_box = []
    for i in range(input_range_layer.shape[0]):
        output_range_box_i = []
        for j in range(input_range_layer.shape[1]): # input_range_layer.shape[1]=1 for fully-connected layer, input_range_layer.shape[2]=2 for any activation layer
            # compute the minimal output
            neuron_j_min = activate(activation, input_range_layer[i][j][0])
            # compute the maximal output
            neuron_j_max = activate(activation, input_range_layer[i][j][1])
            output_range_box_i.append([neuron_j_min,neuron_j_max])
        output_range_box.append(output_range_box_i)

    return output_range_box

#############################################################################################
## Constraints of MILP relaxation for different layers
# convolutional layer
def relaxation_convolutional_layer(x_in, x_out, kernal, bias, stride):
    constraints = []
    for i in range(0, x_in.shape[0]-kernal.shape[0]+1, stride):
        for j in range(0, x_in.shape[1]-kernal.shape[1]+1, stride):
            temp_in = cp.vec(x_in[i:i+kernal.shape[0],j:j+kernal.shape[1]])
            temp_kernal = cp.vec(kernal)
            constraints += [temp_kernal @ temp_in + bias == x_out[i,j]]
    return constraints


# pooling layer
def relaxation_pooling_layer(x_in, x_out, filter_size, pooling_type):
    constraints = []
    if pooling_type == 'max':
        for i in range(round(x_in.shape[0]/filter_size[0])):
            for j in range(round(x_in.shape[1]/filter_size[1])):
            constraints += [cp.max(x_in[i*filter_size[0]:i*(filter_size[0]+1), j*filter_size[1]:j*(filter_size[1]+1)]) == x_out[i,j]]
    if pooling_type == 'average':
        for i in range(round(x_in.shape[0]/filter_size[0])):
            for j in range(round(x_in.shape[1]/filter_size[1])):
            constraints += [cp.sum(x_in[i*filter_size[0]:i*(filter_size[0]+1), j*filter_size[1]:j*(filter_size[1]+1)])/(filter_size[0]*filter_size[1]) == x_out[i,j]]
    return constraints

# flatten layer
def relaxation_flatten_layer(x_in, x_out):
    constraints = []
    for i in range(x_in.shape[0]):
        for j in range(x_in.shape[1]):
            constraints += [x_in[i,j] == x_out[i*x_in.shape[0]+j,0]]
    return constraints


# Relu/tanh/sigmoid activation layer
# input_range_layer is a matrix of two-element lists
def relaxation_activation_layer(x_in, x_out, z0, z1, input_range_layer, activation):

    constraints = []

    for i in range(x_in.shape[0]):
        for j in range(x_in.shape[1]):
            low = input_range_layer[i][j][0][0]
            upp = input_range_layer[i][j][1][0]

            # define slack integers
            constraints += [z0[i,j] + z1[i,j] == 1]
            # The triangle constraint for 0<=x<=u
            constraints += [-x_in[i,j] <= M * (1-z0[i,j])]
            constraints += [x_in[i,j] - upp <= M * (1-z0[i,j])]
            constraints += [x_out[i,j] - activate_de_right(activation,0)*x_in[i,j] - activate(activation,0) <= M * (1-z0[i,j])]
            constraints += [x_out[i,j] - activate_de_left(activation,upp)*(x_in[i,j]-upp) - activate(activation,upp) <= M * (1-z0[i,j])]
            constraints += [-x_out[i,j] + (activate(activation,upp)-activate(activation,0))/upp*x_in[i,j] + activate(activation,j) <= M * (1-z0[i,j])]
            # The triangle constraint for l<=x<=0
            constraints += [x_in[i,j] <= M * (1-z1[i,j])]
            constraints += [-x_in[i,j] + low <= M * (1-z1[i,j])]
            constraints += [-x_out[i,j] + activate_de_left(activation,0)*x_in[i,j] + activate(activation,0) <= M * (1-z1[i,j])]
            constraints += [-x_out[i,j] + activate_de_right(activation,low)*(x_in[i,0]-low) + activate(activation,low) <= M * (1-z1[i,j])]
            constraints += [x_out[i,j] - (activate(activation,low)-activate(activation,0))/low*x_in[i,j] - activate(activation,0) <= M * (1-z1[i,j])]
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



