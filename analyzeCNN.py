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
## layer.type = {'Convolutional', 'Pooling', 'Fully_connected', 'Flatten'}
## layer.weight = weight if type = 'Fully_connected', n*n indentical matrix otherwise, n is the dimension of the output of the previous layer
## layer.bias = bias if type = 'Fully_connected', n*1 zero vector otherwise
## layer.kernal: only for type = 'Convolutional'
## layer.bias: only for type = 'Convolutional'
## layer.stride: only for type = 'Convolutional'
## layer.activation = {'ReLU', 'tanh', 'sigmoid'} if type = 'Fully_connected', 'Convolutional' if type = 'Convolutional', {'max', 'average'} if type = 'Pooling'
## layer.filter_size: only for type = 'Pooling'
# Keep the original properties:
## size_of_inputs: matrix (n*1 matrix for FC network)
## size_of_outputs: n*1 matrix
## num_of_hidden_layers: integer
## network_structure: matrix (n*1 matrix for FC network/layers)
## scale_factor: number
## offset: number

##############################################################
# output range analysis by MILP relaxation for convolutional neural network
def output_range_MILP_CNN(NN, network_input_box, output_index):
    layers = NN.layers
    offset = NN.offset
    scale_factor = NN.scale_factor

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
            weight_i = np.reshape(NN.layers.weight[i][output_index], (1, -1))
            bias_i = np.reshape(NN.layers.bias[i][output_index], (1, -1))
        
        input_range_layer = neuron_range_layer_basic(weight_i, bias_i, output_range_last_layer)
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
        for j in range(network_structure.shape[0]):
            _, range_update = neuron_input_range_fc(NN, i, j, network_input_box, range_update)
    print(str(range_update[-1]))


    def activate(activation, x):
        if activation == 'ReLU':
            return relu(x)
        elif activation == 'sigmoid':
            return sigmoid(x)
        elif activation == 'tanh':
            return tanh(x)

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
    network_in = cp.Variable((NN.num_of_inputs,1))
    # variables in previous layers
    x_in = {}
    x_out = {}
    z0 = {}
    z1 = {}
    for i in range(layer_index):
        x_in[i] = cp.Variable((NN.network_structure[i][0], NN.network_structure[i][1]))
        x_out[i] = cp.Variable((NN.network_structure[i][0], NN.network_structure[i][1]))
        z0[i] = cp.Variable((NN.network_structure[i][0], NN.network_structure[i][1]), boolean=True)
        z1[i] = cp.Variable((NN.network_structure[i][0], NN.network_structure[i][1]), boolean=True)
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
            constraints += [x_in[i] == weight_i @ x_in[i-1] + bias_i]

        # add constraint for activation function relaxation
        constraints += relaxation_activation_layer(x_in[i], x_out[i], z0[i], z1[i], input_range_all[i], activation)
        

    # add constraint for the last layer and the neuron
    weight_neuron = np.reshape(weight_all_layer[layer_index][neuron_index], (1, -1))
    bias_neuron = np.reshape(bias_all_layer[layer_index][neuron_index], (1, -1))
    #print(x_in_neuron.shape)
    #print(weight_neuron.shape)
    #print(x_out[0:len(bias_all_layer[layer_index-1]),layer_index-1:layer_index].shape)
    #print(bias_neuron.shape)
    if layer_index >= 1:
        constraints += [x_in_neuron == weight_neuron @ x_out[0:len(bias_all_layer[layer_index-1]),layer_index-1:layer_index] + bias_neuron]
    else:
        constraints += [x_in_neuron == weight_neuron @ network_in[0:len(network_input_box),0:1] + bias_neuron]    
  
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

        

def neuron_range_layer_basic(weight, bias, output_range_last_layer):
    # solving LPs
    neuron_dim = bias.shape[0]
    input_range_box = []
    for j in range(neuron_dim):
        # c: weight of the j-th dimension
        c = weight[j]
        c = c.transpose()
        #print('c: ' + str(c))
        b = bias[j]
        #print('b: ' + str(b))
        # compute the minimal input
        res_min = linprog(c, bounds=output_range_last_layer, options={"disp": False})
        input_j_min = res_min.fun + b
        # compute the maximal input
        res_max = linprog(-c, bounds=output_range_last_layer, options={"disp": False})
        input_j_max = -res_max.fun + b
        input_range_box.append([input_j_min, input_j_max])
    return input_range_box

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


# Relu/tanh/sigmoid activation layer
# input_range_layer is a matrix of two-element lists
def relaxation_activation_layer(x_in, x_out, z0, z1, input_range_layer, activation):

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

def output_range_layer(weight, bias, input_range_layer, activation):
    # solving LPs
    neuron_dim = bias.shape[0]
    output_range_box = []
    new_weight = []
    for j in range(neuron_dim):
        # c: weight of the j-th dimension
        c = weight[j]
        c = c.transpose()
        #print('c: ' + str(c))
        b = bias[j]
        #print('b: ' + str(b))
        # compute the minimal input
        res_min = linprog(c, bounds=input_range_layer, options={"disp": False})
        input_j_min = res_min.fun + b
        #print('min: ' + str(input_j_min))
        # compute the minimal output
        if activation == 'ReLU':
            if input_j_min < 0:
                output_j_min = np.array([0])
            else:
                output_j_min = input_j_min
        if activation == 'sigmoid':
            output_j_min = 1/(1+np.exp(-input_j_min))
        if activation == 'tanh':
            output_j_min = 2/(1+np.exp(-2*input_j_min))-1
        # compute the maximal input
        res_max = linprog(-c, bounds=input_range_layer, options={"disp": False})
        input_j_max = -res_max.fun + b
        # compute the maximal output
        if activation == 'ReLU':
            if input_j_max < 0:
                output_j_max = np.array([0])
            else:
                output_j_max = input_j_max
                new_weight.append(weight[j])
        if activation == 'sigmoid':
            output_j_max = 1/(1+np.exp(-input_j_max))
        if activation == 'tanh':
            output_j_max = 2/(1+np.exp(-2*input_j_max))-1
        output_range_box.append([output_j_min, output_j_max])
    return output_range_box, new_weight


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



