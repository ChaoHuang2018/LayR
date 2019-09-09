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
## layer.type = {'Convolutional', 'Pooling', 'Fully_connected'}
## layer.weight = weight if type = 'Fully_connected', n*n indentical matrix otherwise, n is the dimension of the output of the previous layer
## layer.bias = bias if type = 'Fully_connected', n*1 zero vector otherwise
## layer.activation = {'ReLU', 'tanh', 'sigmoid'} if type = 'Fully_connected', 'Convolutional' if type = 'Convolutional', {'max', 'average'} if type = 'Pooling'

##############################################################
# output range analysis by MILP relaxation for convolutional neural network
def output_range_MILP_CNN(NN, network_input_box, output_index):
    layers = NN.layers
    layers

##############################################################
# output range analysis by MILP relaxation for fully connected neural network
def output_range_MILP_simpleNN(NN, network_input_box, output_index):
    weight_all_layer = NN.weights
    bias_all_layer = NN.bias
    offset = NN.offset
    scale_factor = NN.scale_factor
    activation_all_layer = NN.activations


    # initialization of the input range of all the neurons by naive method
    input_range_all = []
    layers = len(bias_all_layer)
    output_range_last_layer = network_input_box
    for j in range(layers):
        if j < layers - 1:
            weight_j = weight_all_layer[j]
        else:
            weight_j = np.reshape(weight_all_layer[j][output_index], (1, -1))
        if j < layers - 1:
            bias_j = bias_all_layer[j]
        else:
            bias_j = np.reshape(bias_all_layer[j][output_index], (1, -1))
        input_range_layer = neuron_range_layer_basic(weight_j, bias_j, output_range_last_layer, activation_all_layer[j])
        #print(input_range_layer[0][1][0])
        #print('range of layer ' + str(j) + ': ' + str(input_range_layer))
        input_range_all.append(input_range_layer)
        output_range_last_layer, _ = output_range_layer(weight_j, bias_j, output_range_last_layer, activation_all_layer[j])
    print("intput range by naive method: " + str([input_range_layer[0][0], input_range_layer[0][1]]))
    print("Output range by naive method: " + str([(output_range_last_layer[0][0]-offset)*scale_factor, (output_range_last_layer[0][1]-offset)*scale_factor]))

    layer_index = 1
    neuron_index = 0
    
    #print('Output range by naive test: ' + str([input_range_all[layer_index][neuron_index]]))
    # compute by milp relaxation
    network_last_input,_ = neuron_input_range(weight_all_layer, bias_all_layer, layers-1, output_index, network_input_box, input_range_all, activation_all_layer)
    #network_last_input,_ = neuron_input_range(weight_all_layer, bias_all_layer, layer_index, neuron_index, network_input_box, input_range_all, activation_all_layer)
    print("Output range by MILP relaxation: " + str([(sigmoid(network_last_input[0])-offset)*scale_factor, (sigmoid(network_last_input[1])-offset)*scale_factor]))

    range_update = copy.deepcopy(input_range_all)
    for j in range(layers):
        for i in range(len(bias_all_layer[j])):
            _, range_update = neuron_input_range(weight_all_layer, bias_all_layer, layers-1, output_index, network_input_box, range_update, activation_all_layer)
    print(str(range_update[-1]))
    print(str([(sigmoid(range_update[-1][0][0])-offset)*scale_factor, (sigmoid(range_update[-1][0][1])-offset)*scale_factor]))

    return network_last_input[0], network_last_input[1]

        
# Compute the input range for a specific neuron and return the updated input_range_all
# When layer_index = layers, this function outputs the output range of the neural network
def neuron_input_range(weights, bias, layer_index, neuron_index, network_input_box, input_range_all, activation_all_layer):
    weight_all_layer = weights
    bias_all_layer = bias
    layers = len(bias_all_layer)
    width = max([len(b) for b in bias_all_layer])

    # define large positive number M to enable Big M method
    M = 10e4
    # variables in the input layer
    network_in = cp.Variable((len(network_input_box),1))
    # variables in previous layers
    if layer_index >= 1:
        x_in = cp.Variable((width, layer_index))
        x_out = cp.Variable((width, layer_index))
        z = {}
        z[0] = cp.Variable((width, layer_index), integer=True)
        z[1] = cp.Variable((width, layer_index), integer=True)
    # variables for the specific neuron
    x_in_neuron = cp.Variable()

    constraints = []
    # add constraints for the input layer
    if layer_index >= 1:
        constraints += [ 0 <= z[0] ]
        constraints += [ z[0] <= 1]
        constraints += [ 0 <= z[1]]
        constraints += [ z[1] <= 1]
    for i in range(len(network_input_box)):
        constraints += [network_in[i,0] >= network_input_box[i][0]]
        constraints += [network_in[i,0] <= network_input_box[i][1]]

        #constraints += [network_in[i,0] == 0.7]

    if layer_index >= 1:
        #print(x_in[0,0].shape)
        #print(np.array(weight_all_layer[0]).shape)
        #print(network_in.shape)
        #print((np.array(weight_all_layer[0]) @ network_in).shape)
        #print(bias_all_layer[0].shape)
        constraints += [x_in[:,0:1] == np.array(weight_all_layer[0]) @ network_in + bias_all_layer[0]]

    # add constraints for the layers before the neuron
    for j in range(layer_index):
        weight_j = weight_all_layer[j]
        bias_j = bias_all_layer[j]

        # add constraint for linear transformation between layers
        if j+1 <= layer_index-1:
            weight_j_next = weight_all_layer[j+1]
            bias_j_next = bias_all_layer[j+1]
            
            #print(x_in[:,j+1:j+2].shape)
            #print(weight_j_next.shape)
            #print(x_out[:,j:j+1].shape)
            #print(bias_j_next.shape)
            constraints += [x_in[0:len(bias_j_next),j+1:j+2] == weight_j_next @ x_out[0:len(bias_j),j:j+1] + bias_j_next]

        # add constraint for sigmoid function relaxation
        for i in range(weight_j.shape[0]):
            low = input_range_all[j][i][0][0]
            upp = input_range_all[j][i][1][0]
            
            # define slack integers
            constraints += [z[0][i,j] + z[1][i,j] == 1]
            # The triangle constraint for 0<=x<=u
            constraints += [-x_in[i,j] <= M * (1-z[0][i,j])]
            constraints += [x_in[i,j] - upp <= M * (1-z[0][i,j])]
            constraints += [x_out[i,j] - sigmoid(0)*(1-sigmoid(0))*x_in[i,j]-sigmoid(0) <= M * (1-z[0][i,j])]
            constraints += [x_out[i,j] - sigmoid(upp)*(1-sigmoid(upp))*(x_in[i,j]-upp) - sigmoid(upp) <= M * (1-z[0][i,j])]
            constraints += [-x_out[i,j] + (sigmoid(upp)-sigmoid(0))/upp*x_in[i,j] + sigmoid(0) <= M * (1-z[0][i,j])]
            # The triangle constraint for l<=x<=0
            constraints += [x_in[i,j] <= M * (1-z[1][i,j])]
            constraints += [-x_in[i,j] + low <= M * (1-z[1][i,j])]
            constraints += [-x_out[i,j] + sigmoid(0)*(1-sigmoid(0))*x_in[i,j] + sigmoid(0) <= M * (1-z[1][i,j])]
            constraints += [-x_out[i,j] + sigmoid(low)*(1-sigmoid(low))*(x_in[i,j]-low) + sigmoid(low) <= M * (1-z[1][i,j])]
            constraints += [x_out[i,j] - (sigmoid(low)-sigmoid(0))/low*x_in[i,j] - sigmoid(0) <= M * (1-z[1][i,j])]

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

        

def neuron_range_layer_basic(weight, bias, output_range_last_layer, activation):
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

# Layer for Relu/tanh/sigmoid activation 
def relaxation_activation_layer(x_in, x_out, z, input_range_all, layer, activation):
    j = layer

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


    if len(shape)
    constraints = []
    for i in range(weight_j.shape[0]):
        low = input_range_all[j][i][0][0]
        upp = input_range_all[j][i][1][0]
        
        # define slack integers
        constraints += [z[0][i,j] + z[1][i,j] == 1]
        # The triangle constraint for 0<=x<=u
        constraints += [-x_in[i,j] <= M * (1-z[0][i,j])]
        constraints += [x_in[i,j] - upp <= M * (1-z[0][i,j])]
        constraints += [x_out[i,j] - activate_de_right(activation,0)*x_in[i,j] - activate(activation,0) <= M * (1-z[0][i,j])]
        constraints += [x_out[i,j] - activate_de_left(activation,upp)*(x_in[i,j]-upp) - activate(activation,upp) <= M * (1-z[0][i,j])]
        constraints += [-x_out[i,j] + (activate(activation,upp)-activate(activation,0))/upp*x_in[i,j] + activate(activation,0) <= M * (1-z[0][i,j])]
        # The triangle constraint for l<=x<=0
        constraints += [x_in[i,j] <= M * (1-z[1][i,j])]
        constraints += [-x_in[i,j] + low <= M * (1-z[1][i,j])]
        constraints += [-x_out[i,j] + activate_de_left(activation,0)*x_in[i,j] + activate(activation,0) <= M * (1-z[1][i,j])]
        constraints += [-x_out[i,j] + activate_de_right(activation,low)*(x_in[i,j]-low) + activate(activation,low) <= M * (1-z[1][i,j])]
        constraints += [x_out[i,j] - (activate(activation,low)-activate(activation,0))/low*x_in[i,j] - activate(activation,0) <= M * (1-z[1][i,j])]
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



