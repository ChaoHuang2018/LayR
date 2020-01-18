import numpy as np
import tensorflow as tf
import itertools
import math
import random
import time
import copy
import os
import copy

from ERAN.tf_verify.eran import ERAN
from ERAN.tf_verify.read_net_file import *
from ERAN.tf_verify.read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ERAN.tf_verify.ai_milp import *
import argparse
from ERAN.tf_verify.config import config

class ERANModel(object):
    def __init__(
        self,
        NN
    ):
        # neural networks
        self.NN = NN
        netfolder = 'ERAN/nets'
        filename, file_extension = os.path.splitext(self.NN.name)
        num_pixels = 784 # for mnist

        if file_extension == ".meta" or file_extension == ".pb":
            sess = tf.Session()
            saver = tf.train.import_meta_graph(netfolder + '/' + self.NN.name)
            saver.restore(sess, tf.train.latest_checkpoint(netfolder + '/'))
            ops = sess.graph.get_operations()
            last_layer_index = -1
            while ops[last_layer_index].type in non_layer_operation_types:
                last_layer_index -= 1
            self.eran = ERAN(sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0'), sess)
        elif file_extension == '.pyt':
            model, is_conv, means, stds = read_tensorflow_net(netfolder + '/' + self.NN.name, num_pixels, True)
            self.eran = ERAN(model, is_onnx=False)
        elif file_extension == '.tf':
            model, is_conv, means, stds = read_tensorflow_net(netfolder + '/' + self.NN.name, num_pixels, False)
            self.eran = ERAN(model, is_onnx=False)

    def input_range_eran(self, network_in):
        NN = self.NN
        network_in_lower_bound, network_in_upper_bound = self.network_in_split(network_in)
        label, _, nlb, nub = self.eran.analyze_box(network_in_lower_bound, network_in_upper_bound, 'deepzono', 1, 1,
                                                   False)
        operations = self.eran.optimizer.operations
        # operation 0 is 'Placeholder'
        i = 1
        j = 0
        nlb_eran_raw = []
        nub_eran_raw = []
        while i < len(operations):
            if (operations[i] == 'MatMul') and (operations[i+1] in ['Add', 'BiasAdd']):
                if i < len(operations) - 2:
                    if operations[i+2] == 'Relu':
                        nlb_eran_raw.append(nlb[j])
                        nub_eran_raw.append(nub[j])
                        j = j + 1
                    if operations[i+2] == 'Tanh' or operations[i+2] == 'Sigmoid':
                        nlb_eran_raw.append(nlb[j])
                        nub_eran_raw.append(nub[j])
                        j = j + 2
                    i += 3
                else:
                    nlb_eran_raw.append(nlb[-1])
                    nub_eran_raw.append(nub[-1])
                    i += 1
            elif (operations[i] == 'Conv2D') and (operations[i+1] in ['Add', 'BiasAdd']):
                if operations[i + 2] == 'Relu':
                    nlb_eran_raw.append(nlb[j])
                    nub_eran_raw.append(nub[j])
                    j = j + 1
                i += 3
            else:
                i += 1
        input_range_all = []
        i = 0
        j = 0
        while i < self.NN.num_of_hidden_layers:
            if NN.layers[i].type == 'Activation' or NN.layers[i].type == 'Fully_connected':
                input_range_layer = self.eran_range_reashape(i, nlb_eran_raw[j], nub_eran_raw[j])
                j += 1
                input_range_all.append(input_range_layer)
            i += 1
        return input_range_all

    def network_in_split(self, network_in):
        network_in_lower_bound = []
        network_in_upper_bound = []
        if self.NN.type == 'Convolutional' or self.NN.type == 'Flatten':
            for s in range(self.NN.layers[0].input_dim[2]):
                for i in range(self.NN.layers[0].input_dim[0]):
                    for j in range(self.NN.layers[0].input_dim[1]):
                        network_in_lower_bound.append(network_in[s][i][j][0])
                        network_in_upper_bound.append(network_in[s][i][j][1])
        else:
            for i in range(self.NN.layers[0].input_dim[0]):
                network_in_lower_bound.append(network_in[i][0])
                network_in_upper_bound.append(network_in[i][1])
        return network_in_lower_bound, network_in_upper_bound

    def eran_range_reashape(self, layer_index, nlb_layer, nub_layer):
        layer = self.NN.layers[layer_index]
        print(layer_index)
        print(self.NN.layers[layer_index].type)
        if layer.type == 'Activation':
            nlb_3d = np.reshape(np.array(nlb_layer), (layer.input_dim[2], round(layer.input_dim[0]), round(layer.input_dim[1])))
            nub_3d = np.reshape(np.array(nub_layer), (layer.input_dim[2], round(layer.input_dim[0]), round(layer.input_dim[1])))
            input_range_layer = copy.deepcopy(nlb_3d).tolist()
            for s in range(layer.input_dim[2]):
                for i in range(layer.input_dim[0]):
                    for j in range(layer.input_dim[1]):
                        input_range_layer[s][i][j] = [nlb_3d[s][i][j], nub_3d[s][i][j]]
        else:
            input_range_layer = copy.deepcopy(nlb_layer)
            for i in range(layer.input_dim[0]):
                input_range_layer[i] = [nlb_layer[i], nub_layer[i]]
        return input_range_layer