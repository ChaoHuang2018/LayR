import numpy as np
import tensorflow as tf
import itertools
import math
import random
import time
import copy


class Activation(object):

    def __init__(
        self,
        activation
    ):
        # neural networks
        self.activation = activation

    # uniformly represent the activation function and the derivative
    def activate(self, x):
        if self.activation == 'ReLU':
            return self.relu(x)
        elif self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
        elif self.activation == 'identity':
            return self.identity(x)

    def activate_de_left(self, x):
        if self.activation == 'ReLU':
            return self.relu_de_left(x)
        elif self.activation == 'sigmoid':
            return self.sigmoid_de_left(x)
        elif self.activation == 'tanh':
            return self.tanh_de_left(x)
        elif self.activation == 'identity':
            return self.identity_de_left(x)

    def activate_de_right(self, x):
        if self.activation == 'ReLU':
            return self.relu_de_right(x)
        elif self.activation == 'sigmoid':
            return self.sigmoid_de_right(x)
        elif self.activation == 'tanh':
            return self.tanh_de_right(x)
        elif self.activation == 'identity':
            return self.identity_de_right(x)

    # define relu activation function and its left/right derivative
    @staticmethod
    def relu(x):
        if x >= 0:
            r = x
        else:
            r = 0
        return r

    @staticmethod
    def relu_de_left(x):
        if x <= 0:
            de_l = 0
        else:
            de_l = 1
        return de_l

    @staticmethod
    def relu_de_right(x):
        if x < 0:
            de_r = 0
        else:
            de_r = 1
        return de_r

    # define tanh activation function and its left/right derivative
    @staticmethod
    def tanh(x):
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return t

    @staticmethod
    def tanh_de_left(x):
        de_l = 1 - (self.tanh(x)) ** 2
        if abs(de_l) <= 10e-4:
            de_l = 0
        return de_l

    @staticmethod
    def tanh_de_right(x):
        de_r = self.tanh_de_left(x)
        return de_r

    # define sigmoid activation function and its left/right derivative
    @staticmethod
    def sigmoid(x):
        if x < 0:
            return 1. - 1. / (1. + np.exp(x))
        else:
            return 1. / (1. + np.exp(-x))

    @staticmethod
    def sigmoid_de_left(x):
        sig = self.sigmoid(x)
        de_l = sig * (1. - sig)
        # if abs(de_l)<=10e-4:
        #     de_l = 0
        return de_l

    @staticmethod
    def sigmoid_de_right(x):
        de_r = self.sigmoid_de_left(x)
        return de_r

    # define Indentity activation function and its left/right derivative
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def identity_de_left(x):
        return 1

    @staticmethod
    def identity_de_right(x):
        return 1