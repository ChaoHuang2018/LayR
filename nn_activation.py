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
            return relu(x)
        elif self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'tanh':
            return tanh(x)
        elif self.activation == 'Affine':
            return affine(x)

    def activate_de_left(self, x):
        if self.activation == 'ReLU':
            return relu_de_left(x)
        elif self.activation == 'sigmoid':
            return sigmoid_de_left(x)
        elif self.activation == 'tanh':
            return tanh_de_left(x)
        elif self.activation == 'Affine':
            return affine_de_left(x)

    def activate_de_right(self, x):
        if self.activation == 'ReLU':
            return relu_de_right(x)
        elif self.activation == 'sigmoid':
            return sigmoid_de_right(x)
        elif self.activation == 'tanh':
            return tanh_de_right(x)
        elif self.activation == 'Affine':
            return affine_de_right(x)


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
    de_l = 1 - (tanh(x)) ** 2
    if abs(de_l) <= 10e-4:
        de_l = 0
    return de_l


def tanh_de_right(x):
    de_r = tanh_de_left(x)
    return de_r


# define sigmoid activation function and its left/right derivative
def sigmoid(x):
    if x < 0:
        return 1. - 1. / (1. + np.exp(x))
    else:
        return 1. / (1. + np.exp(-x))


def sigmoid_de_left(x):
    sig = sigmoid(x)
    de_l = sig * (1. - sig)
    # if abs(de_l)<=10e-4:
    #     de_l = 0
    return de_l

def sigmoid_de_right(x):
    de_r = sigmoid_de_left(x)
    return de_r

# define Indentity activation function and its left/right derivative
def affine(x):
    return x

def affine_de_left(x):
    return 1

def affine_de_right(x):
    return 1