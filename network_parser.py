import numpy as np
from neuralnetwork import NN


def nn_controller(filename, activation):
    """
    Return the network controller function
    """
    filename = 'nn_retrained/' + filename
    # Obtain the trained parameters and assign the value to res
    with open(filename) as inputfile:
        lines = inputfile.readlines()
    length = len(lines)
    res = np.zeros(length)
    for i, text in enumerate(lines):
        res[i] = eval(text)

    # Set the controller
    NN_controller = NN(res, activation)
    controller = NN_controller.controller
    return controller


def nn_controller_details(filename, activation):
    """
    Return weights and bias
    """
    filename = 'nn_retrained/' + filename

    with open(filename) as inputfile:
        lines = inputfile.readlines()
    length = len(lines)
    res = np.zeros(length)
    for i, text in enumerate(lines):
        res[i] = eval(text)

    # Set the controller
    NN_controller = NN(res, activation)

    return NN_controller
