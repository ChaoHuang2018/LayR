import numpy as np
from neuralnetwork import NN
from tensorflow.keras.models import model_from_json


def nn_controller(filename, activation, keras=False):
    """
    Return the network controller function
    """
    if not keras:
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
    else:
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights('model.h5')
        print("Loaded kera model from disk.")
        NN_controller = NN(keras=True, model=loaded_model)
        controller = NN_controller.keras_model
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
