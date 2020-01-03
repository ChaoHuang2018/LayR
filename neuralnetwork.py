import json
import numpy as np
from numpy import linalg as LA
from keras import backend as K


class NN(object):
    """
    a neural network with relu activation function
    """
    def __init__(
        self,
        name=None,
        res=None,
        activation=None,
        keras=False,
        model=None,
        model_json=None
    ):
        self.name = name
        if not keras:
            # activation type
            activations = activation.split('_')
            if len(activations) > 1:
                self.activation = activations[0]
                self.last_layer_activation = activations[1]
            else:
                self.activation = activation
                self.last_layer_activation = None
            # affine mapping of the output
            self.offset = res[-2]
            self.scale_factor = res[-1]

            # parse structure of neural networks
            self.num_of_inputs = int(res[0])
            self.num_of_outputs = int(res[1])
            self.num_of_hidden_layers = int(res[2])
            self.network_structure = np.zeros(self.num_of_hidden_layers + 1,
                                              dtype=int)

            self.activations = ([self.activation] *
                                (self.num_of_hidden_layers + 1))
            if self.last_layer_activation is not None:
                self.activations[-1] = self.last_layer_activation

            # pointer is current reading index
            self.pointer = 3

            # num of neurons of each layer
            for i in range(self.num_of_hidden_layers):
                self.network_structure[i] = int(res[self.pointer])
                self.pointer += 1

            # output layer
            self.network_structure[-1] = self.num_of_outputs

            # all values from the text file
            self.param = res

            # store the weights and bias in two lists
            # self.weights
            # self.bias
            self.parse_w_b()
        else:
            self.type = None
            self.layers = None
            params = []
            self.weights = []
            self.bias = []
            self.model = model
            with open(model_json) as json_file:
                self.config = json.load(json_file)
            self.set_type()
            self.set_layer()
            for layer in model.layers:
                params.append(layer.get_weights())  # list of numpy arrays
            for param in params:
                if len(param) == 0:
                    continue
                else:
                    self.weights.append(param[0])
                    self.bias.append(param[1])

    def set_type(self):
        if self.config:
            for class_name in self.config['config']:
                if class_name['class_name'][:4] == 'Conv':
                    self.type = 'Convolutional'
            if not self.type:
                self.type = 'Fully_connected'

    def set_layer(self):
        self.layers = []
        layers_config = self.config['config']
        for idx, layer in enumerate(self.model.layers):
            layer_tmp = Layer()
            layer_activation = None
            layer_config = layers_config[idx]
            layer_detail = layer_config['config']
            if layer_config['class_name'] == 'Flatten':
                layer_tmp._type = 'Flatten'
                layer_tmp._input_dim = layer.input_shape[1:]
                layer_tmp._output_dim = layer.output_shape[1:]
                self.layers.append(layer_tmp)
            elif layer_config['class_name'] == 'Conv2D':
                layer_tmp._type = 'Convolutional'
                layer_tmp._input_dim = layer.input_shape[1:]
                layer_tmp._output_dim = layer.output_shape[1:]
                layer_tmp._kernal = layer.get_weights()[0]
                layer_tmp._bias = layer.get_weights()[1]
                layer_tmp._stride = layer.strides
                layer_tmp._activation = self.activation_function(
                    layer_detail['activation']
                )
                layer_tmp._filter_size = layer.filters
                self.layers.append(layer_tmp)

                # Activation layer
                layer_activation = Layer()
                layer_activation._type = 'Activation'
                layer_activation._activation = layer_tmp.activation
                layer_activation._input_dim, layer_activation._output_dim = (
                    layer_tmp.output_dim,
                    layer_tmp.output_dim
                )
                self.layers.append(layer_activation)

            elif (
                layer_config['class_name'] == 'Dense' and
                layer_detail['activation'] != 'softmax'
            ):
                layer_tmp._type = 'Fully_connected'
                layer_tmp._input_dim = layer.output_shape[1:]
                layer_tmp._output_dim = layer.output_shape[1:]
                layer_tmp._activation = self.activation_function(
                    layer_detail['activation']
                )
                params = layer.get_weights()
                layer_tmp._weight = params[0]
                layer_tmp._bias = params[1]
                self.layers.append(layer_tmp)

            elif layer_config['class_name'] == 'MaxPooling2D':
                layer_tmp._type = 'Pooling'
                layer_tmp._activation = 'max'
                layer_tmp._stride = layer.strides
                layer_tmp._filter_size = layer.pool_size
                layer_tmp._input_dim = layer.input_shape[1:]
                layer_tmp._output_dim = layer.output_shape[1:]
                self.layers.append(layer_tmp)

            elif layer_config['class_name'] == 'AveragePooling2D':
                layer_tmp._type = 'Pooling'
                layer_tmp._activation = 'average'
                layer_tmp._stride = layer.strides
                layer_tmp._filter_size = layer.pool_size
                layer_tmp._input_dim = layer.input_shape[1:]
                layer_tmp._output_dim = layer.output_shape[1:]
                self.layers.append(layer_tmp)

    def activation_function(self, activation_type):
        if activation_type == 'relu':
            return 'ReLU'
        else:
            return activation_type

    def keras_model(self, x):
        if self.model is not None:
            y = self.model.predict(x)
            return y

    def keras_model_pre_softmax(self, x):
        get_output_pre_softmax = K.function([self.model.layers[0].input],
                                            [self.model.layers[6].output])
        layer_output = get_output_pre_softmax([x])[0][0][0]
        return layer_output

    def activate(self, x):
        """
        activation function
        """
        if self.activation == 'ReLU':
            x[x < 0] = 0
        elif self.activation == 'tanh':
            x = np.tanh(x)
        elif self.activation == 'sigmoid':
            x = 1/(1 + np.exp(-x))
        return x

    def last_layer_activate(self, x):
        """
        activation function
        """
        if self.last_layer_activation == 'ReLU':
            x[x < 0] = 0
        elif self.last_layer_activation == 'tanh':
            x = np.tanh(x)
        elif self.last_layer_activation == 'sigmoid':
            x = 1/(1 + np.exp(-x))
        return x

    def parse_w_b(self):
        """
        Parse the input text file
        and store the weights and bias indexed by layer
        Generate: self.weights, self.bias
        """
        # initialize the weights and bias storage space
        self.weights = [None] * (self.num_of_hidden_layers + 1)
        self.bias = [None] * (self.num_of_hidden_layers + 1)

        # compute parameters of the input layer
        weight_matrix0 = np.zeros((self.network_structure[0],
                                   self.num_of_inputs), dtype=np.float64)
        bias_0 = np.zeros((self.network_structure[0], 1), dtype=np.float64)

        for i in range(self.network_structure[0]):
            for j in range(self.num_of_inputs):
                weight_matrix0[i, j] = self.param[self.pointer]
                self.pointer += 1

            bias_0[i] = self.param[self.pointer]
            self.pointer += 1

        # store input layer parameters
        self.weights[0] = weight_matrix0
        self.bias[0] = bias_0

        # compute the hidden layers paramters
        for i in range(self.num_of_hidden_layers):
            weights = np.zeros((self.network_structure[i + 1],
                                self.network_structure[i]), dtype=np.float64)
            bias = np.zeros((self.network_structure[i + 1], 1),
                            dtype=np.float64)

            # read the weight matrix
            for j in range(self.network_structure[i + 1]):
                for k in range(self.network_structure[i]):
                    weights[j][k] = self.param[self.pointer]
                    self.pointer += 1
                bias[j] = self.param[self.pointer]
                self.pointer += 1

            # store parameters of each layer
            self.weights[i + 1] = weights
            self.bias[i + 1] = bias

    def controller(self, x):
        """
        Input: state
        Output: control value after affine transformation
        """
        # transform the input
        length = x.shape[0]
        g = x.reshape([length, 1])

        # pass input through each layer
        for i in range(self.num_of_hidden_layers):
            # linear transformation
            g = self.weights[i] @ g
            g = g + self.bias[i]

            # activation
            g = self.activate(g)

        # output layer
        if self.last_layer_activation is not None:
            # linear transformation
            g = self.weights[self.num_of_hidden_layers] @ g
            g = g + self.bias[self.num_of_hidden_layers]

            # activation
            g = self.last_layer_activate(g)
        else:
            # linear transformation
            g = self.weights[self.num_of_hidden_layers] @ g
            g = g + self.bias[self.num_of_hidden_layers]

            # activation
            g = self.activate(g)

        # affine transformation of output
        y = g - self.offset
        y = y * self.scale_factor

        return y

    @property
    def lips(self):
        if self.activation == 'ReLU':
            scalar = 1
        elif self.activation == 'tanh':
            scalar = 1
        elif self.activation == 'sigmoid':
            scalar = 1/4
        # initialize L cosntant
        L = 1.0
        # multiply norm of weights in each layer
        for i, weight in enumerate(self.weights):
            L *= scalar * LA.norm(weight, 2)

        # activation function of output layer is not the same as other layers
        if self.last_layer_activation is not None:
            if self.activation == 'ReLU':
                L *= 1
            elif self.activation == 'tanh':
                L *= 1
            elif self.activation == 'sigmoid':
                L *= 1/4

        return (L - self.offset) * self.scale_factor

    @property
    def num_of_hidden_layers(self):
        return len(self.layers)


class Layer(object):
    """
    Layer class with following properties:
        type
        weight
        bias
        kernal
        stride
        activation
        filter_size
        input_dim
        output_dim
        layer_idx
    """
    def __init__(self):
        self._type = None
        self._weight = None
        self._bias = None
        self._kernal = None
        self._stride = None
        self._activation = None
        self._filter_size = None
        self._input_dim = None
        self._output_dim = None
        self._layer_idx = None

    @property
    def type(self):
        return self._type

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias

    @property
    def kernal(self):
        return self._kernal

    @property
    def stride(self):
        return self._stride

    @property
    def activation(self):
        return self._activation

    @property
    def filter_size(self):
        return self._filter_size

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim
