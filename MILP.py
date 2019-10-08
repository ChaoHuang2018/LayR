from gurobipy import GRB
import numpy as np


class MILP(object):
    def __init__(self, model):
        self.model = model  # MILP model

    def declare_variables(
        self,
        NN,
        refinement_degrees,
        layer_idx,
        network_idx
    ):
        # input layer variables
        if NN.type == 'Convolutional':
            network_in = []
            for idxChannel in range(NN.layers[0].input_dim[2]):
                network_in.append(
                    self.model.addVars(
                        NN.layers[0].input_dim[0],
                        NN.layers[0].input_dim[1],
                        lb=-GRB.INFINITY,
                        ub=GRB.INFINITY,
                        vtype=GRB.CONTINUOUS,
                        name=(
                            'network_' +
                            str(network_idx) +
                            '_inputs'
                        )
                    )
                )
        else:
            network_in = self.model.addVars(
                NN.layers[0].input_dim[0],
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=(
                    'network_' +
                    str(network_idx) +
                    '_inputs'
                )
            )

        x_in = []  # layer input variables
        x_out = []  # layer output variables
        z0 = []
        z1 = []

        # hidden layers and output layer
        for idxLayer in range(1, layer_idx):
            x_layer_input = []
            x_layer_output = []
            if NN.layers[idxLayer].type == 'Convolutional':
                for idxChannel in range(NN.layers[idxLayer].input_dim[2]):
                    x_layer_input.append(
                        self.model.addVars(
                            NN.layers[idxLayer].input_dim[0],
                            NN.layers[idxLayer].input_dim[1],
                            lb=-GRB.INFINITY,
                            up=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=(
                                'network_' +
                                str(network_idx) +
                                '_layer_input_' +
                                str(idxLayer) +
                                '_channel_' +
                                str(idxChannel)
                            )
                        )
                    )

                x_in.append(x_layer_input)

                for idxChannel in range(
                    NN.layers[idxLayer].output_dim[2]
                ):
                    x_layer_output.append(
                        self.model.addVars(
                            NN.layers[idxLayer].output_dim[0],
                            NN.layers[idxLayer].output_dim[1],
                            lb=-GRB.INFINITY,
                            up=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=(
                                'network_' +
                                str(network_idx) +
                                '_layer_output_' +
                                str(idxLayer) +
                                '_channel_' +
                                str(idxChannel)
                            )
                        )
                    )

                x_out.append(x_layer_output)
                z0.append([])
                z1.append([])

            elif NN.layers[idxLayer].type == 'Activation':
                for idxChannel in range(NN.layers[idxLayer].input_dim[2]):
                    x_layer_input.append(
                        self.model.addVars(
                            NN.layers[idxLayer].input_dim[0],
                            NN.layers[idxLayer].input_dim[1],
                            lb=-GRB.INFINITY,
                            up=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=(
                                'network_' +
                                str(network_idx) +
                                '_layer_input_' +
                                str(idxLayer) +
                                '_channel_' +
                                str(idxChannel)
                            )
                        )
                    )

                x_in.append(x_layer_input)

                for idxChannel in range(
                    NN.layers[idxLayer].output_dim[2]
                ):
                    x_layer_output.append(
                        self.model.addVars(
                            NN.layers[idxLayer].output_dim[0],
                            NN.layers[idxLayer].output_dim[1],
                            lb=-GRB.INFINITY,
                            up=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=(
                                'network_' +
                                str(network_idx) +
                                '_layer_output_' +
                                str(idxLayer) +
                                '_channel_' +
                                str(idxChannel)
                            )
                        )
                    )

                x_out.append(x_layer_output)
                # define slack binary variables
                z0_layer = []
                z1_layer = []

                for idxChannel in range(NN.layers[idxLayer].input_dim[2]):
                    z0_channel = []
                    z1_channel = []
                    for idxRow in range(NN.layers[idxLayer].input_dim[0]):
                        z0_row = []
                        z1_row = []
                        for idxCol in range(
                            NN.layers[idxLayer].input_dim[1]
                        ):
                            if refinement_degrees[idxLayer][
                                idxChannel,
                                idxRow,
                                idxCol
                            ] == 0:
                                z0_row.append([])
                            else:
                                z0_row.append(
                                    self.model.addVars(
                                        refinement_degrees[idxLayer][
                                            idxChannel,
                                            idxRow,
                                            idxCol
                                        ],
                                        vtype=GRB.BINARY
                                    )
                                )
                                z1_row.append(
                                    self.model.addVars(
                                        refinement_degrees[idxLayer][
                                            idxChannel,
                                            idxRow,
                                            idxCol
                                        ],
                                        vtype=GRB.BINARY
                                    )
                                )
                        z0_channel.append(z0_row)
                        z1_channel.append(z1_row)
                    z0_layer.append(z0_channel)
                    z1_layer.append(z1_channel)
                z0.append(z0_layer)
                z1.append(z1_layer)

            elif NN.layers[idxLayer].type == 'Pooling':
                for idxChannel in range(NN.layers[idxLayer].input_dim[2]):
                    x_layer_input.append(
                        self.model.addVars(
                            NN.layers[idxLayer].input_dim[0],
                            NN.layers[idxLayer].input_dim[1],
                            lb=-GRB.INFINITY,
                            up=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=(
                                'network_' +
                                str(network_idx) +
                                '_layer_input_' +
                                str(idxLayer) +
                                '_channel_' +
                                str(idxChannel)
                            )
                        )
                    )

                x_in.append(x_layer_input)

                for idxChannel in range(
                    NN.layers[idxLayer].output_dim[2]
                ):
                    x_layer_output.append(
                        self.model.addVars(
                            NN.layers[idxLayer].output_dim[0],
                            NN.layers[idxLayer].output_dim[1],
                            lb=-GRB.INFINITY,
                            up=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=(
                                'network_' +
                                str(network_idx) +
                                '_layer_output_' +
                                str(idxLayer) +
                                '_channel_' +
                                str(idxChannel)
                            )
                        )
                    )

                x_out.append(x_layer_output)
                z0.append([])
                z1.append([])

            elif NN.layers[idxLayer].type == 'Flatten':
                for idxChannel in range(NN.layers[idxLayer].input_dim[2]):
                    x_layer_input.append(
                        self.model.addVars(
                            NN.layers[idxLayer].input_dim[0],
                            NN.layers[idxLayer].input_dim[1],
                            lb=-GRB.INFINITY,
                            up=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=(
                                'network_' +
                                str(network_idx) +
                                '_layer_input_' +
                                str(idxLayer) +
                                '_channel_' +
                                str(idxChannel)
                            )
                        )
                    )

                x_in.append(x_layer_input)

                for idxChannel in range(
                    NN.layers[idxLayer].output_dim[2]
                ):
                    x_layer_output.append(
                        self.model.addVars(
                            NN.layers[idxLayer].output_dim[0],
                            NN.layers[idxLayer].output_dim[1],
                            lb=-GRB.INFINITY,
                            up=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=(
                                'network_' +
                                str(network_idx) +
                                '_layer_output_' +
                                str(idxLayer) +
                                '_channel_' +
                                str(idxChannel)
                            )
                        )
                    )

                x_out.append(x_layer_output)
                z0.append([])
                z1.append([])

            elif NN.layers[idxLayer].type == 'Fully_connected':
                x_layer_input = self.model.addVars(
                    NN.layers[idxLayer].input_dim[0],
                    lb=-GRB.INFINITY,
                    up=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=(
                        'network_' +
                        str(network_idx) +
                        '_layer_input' +
                        str(idxLayer)
                    )
                )
                x_in.append(x_layer_input)

                x_layer_output = self.model.addVars(
                    NN.layers[idxLayer].input_dim[0],
                    lb=-GRB.INFINITY,
                    up=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=(
                        'network_' +
                        str(network_idx) +
                        '_layer_output' +
                        str(idxLayer)
                    )
                )
                x_out.append(x_layer_output)

                # define slack binary variables
                z0_layer = []
                z1_layer = []
                for idxRow in range(NN.layers[idxLayer].output_dim[0]):
                    z0_layer.append(
                        self.model.addVars(
                            refinement_degrees[idxLayer][idxRow],
                            vtype=GRB.BINARY
                        )
                    )
                    z1_layer.append(
                        self.model.addVars(
                            refinement_degrees[idxLayer][idxRow],
                            vtype=GRB.BINARY
                        )
                    )
                z0.append(z0_layer)
                z1.append(z1_layer)

        # variables for the picked neuron
        x_neuron_output = self.model.addVar(
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name='network_' + str(network_idx) + 'output_neuron'
        )

        all_variables = {
            'input': network_in,
            'layer_input': x_in,
            'layer_output': x_out,
            'binary_1': z0,
            'binary_2': z1,
            'output_neuron': x_neuron_output
        }
        return all_variables

    def add_interlayers_constrainti(self, NN, all_variables, layer_idx):
        network_in = all_variables['input']
        x_in = all_variables['layer_input']
        x_out = all_variables['layer_output']

        current_layer = NN.layers[layer_idx]

        if layer_idx == 0:
            if current_layer.type == 'Fully_connected':
                for idxRow in range(current_layer.output_dim[0]):
                    weight = np.reshape(
                        current_layer.weight[:, 1],
                        (-1, 1)
                    )
                    weight_dic = {}
                    for idxCol in range(weight.shape[0]):
                        weight_dic[idxCol] = weight[idxCol][0]
                    self.model.addConstr(
                        network_in.prod(weight_dic) +
                        NN.layers[layer_idx].bias[idxRow]
                        == x_in[layer_idx][idxRow]
                    )
            elif current_layer.type == 'Convolutional':
                for idxChannel in range(current_layer.input_dim[2]):
                    for idxRow in range(current_layer.input_dim[0]):
                        for idxCol in range(current_layer.input_dim[1]):
                            self.model.addConstr(
                                network_in[idxChannel][idxRow, idxCol]
                                == x_in[layer_idx][idxRow, idxCol]
                            )
        else:
            if current_layer.type == 'Fully_connected':
                for idxRow in range(current_layer.output_dim[0]):
                    weight = np.reshape(
                        current_layer.weight[:, idxRow],
                        (-1, 1)
                    )
                    weight_dic = {}
                    for idxCol in range(weight.shape[0]):
                        weight_dic[idxCol] = weight[idxCol][0]
                    self.model.addConstr(
                        x_out[layer_idx - 1].prod(weight_dic) +
                        current_layer.bias[idxRow]
                        == x_in[layer_idx][idxRow]
                    )
            else:
                for idxChannel in range(current_layer.input_dim[2]):
                    for idxRow in range(current_layer.input_dim[0]):
                        for idxCol in range(current_layer.input_dim[1]):
                            self.model.addConstr(
                                x_out[layer_idx - 1][idxChannel][
                                    idxRow,
                                    idxCol
                                ]
                                == x_in[layer_idx][idxChannel][idxRow, idxCol]
                            )
