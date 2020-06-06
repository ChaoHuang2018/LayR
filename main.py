import time
import json
import csv
import argparse


import numpy as np
import tensorflow as tf


from keras.backend.tensorflow_backend import set_session
from network_parser import nn_controller_details as nn_details
from gurobipy import *
from reachnn import ReachNN


def get_tests(dataset):
    csvfile = open('data/{}_test.csv'.format(dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')
    return tests


def run(args):
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)

    # load neural networks
    eps = args.epsilon

    # result
    dict = {}
    for netname in args.netname:
        dict[netname] = {}
        input_range = []
        NN = nn_details(netname, keras=True)

        # the data, split between train and test sets
        for data_id in eval(args.data_id):
            dict[netname][data_id] = {}
            if args.dataset == 'MNIST':
                tests = get_tests('mnist')
                for idx, test in enumerate(tests):
                    if idx == data_id:
                        break
                image = np.float64(test[1:len(test)]) / np.float64(255)
                data = image.reshape(28, 28, 1)
                data0 = data.reshape(1, 28, 28)
                label = np.int(test[0])
                input_dim = NN.layers[0].input_dim

                for k in range(input_dim[2]):
                    input_range_channel = []
                    for i in range(input_dim[0]):
                        input_range_row = []
                        for j in range(input_dim[1]):
                            input_range_row.append(
                                [max(0, data[i][j][k] - eps),
                                 min(1, data[i][j][k] + eps)]
                            )
                        input_range_channel.append(input_range_row)
                    input_range.append(input_range_channel)
            elif args.dataset == 'CIFAR':
                tests = get_tests('cifar10')
                for idx, test in enumerate(tests):
                    if idx == data_id:
                        break
                image = np.float64(test[1:len(test)]) / np.float64(255)
                data = image.reshape(32, 32, 3)
                data0 = data.reshape(1, 32, 32, 3)
                label = np.int(test[0])
                input_dim = NN.layers[0].input_dim

                for k in range(input_dim[2]):
                    input_range_channel = []
                    for i in range(input_dim[0]):
                        input_range_row = []
                        for j in range(input_dim[1]):
                            input_range_row.append(
                                [(max(0, (data[i][j][k] - eps)) - NN.mean) /
                                 NN.std,
                                 (min(1, (data[i][j][k] + eps)) - NN.mean) /
                                 NN.std]
                            )
                        input_range_channel.append(input_range_row)
                    input_range.append(input_range_channel)

            start_time = time.time()
            nn_analyzer = ReachNN(
                NN, args.dataset, data0, np.array(input_range), args.traceback,
                args.propagation_method, global_robustness_type='L-INFINITY',
                perturbation_bound=eps, approach=args.approach
            )
            old_range, new_range = nn_analyzer.output_range_analysis(
                'METRIC', label, iteration=args.it_num,
                per=eval(args.percentage), is_test=False
            )
            runtime = time.time() - start_time
            print("--- %s seconds ---" % (runtime))
            dict[netname][data_id]['label'] = label
            dict[netname][data_id]['old_range'] = old_range.tolist()
            dict[netname][data_id]['new_range'] = new_range
            dict[netname][data_id]['runtime'] = runtime
            with open(args.store + '.json', 'w') as file:
                json.dump(dict, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--netname', nargs="*", type=str, default=[],
        help='the network name'
    )
    parser.add_argument(
        '--dataset', type=str, default='MNIST', help='the dataset name'
    )
    parser.add_argument(
        '--propagation_method', type=str, default='ERAN',
        help='propagation method for initial solution'
    )
    parser.add_argument(
        '--data_id', type=str, default=[0], help='data id of the test set'
    )
    parser.add_argument(
        '--epsilon', type=float, default=1e-2,
        help='epsilon around the test data point'
    )
    parser.add_argument('--traceback', type=int, default=3, help='traceback')
    parser.add_argument(
        '--it_num', type=int, default=4, help='iteration number'
    )
    parser.add_argument(
        '--percentage', type=str,
        default='[0.0005, 0.2]', help='refinement percentage'
    )
    parser.add_argument(
        '--approach', type=str,
        default='BOTH', help='refinement approach'
    )
    parser.add_argument(
        '--store', type=str, default='result/cifar', help='store result'
    )
    args = parser.parse_args()
    run(args)
