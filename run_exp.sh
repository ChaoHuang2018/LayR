# python main.py --data_id [0,1,2,3] --dataset CIFAR --netname model_CIFAR_CNN_Large model_CIFAR_CNN_Medium --epsilon 0.001 --traceback 2 --percentage [0.0005,0.1] &&
python main.py --data_id [0,1,2,3] --traceback 2 --netname model_MNIST_CNN_Small model_MNIST_CNN_Large --store result/mnist_traceback2_new &&
python main.py --data_id [0,1,2,3] --traceback 1 --netname model_MNIST_CNN_Small model_MNIST_CNN_Large --store result/mnist_traceback1_new &&
python main.py --data_id [0,1,2,3] --netname model_MNIST_CNN_Small model_MNIST_CNN_Large --percentage [0.000125,0.05] --store result/mnist_percentage1_new &&
python main.py --data_id [0,1,2,3] --netname model_MNIST_CNN_Small model_MNIST_CNN_Large --percentage [0.00025,0.1] --store result/mnist_percentage2_new &&
python main.py --data_id [0,1,2,3] --traceback 2 --netname model_MNIST_CNN_Medium --store result/mnist_traceback2_Med_new &&
python main.py --data_id [0,1,2,3] --traceback 1 --netname model_MNIST_CNN_Medium --store result/mnist_traceback1_Med_new &&
python main.py --data_id [0,1,2,3] --netname model_MNIST_CNN_Medium --percentage [0.000125,0.05] --store result/mnist_percentage1_Med_new &&
python main.py --data_id [0,1,2,3] --netname model_MNIST_CNN_Medium --percentage [0.00025,0.1] --store result/mnist_percentage2_Med_new

