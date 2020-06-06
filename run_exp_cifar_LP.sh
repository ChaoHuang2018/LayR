python main.py --data_id [0,1,2,3] --approach UPDATE_RANGE --dataset CIFAR --netname model_CIFAR_CNN_Medium --epsilon 0.001 --traceback 2 --percentage [0.0005,0.1] --store result/LP/CIFAR_Med &&
python main.py --data_id [0,1,2,3] --approach UPDATE_RANGE --dataset CIFAR --netname model_CIFAR_CNN_Large --percentage [0.0005,0.1] --epsilon 0.001 --it_num 3 --store result/LP/CIFAR_Large
