import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import argparse
import matplotlib
import pandas as pd

from tqdm import tqdm
import glob
import os
from datetime import datetime
from itertools import product
import time
import math
import sys
sys.path.append("./src")
from ExplicitModels import SupervisedPredictiveCoding
from torch_utils import *

import warnings
warnings.filterwarnings("ignore")

file_path = os.path.realpath(__file__)
working_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(working_path)
# print(os.getcwd())

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_PC_FashionMNIST_V1.pkl" # THIS LINE NEED TO BE ADJUSTED ACCORDING TO EXPERIMENT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

mnist_dset_train = torchvision.datasets.FashionMNIST('../../../data', train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=20, shuffle=True, num_workers=0)

mnist_dset_test = torchvision.datasets.FashionMNIST('../../../data', train=False, transform=transform, target_transform=None, download=True)
test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=20, shuffle=False, num_workers=0)

activation_type = "sigmoid"
architecture = [784, 500, 10]

neural_lr_start = 0.1
neural_lr_stop = 0.05
neural_lr_rule = "constant"
neural_lr_decay_multiplier = 0.01
neural_dynamic_iterations = 50

lr_start = {'ff' : 0.001}

# model = SupervisedPredictiveCoding(architecture, activation_type)
RESULTS_DF = pd.DataFrame( columns = ['setting_number', 'seed', 'Model', 'Hyperparams', 'Trn_ACC_list', 'Tst_ACC_list'])

seed_list = [10*j for j in range(10)]
n_epochs = 50
lr_decay_multiplier_list = [1, 0.99, 0.95, 0.9]
setting_number = 0

for lr_decay_multiplier in lr_decay_multiplier_list:

    setting_number += 1
    hyperparams_dict = {"lr" : lr_start["ff"], "neural_lr_start" : neural_lr_start, "neural_dynamic_iterations" : neural_dynamic_iterations,
                        "lr_decay_multiplier": lr_decay_multiplier}

    for seed_ in seed_list:
        np.random.seed(seed_)
        torch.manual_seed(seed_)

        model = SupervisedPredictiveCoding(architecture, activation_type)

        trn_acc_list = []
        tst_acc_list = []

        lr = lr_start
        for epoch_ in range(n_epochs):
            lr = {'ff' : lr_start['ff'] * (lr_decay_multiplier)**epoch_}
            for idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y = x.to(device), y.to(device)
                x = activation_inverse(x.view(x.size(0),-1).T, "sigmoid")
                y_one_hot = F.one_hot(y, 10).to(device).T
                y_one_hot = 0.94 * y_one_hot + 0.03 * torch.ones(*y_one_hot.shape, device = device)
                _, pc_loss = model.batch_step(  x, y_one_hot, lr, neural_lr_start, neural_lr_stop, neural_lr_rule,
                                                neural_lr_decay_multiplier, neural_dynamic_iterations,
                                                optimizer = "adam")

            trn_acc = evaluatePC(model, train_loader, device, activation_type = activation_type, printing = False)
            tst_acc = evaluatePC(model, test_loader, device, activation_type = activation_type, printing = False)
            trn_acc_list.append(trn_acc)
            tst_acc_list.append(tst_acc)

        Result_Dict = { "setting_number" : setting_number, "seed" : seed_, "Model" : "PC", 
                        "Hyperparams" : hyperparams_dict, "Trn_ACC_list" : trn_acc_list, "Tst_ACC_list" : tst_acc_list}

        RESULTS_DF = RESULTS_DF.append(Result_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

