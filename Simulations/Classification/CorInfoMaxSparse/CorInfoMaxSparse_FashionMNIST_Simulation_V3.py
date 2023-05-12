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
from ContrastiveModels import ContrastiveCorInfoMaxHopfieldSparse
from torch_utils import *

import warnings
warnings.filterwarnings("ignore")

file_path = os.path.realpath(__file__)
working_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(working_path)
# print(os.getcwd())

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_CorInfoMaxSparse_FashionMNIST_V3.pkl"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

mnist_dset_train = torchvision.datasets.FashionMNIST('../../../data', train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=20, shuffle=True, num_workers=0)

mnist_dset_test = torchvision.datasets.FashionMNIST('../../../data', train=False, transform=transform, target_transform=None, download=True)
test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=20, shuffle=False, num_workers=0)

activation = hard_sigmoid
architecture = [784, 500, 10]

RESULTS_DF = pd.DataFrame( columns = ['setting_number', 'seed', 'Model', 'Hyperparams', 'Trn_ACC_list', 'Tst_ACC_list', 'forward_backward_weight_angle_list'])

############# HYPERPARAMS GRID SEARCH LISTS #########################
beta = 1
lambda_ = 0.99999
epsilon = 0.15
one_over_epsilon = 1 / epsilon
lr_start_list = [{'ff' : np.array([0.35, 0.23]), 'fb': np.array([np.nan, 0.06])}]
lr_decay_multiplier_list = [0.95]
neural_lr_start_list = [0.05]
neural_lr_stop = 0.001
neural_lr_rule_list = ["divide_by_slow_loop_index"]
neural_lr_decay_multiplier = 0.01
neural_dynamic_iterations_nudged = 10
neural_dynamic_iterations_free_list = [20]
hopfield_g_list = [0.3, 0.2]
use_random_sign_beta = True
use_three_phase_list = [False]
STlambda_lr_list = [1e-6, 0.01]
sparse_layers = [1, 2]
n_epochs = 50
seed_list = [10*j for j in range(10)]

setting_number = 0
for lr_start, lr_decay_multiplier, neural_lr_start, neural_lr_rule, neural_dynamic_iterations_free, hopfield_g, use_three_phase in product(lr_start_list, lr_decay_multiplier_list, neural_lr_start_list, neural_lr_rule_list, neural_dynamic_iterations_free_list, hopfield_g_list, use_three_phase_list):
    setting_number += 1
    hyperparams_dict = {"lr_start" : lr_start, "lr_decay_multiplier" : lr_decay_multiplier,
                        "neural_dynamic_iterations_free" : neural_dynamic_iterations_free,
                        "neural_dynamic_iterations_nudged" : neural_dynamic_iterations_nudged, 
                        "neural_lr_rule" : neural_lr_rule, "neural_lr" : neural_lr_start, 
                        "epsilon" : epsilon, "lambda" : lambda_,
                        "architecture" : architecture,
                        "three_phase" : use_three_phase}
    for seed_ in seed_list:
        np.random.seed(seed_)
        torch.manual_seed(seed_)

        trn_acc_list = []
        tst_acc_list = []
        model = ContrastiveCorInfoMaxHopfieldSparse(architecture = architecture, lambda_ = lambda_, 
                                                    epsilon = epsilon, activation = activation, 
                                                    sparse_layers = sparse_layers)
        debug_iteration_point = 1

        for epoch_ in range(n_epochs):
            if epoch_ < 11:
                lr = {'ff' : lr_start['ff'] * (lr_decay_multiplier)**epoch_, 'fb' : lr_start['fb'] * (lr_decay_multiplier)**epoch_}
            else:
                lr = {'ff' : lr_start['ff'] * (0.9)**epoch_, 'fb' : lr_start['fb'] * (0.9)**epoch_}
            for idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0),-1).T
                y_one_hot = F.one_hot(y, 10).to(device).T
                take_debug_logs_ = (idx % 500 == 0)
                if use_random_sign_beta:
                    rnd_sgn = 2*np.random.randint(2) - 1
                    beta = rnd_sgn*beta
                    
                neurons = model.batch_step_hopfield(x, y_one_hot, hopfield_g, 
                                                    lr, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                                    neural_lr_decay_multiplier, neural_dynamic_iterations_free,
                                                    neural_dynamic_iterations_nudged, beta, 
                                                    use_three_phase, take_debug_logs_)
            
            trn_acc = evaluateContrastiveCorInfoMaxHopfieldSparse(  model, train_loader, hopfield_g, neural_lr_start, 
                                                                    neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                                                    neural_lr_decay_multiplier, 
                                                                    neural_dynamic_iterations_free, 
                                                                    device, printing = False)
            tst_acc = evaluateContrastiveCorInfoMaxHopfieldSparse(  model, test_loader, hopfield_g, neural_lr_start, 
                                                                    neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                                                    neural_lr_decay_multiplier, 
                                                                    neural_dynamic_iterations_free, 
                                                                    device, printing = False)
            trn_acc_list.append(trn_acc)
            tst_acc_list.append(tst_acc)

        Result_Dict = {"setting_number" : setting_number, "seed" : seed_, "Model" : "CorInfoMax", 
                        "Hyperparams" : hyperparams_dict, "Trn_ACC_list" : trn_acc_list, "Tst_ACC_list" : tst_acc_list,
                        "forward_backward_weight_angle_list" : model.forward_backward_angles}

        RESULTS_DF = RESULTS_DF.append(Result_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))