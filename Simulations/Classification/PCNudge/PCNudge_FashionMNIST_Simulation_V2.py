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
from ExplicitModels import SupervisedPredictiveCodingNudgedV2_wAutoGrad
from torch_utils import *

import warnings
warnings.filterwarnings("ignore")

file_path = os.path.realpath(__file__)
working_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(working_path)
# print(os.getcwd())

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_PCNudge_FashionMNIST_V2.pkl" # THIS LINE NEED TO BE ADJUSTED ACCORDING TO EXPERIMENT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

mnist_dset_train = torchvision.datasets.FashionMNIST('../../../data', train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=20, shuffle=True, num_workers=0)

mnist_dset_test = torchvision.datasets.FashionMNIST('../../../data', train=False, transform=transform, target_transform=None, download=True)
test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=20, shuffle=False, num_workers=0)

activation = F.relu
architecture = [784, 128, 64, 10]
supervised_lambda_weight_out = 1e-3
neural_lr_start = 0.1 
neural_lr_stop = 0.05 
neural_lr_rule = "constant"
neural_lr_decay_multiplier = 0.005
neural_dynamic_iterations = 50
weight_lr = 1e-3

RESULTS_DF = pd.DataFrame( columns = ['setting_number', 'seed', 'Model', 'Hyperparams', 'Trn_ACC_list', 'Tst_ACC_list'])

random_sign_use_list = [False, True]
seed_list = [10*j for j in range(10)]
n_epochs = 50
setting_number = 0

for random_sign in random_sign_use_list:
    setting_number += 1
    supervised_lambda_weight = supervised_lambda_weight_out
    hyperparams_dict = {"lr" : weight_lr, "supervised_lambda_weight": supervised_lambda_weight, 
                        "neural_lr_start" : neural_lr_start, "neural_dynamic_iterations" : neural_dynamic_iterations,
                        "use_random_sign": str(random_sign)}

    for seed_ in seed_list:
        np.random.seed(seed_)
        torch.manual_seed(seed_)

        trn_acc_list = []
        tst_acc_list = []
        
        model = SupervisedPredictiveCodingNudgedV2_wAutoGrad(architecture, activation, use_stepLR = True, 
                                                             sgd_nesterov = False, optimizer_type = "sgd", 
                                                             optim_lr = weight_lr, stepLR_step_size = 5*3000,)

        for epoch_ in range(n_epochs):
            for idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0),-1).T
                y_one_hot = F.one_hot(y, 10).to(device).T

                if random_sign:
                    rnd_sgn = 2*np.random.randint(2) - 1
                    supervised_lambda_weight = rnd_sgn * supervised_lambda_weight
                
                model.batch_step(   x, y_one_hot, supervised_lambda_weight,
                                    neural_lr_start, neural_lr_stop, neural_lr_rule,
                                    neural_lr_decay_multiplier, neural_dynamic_iterations,
                                    )

            trn_acc = evaluatePC(model, train_loader, device, False, printing = False)
            tst_acc = evaluatePC(model, test_loader, device, False, printing = False)
            trn_acc_list.append(trn_acc)
            tst_acc_list.append(tst_acc)

        Result_Dict = { "setting_number" : setting_number, "seed" : seed_, "Model" : "PCNudge", 
                        "Hyperparams" : hyperparams_dict, "Trn_ACC_list" : trn_acc_list, "Tst_ACC_list" : tst_acc_list}

        RESULTS_DF = RESULTS_DF.append(Result_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))


