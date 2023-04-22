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
from ContrastiveModels import EP
from torch_utils import *

import warnings
warnings.filterwarnings("ignore")

file_path = os.path.realpath(__file__)
working_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(working_path)
# print(os.getcwd())

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_EP_CIFAR10.pkl" # THIS LINE NEED TO BE ADJUSTED ACCORDING TO EXPERIMENT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                            std=(3*0.2023, 3*0.1994, 3*0.2010))])

cifar_dset_train = torchvision.datasets.CIFAR10('../../../data', train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(cifar_dset_train, batch_size=20, shuffle=True, num_workers=0)

cifar_dset_test = torchvision.datasets.CIFAR10('../../../data', train=False, transform=transform, target_transform=None, download=True)
test_loader = torch.utils.data.DataLoader(cifar_dset_test, batch_size=20, shuffle=False, num_workers=0)

activation = hard_sigmoid
criterion = torch.nn.MSELoss(reduction='none').to(device)
architecture = [int(32*32*3), 500, 10]

RESULTS_DF = pd.DataFrame( columns = ['setting_number', 'seed', 'Model', 'Hyperparams', 'Trn_ACC_list', 'Tst_ACC_list'])

############# HYPERPARAMS GRID SEARCH LISTS #########################

alphas_W_list = [[0.1, 0.05], [0.07, 0.03], [0.014, 0.011]]
T1_list = [20]
T2_list = [4]
neural_lr_list = [0.5, 0.3]
seed_list = [10*j for j in range(10)]
random_sign = True
n_epochs = 50

setting_number = 0
for alphas_W, T1, T2, neural_lr in product(alphas_W_list, T1_list, T2_list, neural_lr_list):
    setting_number += 1
    hyperparams_dict = {"alphas_W" : alphas_W, "T1" : T1, "T2" : T2, "neural_lr" : neural_lr}
    for seed_ in seed_list:
        np.random.seed(seed_)
        torch.manual_seed(seed_)

        model = EP(architecture, activation = activation)
        model = model.to(device)

        optim_params = []
        for idx in range(len(model.W)):
            optim_params.append(  {'params': model.W[idx].parameters(), 'lr': alphas_W[idx]}  )
            

        optimizer = torch.optim.SGD( optim_params, momentum=0.0 )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.75)

        mbs = train_loader.batch_size
        iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
        betas = (0.0, 1)
        beta_1, beta_2 = betas

        trn_acc_list = []
        tst_acc_list = []

        debug_iteration_point = 1

        for epoch_ in range(n_epochs):
            model.train()
            for idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y = x.to(device), y.to(device)
                neurons = model.init_neurons(x.size(0), device)
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas
                neurons = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)
                model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)
                optimizer.step()
            # scheduler.step()
            model.eval()
            trn_acc = evaluateEP(model.to(device), train_loader, T1, neural_lr, device, False)
            tst_acc = evaluateEP(model.to(device), test_loader, T1, neural_lr, device, False)
            trn_acc_list.append(trn_acc)
            tst_acc_list.append(tst_acc)

        Result_Dict = {"setting_number" : setting_number, "seed" : seed_, "Model" : "EP", 
                        "Hyperparams" : hyperparams_dict, "Trn_ACC_list" : trn_acc_list, "Tst_ACC_list" : tst_acc_list}

        RESULTS_DF = RESULTS_DF.append(Result_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))