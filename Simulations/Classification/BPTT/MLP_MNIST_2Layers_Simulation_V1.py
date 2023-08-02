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
from ANN import *
from torch_utils import *

import warnings
warnings.filterwarnings("ignore")

file_path = os.path.realpath(__file__)
working_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(working_path)
# print(os.getcwd())

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_MLP_MNIST_2Layers_V1.pkl"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

mnist_dset_train = torchvision.datasets.MNIST('../../../data', train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=20, shuffle=True, num_workers=0)

mnist_dset_test = torchvision.datasets.MNIST('../../../data', train=False, transform=transform, target_transform=None, download=True)
test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=20, shuffle=False, num_workers=0)

activation = F.relu
architecture = [784, 500, 10]

RESULTS_DF = pd.DataFrame( columns = ['setting_number', 'seed', 'Model', 'Hyperparams', 'Trn_ACC_list', 'Tst_ACC_list'])

lr_list = [1e-3]
lr_decay_gamma_list = [0.95, 0.9]
lr_decay_scheduler_step_list = [10, 15, 25]

n_epochs = 50
seed_list = [10*j for j in range(10)]

setting_number = 0
for lr, lr_decay, lr_decay_step in product(lr_list, lr_decay_gamma_list, lr_decay_scheduler_step_list):

    setting_number += 1
    hyperparams_dict = {"lr": lr, "lr_decay": lr_decay, "lr_decay_step": lr_decay_step,
                        "architecture" : architecture}
    for seed_ in seed_list:
        np.random.seed(seed_)
        torch.manual_seed(seed_)

        trn_acc_list = []
        tst_acc_list = []

        criterion = torch.nn.MSELoss().to(device)
        model = MLP(architecture, activation = activation, final_layer_activation = False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay)

        for epoch_ in range(n_epochs):
            model.train()
            for idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y = x.to(device), y.to(device)
                y_one_hot = F.one_hot(y, num_classes=model.nc)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat,y_one_hot.to(torch.float32)) # Use this if criterion = torch.nn.MSELoss().to(device)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
            
            scheduler.step()
            model.eval()

            trn_acc = evaluateClassification(model, train_loader, device, False)
            tst_acc = evaluateClassification(model, test_loader, device, False)
            trn_acc_list.append(trn_acc)
            tst_acc_list.append(tst_acc)

        Result_Dict = {"setting_number" : setting_number, "seed" : seed_, "Model" : "MLP-BPTT", 
                        "Hyperparams" : hyperparams_dict, "Trn_ACC_list" : trn_acc_list, "Tst_ACC_list" : tst_acc_list}

        RESULTS_DF = RESULTS_DF.append(Result_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))