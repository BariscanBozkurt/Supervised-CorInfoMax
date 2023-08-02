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
from torch.autograd import Variable
from biotorch.module.biomodule import BioModule
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

pickle_name_for_results = "simulation_results_FeedbackAlignment_CIFAR10_3Layers_V1.pkl"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                            std=(3*0.2023, 3*0.1994, 3*0.2010))])

cifar_dset_train = torchvision.datasets.CIFAR10('../../../data', train=True, transform=transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(cifar_dset_train, batch_size=20, shuffle=True, num_workers=0)

cifar_dset_test = torchvision.datasets.CIFAR10('../../../data', train=False, transform=transform, target_transform=None, download=True)
test_loader = torch.utils.data.DataLoader(cifar_dset_test, batch_size=20, shuffle=False, num_workers=0)

activation = F.relu
architecture = [int(32*32*3), 1000, 500, 10]

RESULTS_DF = pd.DataFrame( columns = ['setting_number', 'seed', 'Model', 'Hyperparams', 'Trn_ACC_list', 'Tst_ACC_list'])

lr_list = [1e-3]
lr_decay_gamma_list = [0.95, 0.9]
lr_decay_scheduler_step_list = [10, 25]
optimizer_type_list = ["Adam", "SGD"]
final_layer_activation_list = [False, True]
n_epochs = 50
seed_list = [10*j for j in range(10)]

setting_number = 0
for lr, lr_decay, lr_decay_step, optimizer_type, final_layer_activation_ in product(lr_list, lr_decay_gamma_list, lr_decay_scheduler_step_list, optimizer_type_list, final_layer_activation_list):

    setting_number += 1
    hyperparams_dict = {"lr": lr, "lr_decay": lr_decay, "lr_decay_step": lr_decay_step, "optimizer_type": optimizer_type,
                        "final_layer_activation": final_layer_activation_, "architecture" : architecture}
    for seed_ in seed_list:
        np.random.seed(seed_)
        torch.manual_seed(seed_)

        trn_acc_list = []
        tst_acc_list = []

        criterion = torch.nn.MSELoss().to(device)
        model = BioModule(MLP(architecture, activation = activation, final_layer_activation = final_layer_activation_), mode = "fa").to(device)
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr/10)
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay)

        for epoch_ in range(n_epochs):
            model.train()
            for idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y = Variable(x.to(device)), Variable(y.to(device))
                y_one_hot = F.one_hot(y, num_classes=architecture[-1])
                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat,y_one_hot.to(torch.float32)) # Use this if criterion = torch.nn.MSELoss().to(device)
                # backward pass: compute gradient of the loss with respect to model parameters
                model.zero_grad()
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
            
            scheduler.step()
            model.eval()

            trn_acc = evaluateClassification(model, train_loader, device, False)
            tst_acc = evaluateClassification(model, test_loader, device, False)
            trn_acc_list.append(trn_acc)
            tst_acc_list.append(tst_acc)

        Result_Dict = {"setting_number" : setting_number, "seed" : seed_, "Model" : "MLP-FeedBackAlignment", 
                        "Hyperparams" : hyperparams_dict, "Trn_ACC_list" : trn_acc_list, "Tst_ACC_list" : tst_acc_list}

        RESULTS_DF = RESULTS_DF.append(Result_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))