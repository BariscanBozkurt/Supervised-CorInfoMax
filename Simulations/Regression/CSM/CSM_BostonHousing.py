import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import argparse
import matplotlib
from tqdm import tqdm
import glob
from PIL import Image
import os
from datetime import datetime
import time
from itertools import product
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
sys.path.append("./src")
from ContrastiveModels import *
from visualization import *
from dataset import get_boston_housing_dataset_pytorch

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"

if not os.path.exists("../Results"):
    os.mkdir("../Results")

pickle_name_for_results = "simulation_results_CSM_Boston_Housing.pkl"

task = "regression"
activation = hard_sigmoid
criterion = torch.nn.MSELoss(reduction='none').to(device)
architecture = [13, 13, 1]

RESULTS_DF = pd.DataFrame( columns = ['setting_number', 'seed', 'Model', 'Hyperparams', 'Trn_RMSE_list', 'Tst_RMSE_list', 'Trn_MAE_list', 'Tst_MAE_list'])

############# HYPERPARAMS GRID SEARCH LISTS #########################

alphas_W_list = [[0.5, 0.2]]
alphas_M_list = [[-0.001], [-0.007], [-0.005]]
T1_list = [30, 20]
T2_list = [4]
neural_lr_list = [0.2]
random_sign = True
n_epochs = 25
seed_list = [10*j for j in range(10)]

setting_number = 0
for alphas_W, alphas_M, T1, T2, neural_lr in tqdm(product(alphas_W_list, alphas_M_list, T1_list, T2_list, neural_lr_list)):
    setting_number += 1
    hyperparams_dict = {"alphas_W" : alphas_W, "alphas_M" : alphas_M, "T1" : T1, "T2" : T2, "neural_lr" : neural_lr}
    for seed_ in seed_list:
        np.random.seed(seed_)
        torch.manual_seed(seed_)

        train_loader, test_loader, maximum_label_value = get_boston_housing_dataset_pytorch(batch_size = 5, seed = seed_)

        model = CSM(architecture, activation = activation, alphas_W = alphas_W, alphas_M = alphas_M, task = task)
        model = model.to(device)

        optim_params = []
        for idx in range(len(model.W)):
            optim_params.append(  {'params': model.W[idx].parameters(), 'lr': alphas_W[idx]}  )
            
        for idx in range(len(model.M)):
            optim_params.append(  {'params': model.M[idx].parameters(), 'lr': alphas_M[idx]}  )

        optimizer = torch.optim.SGD( optim_params, momentum=0.0 )

        mbs = train_loader.batch_size
        iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
        betas = (0.0, 1)
        beta_1, beta_2 = betas

        trn_rmse_list = []
        tst_rmse_list = []
        trn_mae_list = []
        tst_mae_list = []

        debug_iteration_point = 1

        for epoch_ in range(n_epochs):
            model.train()
            for idx, (x, y) in (enumerate(train_loader)):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                neurons = model.init_neurons(x.size(0), device)
                neurons = model(x, y, neurons, T1, neural_lr = neural_lr, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas
                neurons = model(x, y, neurons, T2, neural_lr = neural_lr, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)
                model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, alphas_M, criterion)
                optimizer.step()
            
            if epoch_ % debug_iteration_point == 0:
                ## Train Evaluations
                gt_list = []
                pred_list = []
                model.eval()
                for x, y in train_loader:
                    neurons = model.init_neurons(x.size(0), device)
                    neurons = model(x, y, neurons, T1, neural_lr = neural_lr, beta=0, criterion=criterion)
                    output = neurons[-1]
                    gt_list += (list(maximum_label_value*torch2numpy(y).reshape(-1,)))
                    pred_list += (list(maximum_label_value*torch2numpy(output).reshape(-1,)))
                train_RMSE = np.sqrt(((np.array(gt_list) - np.array(pred_list)) ** 2).mean())
                train_MAE = mean_absolute_error(np.array(gt_list), np.array(pred_list))
                trn_rmse_list.append(train_RMSE)
                trn_mae_list.append(train_MAE)
                ## Test Evaluation

                gt_list = []
                pred_list = []
                model.eval()
                for x, y in test_loader:
                    neurons = model.init_neurons(x.size(0), device)
                    neurons = model(x, y, neurons, T1, neural_lr = neural_lr, beta=0, criterion=criterion)
                    output = neurons[-1]
                    gt_list += (list(maximum_label_value*torch2numpy(y).reshape(-1,)))
                    pred_list += (list(maximum_label_value*torch2numpy(output).reshape(-1,)))

                test_RMSE = np.sqrt(((np.array(gt_list) - np.array(pred_list)) ** 2).mean())
                test_MAE = mean_absolute_error(np.array(gt_list), np.array(pred_list))
                tst_rmse_list.append(test_RMSE)
                tst_mae_list.append(test_MAE)

            if epoch_ % debug_iteration_point == 0:
                ## Train Evaluations
                gt_list = []
                pred_list = []
                model.eval()
                for x, y in train_loader:
                    neurons = model.init_neurons(x.size(0), device)
                    neurons = model(x, y, neurons, T1, neural_lr = neural_lr, beta=0, criterion=criterion)
                    output = neurons[-1]
                    gt_list += (list(maximum_label_value*torch2numpy(y).reshape(-1,)))
                    pred_list += (list(maximum_label_value*torch2numpy(output).reshape(-1,)))

                test_RMSE = np.sqrt(((np.array(gt_list) - np.array(pred_list)) ** 2).mean())
                test_MAE = mean_absolute_error(np.array(gt_list), np.array(pred_list))
                tst_rmse_list.append(test_RMSE)
                tst_mae_list.append(test_MAE)
                ## Test Evaluation

                gt_list = []
                pred_list = []
                model.eval()
                for x, y in test_loader:
                    neurons = model.init_neurons(x.size(0), device)
                    neurons = model(x, y, neurons, T1, neural_lr = neural_lr, beta=0, criterion=criterion)
                    output = neurons[-1]
                    gt_list += (list(maximum_label_value*torch2numpy(y).reshape(-1,)))
                    pred_list += (list(maximum_label_value*torch2numpy(output).reshape(-1,)))

                test_RMSE = np.sqrt(((np.array(gt_list) - np.array(pred_list)) ** 2).mean())
                test_MAE = mean_absolute_error(np.array(gt_list), np.array(pred_list))
                tst_rmse_list.append(test_RMSE)
                tst_mae_list.append(test_MAE)

        Result_Dict = {"setting_number" : setting_number, "seed" : seed_, "Model" : "CSM", 
                        "Hyperparams" : hyperparams_dict, "Trn_RMSE_list" : trn_rmse_list, "Tst_RMSE_list" : tst_rmse_list,
                        "Trn_MAE_list" : trn_mae_list, "Tst_MAE_list": tst_mae_list}

        RESULTS_DF = RESULTS_DF.append(Result_Dict, ignore_index = True)
        RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

RESULTS_DF.to_pickle(os.path.join("../Results", pickle_name_for_results))

