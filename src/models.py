import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

import os
from datetime import datetime
import time
import math

from itertools import repeat
from torch.nn.parameter import Parameter
import collections
from torch_utils import *

class TwoLayerCorInfoMaxHopfield():
    
    def __init__(self, architecture, lambda_h, lambda_y, epsilon, activation = hard_sigmoid, initialization = "xavier"):
        
        self.architecture = architecture
        self.lambda_h = lambda_h
        self.lambda_y = lambda_y
        self.gam_h = (1 - lambda_h) / lambda_h
        self.gam_y = (1 - lambda_y) / lambda_y
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if initialization == "identity":
            # Feedforward Synapses Initialization
            Wff = []
            for idx in range(len(architecture)-1):
                weight = torch.eye(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
                # weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
                # torch.nn.init.xavier_uniform_(weight)
                bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
                Wff.append({'weight': weight, 'bias': bias})
            Wff = np.array(Wff)
            
            # Feedback Synapses Initialization
            Wfb = []
            for idx in range(len(architecture)-1):
                weight = torch.eye(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
                # weight = torch.randn(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
                # torch.nn.init.xavier_uniform_(weight)
                bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
                Wfb.append({'weight': weight, 'bias': bias})
            Wfb = np.array(Wfb)
            
            # Lateral Synapses Initialization
            B = []
            for idx in range(len(architecture)-1):
                # weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
                # torch.nn.init.xavier_uniform_(weight)
                # weight = weight @ weight.T
                weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
                B.append({'weight': weight})
            B = np.array(B)
        
        elif initialization == "xavier":
            # Feedforward Synapses Initialization
            Wff = []
            for idx in range(len(architecture)-1):
                weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
                torch.nn.init.xavier_uniform_(weight)
                bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
                Wff.append({'weight': weight, 'bias': bias})
            Wff = np.array(Wff)
            
            # Feedback Synapses Initialization
            Wfb = []
            for idx in range(len(architecture)-1):
                weight = torch.randn(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
                torch.nn.init.xavier_uniform_(weight)
                bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
                Wfb.append({'weight': weight, 'bias': bias})
            Wfb = np.array(Wfb)
            
            # Lateral Synapses Initialization
            B = []
            for idx in range(len(architecture)-1):
                weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
                torch.nn.init.xavier_uniform_(weight)
                weight = weight @ weight.T
                B.append({'weight': weight})
            B = np.array(B)
            
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
    def init_neurons(self, mbs, random_initialize = True, device = 'cuda'):
        # Initializing the neurons
        if random_initialize:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.randn((mbs, size), requires_grad=False, device=device).T)       
        else:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.zeros((mbs, size), requires_grad=False, device=device).T)
        return neurons
    
    def calculate_neural_dynamics_grad(self, x, h, vh,  y_hat, vy, y, beta):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        M0 = B[0]['weight'] + torch.eye(B[0]['weight'].shape[0], requires_grad=False, device = self.device)
        M1 = B[1]['weight'] + torch.eye(B[1]['weight'].shape[0], requires_grad=False, device = self.device)
        gam_h = self.gam_h
        gam_y = self.gam_y
        one_over_epsilon = self.one_over_epsilon
        
        grad_h = -vh + 2* gam_h * M0 @ h - one_over_epsilon * (h - (Wff[0]['weight'] @ x + Wff[0]['bias'])) - one_over_epsilon * (h - (Wfb[1]['weight'] @ y_hat + Wfb[1]['bias']))

        grad_y = -vy + gam_y * M1 @ y_hat - one_over_epsilon * (y_hat - (Wff[1]['weight'] @ h + Wff[1]['bias'])) + 2 * beta * (y - y_hat)

        return grad_h, grad_y

    def run_neural_dynamics(self, x, h, y_hat, y, neural_lr, neural_dynamic_iterations, beta, output_sparsity = False, STlambda_lr = 0.01):
        gam_h, gam_y = self.gam_h, self.gam_y
        vh,vy = torch.clone(h), torch.clone(y_hat)
        if output_sparsity:
            mbs = x.size(1)
            STLAMBD = torch.zeros(1, mbs).to(self.device)
        for iter_count in range(neural_dynamic_iterations):
            with torch.no_grad():       
                grad_h, grad_y = self.calculate_neural_dynamics_grad(x, h, vh, y_hat, vy, y, beta)
                vh = vh + neural_lr * grad_h
                h = self.activation(0.5 * (1 / gam_h) * vh)
                if output_sparsity:
                    y_hat = F.relu(y_hat + neural_lr * grad_y - STLAMBD)
                    STLAMBD = (F.relu(STLAMBD + STlambda_lr * (torch.sum(y_hat, 0).view(1, -1) - 1)))
                else:
                    vy = vy + neural_lr * grad_y
                    y_hat = self.activation((1 / gam_y) * vy)
        return h, y_hat
    
    def batch_step(self, x, y_label, lr, neural_lr, neural_dynamic_iterations_free, 
                   neural_dynamic_iterations_nudged, beta, output_sparsity = False, STlambda_lr = 0.01):
        
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_h, lambda_y = self.lambda_h, self.lambda_y
        gam_h, gam_y = self.gam_h, self.gam_y

        h, y_hat = self.init_neurons(x.size(1), device = self.device)

        h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
                                            neural_dynamic_iterations_free, 0, output_sparsity, STlambda_lr)
        neurons1 = [h, y_hat].copy()

        error_hx_free = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_free = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_free = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_free = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])

        h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
                                            neural_dynamic_iterations_nudged, beta, output_sparsity, STlambda_lr)
        neurons2 = [h, y_hat].copy()

        error_hx_nudged = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_nudged = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_nudged = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])
        
        # Wff_old = torch.clone(Wff[0]['weight'])
        ### Weight Updates
        #k = 5  # Below lines output ---> tensor(0., device='cuda:0')
        #torch.norm(outer_prod_broadcasting(error_hx_free.T, x.T)[k] - (torch.outer(error_hx_free[:,k], x[:,k])))
        Wff[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting((error_hx_free - error_hx_nudged).T, x.T), axis = 0)
        # Wfb[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_xh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_xh_nudged.T, neurons2[0].T), axis = 0)
        Wff[1]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_yh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_yh_nudged.T, neurons2[0].T), axis = 0)
        Wfb[1]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_hy_free.T, neurons1[1].T) - outer_prod_broadcasting(error_hy_nudged.T, neurons2[1].T), axis = 0)
        
        Wff[0]['bias'] -= lr['fb'] * torch.mean(error_hx_free - error_hx_nudged , axis = 1, keepdims = True)
        # Wfb[0]['bias'] -= lr['fb'] * torch.mean(error_xh_nudged - error_xh_free, axis = 1, keepdims = True)
        Wff[1]['bias'] -= lr['fb'] * torch.mean(error_yh_free - error_yh_nudged, axis = 1, keepdims = True)
        Wfb[1]['bias'] -= lr['fb'] * torch.mean(error_hy_free - error_hy_nudged, axis = 1, keepdims = True)

        # B[0]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[0].T, neurons2[0].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[0].T, neurons1[0].T), axis = 0))
        # B[1]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[1].T, neurons2[1].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[1].T, neurons1[1].T), axis = 0))

        # zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        # zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        # zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)
        # zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)
        # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_free, zh_free)))
        # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_free, zy_free)))
        # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] + gam_h * (torch.outer(zh_nudged, zh_nudged) - torch.outer(zh_free, zh_free)))
        # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] + gam_y * (torch.outer(zy_nudged, zy_nudged) - torch.outer(zy_free, zy_free)))

        # zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        # zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        # zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)
        # zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)
        # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_nudged, zh_nudged) - torch.outer(zh_free, zh_free)))
        # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_nudged, zy_nudged) - torch.outer(zy_free, zy_free)))
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
        return h, y_hat

class TwoLayerCorInfoMax():
    
    def __init__(self, architecture, lambda_h, lambda_y, epsilon, activation = hard_sigmoid):
        
        self.architecture = architecture
        self.lambda_h = lambda_h
        self.lambda_y = lambda_y
        self.gam_h = (1 - lambda_h) / lambda_h
        self.gam_y = (1 - lambda_y) / lambda_y
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            Wfb.append({'weight': weight, 'bias': bias})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            # weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)
            
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
    def init_neurons(self, mbs, random_initialize = False, device = 'cuda'):
        # Initializing the neurons
        if random_initialize:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.randn((mbs, size), requires_grad=False, device=device).T)       
        else:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.zeros((mbs, size), requires_grad=False, device=device).T)
        return neurons
    
    def calculate_neural_dynamics_grad(self, x, h, y_hat, y, beta):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_h = self.gam_h
        gam_y = self.gam_y
        one_over_epsilon = self.one_over_epsilon
        
        grad_h = 2* gam_h * B[0]['weight'] @ h - one_over_epsilon * (h - (Wff[0]['weight'] @ x + Wff[0]['bias'])) - one_over_epsilon * (h - (Wfb[1]['weight'] @ y_hat + Wfb[1]['bias']))

        grad_y = gam_y * B[1]['weight'] @ y_hat - one_over_epsilon * (y_hat - (Wff[1]['weight'] @ h + Wff[1]['bias'])) + 2 * beta * (y - y_hat)

        return grad_h, grad_y

    def run_neural_dynamics(self, x, h, y_hat, y, neural_lr, neural_dynamic_iterations, beta, output_sparsity = False, STlambda_lr = 0.01):
        if output_sparsity:
            mbs = x.size(1)
            STLAMBD = torch.zeros(1, mbs).to(self.device)
        for iter_count in range(neural_dynamic_iterations):
            with torch.no_grad():       
                grad_h, grad_y = self.calculate_neural_dynamics_grad(x, h, y_hat, y, beta)
                h = self.activation(h + neural_lr * grad_h)
                if output_sparsity:
                    y_hat = F.relu(y_hat + neural_lr * grad_y - STLAMBD)
                    STLAMBD = (F.relu(STLAMBD + STlambda_lr * (torch.sum(y_hat, 0).view(1, -1) - 1)))
                else:
                    y_hat = self.activation(y_hat + neural_lr * grad_y)
        return h, y_hat
    
    def batch_step(self, x, y_label, lr, neural_lr, neural_dynamic_iterations_free, 
                   neural_dynamic_iterations_nudged, beta, output_sparsity = False, STlambda_lr = 0.01):
        
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_h, lambda_y = self.lambda_h, self.lambda_y
        gam_h, gam_y = self.gam_h, self.gam_y

        h, y_hat = self.init_neurons(x.size(1), device = self.device)

        h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
                                            neural_dynamic_iterations_free, 0, output_sparsity, STlambda_lr)
        neurons1 = [h, y_hat].copy()

        error_hx_free = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_free = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_free = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_free = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])

        zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)

        B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_free, zh_free)))
        B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_free, zy_free)))
        self.B = B

        h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
                                            neural_dynamic_iterations_nudged, beta, output_sparsity, STlambda_lr)
        neurons2 = [h, y_hat].copy()

        error_hx_nudged = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_nudged = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_nudged = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])
        
        # Wff_old = torch.clone(Wff[0]['weight'])
        ### Weight Updates
        #k = 5  # Below lines output ---> tensor(0., device='cuda:0')
        #torch.norm(outer_prod_broadcasting(error_hx_free.T, x.T)[k] - (torch.outer(error_hx_free[:,k], x[:,k])))
        # Wff[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting((error_hx_free - error_hx_nudged).T, x.T), axis = 0)
        Wff[0]['weight'] -= (1 / beta) * lr['ff'] * torch.mean(outer_prod_broadcasting(error_hx_free.T, x.T) - outer_prod_broadcasting(error_hx_nudged.T, x.T), axis = 0)
        # Wfb[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_xh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_xh_nudged.T, neurons2[0].T), axis = 0)
        Wff[1]['weight'] -= (1 / beta) * lr['ff'] * torch.mean(outer_prod_broadcasting(error_yh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_yh_nudged.T, neurons2[0].T), axis = 0)
        Wfb[1]['weight'] -= (1 / beta) * lr['fb'] * torch.mean(outer_prod_broadcasting(error_hy_free.T, neurons1[1].T) - outer_prod_broadcasting(error_hy_nudged.T, neurons2[1].T), axis = 0)
        
        Wff[0]['bias'] -= (1 / beta) * lr['ff'] * torch.mean(error_hx_free - error_hx_nudged, axis = 1, keepdims = True)
        # Wfb[0]['bias'] -= lr['fb'] * torch.mean(error_xh_nudged - error_xh_free, axis = 1, keepdims = True)
        Wff[1]['bias'] -= (1 / beta) * lr['ff'] * torch.mean(error_yh_free - error_yh_nudged, axis = 1, keepdims = True)
        Wfb[1]['bias'] -= (1 / beta) * lr['fb'] * torch.mean(error_hy_free - error_hy_nudged, axis = 1, keepdims = True)

        # B[0]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[0].T, neurons2[0].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[0].T, neurons1[0].T), axis = 0))
        # B[1]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[1].T, neurons2[1].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[1].T, neurons1[1].T), axis = 0))

        # # zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        # zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        # # zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)
        # zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)
        # # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_free, zh_free)))
        # # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_free, zy_free)))
        # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_nudged, zh_nudged)))
        # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_nudged, zy_nudged)))

        # # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] + gam_h * (torch.outer(zh_nudged, zh_nudged) - torch.outer(zh_free, zh_free)))
        # # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] + gam_y * (torch.outer(zy_nudged, zy_nudged) - torch.outer(zy_free, zy_free)))

        # # zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        # # zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        # # zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)
        # # zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)
        # # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_nudged, zh_nudged) - torch.outer(zh_free, zh_free)))
        # # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_nudged, zy_nudged) - torch.outer(zy_free, zy_free)))
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
        return h, y_hat

class TwoLayerCorInfoMaxThreePhase():
    
    def __init__(self, architecture, lambda_h, lambda_y, epsilon, activation = hard_sigmoid):
        
        self.architecture = architecture
        self.lambda_h = lambda_h
        self.lambda_y = lambda_y
        self.gam_h = (1 - lambda_h) / lambda_h
        self.gam_y = (1 - lambda_y) / lambda_y
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            Wfb.append({'weight': weight, 'bias': bias})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            # weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)
            
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
    def init_neurons(self, mbs, random_initialize = False, device = 'cuda'):
        # Initializing the neurons
        if random_initialize:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.randn((mbs, size), requires_grad=False, device=device).T)       
        else:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.zeros((mbs, size), requires_grad=False, device=device).T)
        return neurons
    
    def calculate_neural_dynamics_grad(self, x, h, y_hat, y, beta):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_h = self.gam_h
        gam_y = self.gam_y
        one_over_epsilon = self.one_over_epsilon
        
        grad_h = 2* gam_h * B[0]['weight'] @ h - one_over_epsilon * (h - (Wff[0]['weight'] @ x + Wff[0]['bias'])) - one_over_epsilon * (h - (Wfb[1]['weight'] @ y_hat + Wfb[1]['bias']))

        grad_y = gam_y * B[1]['weight'] @ y_hat - one_over_epsilon * (y_hat - (Wff[1]['weight'] @ h + Wff[1]['bias'])) + 2 * beta * (y - y_hat)

        return grad_h, grad_y

    def run_neural_dynamics(self, x, h, y_hat, y, neural_lr, neural_dynamic_iterations, beta, output_sparsity = False, STlambda_lr = 0.01):
        if output_sparsity:
            mbs = x.size(1)
            STLAMBD = torch.zeros(1, mbs).to(self.device)
        for iter_count in range(neural_dynamic_iterations):
            with torch.no_grad():       
                grad_h, grad_y = self.calculate_neural_dynamics_grad(x, h, y_hat, y, beta)
                h = self.activation(h + neural_lr * grad_h)
                if output_sparsity:
                    y_hat = F.relu(y_hat + neural_lr * grad_y - STLAMBD)
                    STLAMBD = (F.relu(STLAMBD + STlambda_lr * (torch.sum(y_hat, 0).view(1, -1) - 1)))
                else:
                    y_hat = self.activation(y_hat + neural_lr * grad_y)
        return h, y_hat
    
    def batch_step(self, x, y_label, lr, neural_lr, neural_dynamic_iterations_free, 
                   neural_dynamic_iterations_nudged, beta, output_sparsity = False, STlambda_lr = 0.01):
        
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_h, lambda_y = self.lambda_h, self.lambda_y
        gam_h, gam_y = self.gam_h, self.gam_y

        h, y_hat = self.init_neurons(x.size(1), device = self.device)

        h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
                                            neural_dynamic_iterations_free, 0, output_sparsity, STlambda_lr)
        neurons1 = [h, y_hat].copy()

        error_hx_free = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_free = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_free = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_free = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])

        h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
                                            neural_dynamic_iterations_nudged, beta, output_sparsity, STlambda_lr)
        neurons2 = [h, y_hat].copy()

        error_hx_nudged = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_nudged = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_nudged = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])
        
        h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
                                            neural_dynamic_iterations_nudged, -beta, output_sparsity, STlambda_lr)
        neurons3 = [h, y_hat].copy()

        error_hx_nudged2 = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_nudged2 = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_nudged2 = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])

        # Wff_old = torch.clone(Wff[0]['weight'])
        ### Weight Updates
        #k = 5  # Below lines output ---> tensor(0., device='cuda:0')
        #torch.norm(outer_prod_broadcasting(error_hx_free.T, x.T)[k] - (torch.outer(error_hx_free[:,k], x[:,k])))
        # Wff[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting((error_hx_free - error_hx_nudged).T, x.T), axis = 0)
        Wff[0]['weight'] -= (1 / (2*beta)) * lr['ff'] * torch.mean(outer_prod_broadcasting(error_hx_nudged2.T, x.T) - outer_prod_broadcasting(error_hx_nudged.T, x.T), axis = 0)
        # Wfb[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_xh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_xh_nudged.T, neurons2[0].T), axis = 0)
        Wff[1]['weight'] -= (1 / (2*beta)) * lr['ff'] * torch.mean(outer_prod_broadcasting(error_yh_nudged2.T, neurons3[0].T) - outer_prod_broadcasting(error_yh_nudged.T, neurons2[0].T), axis = 0)
        Wfb[1]['weight'] -= (1 / (2*beta)) * lr['fb'] * torch.mean(outer_prod_broadcasting(error_hy_nudged2.T, neurons3[1].T) - outer_prod_broadcasting(error_hy_nudged.T, neurons2[1].T), axis = 0)
        
        Wff[0]['bias'] -= (1 / (2*beta)) * lr['fb'] * torch.mean(error_hx_nudged2 - error_hx_nudged, axis = 1, keepdims = True)
        # Wfb[0]['bias'] -= lr['fb'] * torch.mean(error_xh_nudged - error_xh_free, axis = 1, keepdims = True)
        Wff[1]['bias'] -= (1 / (2*beta)) * lr['fb'] * torch.mean(error_yh_nudged2 - error_yh_nudged, axis = 1, keepdims = True)
        Wfb[1]['bias'] -= (1 / (2*beta)) * lr['fb'] * torch.mean(error_hy_nudged2 - error_hy_nudged, axis = 1, keepdims = True)

        # B[0]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[0].T, neurons2[0].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[0].T, neurons1[0].T), axis = 0))
        # B[1]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[1].T, neurons2[1].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[1].T, neurons1[1].T), axis = 0))

        # zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        # zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)
        zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)
        # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_free, zh_free)))
        # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_free, zy_free)))
        B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_nudged, zh_nudged)))
        B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_nudged, zy_nudged)))

        # # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] + gam_h * (torch.outer(zh_nudged, zh_nudged) - torch.outer(zh_free, zh_free)))
        # # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] + gam_y * (torch.outer(zy_nudged, zy_nudged) - torch.outer(zy_free, zy_free)))

        # # zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        # # zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        # # zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)
        # # zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)
        # # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_nudged, zh_nudged) - torch.outer(zh_free, zh_free)))
        # # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_nudged, zy_nudged) - torch.outer(zy_free, zy_free)))
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
        return h, y_hat


class TwoLayerCorInfoMaxV2():
    
    def __init__(self, architecture, lambda_h, lambda_y, epsilon, activation = hard_sigmoid):
        
        self.architecture = architecture
        self.lambda_h = lambda_h
        self.lambda_y = lambda_y
        self.gam_h = (1 - lambda_h) / lambda_h
        self.gam_y = (1 - lambda_y) / lambda_y
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            Wfb.append({'weight': weight, 'bias': bias})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            # weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            # torch.nn.init.xavier_uniform_(weight)
            # weight = weight @ weight.T
            weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)
            
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
    def init_neurons(self, mbs, random_initialize = True, device = 'cuda'):
        # Initializing the neurons
        if random_initialize:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.randn((mbs, size), requires_grad=False, device=device).T)       
        else:
            neurons = []
            append = neurons.append
            for size in self.architecture[1:]:  
                append(torch.zeros((mbs, size), requires_grad=False, device=device).T)
        return neurons
    
    def calculate_neural_dynamics_grad(self, x, h, y_hat, y, beta):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_h = self.gam_h
        gam_y = self.gam_y
        one_over_epsilon = self.one_over_epsilon
        
        grad_h = 2* gam_h * B[0]['weight'] @ h - one_over_epsilon * (h - (Wff[0]['weight'] @ x + Wff[0]['bias'])) - one_over_epsilon * (h - (Wfb[1]['weight'] @ y_hat + Wfb[1]['bias']))

        grad_y = gam_y * B[1]['weight'] @ y_hat - one_over_epsilon * (y_hat - (Wff[1]['weight'] @ h + Wff[1]['bias'])) + 2 * beta * (y - y_hat)

        return grad_h, grad_y

    def run_neural_dynamics(self, x, h, y_hat, y, neural_lr, neural_dynamic_iterations, beta, output_sparsity = False, STlambda_lr = 0.01):
        if output_sparsity:
            mbs = x.size(1)
            STLAMBD = torch.zeros(1, mbs).to(self.device)
        for iter_count in range(neural_dynamic_iterations):
            with torch.no_grad():       
                grad_h, grad_y = self.calculate_neural_dynamics_grad(x, h, y_hat, y, beta)
                h = self.activation(h + neural_lr * grad_h)
                if output_sparsity:
                    y_hat = F.relu(y_hat + neural_lr * grad_y - STLAMBD)
                    STLAMBD = self.activation(F.relu(STLAMBD + STlambda_lr * (torch.sum(y_hat, 0).view(1, -1) - 1)))
                else:
                    y_hat = self.activation(y_hat + neural_lr * grad_y)
        return h, y_hat
    
    def batch_step(self, x, y_label, lr, neural_lr, neural_dynamic_iterations_free, 
                   neural_dynamic_iterations_nudged, beta, output_sparsity = False, STlambda_lr = 0.01):
        
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_h, lambda_y = self.lambda_h, self.lambda_y
        gam_h, gam_y = self.gam_h, self.gam_y

        h, y_hat = self.init_neurons(x.size(1), device = self.device)

        h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
                                            neural_dynamic_iterations_free, 0, output_sparsity, STlambda_lr)
        neurons1 = [h, y_hat].copy()

        error_hx_free = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_free = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_free = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_free = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])

        h, y_hat = self.run_neural_dynamics(x, neurons1[0], neurons1[1], y_label, neural_lr, 
                                            neural_dynamic_iterations_nudged, beta, output_sparsity, STlambda_lr)
        neurons2 = [h, y_hat].copy()

        # h, y_hat = self.run_neural_dynamics(x, neurons1[0], neurons1[1], y_label, neural_lr, 
        #                                     neural_dynamic_iterations_nudged, -beta, output_sparsity, STlambda_lr)
        # neurons3 = [h, y_hat].copy()

        error_hx_nudged = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_nudged = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        error_hy_nudged = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])
        
        # Wff_old = torch.clone(Wff[0]['weight'])
        ### Weight Updates
        #k = 5  # Below lines output ---> tensor(0., device='cuda:0')
        #torch.norm(outer_prod_broadcasting(error_hx_free.T, x.T)[k] - (torch.outer(error_hx_free[:,k], x[:,k])))
        Wff[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting((error_hx_free - error_hx_nudged).T, x.T), axis = 0)
        # Wfb[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_xh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_xh_nudged.T, neurons2[0].T), axis = 0)
        Wff[1]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_yh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_yh_nudged.T, neurons2[0].T), axis = 0)
        Wfb[1]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_hy_free.T, neurons1[1].T) - outer_prod_broadcasting(error_hy_nudged.T, neurons2[1].T), axis = 0)
        
        Wff[0]['bias'] -= lr['fb'] * torch.mean(error_hx_nudged - error_hx_free, axis = 1, keepdims = True)
        # Wfb[0]['bias'] -= lr['fb'] * torch.mean(error_xh_nudged - error_xh_free, axis = 1, keepdims = True)
        Wff[1]['bias'] -= lr['fb'] * torch.mean(error_yh_nudged - error_yh_free, axis = 1, keepdims = True)
        Wfb[1]['bias'] -= lr['fb'] * torch.mean(error_hy_nudged - error_hy_free, axis = 1, keepdims = True)

        # B[0]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[0].T, neurons2[0].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[0].T, neurons1[0].T), axis = 0))
        # B[1]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[1].T, neurons2[1].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[1].T, neurons1[1].T), axis = 0))

        zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)
        zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)
        B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_free, zh_free)))
        B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_free, zy_free)))
        # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] + gam_h * (torch.outer(zh_nudged, zh_nudged) - torch.outer(zh_free, zh_free)))
        # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] + gam_y * (torch.outer(zy_nudged, zy_nudged) - torch.outer(zy_free, zy_free)))

        # zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        # zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        # zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)
        # zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)
        # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_nudged, zh_nudged) - torch.outer(zh_free, zh_free)))
        # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_nudged, zy_nudged) - torch.outer(zy_free, zy_free)))
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
        return h, y_hat


# class TwoLayerCorInfoMax():
    
#     def __init__(self, architecture, lambda_h, lambda_y, epsilon, activation = hard_sigmoid):
        
#         self.architecture = architecture
#         self.lambda_h = lambda_h
#         self.lambda_y = lambda_y
#         self.epsilon = epsilon
#         self.one_over_epsilon = 1 / epsilon
#         self.activation = activation
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
#         # Feedforward Synapses Initialization
#         Wff = []
#         for idx in range(len(architecture)-1):
#             weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
#             torch.nn.init.xavier_uniform_(weight)
#             bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
#             Wff.append({'weight': weight, 'bias': bias})
#         Wff = np.array(Wff)
        
#         # Feedback Synapses Initialization
#         Wfb = []
#         for idx in range(len(architecture)-1):
#             weight = torch.randn(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
#             torch.nn.init.xavier_uniform_(weight)
#             bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
#             Wfb.append({'weight': weight, 'bias': bias})
#         Wfb = np.array(Wfb)
        
#         # Lateral Synapses Initialization
#         B = []
#         for idx in range(len(architecture)-1):
#             weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
#             torch.nn.init.xavier_uniform_(weight)
#             weight = weight @ weight.T
#             B.append({'weight': weight})
#         B = np.array(B)
#         # # Feedforward Synapses Initialization
#         # Wff = []
#         # for idx in range(len(architecture)-1):
#         #     weight = torch.eye(architecture[idx + 1], architecture[idx], requires_grad = False).to(device)
#         #     #torch.nn.init.xavier_uniform_(weight)
#         #     bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(device)
#         #     Wff.append({'weight': weight, 'bias': bias})

#         # # Feedback Synapses Initialization
#         # Wfb = []
#         # for idx in range(len(architecture)-1):
#         #     weight = torch.eye(architecture[idx], architecture[idx + 1], requires_grad = False).to(device)
#         #     #torch.nn.init.xavier_uniform_(weight)
#         #     bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(device)
#         #     Wfb.append({'weight': weight, 'bias': bias})

#         # # Lateral Synapses Initialization
#         # B = []
#         # for idx in range(len(architecture)-1):
#         #     weight = 10*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(device)
#         #     #torch.nn.init.xavier_uniform_(weight)
#         #     #weight = weight @ weight.T
#         #     B.append({'weight': weight})
            
#         self.Wff = Wff
#         self.Wfb = Wfb
#         self.B = B
        
#     def init_neurons(self, mbs, random_initialize = True, device = 'cuda'):
#         # Initializing the neurons
#         if random_initialize:
#             neurons = []
#             append = neurons.append
#             for size in self.architecture[1:]:  
#                 append(torch.randn((mbs, size), requires_grad=False, device=device).T)       
#         else:
#             neurons = []
#             append = neurons.append
#             for size in self.architecture[1:]:  
#                 append(torch.zeros((mbs, size), requires_grad=False, device=device).T)
#         return neurons
    
#     def calculate_neural_dynamics_grad(self, x, h, y_hat, y, beta):
#         Wff = self.Wff
#         Wfb = self.Wfb
#         B = self.B
#         lambda_h = self.lambda_h
#         lambda_y = self.lambda_y
#         one_over_epsilon = self.one_over_epsilon
        
#         grad_h = 0.5*(one_over_epsilon * Wfb[0]['weight'].T @ (x - (Wfb[0]['weight'] @ h + Wfb[0]['bias'])) + 
#              ((1 - lambda_h) / lambda_h) * B[0]['weight'] @ h -
#              one_over_epsilon * (h - (Wff[0]['weight'] @ x + Wff[0]['bias'])))

#         grad_y = 0.5*(one_over_epsilon * Wfb[1]['weight'].T @ (h - (Wfb[1]['weight'] @ y_hat + Wfb[1]['bias'])) +
#              ((1 - lambda_y) / lambda_y) * B[1]['weight'] @ y_hat - 
#              one_over_epsilon * (y_hat - (Wff[1]['weight'] @ h + Wff[1]['bias']))) + 2 * beta * (y - y_hat)

#         return grad_h, grad_y

#     def run_neural_dynamics(self, x, h, y_hat, y, neural_lr, neural_dynamic_iterations, beta):
#         for iter_count in range(neural_dynamic_iterations):
#             with torch.no_grad():       
#                 grad_h, grad_y = self.calculate_neural_dynamics_grad(x, h, y_hat, y, beta)
#                 h = self.activation(h + neural_lr * grad_h)
#                 y_hat = self.activation(y_hat + neural_lr * grad_y)
#         return h, y_hat
    
#     def batch_step(self, x, y_label, lr, neural_lr, neural_dynamic_iterations_free, 
#                    neural_dynamic_iterations_nudged, beta):
        
#         Wff, Wfb, B = self.Wff, self.Wfb, self.B
        
#         h, y_hat = self.init_neurons(x.size(1), device = self.device)

#         h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
#                                             neural_dynamic_iterations_free, 0)
#         neurons1 = [h, y_hat].copy()

#         error_hx_free = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
#         error_xh_free = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

#         error_yh_free = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
#         error_hy_free = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])

#         h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
#                                             neural_dynamic_iterations_nudged, beta)
#         neurons2 = [h, y_hat].copy()

#         error_hx_nudged = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
#         error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

#         error_yh_nudged = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
#         error_hy_nudged = h - (self.Wfb[1]['weight'] @ y_hat + self.Wfb[1]['bias'])
        
#         Wff_old = torch.clone(Wff[0]['weight'])
#         ### Weight Updates
#         #k = 5  # Below lines output ---> tensor(0., device='cuda:0')
#         #torch.norm(outer_prod_broadcasting(error_hx_free.T, x.T)[k] - (torch.outer(error_hx_free[:,k], x[:,k])))
#         Wff[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting((error_hx_free - error_hx_nudged).T, x.T), axis = 0)
#         Wfb[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_xh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_xh_nudged.T, neurons2[0].T), axis = 0)
#         Wff[1]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_yh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_yh_nudged.T, neurons2[0].T), axis = 0)
#         Wfb[1]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_hy_free.T, neurons1[1].T) - outer_prod_broadcasting(error_hy_nudged.T, neurons2[1].T), axis = 0)
        
#         Wff[0]['bias'] -= lr['fb'] * torch.mean(error_hx_nudged - error_hx_free, axis = 1, keepdims = True)
#         Wfb[0]['bias'] -= lr['fb'] * torch.mean(error_xh_nudged - error_xh_free, axis = 1, keepdims = True)
#         Wff[1]['bias'] -= lr['fb'] * torch.mean(error_yh_nudged - error_yh_free, axis = 1, keepdims = True)
#         Wfb[1]['bias'] -= lr['fb'] * torch.mean(error_hy_nudged - error_hy_free, axis = 1, keepdims = True)

#         B[0]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[0].T, neurons2[0].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[0].T, neurons1[0].T), axis = 0))
#         B[1]['weight'] -= lr['lat'] * (torch.mean(outer_prod_broadcasting(neurons2[1].T, neurons2[1].T), axis = 0) - torch.mean(outer_prod_broadcasting(neurons1[1].T, neurons1[1].T), axis = 0))
        
#         self.Wff = Wff
#         self.Wfb = Wfb
#         self.B = B
        
#         return h, y_hat






# class CorInfoMax(torch.nn.Module):
#     def __init__(self, architecture, zeta = 1e-5, activation = hard_sigmoid):
#         super(CorInfoMax, self).__init__()
        
#         self.activation = activation
#         self.architecture = architecture 
#         self.nc = self.architecture[-1]
#         self.zeta = zeta
#         # Feedforward and Feedboack Synapses Initialization
#         self.W = torch.nn.ModuleList()
#         for idx in range(len(architecture)-1):
#             m = torch.nn.Linear(architecture[idx], architecture[idx+1], bias = True)
#             torch.nn.init.xavier_uniform_(m.weight)
#             # m.weight.data.mul_(torch.tensor([1]))
#             if m.bias is not None:
#                 m.bias.data.mul_(0)
#             self.W.append(m)

#         # Lateral Synapses Initialization
#         self.M = torch.nn.ModuleList()
#         for idx in range(1,len(architecture)):
#             m = torch.nn.Linear(architecture[idx], architecture[idx], bias = False)
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.weight.data = m.weight.data @ m.weight.data.T
#             self.M.append(m)

#     def init_neurons(self, mbs, device):
#         # Initializing the neurons
#         neurons = []
#         append = neurons.append
#         for size in self.architecture[1:]:  
#             append(torch.zeros((mbs, size), requires_grad=True, device=device))
#         return neurons

#     def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
#         # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
#         beta_1, beta_2 = betas
        
#         self.zero_grad()            # p.grad is zero
#         if not(check_thm):
#             phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
#         else:
#             phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
#         phi_1 = phi_1.mean()
        
#         phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
#         phi_2 = phi_2.mean()
        
#         delta_phi = (phi_2 - phi_1)/(beta_2 - beta_1)        
#         delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem

#     def Phi(self, x, y, neurons, beta, criterion, use_logdet_cost = False):
#         # Computes the primitive function given static input x, label y, neurons is the sequence of hidden layers neurons
#         # criterion is the loss
#         x = x.view(x.size(0),-1) # flattening the input
#         zeta = self.zeta
#         layers = [x] + neurons  # concatenate the input to other layers
        
#         # Primitive function computation
#         phi = 0.0
#         for idx in range(len(self.W)): # Linear Terms and Quadratic Terms
#             phi += torch.norm(self.W[idx](layers[idx]) - layers[idx + 1], dim = 1)

#         if beta!=0.0: # Nudging the output layer when beta is non zero 
#             if criterion.__class__.__name__.find('MSE')!=-1:
#                 y = F.one_hot(y, num_classes=self.nc)
#                 L = criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
#             else:
#                 L = criterion(layers[-1].float(), y).squeeze()     
#             phi += beta*L
        
#         return phi