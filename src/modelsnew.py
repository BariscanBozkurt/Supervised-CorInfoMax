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



class TwoLayerCorInfoMax():
    
    def __init__(self, architecture, lambda_h, lambda_y,psiv, epsilon, activation = hard_sigmoid):
        
        self.architecture = architecture
        self.lambda_h = lambda_h
        self.lambda_y = lambda_y
        self.gam_h = (1 - lambda_h) / lambda_h
        self.gam_y = (1 - lambda_y) / lambda_y
        self.epsilon = epsilon
        self.psi=psiv
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
            # weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
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
        psiv=self.psi
        
        grad_h = gam_h*(1-psiv) * B[0]['weight'] @ h - (1-psiv)*one_over_epsilon * (h - (Wff[0]['weight'] @ x + Wff[0]['bias'])) + psiv*one_over_epsilon * (h - (Wfb[1]['weight'] @ (y_hat-y) + Wfb[1]['bias']))

        grad_y = gam_y *(1-psiv)* B[1]['weight'] @ (y_hat) - (1-psiv)*one_over_epsilon * (y_hat - (Wff[1]['weight'] @ h + Wff[1]['bias'])) + 2 * beta * (y - y_hat)

        return grad_h, grad_y

    def run_neural_dynamics(self, x, h, y_hat, y, neural_lr_start, neural_lr_stop,
                            neural_dynamic_iterations, beta, lr_rule = "constant", 
                            lr_decay_multiplier = 0.01, output_sparsity = False, STlambda_lr = 0.01):
        if output_sparsity:
            mbs = x.size(1)
            STLAMBD = torch.zeros(1, mbs).to(self.device)
        for iter_count in range(neural_dynamic_iterations):
            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                grad_h, grad_y = self.calculate_neural_dynamics_grad(x, h, y_hat, y, beta)
                h = self.activation(h + neural_lr * grad_h)
                if output_sparsity:
                    y_hat = F.relu(y_hat + neural_lr * grad_y - STLAMBD)
                    STLAMBD = (F.relu(STLAMBD + STlambda_lr * (torch.sum(y_hat, 0).view(1, -1) - 1)))
                else:
                    y_hat = self.activation(y_hat + neural_lr * grad_y)
        return h, y_hat
    
    def batch_step(self, x, y_label, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "divide_by_slow_loop_index",
                   neural_lr_decay_multiplier = 0.01, neural_dynamic_iterations_free = 20, 
                   neural_dynamic_iterations_nudged = 4, beta = 1, output_sparsity = False, STlambda_lr = 0.01):
        
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_h, lambda_y = self.lambda_h, self.lambda_y
        gam_h, gam_y = self.gam_h, self.gam_y

        h, y_hat = self.init_neurons(x.size(1), device = self.device)

        # h, y_hat = self.run_neural_dynamics(x, h, y_hat, y_label, neural_lr, 
        #                                     neural_dynamic_iterations_free, 0, output_sparsity, STlambda_lr)
        # neurons1 = [h, y_hat].copy()

        h, y_hat = self.run_neural_dynamics( x, h, y_hat, y_label, neural_lr_start, neural_lr_stop,
                                             neural_dynamic_iterations_nudged, beta, neural_lr_rule,
                                             neural_lr_decay_multiplier, output_sparsity, STlambda_lr)
        neurons2 = [h, y_hat].copy()

        error_hx_nudged = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_nudged = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        err = y_hat-y_label
        error_herr_nudged = (h - (self.Wfb[1]['weight'] @ err + self.Wfb[1]['bias']))
        
        # Wff_old = torch.clone(Wff[0]['weight'])
        ### Weight Updates
        #k = 5  # Below lines output ---> tensor(0., device='cuda:0')
        #torch.norm(outer_prod_broadcasting(error_hx_free.T, x.T)[k] - (torch.outer(error_hx_free[:,k], x[:,k])))
        # Wff[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting((error_hx_free - error_hx_nudged).T, x.T), axis = 0)
        Wff[0]['weight'] += lr['ff'] * torch.mean( outer_prod_broadcasting(error_hx_nudged.T, x.T), axis = 0)
        # Wfb[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_xh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_xh_nudged.T, neurons2[0].T), axis = 0)
        Wff[1]['weight'] += lr['ff'] * torch.mean(  outer_prod_broadcasting(error_yh_nudged.T, neurons2[0].T), axis = 0)
        Wfb[1]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(error_herr_nudged.T, err.T), axis = 0)
        
        Wff[0]['bias'] += lr['ff'] * torch.mean( error_hx_nudged, axis = 1, keepdims = True)
        # Wfb[0]['bias'] -= lr['fb'] * torch.mean(error_xh_nudged - error_xh_free, axis = 1, keepdims = True)
        Wff[1]['bias'] += lr['ff'] * torch.mean( error_yh_nudged, axis = 1, keepdims = True)
        Wfb[1]['bias'] += lr['fb'] * torch.mean( error_herr_nudged, axis = 1, keepdims = True)

        zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)

        B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_nudged, zh_nudged)))
        B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_nudged, zy_nudged)))
        self.B = B

        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
        return h, y_hat

    def batch_step_EP(self, x, y_label, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "divide_by_slow_loop_index",
                   neural_lr_decay_multiplier = 0.01, neural_dynamic_iterations_free = 20, 
                   neural_dynamic_iterations_nudged = 4, beta = 1, output_sparsity = False, STlambda_lr = 0.01):
        
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_h, lambda_y = self.lambda_h, self.lambda_y
        gam_h, gam_y = self.gam_h, self.gam_y

        h, y_hat = self.init_neurons(x.size(1), device = self.device)

        h, y_hat = self.run_neural_dynamics( x, h, y_hat, y_label, neural_lr_start, neural_lr_stop,
                                             neural_dynamic_iterations_nudged, beta, neural_lr_rule,
                                             neural_lr_decay_multiplier, output_sparsity, STlambda_lr)
        neurons1 = [h, y_hat].copy()

        error_hx_free = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_free = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        err_free = y_hat - y_label
        error_herr_free = (h - (self.Wfb[1]['weight'] @ err_free + self.Wfb[1]['bias']))

        zh_free = torch.mean(B[0]['weight'] @ neurons1[0], 1)
        zy_free = torch.mean(B[1]['weight'] @ neurons1[1], 1)

        B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_free, zh_free)))
        B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_free, zy_free)))
        
        h, y_hat = self.run_neural_dynamics( x, h, y_hat, y_label, neural_lr_start, neural_lr_stop,
                                             neural_dynamic_iterations_nudged, beta, neural_lr_rule,
                                             neural_lr_decay_multiplier, output_sparsity, STlambda_lr)
        neurons2 = [h, y_hat].copy()

        error_hx_nudged = h - (self.Wff[0]['weight'] @ x + self.Wff[0]['bias'])
        # error_xh_nudged = x - (self.Wfb[0]['weight'] @ h + self.Wfb[0]['bias'])

        error_yh_nudged = y_hat - (self.Wff[1]['weight'] @ h + self.Wff[1]['bias'])
        err_nudged = y_hat - y_label
        error_herr_nudged = (h - (self.Wfb[1]['weight'] @ err_nudged + self.Wfb[1]['bias']))
        
        # Wff_old = torch.clone(Wff[0]['weight'])
        ### Weight Updates
        #k = 5  # Below lines output ---> tensor(0., device='cuda:0')
        #torch.norm(outer_prod_broadcasting(error_hx_free.T, x.T)[k] - (torch.outer(error_hx_free[:,k], x[:,k])))
        # Wff[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting((error_hx_free - error_hx_nudged).T, x.T), axis = 0)
        # Wff[0]['weight'] += lr['ff'] * torch.mean( outer_prod_broadcasting(error_hx_nudged.T, x.T), axis = 0)
        # # Wfb[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_xh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_xh_nudged.T, neurons2[0].T), axis = 0)
        # Wff[1]['weight'] += lr['ff'] * torch.mean(  outer_prod_broadcasting(error_yh_nudged.T, neurons2[0].T), axis = 0)
        # Wfb[1]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(error_herr_nudged.T, err.T), axis = 0)
        
        # Wff[0]['bias'] += lr['ff'] * torch.mean( error_hx_nudged, axis = 1, keepdims = True)
        # # Wfb[0]['bias'] -= lr['fb'] * torch.mean(error_xh_nudged - error_xh_free, axis = 1, keepdims = True)
        # Wff[1]['bias'] += lr['ff'] * torch.mean( error_yh_nudged, axis = 1, keepdims = True)
        # Wfb[1]['bias'] += lr['fb'] * torch.mean( error_herr_nudged, axis = 1, keepdims = True)

        # zh_nudged = torch.mean(B[0]['weight'] @ neurons2[0], 1)
        # zy_nudged = torch.mean(B[1]['weight'] @ neurons2[1], 1)

        # B[0]['weight'] = (1 / lambda_h) * (B[0]['weight'] - gam_h * (torch.outer(zh_nudged, zh_nudged)))
        # B[1]['weight'] = (1 / lambda_y) * (B[1]['weight'] - gam_y * (torch.outer(zy_nudged, zy_nudged)))

        Wff[0]['weight'] -= (1 / beta) * lr['ff'] * torch.mean(outer_prod_broadcasting(error_hx_free.T, x.T) - outer_prod_broadcasting(error_hx_nudged.T, x.T), axis = 0)
        # Wfb[0]['weight'] -= lr['ff'] * torch.mean(outer_prod_broadcasting(error_xh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_xh_nudged.T, neurons2[0].T), axis = 0)
        Wff[1]['weight'] -= (1 / beta) * lr['ff'] * torch.mean(outer_prod_broadcasting(error_yh_free.T, neurons1[0].T) - outer_prod_broadcasting(error_yh_nudged.T, neurons2[0].T), axis = 0)
        Wfb[1]['weight'] -= (1 / beta) * lr['fb'] * torch.mean(outer_prod_broadcasting(error_herr_free.T, err_free.T) - outer_prod_broadcasting(error_herr_nudged.T, err_nudged.T), axis = 0)
        
        Wff[0]['bias'] -= (1 / beta) * lr['ff'] * torch.mean(error_hx_free - error_hx_nudged, axis = 1, keepdims = True)
        # Wfb[0]['bias'] -= lr['fb'] * torch.mean(error_xh_nudged - error_xh_free, axis = 1, keepdims = True)
        Wff[1]['bias'] -= (1 / beta) * lr['ff'] * torch.mean(error_yh_free - error_yh_nudged, axis = 1, keepdims = True)
        Wfb[1]['bias'] -= (1 / beta) * lr['fb'] * torch.mean(error_herr_free - error_herr_nudged, axis = 1, keepdims = True)

        
        self.B = B

        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
        return h, y_hat