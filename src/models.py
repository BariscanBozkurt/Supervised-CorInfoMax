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

class CorInfoMax():
    
    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid, output_sparsity = True, STlambda_lr = 0.01):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.output_sparsity = output_sparsity
        self.STlambda_lr = STlambda_lr
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
            weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
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

    def fast_forward(self, x):
        Wff = self.Wff
        neurons = []
        for jj in range(len(Wff)):
            if jj == 0:
                neurons.append(self.activation(Wff[jj]['weight'] @ x + Wff[jj]['bias']))
            else:
                neurons.append(self.activation(Wff[jj]['weight'] @ neurons[-1] + Wff[jj]['bias']))
        return neurons

    def calculate_neural_dynamics_grad(self, x, y, neurons, beta, mode = "train"):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_ = self.gam_
        one_over_epsilon = self.one_over_epsilon

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]

        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                if mode == "train":
                    init_grads[jj] = torch.zeros(*layers[jj + 1].shape, device = self.device)
                else:
                    init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + 2 * beta * (y - layers[jj + 1])
            else:
                init_grads[jj] = 2 * gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) - one_over_epsilon * (layers[jj + 1] - (Wfb[jj + 1]['weight'] @ layers[jj + 2] + Wfb[jj + 1]['bias']))
        return init_grads

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10, beta = 1, mode = "train"):
        if self.output_sparsity:
            mbs = x.size(1)
            STLAMBD = torch.zeros(1, mbs).to(self.device)
            STlambda_lr = self.STlambda_lr
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                neuron_grads = self.calculate_neural_dynamics_grad(x, y, neurons, beta, mode)

                for neuron_iter in range(len(neurons)):
                    if neuron_iter == len(neurons) - 1:
                        if self.output_sparsity:
                            neurons[neuron_iter] = F.relu(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter] - STLAMBD)
                            STLAMBD = F.relu(STLAMBD + STlambda_lr * (torch.sum(neurons[neuron_iter], 0).view(1, -1) - 1))
                        else:
                            neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
                    else:
                        neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        return neurons

    def batch_step(self, x, y, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, beta = 1, mode = "train"):
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.fast_forward(x)

        if mode == "train":
            neurons[-1] = y.to(torch.float)

        neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations, beta)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ neurons[jj]
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        self.B = B

        layers = [x] + neurons
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [layers[jj] - (Wfb[jj]['weight'] @ layers[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += lr['ff'] * torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers[jj].T), axis = 0)
            Wff[jj]['bias'] += lr['ff'] * torch.mean(forward_errors[jj], axis = 1, keepdims = True)

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
            Wfb[jj]['bias'] += lr['fb'] * torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

        self.Wff = Wff
        self.Wfb = Wfb
        return neurons



class CorInfoMaxV2():
    
    def __init__(self, architecture, lambda_, epsilon, activation_type = "sigmoid", output_sparsity = True, STlambda_lr = 0.01):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation_type = activation_type
        self.output_sparsity = output_sparsity
        self.STlambda_lr = STlambda_lr
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
            weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)
            
        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        
    def activation_func(self, x, type_ = "linear"):
        if type_ == "linear":
            f_x = x
            fp_x = torch.ones(*x.shape, device = x.device)
        elif type_ == "tanh":
            f_x = torch.tanh(x)
            fp_x = torch.ones(*x.shape, device = x.device) - f_x ** 2
        elif type_ == "sigmoid":
            ones_vec = torch.ones(*x.shape, device = x.device)
            f_x = 1 / (ones_vec + torch.exp(-x))
            fp_x = f_x * (ones_vec - f_x)
        elif type_ == "relu":
            f_x = torch.maximum(x, torch.tensor([0], device = x.device))
            fp_x = 1 * (x > 0)
        elif type_ == "exp":
            f_x = torch.exp(x)
            fp_x = f_x
        else: # Use linear
            f_x = x
            fp_x = torch.ones(*x.shape, device = x.device)
            
        return f_x, fp_x

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

    def fast_forward(self, x):
        Wff = self.Wff
        neurons = []
        for jj in range(len(Wff)):
            if jj == 0:
                neurons.append(Wff[jj]['weight'] @ self.activation_func(x, self.activation_type)[0] + Wff[jj]['bias'])
            else:
                neurons.append(Wff[jj]['weight'] @ self.activation_func(neurons[-1], self.activation_type)[0] + Wff[jj]['bias'])
        return neurons

    def calculate_neural_dynamics_grad(self, x, y, neurons, beta, mode = "train"):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_ = self.gam_
        one_over_epsilon = self.one_over_epsilon

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [layers_after_activation[jj][0] - (Wfb[jj]['weight'] @ layers[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                if mode == "train":
                    init_grads[jj] = torch.zeros(*layers[jj + 1].shape, device = self.device)
                else:
                    init_grads[jj] = gam_ * B[jj]['weight'] @ layers_after_activation[jj + 1][0] - one_over_epsilon * (forward_errors[jj]) 
            else:
                init_grads[jj] = 2 * gam_ * B[jj]['weight'] @ layers_after_activation[jj + 1][0] - one_over_epsilon * (forward_errors[jj]) - one_over_epsilon * (backward_errors[jj] * layers_after_activation[jj+1][1])
        return init_grads

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10, beta = 1, mode = "train"):
        if self.output_sparsity:
            mbs = x.size(1)
            STLAMBD = torch.zeros(1, mbs).to(self.device)
            STlambda_lr = self.STlambda_lr
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                neuron_grads = self.calculate_neural_dynamics_grad(x, y, neurons, beta, mode)
                for neuron_iter in range(len(neurons)):
                    neurons[neuron_iter] = neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter]
                # for neuron_iter in range(len(neurons)):
                #     if neuron_iter == len(neurons) - 1:
                #         if self.output_sparsity:
                #             neurons[neuron_iter] = F.relu(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter] - STLAMBD)
                #             STLAMBD = F.relu(STLAMBD + STlambda_lr * (torch.sum(neurons[neuron_iter], 0).view(1, -1) - 1))
                #         else:
                #             neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
                #     else:
                #         neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        return neurons

    def batch_step(self, x, y, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, beta = 1, mode = "train"):
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.fast_forward(x)

        if mode == "train":
            neurons[-1] = y.to(torch.float)

        neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations, beta)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ neurons[jj]
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        self.B = B

        layers = [x] + neurons
        # ## Compute forward errors
        # forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        # ## Compute backward errors
        # backward_errors = [layers[jj] - (Wfb[jj]['weight'] @ layers[jj + 1] - Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [layers_after_activation[jj][0] - (Wfb[jj]['weight'] @ layers[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += lr['ff'] * torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers_after_activation[jj][0].T), axis = 0)
            Wff[jj]['bias'] += lr['ff'] * torch.mean(forward_errors[jj], axis = 1, keepdims = True)

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
            Wfb[jj]['bias'] += lr['fb'] * torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

        self.Wff = Wff
        self.Wfb = Wfb
        return neurons