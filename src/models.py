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
        self.t = 0 # Used if optimizer is Adam
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
            
        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            m_Wff_moment.append({'weight': weight, 'bias': bias})
        m_Wff_moment = np.array(m_Wff_moment)

        v_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            v_Wff_moment.append({'weight': weight, 'bias': bias})
        v_Wff_moment = np.array(v_Wff_moment)

        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wfb_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            m_Wfb_moment.append({'weight': weight, 'bias': bias})
        m_Wfb_moment = np.array(m_Wfb_moment)

        v_Wfb_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            v_Wfb_moment.append({'weight': weight, 'bias': bias})
        v_Wfb_moment = np.array(v_Wfb_moment)

        self.Wff = Wff
        self.Wfb = Wfb
        self.m_Wff_moment = m_Wff_moment
        self.m_Wfb_moment = m_Wfb_moment
        self.v_Wff_moment = v_Wff_moment
        self.v_Wfb_moment = v_Wfb_moment
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
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, beta = 1, mode = "train",
                   optimizer = "sgd", adam_opt_params = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}):

        if optimizer == "adam":
            t = self.t
            m_Wff_moment = self.m_Wff_moment
            v_Wff_moment = self.v_Wff_moment
            m_Wfb_moment = self.m_Wfb_moment
            v_Wfb_moment = self.v_Wfb_moment

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

        if optimizer == "sgd":
            ### Learning updates for feed-forward and backward weights
            for jj in range(len(Wff)):
                Wff[jj]['weight'] += lr['ff'] * torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers[jj].T), axis = 0)
                Wff[jj]['bias'] += lr['ff'] * torch.mean(forward_errors[jj], axis = 1, keepdims = True)

            for jj in range(1, len(Wfb)):
                Wfb[jj]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
                Wfb[jj]['bias'] += lr['fb'] * torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

            self.Wff = Wff
            self.Wfb = Wfb

        elif optimizer == "adam":
            t += 1
            self.t = t
            for jj in range(len(Wff)):
                grad_Wff_weight = torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers[jj].T), axis = 0)
                grad_Wff_bias = torch.mean(forward_errors[jj], axis = 1, keepdims = True)

                m_Wff_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wff_weight
                m_Wff_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wff_bias

                v_Wff_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_weight ** 2)
                v_Wff_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_bias ** 2)

                Wff[jj]['weight'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["weight"] / (torch.sqrt(v_Wff_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wff[jj]['bias'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["bias"] / (torch.sqrt(v_Wff_moment[jj]["bias"]) + adam_opt_params["eps"])

            # for jj in range(1, len(Wfb)):
            #     Wfb[jj]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
            #     Wfb[jj]['bias'] += lr['fb'] * torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

            for jj in range(1, len(Wfb)):
                grad_Wfb_weight = torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
                grad_Wfb_bias = torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

                m_Wfb_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wfb_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wfb_weight
                m_Wfb_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wfb_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wfb_bias

                v_Wfb_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wfb_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wfb_weight ** 2)
                v_Wfb_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wfb_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wfb_bias ** 2)

                Wfb[jj]['weight'] += lr['fb'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wfb_moment[jj]["weight"] / (torch.sqrt(v_Wfb_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wfb[jj]['bias'] += lr['fb'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wfb_moment[jj]["bias"] / (torch.sqrt(v_Wfb_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            self.Wff = Wff
            self.Wfb = Wfb
            self.m_Wff_moment = m_Wff_moment
            self.v_Wff_moment = v_Wff_moment
            self.m_Wfb_moment = m_Wfb_moment
            self.v_Wfb_moment = v_Wfb_moment
        return neurons


class CorInfoMaxErrorProp():
    
    def __init__(self, architecture, lambda_, epsilon, psiv, activation_type = "sigmoid", output_sparsity = True, STlambda_lr = 0.01):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.psiv = psiv
        self.one_over_epsilon = 1 / epsilon
        self.activation_type = activation_type
        self.output_sparsity = output_sparsity
        self.STlambda_lr = STlambda_lr
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.t = 0 # Used if optimizer is Adam
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = (2 * torch.rand(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device) - 1) * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
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
            
        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            m_Wff_moment.append({'weight': weight, 'bias': bias})
        m_Wff_moment = np.array(m_Wff_moment)

        v_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            v_Wff_moment.append({'weight': weight, 'bias': bias})
        v_Wff_moment = np.array(v_Wff_moment)

        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wfb_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            m_Wfb_moment.append({'weight': weight, 'bias': bias})
        m_Wfb_moment = np.array(m_Wfb_moment)

        v_Wfb_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            v_Wfb_moment.append({'weight': weight, 'bias': bias})
        v_Wfb_moment = np.array(v_Wfb_moment)

        self.Wff = Wff
        self.Wfb = Wfb
        self.m_Wff_moment = m_Wff_moment
        self.m_Wfb_moment = m_Wfb_moment
        self.v_Wff_moment = v_Wff_moment
        self.v_Wfb_moment = v_Wfb_moment
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
        psiv = self.psiv

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [neurons[-1] - y]
        for jj in reversed(range(1, len(Wfb))):
            activated = backward_errors[-1]
            backward_errors.append(neurons[jj - 1] - (Wfb[jj]['weight'] @ activated + Wfb[jj]['bias']))
        
        backward_errors = list(reversed(backward_errors))
        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                init_grads[jj] = (1 - psiv) * gam_ * B[jj]['weight'] @ layers[jj + 1] - (1 - psiv) * one_over_epsilon * (forward_errors[jj]) + 2 * beta * (y - layers[jj + 1])
            else:
                init_grads[jj] = (1 - 2 * psiv) * gam_ * B[jj]['weight'] @ layers[jj + 1] - (1 - psiv) * one_over_epsilon * (forward_errors[jj]) + psiv * one_over_epsilon * (backward_errors[jj])
        return init_grads

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10, beta = 1, clip_grad_updates = False):
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                neuron_grads = self.calculate_neural_dynamics_grad(x, y, neurons, beta)
                for neuron_iter in range(len(neurons)):
                    if clip_grad_updates:
                        neurons[neuron_iter] = hard_sigmoid(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
                    else:
                        neurons[neuron_iter] = neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter]

        return neurons

    def batch_step(self, x, y, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, beta = 1, mode = "train",
                   clip_neural_grad_updates = False, optimizer = "sgd", adam_opt_params = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}):

        if optimizer == "adam":
            t = self.t
            m_Wff_moment = self.m_Wff_moment
            v_Wff_moment = self.v_Wff_moment
            m_Wfb_moment = self.m_Wfb_moment
            v_Wfb_moment = self.v_Wfb_moment

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.fast_forward(x)

        # if mode == "train":
        #     neurons[-1] = y.to(torch.float)

        neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations, beta, clip_neural_grad_updates)

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
        
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers) - 1)] + [neurons[-1]]
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        ## Compute backward errors
        backward_errors = [neurons[-1] - y]
        for jj in reversed(range(1, len(Wfb))):
            activated = backward_errors[-1]
            backward_errors.append(neurons[jj - 1] - (Wfb[jj]['weight'] @ activated + Wfb[jj]['bias']))
        
        backward_errors = list(reversed(backward_errors))
        if optimizer == "sgd":
            ### Learning updates for feed-forward and backward weights
            for jj in range(len(Wff)):
                Wff[jj]['weight'] += lr['ff'] * torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers_after_activation[jj][0].T), axis = 0)
                Wff[jj]['bias'] += lr['ff'] * torch.mean(forward_errors[jj], axis = 1, keepdims = True)

            for jj in range(1, len(Wfb)):
                Wfb[jj]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, backward_errors[jj].T), axis = 0)
                Wfb[jj]['bias'] += lr['fb'] * torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

            self.Wff = Wff
            self.Wfb = Wfb

        elif optimizer == "adam":
            t += 1
            self.t = t
            for jj in range(len(Wff)):
                grad_Wff_weight = torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers_after_activation[jj][0].T), axis = 0)
                grad_Wff_bias = torch.mean(forward_errors[jj], axis = 1, keepdims = True)

                m_Wff_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wff_weight
                m_Wff_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wff_bias

                v_Wff_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_weight ** 2)
                v_Wff_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_bias ** 2)

                Wff[jj]['weight'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["weight"] / (torch.sqrt(v_Wff_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wff[jj]['bias'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["bias"] / (torch.sqrt(v_Wff_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            for jj in range(1, len(Wfb)):
                grad_Wfb_weight = torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, backward_errors[jj].T), axis = 0)
                # print(torch.norm(grad_Wfb_weight))
                grad_Wfb_bias = torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

                m_Wfb_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wfb_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wfb_weight
                m_Wfb_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wfb_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wfb_bias

                v_Wfb_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wfb_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wfb_weight ** 2)
                v_Wfb_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wfb_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wfb_bias ** 2)

                Wfb[jj]['weight'] += lr['fb'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wfb_moment[jj]["weight"] / (torch.sqrt(v_Wfb_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wfb[jj]['bias'] += lr['fb'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wfb_moment[jj]["bias"] / (torch.sqrt(v_Wfb_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            self.Wff = Wff
            self.Wfb = Wfb
            self.m_Wff_moment = m_Wff_moment
            self.v_Wff_moment = v_Wff_moment
            self.m_Wfb_moment = m_Wfb_moment
            self.v_Wfb_moment = v_Wfb_moment
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
        self.t = 0 # Used if optimizer is Adam

        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = torch.eye(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = torch.eye(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            Wfb.append({'weight': weight, 'bias': bias})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            weight = 0.1*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)

        # Lateral Synapses Initialization
        Bsigma = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            weight = 0.1*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            Bsigma.append({'weight': weight})
        Bsigma = np.array(Bsigma)
            
        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            m_Wff_moment.append({'weight': weight, 'bias': bias})
        m_Wff_moment = np.array(m_Wff_moment)

        v_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            v_Wff_moment.append({'weight': weight, 'bias': bias})
        v_Wff_moment = np.array(v_Wff_moment)

        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wfb_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            m_Wfb_moment.append({'weight': weight, 'bias': bias})
        m_Wfb_moment = np.array(m_Wfb_moment)

        v_Wfb_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            v_Wfb_moment.append({'weight': weight, 'bias': bias})
        v_Wfb_moment = np.array(v_Wfb_moment)

        self.Wff = Wff
        self.Wfb = Wfb
        self.m_Wff_moment = m_Wff_moment
        self.m_Wfb_moment = m_Wfb_moment
        self.v_Wff_moment = v_Wff_moment
        self.v_Wfb_moment = v_Wfb_moment
        self.B = B
        self.Bsigma = Bsigma
        
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
                neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
            else:
                neurons.append(Wff[jj]['weight'] @ self.activation_func(neurons[-1], self.activation_type)[0] + Wff[jj]['bias'])
        return neurons

    def calculate_neural_dynamics_grad(self, x, y, neurons, beta, mode = "train"):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        Bsigma = self.Bsigma
        gam_ = self.gam_
        one_over_epsilon = self.one_over_epsilon

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]
        # layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        layers_after_activation = [[x, 1]] + [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(1, len(layers) - 1)] + [neurons[-1]]
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [layers_after_activation[jj][0] - (Wfb[jj]['weight'] @ layers[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                if mode == "train":
                    init_grads[jj] = torch.zeros(*layers[jj + 1].shape, device = self.device)
                else:
                    init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (forward_errors[jj]) 
            else:
                init_grads[jj] = gam_ * Bsigma[jj]['weight'] @ (layers_after_activation[jj + 1][0] * layers_after_activation[jj + 1][1]) + gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (forward_errors[jj]) - one_over_epsilon * (backward_errors[jj] * layers_after_activation[jj+1][1])
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
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, beta = 1, mode = "train",
                   optimizer = "sgd", adam_opt_params = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}):

        if optimizer == "adam":
            t = self.t
            m_Wff_moment = self.m_Wff_moment
            v_Wff_moment = self.v_Wff_moment
            m_Wfb_moment = self.m_Wfb_moment
            v_Wfb_moment = self.v_Wfb_moment

        Wff, Wfb, B, Bsigma = self.Wff, self.Wfb, self.B, self.Bsigma
        lambda_ = self.lambda_
        gam_ = self.gam_

        neurons = self.init_neurons(x.size(1), device = self.device)
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

        for jj in range(len(Bsigma)):
            z = Bsigma[jj]['weight'] @ self.activation_func(neurons[jj], self.activation_type)[0]
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            Bsigma[jj]['weight'] = (1 / lambda_) * (Bsigma[jj]['weight'] - gam_ * B_update)

        self.Bsigma = Bsigma

        layers = [x] + neurons
        # ## Compute forward errors
        # forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        # ## Compute backward errors
        # backward_errors = [layers[jj] - (Wfb[jj]['weight'] @ layers[jj + 1] - Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        # layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        layers_after_activation = [[x, 1]] + [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(1, len(layers) - 1)] + [neurons[-1]]
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [layers_after_activation[jj][0] - (Wfb[jj]['weight'] @ layers[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        if optimizer == "sgd":
            ### Learning updates for feed-forward and backward weights
            for jj in range(len(Wff)):
                Wff[jj]['weight'] += lr['ff'] * torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers_after_activation[jj][0].T), axis = 0)
                Wff[jj]['bias'] += lr['ff'] * torch.mean(forward_errors[jj], axis = 1, keepdims = True)

            for jj in range(1, len(Wfb)):
                Wfb[jj]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
                Wfb[jj]['bias'] += lr['fb'] * torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

            self.Wff = Wff
            self.Wfb = Wfb

        elif optimizer == "adam":
            t += 1
            self.t = t
            for jj in range(len(Wff)):
                grad_Wff_weight = torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers_after_activation[jj][0].T), axis = 0)
                grad_Wff_bias = torch.mean(forward_errors[jj], axis = 1, keepdims = True)

                m_Wff_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wff_weight
                m_Wff_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wff_bias

                v_Wff_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_weight ** 2)
                v_Wff_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_bias ** 2)

                Wff[jj]['weight'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["weight"] / (torch.sqrt(v_Wff_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wff[jj]['bias'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["bias"] / (torch.sqrt(v_Wff_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            for jj in range(1, len(Wfb)):
                grad_Wfb_weight = torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
                # print(torch.norm(grad_Wfb_weight))
                grad_Wfb_bias = torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

                m_Wfb_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wfb_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wfb_weight
                m_Wfb_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wfb_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wfb_bias

                v_Wfb_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wfb_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wfb_weight ** 2)
                v_Wfb_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wfb_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wfb_bias ** 2)

                Wfb[jj]['weight'] += lr['fb'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wfb_moment[jj]["weight"] / (torch.sqrt(v_Wfb_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wfb[jj]['bias'] += lr['fb'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wfb_moment[jj]["bias"] / (torch.sqrt(v_Wfb_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            self.Wff = Wff
            self.Wfb = Wfb
            self.m_Wff_moment = m_Wff_moment
            self.v_Wff_moment = v_Wff_moment
            self.m_Wfb_moment = m_Wfb_moment
            self.v_Wfb_moment = v_Wfb_moment
        return neurons

class CorInfoMaxV3():
    
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
        self.t = 0 # Used if optimizer is Adam

        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = torch.eye(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = torch.eye(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            Wfb.append({'weight': weight, 'bias': bias})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            # weight = 1*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)

        # Lateral Synapses Initialization
        Bsigma = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            # weight = 1*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            Bsigma.append({'weight': weight})
        Bsigma = np.array(Bsigma)
            
        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            m_Wff_moment.append({'weight': weight, 'bias': bias})
        m_Wff_moment = np.array(m_Wff_moment)

        v_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            v_Wff_moment.append({'weight': weight, 'bias': bias})
        v_Wff_moment = np.array(v_Wff_moment)

        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wfb_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            m_Wfb_moment.append({'weight': weight, 'bias': bias})
        m_Wfb_moment = np.array(m_Wfb_moment)

        v_Wfb_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            v_Wfb_moment.append({'weight': weight, 'bias': bias})
        v_Wfb_moment = np.array(v_Wfb_moment)

        self.Wff = Wff
        self.Wfb = Wfb
        self.m_Wff_moment = m_Wff_moment
        self.m_Wfb_moment = m_Wfb_moment
        self.v_Wff_moment = v_Wff_moment
        self.v_Wfb_moment = v_Wfb_moment
        self.B = B
        self.Bsigma = Bsigma
        
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
        elif type_ == "softmax":
            f_x = F.softmax(x, 0)
            fp_x = torch.diag_embed(f_x.T) - torch.einsum('ij, ik -> ijk', f_x.T, f_x.T)
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
                neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
            else:
                neurons.append(Wff[jj]['weight'] @ self.activation_func(neurons[-1], self.activation_type)[0] + Wff[jj]['bias'])
        return neurons

    def calculate_neural_dynamics_grad(self, x, y, neurons, beta, mode = "train"):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        Bsigma = self.Bsigma
        gam_ = self.gam_
        one_over_epsilon = self.one_over_epsilon

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]
        # layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        layers_after_activation = [[x, 1]] + [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(1, len(layers) - 1)] + [neurons[-1]]
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [layers_after_activation[jj][0] - (Wfb[jj]['weight'] @ layers[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                if mode == "train":
                    init_grads[jj] = torch.zeros(*layers[jj + 1].shape, device = self.device)
                else:
                    init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (forward_errors[jj]) 
            else:
                try:
                    init_grads[jj] = gam_ * Bsigma[jj]['weight'] @ (layers_after_activation[jj + 1][0] * layers_after_activation[jj + 1][1]) + gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (forward_errors[jj]) - one_over_epsilon * (backward_errors[jj] * layers_after_activation[jj+1][1])
                except:
                    init_grads[jj] = torch.bmm(layers_after_activation[jj + 1][1], (gam_ * Bsigma[jj]['weight'] @ (layers_after_activation[jj + 1][0])).T.unsqueeze(2)).squeeze(2).T + gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (forward_errors[jj]) - one_over_epsilon * torch.bmm(layers_after_activation[jj+1][1], backward_errors[jj].T.unsqueeze(2)).squeeze(2).T
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
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, beta = 1, mode = "train",
                   optimizer = "sgd", adam_opt_params = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}):

        if optimizer == "adam":
            t = self.t
            m_Wff_moment = self.m_Wff_moment
            v_Wff_moment = self.v_Wff_moment
            m_Wfb_moment = self.m_Wfb_moment
            v_Wfb_moment = self.v_Wfb_moment

        Wff, Wfb, B, Bsigma = self.Wff, self.Wfb, self.B, self.Bsigma
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

        for jj in range(len(Bsigma)):
            z = Bsigma[jj]['weight'] @ self.activation_func(neurons[jj], self.activation_type)[0]
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            Bsigma[jj]['weight'] = (1 / lambda_) * (Bsigma[jj]['weight'] - gam_ * B_update)

        self.Bsigma = Bsigma

        layers = [x] + neurons
        # ## Compute forward errors
        # forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        # ## Compute backward errors
        # backward_errors = [layers[jj] - (Wfb[jj]['weight'] @ layers[jj + 1] - Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        # layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        layers_after_activation = [[x, 1]] + [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(1, len(layers) - 1)] + [neurons[-1]]
        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [layers_after_activation[jj][0] - (Wfb[jj]['weight'] @ layers[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        
        if optimizer == "sgd":
            ### Learning updates for feed-forward and backward weights
            for jj in range(len(Wff)):
                Wff[jj]['weight'] += lr['ff'] * torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers_after_activation[jj][0].T), axis = 0)
                Wff[jj]['bias'] += lr['ff'] * torch.mean(forward_errors[jj], axis = 1, keepdims = True)

            for jj in range(1, len(Wfb)):
                Wfb[jj]['weight'] += lr['fb'] * torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
                Wfb[jj]['bias'] += lr['fb'] * torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

            self.Wff = Wff
            self.Wfb = Wfb

        elif optimizer == "adam":
            t += 1
            self.t = t
            for jj in range(len(Wff)):
                grad_Wff_weight = torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers_after_activation[jj][0].T), axis = 0)
                grad_Wff_bias = torch.mean(forward_errors[jj], axis = 1, keepdims = True)

                m_Wff_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wff_weight
                m_Wff_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wff_bias

                v_Wff_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_weight ** 2)
                v_Wff_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_bias ** 2)

                Wff[jj]['weight'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["weight"] / (torch.sqrt(v_Wff_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wff[jj]['bias'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["bias"] / (torch.sqrt(v_Wff_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            for jj in range(1, len(Wfb)):
                grad_Wfb_weight = torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
                # print(torch.norm(grad_Wfb_weight))
                grad_Wfb_bias = torch.mean(backward_errors[jj - 1], axis = 1, keepdims = True)

                m_Wfb_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wfb_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wfb_weight
                m_Wfb_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wfb_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wfb_bias

                v_Wfb_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wfb_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wfb_weight ** 2)
                v_Wfb_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wfb_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wfb_bias ** 2)

                Wfb[jj]['weight'] += lr['fb'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wfb_moment[jj]["weight"] / (torch.sqrt(v_Wfb_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wfb[jj]['bias'] += lr['fb'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wfb_moment[jj]["bias"] / (torch.sqrt(v_Wfb_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            self.Wff = Wff
            self.Wfb = Wfb
            self.m_Wff_moment = m_Wff_moment
            self.v_Wff_moment = v_Wff_moment
            self.m_Wfb_moment = m_Wfb_moment
            self.v_Wfb_moment = v_Wfb_moment
        return neurons

class CorInfoMaxV0():
    
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
        self.t = 0 # Used if optimizer is Adam
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)
            
        ### Following moments are used if Adam optimizer is selected in the batch step function
        m_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            m_Wff_moment.append({'weight': weight, 'bias': bias})
        m_Wff_moment = np.array(m_Wff_moment)

        v_Wff_moment = []
        for idx in range(len(architecture)-1):
            weight = torch.zeros(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            v_Wff_moment.append({'weight': weight, 'bias': bias})
        v_Wff_moment = np.array(v_Wff_moment)

        self.Wff = Wff
        self.m_Wff_moment = m_Wff_moment
        self.v_Wff_moment = v_Wff_moment
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

    def fast_forward(self, x):
        Wff = self.Wff
        neurons = []
        for jj in range(len(Wff)):
            if jj == 0:
                neurons.append(Wff[jj]['weight'] @ self.activation_func(x, self.activation_type)[0] + Wff[jj]['bias'])
            else:
                neurons.append(Wff[jj]['weight'] @ self.activation_func(neurons[-1], self.activation_type)[0] + Wff[jj]['bias'])
        return neurons

    def calculate_neural_dynamics_grad(self, x, y, neurons, mode = "train"):
        Wff = self.Wff
        B = self.B
        gam_ = self.gam_
        one_over_epsilon = self.one_over_epsilon

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        error_layers = [(layers[jj+1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias'])) for jj in range(len(layers) - 1)]
        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                if mode == "train":
                    init_grads[jj] = torch.zeros(*layers[jj + 1].shape, device = self.device)
                else:
                    init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + 2 * beta * (y - layers[jj + 1])
            else:
                init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * error_layers[jj] + one_over_epsilon * (Wff[jj + 1]['weight'].T @ error_layers[jj + 1]) * layers_after_activation[jj + 1][1]
        return init_grads

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10 ):
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                neuron_grads = self.calculate_neural_dynamics_grad(x, y, neurons)
                for neuron_iter in range(len(neurons)):
                    neurons[neuron_iter] = neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter]

        return neurons

    def batch_step(self, x, y, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, mode = "train",
                   optimizer = "sgd", adam_opt_params = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}):
        
        if optimizer == "adam":
            t = self.t
            m_Wff_moment = self.m_Wff_moment
            v_Wff_moment = self.v_Wff_moment

        Wff = self.Wff
        B = self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        neurons = self.fast_forward(x)

        if mode == "train":
            neurons[-1] = y.to(torch.float)

        neurons = self.run_neural_dynamics( x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                            neural_lr_decay_multiplier, neural_dynamic_iterations)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ neurons[jj]
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        self.B = B

        layers = [x] + neurons  # concatenate the input to other layers
        # pc_loss = self.PC_loss(x, neurons).mean()
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers) - 1)] + [neurons[-1]]
        error_layers = [(layers[jj+1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias'])) for jj in range(len(layers) - 1)]

        if optimizer == "sgd":
            ### Learning updates for feed-forward and backward weights
            for jj in range(len(Wff)):
                Wff[jj]['weight'] += lr['ff'] * torch.mean(outer_prod_broadcasting(error_layers[jj].T, layers_after_activation[jj][0].T), axis = 0)
                Wff[jj]['bias'] += lr['ff'] * torch.mean(error_layers[jj], axis = 1, keepdims = True)

            self.Wff = Wff
            
        elif optimizer == "adam":
            t += 1
            self.t = t
            for jj in range(len(Wff)):
                grad_Wff_weight = torch.mean(outer_prod_broadcasting(error_layers[jj].T, layers_after_activation[jj][0].T), axis = 0)
                grad_Wff_bias = torch.mean(error_layers[jj], axis = 1, keepdims = True)

                m_Wff_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wff_weight
                m_Wff_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wff_bias

                v_Wff_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_weight ** 2)
                v_Wff_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_bias ** 2)

                Wff[jj]['weight'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["weight"] / (torch.sqrt(v_Wff_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wff[jj]['bias'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["bias"] / (torch.sqrt(v_Wff_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            self.Wff = Wff
            self.m_Wff_moment = m_Wff_moment
            self.v_Wff_moment = v_Wff_moment

        return neurons