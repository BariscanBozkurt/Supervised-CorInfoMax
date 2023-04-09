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
import matplotlib


class EP(torch.nn.Module):
    #TODO : Add structured docstring for understandibility
    """
    Modified from https://github.com/Laborieux-Axel/Equilibrium-Propagation/blob/master/model_utils.py
    This EP Class is a little bit different from the one taken from the above github page. The above one uses fixed point iteration in the 
    neural dynamics, i.e., s_(t+1) = sigma( dPhi/ds ), whereas in this implementation we use s_(t+1) = s(t) - neural_lr * sigma( dPhi/ds )
    """
    def __init__(self, architecture, activation = hard_sigmoid):
        super(EP, self).__init__()
        
        self.activation = activation
        self.architecture = architecture 
        self.nc = self.architecture[-1]

        # Feedforward and Feedback Synapses Initialization
        self.W = torch.nn.ModuleList()
        for idx in range(len(architecture)-1):
            m = torch.nn.Linear(architecture[idx], architecture[idx+1], bias=True)
            torch.nn.init.xavier_uniform_(m.weight)
            # m.weight.data.mul_(torch.tensor([1]))
            if m.bias is not None:
                m.bias.data.mul_(0)
            self.W.append(m)

    def Phi(self, x, y, neurons, beta, criterion):
        # Computes the primitive function given static input x, label y, neurons is the sequence of hidden layers neurons
        # criterion is the loss 
        x = x.view(x.size(0),-1) # flattening the input
        
        layers = [x] + neurons  # concatenate the input to other layers
        
        # Primitive function computation
        phi = 0.0
        for idx in range(len(neurons)): # Squared Norms
            phi += 0.5*torch.sum( neurons[idx] * neurons[idx], dim=1).squeeze() # Scalar product s_n.s_n
        for idx in range(len(self.W)): # Linear Terms and Quadratic Terms
            phi -= torch.sum( self.W[idx](layers[idx]) * layers[idx+1], dim=1).squeeze() # Scalar product s_n.W.s_n-1

        if beta!=0.0: # Nudging the output layer when beta is non zero 
            if criterion.__class__.__name__.find('MSE')!=-1:
                y = F.one_hot(y, num_classes=self.nc)
                L = criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].float(), y).squeeze()     
            phi += beta*L
        
        return phi
    
    
    def forward(self, x, y, neurons, T, neural_lr = 0.5, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion) # Computing Phi
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm) # dPhi/ds

            with torch.no_grad():
                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation(neurons[idx] - neural_lr * grads[idx] )  # s_(t+1) = s_(t) - neural_lr * sigma( dPhi/ds )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True
             
            if not_mse:
                neurons[-1] = grads[-1]
            else:
                with torch.no_grad():
                    neurons[-1] = self.activation(neurons[-1] - neural_lr * grads[-1] )

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons


    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neurons = []
        append = neurons.append
        for size in self.architecture[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_2 - beta_1)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem

class CSM(torch.nn.Module):
    """
    Contrastive Similarity Matching for Supervised Learning.
    Paper :                             https://arxiv.org/abs/2002.10378
    Published Official Theano Code :    https://github.com/Pehlevan-Group/Supervised-Similarity-Matching
    """
    def __init__(self, architecture, activation, alphas_W, alphas_M, task = "classification"):
        super(CSM, self).__init__()
        
        self.activation = activation
        self.architecture = architecture 
        self.nc = self.architecture[-1]
        self.task = task
        # Feedforward and Feedboack Synapses Initialization
        self.W = torch.nn.ModuleList()
        for idx in range(len(architecture)-1):
            m = torch.nn.Linear(architecture[idx], architecture[idx+1], bias=True)
            torch.nn.init.xavier_uniform_(m.weight)
            # m.weight.data.mul_(torch.tensor([1]))
            if m.bias is not None:
                m.bias.data.mul_(0)
            self.W.append(m)

        # Lateral Synapses Initialization
        self.M = torch.nn.ModuleList()
        for idx in range(1,len(architecture)-1):
            m = torch.nn.Linear(architecture[idx], architecture[idx], bias = False)
            torch.nn.init.xavier_uniform_(m.weight)
            m.weight.data = m.weight.data @ m.weight.data.T
            self.M.append(m)

        self.M_copy = torch.nn.ModuleList()
        for idx in range(1, len(architecture) - 1):
            m = torch.nn.Linear(architecture[idx], architecture[idx], bias = False)
            m.weight.data = self.M[idx-1].weight.data
            m.weight.data.requires_grad_(False)
            self.M_copy.append(m)

        optim_params = []
        for idx in range(len(self.W)):
            optim_params.append(  {'params': self.W[idx].parameters(), 'lr': alphas_W[idx]}  )
            
        for idx in range(len(self.M)):
            optim_params.append(  {'params': self.M[idx].parameters(), 'lr': alphas_M[idx]}  )

        optimizer = torch.optim.SGD( optim_params, momentum=0.0 )
        self.optimizer = optimizer

    def Phi(self, x, y, neurons, beta, criterion):
        # Computes the primitive function given static input x, label y, neurons is the sequence of hidden layers neurons
        # criterion is the loss
        x = x.view(x.size(0),-1) # flattening the input
        
        layers = [x] + neurons  # concatenate the input to other layers
        
        # Primitive function computation
        phi = 0.0
        for idx in range(len(neurons)): # Squared Norms
            phi += 0.5*torch.sum( neurons[idx] * neurons[idx], dim=1).squeeze() # Scalar product s_n.s_n
        for idx in range(len(self.W)): # Linear Terms and Quadratic Terms
            phi -= torch.sum( self.W[idx](layers[idx]) * layers[idx+1], dim=1).squeeze() # Scalar product s_n.W.s_n-1
        for idx in range(len(self.M)): # Lateral Terms
            if beta != 0.0:
                phi += 0.5*torch.sum( self.M[idx](layers[idx+1]) * layers[idx+1], dim=1).squeeze() # Scalar product s_n.M.s_n
            else:
                phi += 0.5*torch.sum( self.M_copy[idx](layers[idx+1]) * layers[idx+1], dim=1).squeeze() # Scalar product s_n.M.s_n

        if beta!=0.0: # Nudging the output layer when beta is non zero 
            if criterion.__class__.__name__.find('MSE')!=-1:
                if self.task == "classification":
                    y = F.one_hot(y, num_classes=self.nc)
                L = criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].float(), y).squeeze()     
            phi += beta*L
        
        return phi
    
    
    def forward(self, x, y, neurons, T, neural_lr = 0.5, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion) # Computing Phi
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm) # dPhi/ds
            with torch.no_grad():
                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation(neurons[idx] - neural_lr * grads[idx] )  # s_(t+1) = s_(t) - neural_lr * sigma( dPhi/ds )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True
             
            if not_mse:
                neurons[-1] = grads[-1]
            else:
                with torch.no_grad():
                    neurons[-1] = self.activation(neurons[-1] - neural_lr * grads[-1] )

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons

    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neurons = []
        append = neurons.append
        for size in self.architecture[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, alphas_M, criterion, check_thm=False):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_2 - beta_1)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
        # # Contrastive Similarity Matching Lateral Weight Update additional term is added below (before optimizer step)
        # with torch.no_grad(): # Check line 306 in https://github.com/Pehlevan-Group/Supervised-Similarity-Matching/blob/master/Main/model_wlat_smep_mod.py
        #     for kk in range(len(self.M)):
        #         Mweight = self.M[kk].weight.data
        #         self.M[kk].weight.data = Mweight + (alphas_M[kk]) * Mweight/(2 * np.abs(beta_2))
        self.optimizer.step()

        for idx in range(len(self.M)):
            self.M_copy[idx].weight.data = self.M[idx].weight.data
            self.M_copy[idx].weight.data.requires_grad_(False)

class ContrastiveCorInfoMax():
    
    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid, output_sparsity = False, STlambda_lr = 0.01):
        
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
            
        self.Rh1 = torch.eye(architecture[1], architecture[1]).to(self.device) # For checking the true correlation matrix
        self.Rh2 = (0*torch.eye(architecture[1], architecture[1])).to(self.device) # For checking the true correlation matrix
        self.Bhdiag_list = []

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

    def calculate_neural_dynamics_grad(self, x, y, neurons, beta):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_ = self.gam_
        one_over_epsilon = self.one_over_epsilon

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]

        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + 2 * beta * (y - layers[jj + 1])
            else:
                init_grads[jj] = 2 * gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) - one_over_epsilon * (layers[jj + 1] - (Wfb[jj + 1]['weight'] @ layers[jj + 2] + Wfb[jj + 1]['bias']))
        return init_grads

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10, beta = 1):
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
                neuron_grads = self.calculate_neural_dynamics_grad(x, y, neurons, beta)

                for neuron_iter in range(len(neurons)):
                    if neuron_iter == len(neurons) - 1:
                        if self.output_sparsity:
                            neurons[neuron_iter] = F.relu(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter] - STLAMBD)
                            STLAMBD = STLAMBD + STlambda_lr * (torch.sum(neurons[neuron_iter], 0).view(1, -1) - 1)
                        else:
                            neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
                    else:
                        neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        return neurons

    def batch_step(self, x, y, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, neural_dynamic_iterations_nudged = 10, 
                   beta = 1, make_B_off_diag_nonpositive = False):
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)

        neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations_free, 0)
        
        neurons1 = neurons.copy()
        # ### Lateral Weight Updates
        # for jj in range(len(B)):
        #     z = B[jj]['weight'] @ neurons[jj]
        #     B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
        #     B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        # self.B = B

        neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, beta)

        neurons2 = neurons.copy()

        layers_free = [x] + neurons1
        layers_nudged = [x] + neurons2

        ## Compute forward errors
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [layers_free[jj] - (Wfb[jj]['weight'] @ layers_free[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [layers_nudged[jj] - (Wfb[jj]['weight'] @ layers_nudged[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] -= (1/beta) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            Wff[jj]['bias'] -= (1/beta) * lr['ff'][jj] * torch.mean(forward_errors_free[jj] - forward_errors_nudged[jj], axis = 1, keepdims = True)

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] -= (1/beta) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            Wfb[jj]['bias'] -= (1/beta) * lr['fb'][jj] * torch.mean(backward_errors_free[jj - 1] - backward_errors_nudged[jj - 1], axis = 1, keepdims = True)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ neurons[jj]
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)
        
        if make_B_off_diag_nonpositive:
            for jj in range(len(B)):
                B[jj]['weight'] = torch_make_off_diag_nonpositive(B[jj]['weight'])
        
        self.Bhdiag_list.append(torch.diag(B[0]['weight']))
        self.Rh1 = lambda_ * self.Rh1 + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons[0].T, neurons[0].T), axis = 0)
        self.Rh2 = lambda_ * self.Rh2 + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons[0].T, neurons[0].T), axis = 0)
        self.B = B
        self.Wff = Wff
        self.Wfb = Wfb
        return neurons

    def batch_step_noEP(self, x, y, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                        neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, neural_dynamic_iterations_nudged = 10, beta = 1):
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)

        neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations_free, 0)
        
        neurons1 = neurons.copy()
        # ### Lateral Weight Updates
        # for jj in range(len(B)):
        #     z = B[jj]['weight'] @ neurons[jj]
        #     B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
        #     B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        # self.B = B

        neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, beta)

        neurons2 = neurons.copy()

        layers_free = [x] + neurons1
        layers_nudged = [x] + neurons2

        ## Compute forward errors
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [layers_free[jj] - (Wfb[jj]['weight'] @ layers_free[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [layers_nudged[jj] - (Wfb[jj]['weight'] @ layers_nudged[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] -= (1/beta) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            Wff[jj]['bias'] -= (1/beta) * lr['ff'][jj] * torch.mean(forward_errors_free[jj] - forward_errors_nudged[jj], axis = 1, keepdims = True)

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] -= (1/beta) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            Wfb[jj]['bias'] -= (1/beta) * lr['fb'][jj] * torch.mean(backward_errors_free[jj - 1] - backward_errors_nudged[jj - 1], axis = 1, keepdims = True)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ neurons[jj]
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        self.B = B
        self.Wff = Wff
        self.Wfb = Wfb
        return neurons



class ContrastiveCorInfoMax_wCWU():
    
    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid, output_sparsity = False, STlambda_lr = 0.01):
        
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

    def calculate_neural_dynamics_grad(self, x, y, neurons, beta):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_ = self.gam_
        one_over_epsilon = self.one_over_epsilon

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]

        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + 2 * beta * (y - layers[jj + 1])
            else:
                init_grads[jj] = 2 * gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) - one_over_epsilon * (layers[jj + 1] - (Wfb[jj + 1]['weight'] @ layers[jj + 2] + Wfb[jj + 1]['bias']))
        return init_grads

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10, beta = 1):
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
                neuron_grads = self.calculate_neural_dynamics_grad(x, y, neurons, beta)

                for neuron_iter in range(len(neurons)):
                    if neuron_iter == len(neurons) - 1:
                        if self.output_sparsity:
                            neurons[neuron_iter] = F.relu(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter] - STLAMBD)
                            STLAMBD = STLAMBD + STlambda_lr * (torch.sum(neurons[neuron_iter], 0).view(1, -1) - 1)
                        else:
                            neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
                    else:
                        neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        return neurons

    def batch_step(self, x, y, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, neural_dynamic_iterations_nudged = 10, beta = 1):
        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)

        neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations_free, 0)
        
        neurons1 = neurons.copy()
        # ### Lateral Weight Updates
        # for jj in range(len(B)):
        #     z = B[jj]['weight'] @ neurons[jj]
        #     B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
        #     B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        # self.B = B
        layers_free = [x] + neurons1
        for k in range(neural_dynamic_iterations_nudged):
            neurons = self.run_neural_dynamics(x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                            neural_lr_decay_multiplier, 1, beta)

            neurons2 = neurons.copy()

            layers_nudged = [x] + neurons2

            ## Compute forward errors
            forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
            forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
            ## Compute backward errors
            backward_errors_free = [layers_free[jj] - (Wfb[jj]['weight'] @ layers_free[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]
            backward_errors_nudged = [layers_nudged[jj] - (Wfb[jj]['weight'] @ layers_nudged[jj + 1] + Wfb[jj]['bias']) for jj in range(1, len(Wfb))]

            ### Learning updates for feed-forward and backward weights
            for jj in range(len(Wff)):
                Wff[jj]['weight'] -= (1/beta) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
                Wff[jj]['bias'] -= (1/beta) * lr['ff'][jj] * torch.mean(forward_errors_free[jj] - forward_errors_nudged[jj], axis = 1, keepdims = True)

            for jj in range(1, len(Wfb)):
                Wfb[jj]['weight'] -= (1/beta) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
                Wfb[jj]['bias'] -= (1/beta) * lr['fb'][jj] * torch.mean(backward_errors_free[jj - 1] - backward_errors_nudged[jj - 1], axis = 1, keepdims = True)

            layers_free = layers_nudged
        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ neurons[jj]
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        self.B = B
        self.Wff = Wff
        self.Wfb = Wfb
        return neurons