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

# The libraries for debug plotting
import pylab as pl
from IPython.display import Latex, Math, clear_output, display

class ContrastiveCorInfoMaxHopfield():
    """This is the algorithm to be used in the paper. The summary will be added later.
    """

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
        # self.run_neural_dynamics = self.run_neural_dynamics_hopfield
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)

            # torch.nn.init.kaiming_uniform_(weight)
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(bias, -bound, bound)

            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            # weight = torch.randn(architecture[idx] + 1, architecture[idx + 1], requires_grad = False).to(self.device)
            weight = torch.eye(architecture[idx] + 1, architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            # torch.nn.init.kaiming_uniform_(weight)
            # bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            Wfb.append({'weight': weight})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1] + 1, architecture[idx + 1] + 1, requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            # weight = 1.0*torch.eye(architecture[idx + 1] + 1, architecture[idx + 1] + 1, requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)

        # Correlation Matrices (Only for debugging)
        Rfree = []
        for idx in range(len(architecture) - 1):
            weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            Rfree.append({'weight': weight})

        Rfree = np.array(Rfree)

        # Correlation Matrices (Only for debugging)
        Rnudged = []
        for idx in range(len(architecture) - 1):
            weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            Rnudged.append({'weight': weight})

        Rnudged = np.array(Rnudged)

        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        self.Rfree = Rfree
        self.Rnudged = Rnudged
        
        ############ Some Debugging Logs ##########################
        self.forward_backward_angles = []
        self.layerwise_forward_corinfo_list_free = []
        self.layerwise_backward_corinfo_list_free = []
        self.layerwise_forward_corinfo_list_nudged = []
        self.layerwise_backward_corinfo_list_nudged = []

        self.neural_dynamics_free_forward_info_list = []
        self.neural_dynamics_free_backward_info_list = []
        self.neural_dynamics_nudged_forward_info_list = []
        self.neural_dynamics_nudged_backward_info_list = []

    ###############################################################
    ############### HELPER METHODS ################################
    ###############################################################
    def copy_neurons(self, neurons):
        copy = []
        for n in neurons:
            copy.append(torch.empty_like(n).copy_(n.data))#.requires_grad_())
        return copy
        
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

    def append_ones_row_vector_to_tensor(self, inp_vector):
        return torch.cat((inp_vector, torch.ones(1, inp_vector.shape[1]).to(inp_vector.device)), 0)
    
    def plot_for_debug(self,):
        # pl.clf()
        # pl.subplot(2,1,1)
        # pl.plot(self.forward_backward_angles, linewidth=2)
        # pl.xlabel("Number of Debug Points", fontsize=45)
        # pl.ylabel("Angle (Degree)", fontsize=45)
        # pl.title("Angles of Forward and Backward Weights", fontsize=45)
        # pl.grid()
        # pl.xticks(fontsize=45)
        # pl.yticks(fontsize=45)

        # clear_output(wait=True)
        # display(pl.gcf())
        pass
    ###############################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ##############
    ###############################################################
    def angle_between_two_matrices(self, A, B):
        """Computes the angle between two matrices A and B.

        Args:
            A (torch.Tensor): Pytorch tensor of size m times n
            B (torch.Tensor): Pytorch tensor of size m times n

        Returns:
            angle: angle between the matrices A and B. The formula is given by the following:
                (180/pi) * acos[ Tr(A @ B.T) / sqrt(Tr(A @ A.T) * Tr(B @ B.T))] 
        """

        angle = (180 / torch.pi) * torch.acos(torch.trace(A @ B.T) / torch.sqrt(torch.trace(A @ A.T) * torch.trace(B @ B.T)))
        return angle

    def layerwise_forward_and_backward_correlative_information(self, layers, phase = "free"):
        Wff = self.Wff
        Wfb = self.Wfb
        if phase == "free":
            R = self.Rfree 
        elif phase == "nudged":
            R = self.Rnudged
        epsilon = self.epsilon
        one_over_epsilon = self.one_over_epsilon
        device = self.device
        architecture = self.architecture

        # epsilon_tensor = torch.Tensor([epsilon]).to(device)
        batch_size = layers[0].shape[1]
        batch_size_sqrt_root = np.sqrt(batch_size)
        log_epsilon = np.log(epsilon)

        forward_info_list = []
        backward_info_list = []

        for jj in range(len(architecture) - 2):
            Identity_Matrix = epsilon * torch.eye(*R[jj + 1]['weight'].shape).to(device)
            forward_info_jj= (torch.logdet(R[jj + 1]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(layers[jj + 2] - Wff[jj + 1]['weight'] @ layers[jj + 1] - Wff[jj + 1]['bias']) ** 2 - layers[jj + 2].shape[0] * log_epsilon)).item()

            forward_info_list.append(forward_info_jj)

        for jj in range(len(architecture) - 2):
            Identity_Matrix = epsilon * torch.eye(*R[jj]['weight'].shape).to(device)
            backward_info_jj = (torch.logdet(R[jj]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(self.append_ones_row_vector_to_tensor(layers[jj + 1]) - Wfb[jj + 1]['weight'] @ layers[jj + 2]) ** 2 - (layers[jj + 1].shape[0] + 1) * log_epsilon)).item()

            backward_info_list.append(backward_info_jj)

            return forward_info_list, backward_info_list

    # def layerwise_forward_and_backward_correlative_information(self, layers_free, layers_nudged):
    #     Wff = self.Wff
    #     Wfb = self.Wfb
    #     Rfree = self.Rfree 
    #     Rnudged = self.Rnudged
    #     epsilon = self.epsilon
    #     one_over_epsilon = self.one_over_epsilon
    #     device = self.device
    #     architecture = self.architecture

    #     # epsilon_tensor = torch.Tensor([epsilon]).to(device)
    #     batch_size = layers_free[0].shape[1]
    #     batch_size_sqrt_root = np.sqrt(batch_size)
    #     log_epsilon = np.log(epsilon)

    #     forward_info_list_free = []
    #     backward_info_list_free = []
    #     forward_info_list_nudged = []
    #     backward_info_list_nudged = []

    #     # for jj in range(len(architecture) - 2):
    #     #     Identity_Matrix = epsilon * torch.eye(*Rnudged[jj + 1]['weight'].shape).to(device)
    #     #     forward_info_jj_nudged = (torch.logdet(Rnudged[jj + 1]['weight'] + Identity_Matrix) - batch_size_sqrt_root * one_over_epsilon * torch.norm(layers_nudged[jj + 2] - Wff[jj + 1]['weight'] @ layers_nudged[jj + 1] - Wff[jj + 1]['bias']) ** 2 + layers_nudged[jj + 2].shape[0] * log_epsilon).item()
    #     #     forward_info_jj_free = (torch.logdet(Rfree[jj + 1]['weight'] + Identity_Matrix) - batch_size_sqrt_root * one_over_epsilon * torch.norm(layers_free[jj + 2] - Wff[jj + 1]['weight'] @ layers_free[jj + 1] - Wff[jj + 1]['bias']) ** 2 + layers_free[jj + 2].shape[0] * log_epsilon).item()

    #     #     forward_info_list_nudged.append(forward_info_jj_nudged)
    #     #     forward_info_list_free.append(forward_info_jj_free)

    #     # for jj in range(len(architecture) - 2):
    #     #     Identity_Matrix = epsilon * torch.eye(*Rnudged[jj]['weight'].shape).to(device)
    #     #     backward_info_jj_nudged = (torch.logdet(Rnudged[jj]['weight'] + Identity_Matrix) - batch_size_sqrt_root * one_over_epsilon * torch.norm(self.append_ones_row_vector_to_tensor(layers_nudged[jj + 1]) - Wfb[jj + 1]['weight'] @ layers_nudged[jj + 2]) ** 2 + (layers_nudged[jj + 1].shape[0] + 1) * log_epsilon).item()
    #     #     backward_info_jj_free = (torch.logdet(Rfree[jj]['weight'] + Identity_Matrix) - batch_size_sqrt_root * one_over_epsilon * torch.norm(self.append_ones_row_vector_to_tensor(layers_free[jj + 1]) - Wfb[jj + 1]['weight'] @ layers_free[jj + 2]) ** 2 + (layers_free[jj + 1].shape[0] + 1) * log_epsilon).item()

    #     #     backward_info_list_nudged.append(backward_info_jj_nudged)
    #     #     backward_info_list_free.append(backward_info_jj_free)
    #     for jj in range(len(architecture) - 2):
    #         Identity_Matrix = epsilon * torch.eye(*Rnudged[jj + 1]['weight'].shape).to(device)
    #         forward_info_jj_nudged = (torch.logdet(Rnudged[jj + 1]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(layers_nudged[jj + 2] - Wff[jj + 1]['weight'] @ layers_nudged[jj + 1] - Wff[jj + 1]['bias']) ** 2 - layers_nudged[jj + 2].shape[0] * log_epsilon)).item()
    #         forward_info_jj_free = (torch.logdet(Rfree[jj + 1]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(layers_free[jj + 2] - Wff[jj + 1]['weight'] @ layers_free[jj + 1] - Wff[jj + 1]['bias']) ** 2 - layers_free[jj + 2].shape[0] * log_epsilon)).item()

    #         forward_info_list_nudged.append(forward_info_jj_nudged)
    #         forward_info_list_free.append(forward_info_jj_free)

    #     for jj in range(len(architecture) - 2):
    #         Identity_Matrix = epsilon * torch.eye(*Rnudged[jj]['weight'].shape).to(device)
    #         backward_info_jj_nudged = (torch.logdet(Rnudged[jj]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(self.append_ones_row_vector_to_tensor(layers_nudged[jj + 1]) - Wfb[jj + 1]['weight'] @ layers_nudged[jj + 2]) ** 2 - (layers_nudged[jj + 1].shape[0] + 1) * log_epsilon)).item()
    #         backward_info_jj_free = (torch.logdet(Rfree[jj]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(self.append_ones_row_vector_to_tensor(layers_free[jj + 1]) - Wfb[jj + 1]['weight'] @ layers_free[jj + 2]) ** 2 - (layers_free[jj + 1].shape[0] + 1) * log_epsilon)).item()

    #         backward_info_list_nudged.append(backward_info_jj_nudged)
    #         backward_info_list_free.append(backward_info_jj_free)
            
    #     return forward_info_list_free, backward_info_list_free, forward_info_list_nudged, backward_info_list_nudged
    
    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    def run_neural_dynamics_hopfield(self, x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, 
                                     lr_rule = "constant", lr_decay_multiplier = 0.1, 
                                     neural_dynamic_iterations = 10, beta = 1, take_debug_logs = False):

        # if take_debug_logs:
        if beta != 0:
            phase = "free"
        else:
            phase = "nudged"
        forward_info = []
        backward_info = []
            
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_ = self.gam_
        epsilon = self.epsilon
        one_over_epsilon = self.one_over_epsilon

        neurons_intermediate = self.copy_neurons(neurons)
        layers = [x] + neurons  # concatenate the input to other layers
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                for jj in range(len(neurons)):
                    if jj == len(neurons) - 1:
                        # print("here if")
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']
                        apical_voltage = (gam_ * B[jj]['weight'][:-1] @ self.append_ones_row_vector_to_tensor( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - beta * (layers[jj + 1] - y)
                        gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                        neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        neurons[jj] = self.activation(neurons_intermediate[jj])
                        # init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + 2 * beta * (y - layers[jj + 1])
                    else:
                        # print("here else")
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']
                        apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'][:-1] @ self.append_ones_row_vector_to_tensor(layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2])[:-1] #+ Wfb[jj + 1]['bias']
                        gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                        neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        neurons[jj] = self.activation(neurons_intermediate[jj])
                    layers = [x] + neurons  # concatenate the input to other layers

            if take_debug_logs:
                info_measures = self.layerwise_forward_and_backward_correlative_information(layers, phase)
                forward_info.append(np.sum(info_measures[0]))
                backward_info.append(np.sum(info_measures[1]))
                    # neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        return neurons, forward_info, backward_info

    ###############################################################
    ############### BATCH STEP ALGORITHMS #########################
    ###############################################################
    def batch_step_hopfield(self, x, y, hopfield_g, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1, use_three_phase = False, take_debug_logs = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        Rfree = self.Rfree # For debugging to check the correlation matrices vs inverse correlation matrices
        Rnudged = self.Rnudged # For debugging to check the correlation matrices vs inverse correlation matrices

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)

        (neurons,
         free_forward_info,
         free_backward_info
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                             neural_lr_decay_multiplier, neural_dynamic_iterations_free, 0, take_debug_logs)

        
        neurons1 = neurons.copy()
        layers_free_ = [x] + neurons1

        ### Lateral Weight Updates
        for jj in range(len(B)):
            # z = B[jj]['weight'] @ self.append_ones_row_vector_to_tensor(neurons1[jj])
            # B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            # B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

            Rfree[jj]['weight'] = lambda_ * Rfree[jj]['weight'] + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons1[jj].T, neurons1[jj].T), axis = 0)

        (neurons,
         nudged_forward_info,
         nudged_backward_info 
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, beta, take_debug_logs)


        neurons2 = neurons.copy()

        if use_three_phase:
            neurons, _, _ = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, -beta, take_debug_logs)

            neurons3 = neurons.copy()

            layers_free = [x] + neurons3
        else:
            layers_free = [x] + neurons1

        layers_nudged = [x] + neurons2

        ## Compute forward errors
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [self.append_ones_row_vector_to_tensor(layers_free[jj]) - (Wfb[jj]['weight'] @ layers_free[jj + 1]) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [self.append_ones_row_vector_to_tensor(layers_nudged[jj]) - (Wfb[jj]['weight'] @ layers_nudged[jj + 1]) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            Wff[jj]['bias'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(forward_errors_free[jj] - forward_errors_nudged[jj], axis = 1, keepdims = True)

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            # Wfb[jj]['bias'] -= (1/beta) * lr['fb'][jj] * torch.mean(backward_errors_free[jj - 1] - backward_errors_nudged[jj - 1], axis = 1, keepdims = True)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ self.append_ones_row_vector_to_tensor(neurons2[jj])
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

            Rnudged[jj]['weight'] = lambda_ * Rnudged[jj]['weight'] + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons2[jj].T, neurons2[jj].T), axis = 0)
                 
        self.B = B
        self.Wff = Wff
        self.Wfb = Wfb
        self.Rfree = Rfree
        self.Rnudged = Rnudged

        if take_debug_logs:
            instant_forward_backward_angles = []
            for jj in range(1, len(Wff)):
                instant_forward_backward_angles.append(self.angle_between_two_matrices(torch.cat((self.Wff[jj]['weight'], self.Wff[jj]['bias']), 1), self.Wfb[jj]['weight'].T).item())
            
            self.forward_backward_angles.append(instant_forward_backward_angles)

            # (forward_info_list_free, 
            #  backward_info_list_free, 
            #  forward_info_list_nudged, 
            #  backward_info_list_nudged
            # ) = self.layerwise_forward_and_backward_correlative_information(layers_free_, layers_nudged)

            (forward_info_list_free, 
             backward_info_list_free, 
            ) = self.layerwise_forward_and_backward_correlative_information(layers_free_, "free")

            (forward_info_list_nudged, 
             backward_info_list_nudged, 
            ) = self.layerwise_forward_and_backward_correlative_information(layers_free_, "nudged")

            self.layerwise_forward_corinfo_list_free.append(forward_info_list_free)
            self.layerwise_backward_corinfo_list_free.append(backward_info_list_free)
            self.layerwise_forward_corinfo_list_nudged.append(forward_info_list_nudged)
            self.layerwise_backward_corinfo_list_nudged.append(backward_info_list_nudged)

            self.neural_dynamics_free_forward_info_list.append(free_forward_info)
            self.neural_dynamics_free_backward_info_list.append(free_backward_info)
            self.neural_dynamics_nudged_forward_info_list.append(nudged_forward_info)
            self.neural_dynamics_nudged_backward_info_list.append(nudged_backward_info)
        return neurons

class ContrastiveCorInfoMaxHopfield2():
    """This is the algorithm to be used in the paper. The summary will be added later.
    """

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
        # self.run_neural_dynamics = self.run_neural_dynamics_hopfield
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)

            # torch.nn.init.kaiming_uniform_(weight)
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(bias, -bound, bound)

            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            # weight = torch.randn(architecture[idx] + 1, architecture[idx + 1], requires_grad = False).to(self.device)
            weight = torch.eye(architecture[idx] + 1, architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            # torch.nn.init.kaiming_uniform_(weight)
            # bias = torch.zeros(architecture[idx], 1, requires_grad = False).to(self.device)
            Wfb.append({'weight': weight})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1] + 1, architecture[idx + 1] + 1, requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            # weight = 1.0*torch.eye(architecture[idx + 1] + 1, architecture[idx + 1] + 1, requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)

        # Correlation Matrices (Only for debugging)
        Rfree = []
        for idx in range(len(architecture) - 1):
            weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            Rfree.append({'weight': weight})

        Rfree = np.array(Rfree)

        # Correlation Matrices (Only for debugging)
        Rnudged = []
        for idx in range(len(architecture) - 1):
            weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            Rnudged.append({'weight': weight})

        Rnudged = np.array(Rnudged)

        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        self.Rfree = Rfree
        self.Rnudged = Rnudged
        
        ############ Some Debugging Logs ##########################
        self.forward_backward_angles = []
        self.layerwise_forward_corinfo_list_free = []
        self.layerwise_backward_corinfo_list_free = []
        self.layerwise_forward_corinfo_list_nudged = []
        self.layerwise_backward_corinfo_list_nudged = []

        self.neural_dynamics_free_forward_info_list = []
        self.neural_dynamics_free_backward_info_list = []
        self.neural_dynamics_nudged_forward_info_list = []
        self.neural_dynamics_nudged_backward_info_list = []

    ###############################################################
    ############### HELPER METHODS ################################
    ###############################################################
    def copy_neurons(self, neurons):
        copy = []
        for n in neurons:
            copy.append(torch.empty_like(n).copy_(n.data))#.requires_grad_())
        return copy
        
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

    def append_ones_row_vector_to_tensor(self, inp_vector):
        return torch.cat((inp_vector, torch.ones(1, inp_vector.shape[1]).to(inp_vector.device)), 0)
    
    ###############################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ##############
    ###############################################################
    def angle_between_two_matrices(self, A, B):
        """Computes the angle between two matrices A and B.

        Args:
            A (torch.Tensor): Pytorch tensor of size m times n
            B (torch.Tensor): Pytorch tensor of size m times n

        Returns:
            angle: angle between the matrices A and B. The formula is given by the following:
                (180/pi) * acos[ Tr(A @ B.T) / sqrt(Tr(A @ A.T) * Tr(B @ B.T))] 
        """

        angle = (180 / torch.pi) * torch.acos(torch.trace(A @ B.T) / torch.sqrt(torch.trace(A @ A.T) * torch.trace(B @ B.T)))
        return angle

    def layerwise_forward_and_backward_correlative_information(self, layers, phase = "free"):
        Wff = self.Wff
        Wfb = self.Wfb
        if phase == "free":
            R = self.Rfree 
        elif phase == "nudged":
            R = self.Rnudged
        epsilon = self.epsilon
        one_over_epsilon = self.one_over_epsilon
        device = self.device
        architecture = self.architecture

        # epsilon_tensor = torch.Tensor([epsilon]).to(device)
        batch_size = layers[0].shape[1]
        batch_size_sqrt_root = np.sqrt(batch_size)
        log_epsilon = np.log(epsilon)

        forward_info_list = []
        backward_info_list = []

        for jj in range(len(architecture) - 2):
            Identity_Matrix = epsilon * torch.eye(*R[jj + 1]['weight'].shape).to(device)
            forward_info_jj= (torch.logdet(R[jj + 1]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(layers[jj + 2] - Wff[jj + 1]['weight'] @ layers[jj + 1] - Wff[jj + 1]['bias']) ** 2 - layers[jj + 2].shape[0] * log_epsilon)).item()

            forward_info_list.append(forward_info_jj)

        for jj in range(len(architecture) - 2):
            Identity_Matrix = epsilon * torch.eye(*R[jj]['weight'].shape).to(device)
            backward_info_jj = (torch.logdet(R[jj]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(self.append_ones_row_vector_to_tensor(layers[jj + 1]) - Wfb[jj + 1]['weight'] @ layers[jj + 2]) ** 2 - (layers[jj + 1].shape[0] + 1) * log_epsilon)).item()

            backward_info_list.append(backward_info_jj)

            return forward_info_list, backward_info_list
    
    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    def run_neural_dynamics_hopfield(self, x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, 
                                     lr_rule = "constant", lr_decay_multiplier = 0.1, 
                                     neural_dynamic_iterations = 10, beta = 1, take_debug_logs = False):

        # if take_debug_logs:
        if beta != 0:
            phase = "free"
        else:
            phase = "nudged"
        forward_info = []
        backward_info = []
            
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_ = self.gam_
        epsilon = self.epsilon
        one_over_epsilon = self.one_over_epsilon

        neurons_intermediate = self.copy_neurons(neurons)
        layers = [x] + neurons  # concatenate the input to other layers
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                for jj in range(len(neurons)):
                    if jj == len(neurons) - 1:
                        # print("here if")
                        # basal_voltage = Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']
                        # apical_voltage = (gam_ * B[jj]['weight'][:-1] @ self.append_ones_row_vector_to_tensor( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - beta * (layers[jj + 1] - y)
                        # gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                        # neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        # neurons[jj] = self.activation(neurons_intermediate[jj])
                        neuron_gradient = - gam_ * B[jj]['weight'][:-1] @ self.append_ones_row_vector_to_tensor( layers[jj + 1]) + one_over_epsilon * (neurons[jj] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + beta * (neurons[jj] - y)
                        neurons[jj] = self.activation(neurons[jj] - neural_lr * neuron_gradient)
                        # init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + 2 * beta * (y - layers[jj + 1])
                    else:
                        # print("here else")
                        # basal_voltage = Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']
                        # apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'][:-1] @ self.append_ones_row_vector_to_tensor(layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2])[:-1] #+ Wfb[jj + 1]['bias']
                        # gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                        # neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        # neurons[jj] = self.activation(neurons_intermediate[jj])
                        neuron_gradient = - 2 * gam_ * B[jj]['weight'][:-1] @ self.append_ones_row_vector_to_tensor(layers[jj + 1]) + one_over_epsilon * (neurons[jj] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + one_over_epsilon * (neurons[jj] - (Wfb[jj + 1]['weight'] @ layers[jj + 2])[:-1])
                        neurons[jj] = self.activation(neurons[jj] - neural_lr * neuron_gradient)
                    layers = [x] + neurons  # concatenate the input to other layers

            if take_debug_logs:
                info_measures = self.layerwise_forward_and_backward_correlative_information(layers, phase)
                forward_info.append(np.sum(info_measures[0]))
                backward_info.append(np.sum(info_measures[1]))
                    # neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        return neurons, forward_info, backward_info

    ###############################################################
    ############### BATCH STEP ALGORITHMS #########################
    ###############################################################
    def batch_step_hopfield(self, x, y, hopfield_g, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1, use_three_phase = False, take_debug_logs = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        Rfree = self.Rfree # For debugging to check the correlation matrices vs inverse correlation matrices
        Rnudged = self.Rnudged # For debugging to check the correlation matrices vs inverse correlation matrices

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)

        (neurons,
         free_forward_info,
         free_backward_info
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                             neural_lr_decay_multiplier, neural_dynamic_iterations_free, 0, take_debug_logs)

        
        neurons1 = neurons.copy()
        layers_free_ = [x] + neurons1

        ### Lateral Weight Updates
        for jj in range(len(B)):

            Rfree[jj]['weight'] = lambda_ * Rfree[jj]['weight'] + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons1[jj].T, neurons1[jj].T), axis = 0)

        (neurons,
         nudged_forward_info,
         nudged_backward_info 
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, beta, take_debug_logs)


        neurons2 = neurons.copy()

        if use_three_phase:
            neurons, _, _ = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, -beta, take_debug_logs)

            neurons3 = neurons.copy()

            layers_free = [x] + neurons3
        else:
            layers_free = [x] + neurons1

        layers_nudged = [x] + neurons2

        ## Compute forward errors
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [self.append_ones_row_vector_to_tensor(layers_free[jj]) - (Wfb[jj]['weight'] @ layers_free[jj + 1]) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [self.append_ones_row_vector_to_tensor(layers_nudged[jj]) - (Wfb[jj]['weight'] @ layers_nudged[jj + 1]) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            Wff[jj]['bias'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(forward_errors_free[jj] - forward_errors_nudged[jj], axis = 1, keepdims = True)

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            # Wfb[jj]['bias'] -= (1/beta) * lr['fb'][jj] * torch.mean(backward_errors_free[jj - 1] - backward_errors_nudged[jj - 1], axis = 1, keepdims = True)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ self.append_ones_row_vector_to_tensor(neurons2[jj])
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

            Rnudged[jj]['weight'] = lambda_ * Rnudged[jj]['weight'] + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons2[jj].T, neurons2[jj].T), axis = 0)
                 
        self.B = B
        self.Wff = Wff
        self.Wfb = Wfb
        self.Rfree = Rfree
        self.Rnudged = Rnudged

        if take_debug_logs:
            instant_forward_backward_angles = []
            for jj in range(1, len(Wff)):
                instant_forward_backward_angles.append(self.angle_between_two_matrices(torch.cat((self.Wff[jj]['weight'], self.Wff[jj]['bias']), 1), self.Wfb[jj]['weight'].T).item())
            
            self.forward_backward_angles.append(instant_forward_backward_angles)

            (forward_info_list_free, 
             backward_info_list_free, 
            ) = self.layerwise_forward_and_backward_correlative_information(layers_free_, "free")

            (forward_info_list_nudged, 
             backward_info_list_nudged, 
            ) = self.layerwise_forward_and_backward_correlative_information(layers_free_, "nudged")

            self.layerwise_forward_corinfo_list_free.append(forward_info_list_free)
            self.layerwise_backward_corinfo_list_free.append(backward_info_list_free)
            self.layerwise_forward_corinfo_list_nudged.append(forward_info_list_nudged)
            self.layerwise_backward_corinfo_list_nudged.append(backward_info_list_nudged)

            self.neural_dynamics_free_forward_info_list.append(free_forward_info)
            self.neural_dynamics_free_backward_info_list.append(free_backward_info)
            self.neural_dynamics_nudged_forward_info_list.append(nudged_forward_info)
            self.neural_dynamics_nudged_backward_info_list.append(nudged_backward_info)
        return neurons

    def batch_step_hopfield_nudged(self, x, y, hopfield_g, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1, use_three_phase = False, take_debug_logs = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        Rfree = self.Rfree # For debugging to check the correlation matrices vs inverse correlation matrices
        Rnudged = self.Rnudged # For debugging to check the correlation matrices vs inverse correlation matrices

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)

        (neurons,
         nudged_forward_info,
         nudged_backward_info 
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, beta, take_debug_logs)


        neurons2 = neurons.copy()

        if use_three_phase:
            neurons, _, _ = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, -beta, take_debug_logs)

            neurons3 = neurons.copy()

            layers_free = [x] + neurons3
        else:
            layers_free = [x] + neurons
            layers_free_ = layers_free

        layers_nudged = [x] + neurons2

        ## Compute forward errors
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_nudged = [self.append_ones_row_vector_to_tensor(layers_nudged[jj]) - (Wfb[jj]['weight'] @ layers_nudged[jj + 1]) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(- outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            Wff[jj]['bias'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(- forward_errors_nudged[jj], axis = 1, keepdims = True)

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['fb'][jj] * torch.mean(- outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            # Wfb[jj]['bias'] -= (1/beta) * lr['fb'][jj] * torch.mean(backward_errors_free[jj - 1] - backward_errors_nudged[jj - 1], axis = 1, keepdims = True)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ self.append_ones_row_vector_to_tensor(neurons2[jj])
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

            Rnudged[jj]['weight'] = lambda_ * Rnudged[jj]['weight'] + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons2[jj].T, neurons2[jj].T), axis = 0)
                 
        self.B = B
        self.Wff = Wff
        self.Wfb = Wfb
        self.Rnudged = Rnudged

        if take_debug_logs:
            instant_forward_backward_angles = []
            for jj in range(1, len(Wff)):
                instant_forward_backward_angles.append(self.angle_between_two_matrices(torch.cat((self.Wff[jj]['weight'], self.Wff[jj]['bias']), 1), self.Wfb[jj]['weight'].T).item())
            
            self.forward_backward_angles.append(instant_forward_backward_angles)

            (forward_info_list_nudged, 
             backward_info_list_nudged, 
            ) = self.layerwise_forward_and_backward_correlative_information(layers_free_, "nudged")

            self.layerwise_forward_corinfo_list_nudged.append(forward_info_list_nudged)
            self.layerwise_backward_corinfo_list_nudged.append(backward_info_list_nudged)

            self.neural_dynamics_nudged_forward_info_list.append(nudged_forward_info)
            self.neural_dynamics_nudged_backward_info_list.append(nudged_backward_info)
        return neurons

class CorInfoMaxHopfield():
    """This is the algorithm to be used in the paper. The summary will be added later.
    """

    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
            weight = 1.0*torch.eye(architecture[idx + 1] + 1, architecture[idx + 1] + 1, requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)

        # Correlation Matrices (Only for debugging)
        R = []
        for idx in range(len(architecture) - 1):
            weight = 1.0*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            R.append({'weight': weight})

        R = np.array(R)

        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B
        self.R = R
        
        ############ Some Debugging Logs ##########################
        self.forward_backward_angles = []
        self.layerwise_forward_corinfo_list = []
        self.layerwise_backward_corinfo_list = []

    def fast_forward(self, x):
        Wff = self.Wff

        neurons = []
        for jj in range(len(Wff)):
            if jj == 0:
                neurons.append(self.activation(Wff[jj]['weight'] @ x + Wff[jj]['bias']))
            else:
                neurons.append(self.activation(Wff[jj]['weight'] @ neurons[-1] + Wff[jj]['bias']))
        return neurons

    ###############################################################
    ############### HELPER METHODS ################################
    ###############################################################
    def copy_neurons(self, neurons):
        copy = []
        for n in neurons:
            copy.append(torch.empty_like(n).copy_(n.data))#.requires_grad_())
        return copy
        
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

    def append_ones_row_vector_to_tensor(self, inp_vector):
        return torch.cat((inp_vector, torch.ones(1, inp_vector.shape[1]).to(inp_vector.device)), 0)
    
    ###############################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ##############
    ###############################################################
    def angle_between_two_matrices(self, A, B):
        """Computes the angle between two matrices A and B.

        Args:
            A (torch.Tensor): Pytorch tensor of size m times n
            B (torch.Tensor): Pytorch tensor of size m times n

        Returns:
            angle: angle between the matrices A and B. The formula is given by the following:
                (180/pi) * acos[ Tr(A @ B.T) / sqrt(Tr(A @ A.T) * Tr(B @ B.T))] 
        """
        angle = (180 / torch.pi) * torch.acos(torch.trace(A @ B.T) / torch.sqrt(torch.trace(A @ A.T) * torch.trace(B @ B.T)))
        return angle

    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    def run_neural_dynamics_hopfield(self, x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, 
                                     lr_rule = "constant", lr_decay_multiplier = 0.1, 
                                     neural_dynamic_iterations = 10, beta = 1):
        Wff = self.Wff
        Wfb = self.Wfb
        B = self.B
        gam_ = self.gam_
        epsilon = self.epsilon
        one_over_epsilon = self.one_over_epsilon

        neurons_intermediate = self.copy_neurons(neurons)
        layers = [x] + neurons  # concatenate the input to other layers
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            with torch.no_grad():       
                for jj in range(len(neurons)):
                    if jj == len(neurons) - 1:
                        pass
                        # print("here if")
                        # basal_voltage = Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']
                        # apical_voltage = (gam_ * B[jj]['weight'][:-1] @ self.append_ones_row_vector_to_tensor( layers[jj + 1]) + hopfield_g * layers[jj + 1]) + beta * (y - layers[jj + 1])
                        # gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                        # neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        # neurons[jj] = self.activation(neurons_intermediate[jj])
                        # init_grads[jj] = gam_ * B[jj]['weight'] @ layers[jj + 1] - one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) + 2 * beta * (y - layers[jj + 1])
                    else:
                        # print("here else")
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']
                        apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'][:-1] @ self.append_ones_row_vector_to_tensor(layers[jj + 1]) + hopfield_g * layers[jj + 1]) + Wfb[jj + 1]['weight'] @ layers[jj + 2] #+ Wfb[jj + 1]['bias']
                        gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                        neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        neurons[jj] = self.activation(neurons_intermediate[jj])
                    layers = [x] + neurons  # concatenate the input to other layers
                    # neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        return neurons

    ###############################################################
    ############### BATCH STEP ALGORITHMS #########################
    ###############################################################
    def batch_step_hopfield(self, x, y, hopfield_g, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1, take_debug_logs = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        R = self.R # For debugging to check the correlation matrices vs inverse correlation matrices

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)
        neurons[-1] = y.to(torch.float)

        neurons = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                           neural_lr_decay_multiplier, neural_dynamic_iterations_free, 0)
        

        layers= [x] + neurons

        ## Compute forward errors
        forward_errors = [layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors = [layers[jj] - (Wfb[jj]['weight'] @ layers[jj + 1]) for jj in range(1, len(Wfb))]


        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors[jj].T, layers[jj].T) , axis = 0)
            Wff[jj]['bias'] += lr['ff'][jj] * torch.mean(forward_errors[jj] , axis = 1, keepdims = True)

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors[jj - 1].T, layers[jj + 1].T), axis = 0)
            # Wfb[jj]['bias'] -= (1/beta) * lr['fb'][jj] * torch.mean(backward_errors_free[jj - 1] - backward_errors_nudged[jj - 1], axis = 1, keepdims = True)

        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ self.append_ones_row_vector_to_tensor(neurons[jj])
            B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
            B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

            R[jj]['weight'] = lambda_ * R[jj]['weight'] + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons[jj].T, neurons[jj].T), axis = 0)
                 
        self.B = B
        self.Wff = Wff
        self.Wfb = Wfb
        self.R = R

        if take_debug_logs:
            instant_forward_backward_angles = []
            for jj in range(1, len(Wff)):
                instant_forward_backward_angles.append(self.angle_between_two_matrices(self.Wff[jj]['weight'], self.Wfb[jj]['weight'].T).item())
            
            self.forward_backward_angles.append(instant_forward_backward_angles)

        return neurons