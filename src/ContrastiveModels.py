import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

class ContrastiveCorInfoMaxHopfield():

    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid, device = None):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)

            Wff.append({'weight': weight})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)

            Wfb.append({'weight': weight})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
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
            forward_info_jj= (torch.logdet(R[jj + 1]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(layers[jj + 2] - Wff[jj + 1]['weight'] @ layers[jj + 1]) ** 2 - layers[jj + 2].shape[0] * log_epsilon)).item()

            forward_info_list.append(forward_info_jj)

        for jj in range(len(architecture) - 2):
            Identity_Matrix = epsilon * torch.eye(*R[jj]['weight'].shape).to(device)
            backward_info_jj = (torch.logdet(R[jj]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm((layers[jj + 1]) - Wfb[jj + 1]['weight'] @ layers[jj + 2]) ** 2 - (layers[jj + 1].shape[0] + 1) * log_epsilon)).item()

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
                        
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] #+ Wff[jj]['bias']
                        apical_voltage = (gam_ * B[jj]['weight'] @ ( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - beta * (layers[jj + 1] - y)
                        gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) 
                        neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        neurons[jj] = self.activation(neurons_intermediate[jj])
                        
                    else:
                        
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] #+ Wff[jj]['bias']
                        apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'] @ (layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2]) 
                        gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                        neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        neurons[jj] = self.activation(neurons_intermediate[jj])
                    layers = [x] + neurons  # concatenate the input to other layers

            if take_debug_logs:
                info_measures = self.layerwise_forward_and_backward_correlative_information(layers, phase)
                forward_info.append(np.sum(info_measures[0]))
                backward_info.append(np.sum(info_measures[1]))
                    
        return neurons, forward_info, backward_info

    ###############################################################
    ############### BATCH STEP ALGORITHMS #########################
    ###############################################################
    def batch_step_hopfield(self, x, y, hopfield_g, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1, use_three_phase = False, 
                            take_debug_logs = False, weight_decay = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_
        epsilon = self.epsilon

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
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj]) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj]) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [(layers_free[jj]) - (Wfb[jj]['weight'] @ layers_free[jj + 1]) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [(layers_nudged[jj]) - (Wfb[jj]['weight'] @ layers_nudged[jj + 1]) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            if weight_decay:
                Wff[jj]['weight'] -= lr['ff'][jj] * epsilon * Wff[jj]['weight']

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            if weight_decay:
                Wfb[jj]['weight'] -= lr['fb'][jj] * epsilon * Wfb[jj]['weight']
        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ (neurons2[jj])
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
                instant_forward_backward_angles.append(self.angle_between_two_matrices(self.Wff[jj]['weight'], self.Wfb[jj]['weight'].T).item())
            
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

    def save_model_weights(self, pickle_name = "CorInfoWeights"):
        Wff_save = []
        for idx in range(len(self.Wff)):
            weight = torch2numpy(self.Wff[idx]['weight'])
            Wff_save.append({'weight': weight})
            
        Wfb_save = []
        for idx in range(len(self.Wfb)):
            weight = torch2numpy(self.Wfb[idx]['weight'])
            Wfb_save.append({'weight': weight})
            
        B_save = []
        for idx in range(len(self.B)):
            weight = torch2numpy(self.B[idx]['weight'])
            B_save.append({'weight': weight})
            
        model_params = pd.DataFrame(columns = ['Wff', 'Wfb', 'B'])

        model_params['Wff'] = Wff_save
        model_params['Wfb'] = Wfb_save
        model_params['B'] = B_save

        model_params.to_pickle(pickle_name + ".pkl")

    def load_model_weights(self, pickle_name):
        model_params_load = pd.read_pickle(pickle_name + ".pkl")
        for idx in range(len(self.Wff)):
            self.Wff[idx]['weight'] = torch.tensor(model_params_load['Wff'].iloc[idx]['weight'], requires_grad = False).to(self.device)
            
        for idx in range(len(self.Wfb)):
            self.Wfb[idx]['weight'] = torch.tensor(model_params_load['Wfb'].iloc[idx]['weight'], requires_grad = False).to(self.device)
            
        for idx in range(len(self.B)):
            self.B[idx]['weight'] = torch.tensor(model_params_load['B'].iloc[idx]['weight'], requires_grad = False).to(self.device)
                 
class ContrastiveCorInfoMaxHopfieldSparse(ContrastiveCorInfoMaxHopfield):
    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid, sparse_layers = [], device = None):
        self.sparse_layers = sparse_layers
        super().__init__(architecture, lambda_, epsilon, activation, device)
        
    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    def run_neural_dynamics_hopfield(self, x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list,
                                     lr_rule = "constant", lr_decay_multiplier = 0.1, 
                                     neural_dynamic_iterations = 10, beta = 1, take_debug_logs = False):
        mbs = x.size(1)
        if beta == 0:
            STLAMBD_list = [torch.zeros(1, mbs).to(self.device) for _ in range(len(neurons))]
        else:
            STLAMBD_list = self.STLAMBD_list
        STLAMBD_list_intermediate = self.copy_neurons(STLAMBD_list)
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
                        basal_voltage = Wff[jj]['weight'] @ layers[jj]
                        # apical_voltage = (gam_ * B[jj]['weight'] @ ( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - beta * (layers[jj + 1] - y)
                        # gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                        
                        if (jj + 1) in self.sparse_layers:
                            apical_voltage = (gam_ * B[jj]['weight'] @ ( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - STLAMBD_list[jj] - beta * (layers[jj + 1] - y)
                            gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                            neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons 
                            neurons[jj] = F.relu(neurons_intermediate[jj])
                            
                            STLAMBD_list_intermediate[jj] = STLAMBD_list_intermediate[jj] + STlambda_lr_list[jj] * (-STLAMBD_list_intermediate[jj] + (torch.sum(neurons[jj], 0).view(1, -1) - 1) + STLAMBD_list[jj])
                            STLAMBD_list[jj] = F.relu(STLAMBD_list_intermediate[jj])
                        else:
                            apical_voltage = (gam_ * B[jj]['weight'] @ ( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - beta * (layers[jj + 1] - y)
                            gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                            neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons 
                            neurons[jj] = self.activation(neurons_intermediate[jj])
                    else:
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] 
                        # apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'] @ (layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2]) #+ Wfb[jj + 1]['bias']
                        # gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                        
                        if (jj + 1) in self.sparse_layers:
                            apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'] @ (layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2]) - epsilon * STLAMBD_list[jj]#+ Wfb[jj + 1]['bias']
                            gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                            neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                            neurons[jj] = F.relu(neurons_intermediate[jj])
                            STLAMBD_list_intermediate[jj] = STLAMBD_list_intermediate[jj] + STlambda_lr_list[jj] * (-STLAMBD_list_intermediate[jj] + (torch.sum(neurons[jj], 0).view(1, -1) - 1) + STLAMBD_list[jj])
                            STLAMBD_list[jj] = F.relu(STLAMBD_list_intermediate[jj])
                        else:
                            apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'] @ (layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2]) #+ Wfb[jj + 1]['bias']
                            gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                            neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons 
                            neurons[jj] = self.activation(neurons_intermediate[jj])
                        # neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        # neurons[jj] = self.activation(neurons_intermediate[jj])
                    layers = [x] + neurons  # concatenate the input to other layers

            if take_debug_logs:
                info_measures = self.layerwise_forward_and_backward_correlative_information(layers, phase)
                forward_info.append(np.sum(info_measures[0]))
                backward_info.append(np.sum(info_measures[1]))
                    # neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        self.STLAMBD_list = STLAMBD_list
        return neurons, forward_info, backward_info

    ###############################################################
    ############### BATCH STEP ALGORITHMS #########################
    ###############################################################
    def batch_step_hopfield(self, x, y, hopfield_g, lr, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1, use_three_phase = False, 
                            take_debug_logs = False, weight_decay = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_
        epsilon = self.epsilon

        Rfree = self.Rfree # For debugging to check the correlation matrices vs inverse correlation matrices
        Rnudged = self.Rnudged # For debugging to check the correlation matrices vs inverse correlation matrices

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)

        (neurons,
         free_forward_info,
         free_backward_info
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                             neural_lr_decay_multiplier, neural_dynamic_iterations_free, 0, take_debug_logs)

        
        neurons1 = neurons.copy()
        layers_free_ = [x] + neurons1

        for jj in range(len(B)):

            Rfree[jj]['weight'] = lambda_ * Rfree[jj]['weight'] + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons1[jj].T, neurons1[jj].T), axis = 0)

        (neurons,
         nudged_forward_info,
         nudged_backward_info 
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, beta, take_debug_logs)


        neurons2 = neurons.copy()

        if use_three_phase:
            neurons, _, _ = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, -beta, take_debug_logs)

            neurons3 = neurons.copy()

            layers_free = [x] + neurons3
        else:
            layers_free = [x] + neurons1

        layers_nudged = [x] + neurons2

        ## Compute forward errors
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj]) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj]) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [(layers_free[jj]) - (Wfb[jj]['weight'] @ layers_free[jj + 1]) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [(layers_nudged[jj]) - (Wfb[jj]['weight'] @ layers_nudged[jj + 1]) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            if weight_decay:
                Wff[jj]['weight'] -= lr['ff'][jj] * epsilon * Wff[jj]['weight']

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            if weight_decay:
                Wfb[jj]['weight'] -= lr['fb'][jj] * epsilon * Wfb[jj]['weight']
        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ (neurons2[jj])
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
                instant_forward_backward_angles.append(self.angle_between_two_matrices(self.Wff[jj]['weight'], self.Wfb[jj]['weight'].T).item())
            
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

class ContrastiveCorInfoMaxHopfieldSparseV2(ContrastiveCorInfoMaxHopfield):
    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid, sparse_layers = [], device = None):
        self.sparse_layers = sparse_layers
        super().__init__(architecture, lambda_, epsilon, activation, device)
        
    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    def run_neural_dynamics_hopfield(self, x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list,
                                     lr_rule = "constant", lr_decay_multiplier = 0.1, 
                                     neural_dynamic_iterations = 10, beta = 1, take_debug_logs = False):
        mbs = x.size(1)
        if beta == 0:
            STLAMBD_list = [torch.zeros(1, mbs).to(self.device) for _ in range(len(neurons))]
        else:
            STLAMBD_list = self.STLAMBD_list
        STLAMBD_list_intermediate = self.copy_neurons(STLAMBD_list)
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
                        basal_voltage = Wff[jj]['weight'] @ layers[jj]
                        # apical_voltage = (gam_ * B[jj]['weight'] @ ( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - beta * (layers[jj + 1] - y)
                        # gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                        
                        if (jj + 1) in self.sparse_layers:
                            apical_voltage = (gam_ * B[jj]['weight'] @ ( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - STLAMBD_list[jj] - beta * (layers[jj + 1] - y)
                            gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                            neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons 
                            neurons[jj] = F.relu(neurons_intermediate[jj])
                            
                            STLAMBD_list_intermediate[jj] = STLAMBD_list_intermediate[jj] + STlambda_lr_list[jj] * ((torch.sum(neurons[jj], 0).view(1, -1) - 1))
                            STLAMBD_list[jj] = F.relu(STLAMBD_list_intermediate[jj])
                        else:
                            apical_voltage = (gam_ * B[jj]['weight'] @ ( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - beta * (layers[jj + 1] - y)
                            gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                            neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons 
                            neurons[jj] = self.activation(neurons_intermediate[jj])
                    else:
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] 
                        # apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'] @ (layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2]) #+ Wfb[jj + 1]['bias']
                        # gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                        
                        if (jj + 1) in self.sparse_layers:
                            apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'] @ (layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2]) - epsilon * STLAMBD_list[jj]#+ Wfb[jj + 1]['bias']
                            gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                            neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                            neurons[jj] = F.relu(neurons_intermediate[jj])
                            STLAMBD_list_intermediate[jj] = STLAMBD_list_intermediate[jj] + STlambda_lr_list[jj] * ((torch.sum(neurons[jj], 0).view(1, -1) - 1))
                            STLAMBD_list[jj] = F.relu(STLAMBD_list_intermediate[jj])
                        else:
                            apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'] @ (layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2]) #+ Wfb[jj + 1]['bias']
                            gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                            neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons 
                            neurons[jj] = self.activation(neurons_intermediate[jj])
                        # neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        # neurons[jj] = self.activation(neurons_intermediate[jj])
                    layers = [x] + neurons  # concatenate the input to other layers

            if take_debug_logs:
                info_measures = self.layerwise_forward_and_backward_correlative_information(layers, phase)
                forward_info.append(np.sum(info_measures[0]))
                backward_info.append(np.sum(info_measures[1]))
                    # neurons[neuron_iter] = self.activation(neurons[neuron_iter] + neural_lr * neuron_grads[neuron_iter])
        self.STLAMBD_list = STLAMBD_list
        return neurons, forward_info, backward_info

    ###############################################################
    ############### BATCH STEP ALGORITHMS #########################
    ###############################################################
    def batch_step_hopfield(self, x, y, hopfield_g, lr, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1, use_three_phase = False, 
                            take_debug_logs = False, weight_decay = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_
        epsilon = self.epsilon

        Rfree = self.Rfree # For debugging to check the correlation matrices vs inverse correlation matrices
        Rnudged = self.Rnudged # For debugging to check the correlation matrices vs inverse correlation matrices

        # neurons = self.init_neurons(x.size(1), device = self.device)
        neurons = self.init_neurons(x.size(1), device = self.device)

        (neurons,
         free_forward_info,
         free_backward_info
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                             neural_lr_decay_multiplier, neural_dynamic_iterations_free, 0, take_debug_logs)

        
        neurons1 = neurons.copy()
        layers_free_ = [x] + neurons1

        for jj in range(len(B)):

            Rfree[jj]['weight'] = lambda_ * Rfree[jj]['weight'] + (1 - lambda_) * torch.mean(outer_prod_broadcasting(neurons1[jj].T, neurons1[jj].T), axis = 0)

        (neurons,
         nudged_forward_info,
         nudged_backward_info 
        ) = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, beta, take_debug_logs)


        neurons2 = neurons.copy()

        if use_three_phase:
            neurons, _, _ = self.run_neural_dynamics_hopfield(x, y, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, 
                                                              neural_lr_decay_multiplier, neural_dynamic_iterations_nudged, -beta, take_debug_logs)

            neurons3 = neurons.copy()

            layers_free = [x] + neurons3
        else:
            layers_free = [x] + neurons1

        layers_nudged = [x] + neurons2

        ## Compute forward errors
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj]) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj]) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [(layers_free[jj]) - (Wfb[jj]['weight'] @ layers_free[jj + 1]) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [(layers_nudged[jj]) - (Wfb[jj]['weight'] @ layers_nudged[jj + 1]) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            if weight_decay:
                Wff[jj]['weight'] -= lr['ff'][jj] * epsilon * Wff[jj]['weight']

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            if weight_decay:
                Wfb[jj]['weight'] -= lr['fb'][jj] * epsilon * Wfb[jj]['weight']
        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ (neurons2[jj])
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
                instant_forward_backward_angles.append(self.angle_between_two_matrices(self.Wff[jj]['weight'], self.Wfb[jj]['weight'].T).item())
            
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
            if m.bias is not None:
                m.bias.data.mul_(0)
            self.W.append(m)

    def Phi(self, x, y, neurons, beta, criterion):

        x = x.view(x.size(0),-1) 
        
        layers = [x] + neurons 
        
        phi = 0.0
        for idx in range(len(neurons)): 
            phi += 0.5*torch.sum( neurons[idx] * neurons[idx], dim=1).squeeze() 
        for idx in range(len(self.W)): 
            phi -= torch.sum( self.W[idx](layers[idx]) * layers[idx+1], dim=1).squeeze() 

        if beta!=0.0: 
            if criterion.__class__.__name__.find('MSE')!=-1:
                y = F.one_hot(y, num_classes=self.nc)
                L = criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].float(), y).squeeze()     
            phi += beta*L
        
        return phi
    
    
    def forward(self, x, y, neurons, T, neural_lr = 0.5, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion) 
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) 
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm) 

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
        neurons = []
        append = neurons.append
        for size in self.architecture[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        beta_1, beta_2 = betas
        
        self.zero_grad()           
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_2 - beta_1)        
        delta_phi.backward() 

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

        x = x.view(x.size(0),-1) 
        
        layers = [x] + neurons  
        

        phi = 0.0
        for idx in range(len(neurons)): 
            phi += 0.5*torch.sum( neurons[idx] * neurons[idx], dim=1).squeeze() 
        for idx in range(len(self.W)): 
            phi -= torch.sum( self.W[idx](layers[idx]) * layers[idx+1], dim=1).squeeze() 
        for idx in range(len(self.M)): 
            if beta != 0.0:
                phi += 0.5*torch.sum( self.M[idx](layers[idx+1]) * layers[idx+1], dim=1).squeeze() 
            else:
                phi += 0.5*torch.sum( self.M_copy[idx](layers[idx+1]) * layers[idx+1], dim=1).squeeze() 

        if beta!=0.0: 
            if criterion.__class__.__name__.find('MSE')!=-1:
                if self.task == "classification":
                    y = F.one_hot(y, num_classes=self.nc)
                L = criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].float(), y).squeeze()     
            phi += beta*L
        
        return phi
    
    def forward(self, x, y, neurons, T, neural_lr = 0.5, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):

        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion) 
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) 
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm) 
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
        neurons = []
        append = neurons.append
        for size in self.architecture[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, alphas_M, criterion, check_thm=False):
        
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
        delta_phi.backward() 

        self.optimizer.step()
        # Contrastive Similarity Matching Lateral Weight Update additional term is added below (before optimizer step)
        with torch.no_grad(): # Check line 306 in https://github.com/Pehlevan-Group/Supervised-Similarity-Matching/blob/master/Main/model_wlat_smep_mod.py
            for kk in range(len(self.M)):
                Mweight = self.M[kk].weight.data
                self.M[kk].weight.data = Mweight + (alphas_M[kk]) * Mweight/(2 * np.abs(beta_2))
                
        for idx in range(len(self.M)):
            self.M_copy[idx].weight.data = self.M[idx].weight.data
            self.M_copy[idx].weight.data.requires_grad_(False)

###### Debugging #####

class ContrastiveCorInfoMaxHopfieldDebug1():
    """This is the algorithm to be used in the paper. The summary will be added later.
    """

    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid, device = None):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # self.run_neural_dynamics = self.run_neural_dynamics_hopfield
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)

            Wff.append({'weight': weight})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)

            Wfb.append({'weight': weight})
        Wfb = np.array(Wfb)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
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
            forward_info_jj= (torch.logdet(R[jj + 1]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm(layers[jj + 2] - Wff[jj + 1]['weight'] @ layers[jj + 1]) ** 2 - layers[jj + 2].shape[0] * log_epsilon)).item()

            forward_info_list.append(forward_info_jj)

        for jj in range(len(architecture) - 2):
            Identity_Matrix = epsilon * torch.eye(*R[jj]['weight'].shape).to(device)
            backward_info_jj = (torch.logdet(R[jj]['weight'] + Identity_Matrix) - (1 / batch_size) * (one_over_epsilon * torch.norm((layers[jj + 1]) - Wfb[jj + 1]['weight'] @ layers[jj + 2]) ** 2 - (layers[jj + 1].shape[0] + 1) * log_epsilon)).item()

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
                        
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] #+ Wff[jj]['bias']
                        apical_voltage = (gam_ * B[jj]['weight'] @ ( layers[jj + 1]) + hopfield_g * layers[jj + 1]) - beta * (layers[jj + 1] - y)
                        gradient_neurons = -hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + (apical_voltage - neurons_intermediate[jj]) #+ 2 * beta * (y - layers[jj + 1])
                        neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        neurons[jj] = self.activation(neurons_intermediate[jj])
                        
                    else:
                        
                        basal_voltage = Wff[jj]['weight'] @ layers[jj] #+ Wff[jj]['bias']
                        apical_voltage = epsilon * (2 * gam_ * B[jj]['weight'] @ (layers[jj + 1]) + hopfield_g * layers[jj + 1]) + (Wfb[jj + 1]['weight'] @ layers[jj + 2]) #+ Wfb[jj + 1]['bias']
                        gradient_neurons = - hopfield_g * neurons_intermediate[jj] + one_over_epsilon * (basal_voltage - neurons_intermediate[jj]) + one_over_epsilon * (apical_voltage - neurons_intermediate[jj])
                        neurons_intermediate[jj] = neurons_intermediate[jj] + neural_lr * gradient_neurons
                        neurons[jj] = self.activation(neurons_intermediate[jj])
                    layers = [x] + neurons  # concatenate the input to other layers

            if take_debug_logs:
                info_measures = self.layerwise_forward_and_backward_correlative_information(layers, phase)
                forward_info.append(np.sum(info_measures[0]))
                backward_info.append(np.sum(info_measures[1]))
                    
        return neurons, forward_info, backward_info

    ###############################################################
    ############### BATCH STEP ALGORITHMS #########################
    ###############################################################
    def batch_step_hopfield(self, x, y, hopfield_g, lr, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                            neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations_free = 20, 
                            neural_dynamic_iterations_nudged = 10, beta = 1, use_three_phase = False, 
                            take_debug_logs = False, weight_decay = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_
        epsilon = self.epsilon

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
        forward_errors_free = [layers_free[jj + 1] - (Wff[jj]['weight'] @ layers_free[jj]) for jj in range(len(Wff))]
        forward_errors_nudged = [layers_nudged[jj + 1] - (Wff[jj]['weight'] @ layers_nudged[jj]) for jj in range(len(Wff))]
        ## Compute backward errors
        backward_errors_free = [(layers_free[jj]) - (Wfb[jj]['weight'] @ layers_free[jj + 1]) for jj in range(1, len(Wfb))]
        backward_errors_nudged = [(layers_nudged[jj]) - (Wfb[jj]['weight'] @ layers_nudged[jj + 1]) for jj in range(1, len(Wfb))]

        ### Learning updates for feed-forward and backward weights
        for jj in range(len(Wff)):
            Wff[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['ff'][jj] * torch.mean(outer_prod_broadcasting(forward_errors_free[jj].T, layers_free[jj].T) - outer_prod_broadcasting(forward_errors_nudged[jj].T, layers_nudged[jj].T), axis = 0)
            if weight_decay:
                Wff[jj]['weight'] -= lr['ff'][jj] * epsilon * Wff[jj]['weight']

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            if weight_decay:
                Wfb[jj]['weight'] -= lr['fb'][jj] * epsilon * Wfb[jj]['weight']
        ### Lateral Weight Updates
        for jj in range(len(B)):
            z = B[jj]['weight'] @ (neurons2[jj])
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
                instant_forward_backward_angles.append(self.angle_between_two_matrices(self.Wff[jj]['weight'], self.Wfb[jj]['weight'].T).item())
            
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

    def save_model_weights(self, pickle_name = "CorInfoWeights"):
        Wff_save = []
        for idx in range(len(self.Wff)):
            weight = torch2numpy(self.Wff[idx]['weight'])
            Wff_save.append({'weight': weight})
            
        Wfb_save = []
        for idx in range(len(self.Wfb)):
            weight = torch2numpy(self.Wfb[idx]['weight'])
            Wfb_save.append({'weight': weight})
            
        B_save = []
        for idx in range(len(self.B)):
            weight = torch2numpy(self.B[idx]['weight'])
            B_save.append({'weight': weight})
            
        model_params = pd.DataFrame(columns = ['Wff', 'Wfb', 'B'])

        model_params['Wff'] = Wff_save
        model_params['Wfb'] = Wfb_save
        model_params['B'] = B_save

        model_params.to_pickle(pickle_name + ".pkl")

    def load_model_weights(self, pickle_name):
        model_params_load = pd.read_pickle(pickle_name + ".pkl")
        for idx in range(len(self.Wff)):
            self.Wff[idx]['weight'] = torch.tensor(model_params_load['Wff'].iloc[idx]['weight'], requires_grad = False).to(self.device)
            
        for idx in range(len(self.Wfb)):
            self.Wfb[idx]['weight'] = torch.tensor(model_params_load['Wfb'].iloc[idx]['weight'], requires_grad = False).to(self.device)
            
        for idx in range(len(self.B)):
            self.B[idx]['weight'] = torch.tensor(model_params_load['B'].iloc[idx]['weight'], requires_grad = False).to(self.device)
       
class ContrastiveCorInfoMaxHopfieldDebugWithBias():
    """This is the algorithm to be used in the paper. The summary will be added later.
    """

    def __init__(self, architecture, lambda_, epsilon, activation = hard_sigmoid, output_sparsity = False, STlambda_lr = 0.01, device = None):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.output_sparsity = output_sparsity
        self.STlambda_lr = STlambda_lr
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
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
                            neural_dynamic_iterations_nudged = 10, beta = 1, use_three_phase = False, 
                            take_debug_logs = False, weight_decay = False):

        Wff, Wfb, B = self.Wff, self.Wfb, self.B
        lambda_ = self.lambda_
        gam_ = self.gam_
        epsilon = self.epsilon

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
            if weight_decay:
                Wff[jj]['weight'] -= lr['ff'][jj] * epsilon * Wff[jj]['weight']
                Wff[jj]['bias'] -= lr['ff'][jj] * epsilon * Wff[jj]['bias']

        for jj in range(1, len(Wfb)):
            Wfb[jj]['weight'] += -(1/(beta * (int(use_three_phase) + 1))) * lr['fb'][jj] * torch.mean(outer_prod_broadcasting(backward_errors_free[jj - 1].T, layers_free[jj + 1].T) - outer_prod_broadcasting(backward_errors_nudged[jj - 1].T, layers_nudged[jj + 1].T), axis = 0)
            # Wfb[jj]['bias'] -= (1/beta) * lr['fb'][jj] * torch.mean(backward_errors_free[jj - 1] - backward_errors_nudged[jj - 1], axis = 1, keepdims = True)
            if weight_decay:
                Wfb[jj]['weight'] -= lr['fb'][jj] * epsilon * Wfb[jj]['weight']
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

    def save_model_weights(self, pickle_name = "CorInfoWeights"):
        Wff_save = []
        for idx in range(len(self.Wff)):
            weight, bias = torch2numpy(self.Wff[idx]['weight']), torch2numpy(self.Wff[idx]['bias'])
            Wff_save.append({'weight': weight, 'bias': bias})
            
        Wfb_save = []
        for idx in range(len(self.Wfb)):
            weight = torch2numpy(self.Wfb[idx]['weight'])
            Wfb_save.append({'weight': weight})
            
        B_save = []
        for idx in range(len(self.B)):
            weight = torch2numpy(self.B[idx]['weight'])
            B_save.append({'weight': weight})
            
        model_params = pd.DataFrame(columns = ['Wff', 'Wfb', 'B'])

        model_params['Wff'] = Wff_save
        model_params['Wfb'] = Wfb_save
        model_params['B'] = B_save

        model_params.to_pickle(pickle_name + ".pkl")

    def load_model_weights(self, pickle_name):
        model_params_load = pd.read_pickle(pickle_name + ".pkl")
        for idx in range(len(self.Wff)):
            self.Wff[idx]['weight'] = torch.tensor(model_params_load['Wff'].iloc[idx]['weight'], requires_grad = False).to(self.device)
            self.Wff[idx]['bias'] = torch.tensor(model_params_load['Wff'].iloc[idx]['bias'], requires_grad = False).to(self.device)
            
        for idx in range(len(self.Wfb)):
            self.Wfb[idx]['weight'] = torch.tensor(model_params_load['Wfb'].iloc[idx]['weight'], requires_grad = False).to(self.device)
            
        for idx in range(len(self.B)):
            self.B[idx]['weight'] = torch.tensor(model_params_load['B'].iloc[idx]['weight'], requires_grad = False).to(self.device)
        
