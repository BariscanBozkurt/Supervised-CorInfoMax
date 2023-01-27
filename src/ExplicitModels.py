import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
from torch_utils import *

class SupervisedPredictiveCoding():
    
    def __init__(self, architecture, activation_type = "sigmoid"):
        
        self.architecture = architecture

        self.activation_type = activation_type
        self.variances = torch.ones(len(architecture))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.t = 0 # Used if optimizer is Adam
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            # weight = torch.randn(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device)
            # torch.nn.init.xavier_uniform_(weight)
            weight = (2 * torch.rand(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device) - 1) * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)

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

        layers = [x] + neurons  # concatenate the input to other layers
        init_grads = [torch.zeros(*neurons_.shape, dtype = torch.float, device = self.device) for neurons_ in neurons]
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        error_layers = [(layers[jj+1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias'])) / self.variances[jj + 1] for jj in range(len(layers) - 1)]
        for jj in range(len(init_grads)):
            if jj == len(init_grads) - 1:
                init_grads[jj] = torch.zeros(*layers[jj + 1].shape, device = self.device)
            else:
                # f_layer, fp_layer = self.activation(layers[jj], self.activation_type)
                # error_layer = (layers[jj + 1] - (Wff[jj]['weight'] @ f_layer + Wff[jj]['bias'])) / self.variances[jj + 1]
                init_grads[jj] = - error_layers[jj] + (Wff[jj + 1]['weight'].T @ error_layers[jj + 1]) * layers_after_activation[jj + 1][1]
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
        neurons = self.fast_forward(x)

        if mode == "train":
            neurons[-1] = y.to(torch.float)

        neurons = self.run_neural_dynamics( x, y, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, 
                                            neural_lr_decay_multiplier, neural_dynamic_iterations)

        layers = [x] + neurons  # concatenate the input to other layers
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers))]
        error_layers = [(layers[jj+1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias'])) / self.variances[jj + 1] for jj in range(len(layers) - 1)]

        if optimizer == "sgd":
            ### Learning updates for feed-forward and backward weights
            for jj in range(len(Wff)):
                Wff[jj]['weight'] += lr['ff'] * torch.mean(outer_prod_broadcasting(error_layers[jj].T, layers_after_activation[jj][0].T), axis = 0)
                Wff[jj]['bias'] += lr['ff'] * torch.mean(error_layers[jj], axis = 1, keepdims = True)

            self.Wff = Wff
            
        elif optimizer == "adam":
            for jj in range(len(Wff)):
                grad_Wff_weight = torch.mean(outer_prod_broadcasting(error_layers[jj].T, layers_after_activation[jj][0].T), axis = 0)
                grad_Wff_bias = torch.mean(error_layers[jj], axis = 1, keepdims = True)

                m_Wff_moment[jj]["weight"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta1"]) * grad_Wff_weight
                m_Wff_moment[jj]["bias"] = adam_opt_params["beta1"] * m_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta1"]) * grad_Wff_bias

                v_Wff_moment[jj]["weight"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["weight"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_weight ** 2)
                v_Wff_moment[jj]["bias"] =  adam_opt_params["beta2"] * v_Wff_moment[jj]["bias"] + (1 - adam_opt_params["beta2"]) * (grad_Wff_bias ** 2)

                t += 1
                self.t = t

                Wff[jj]['weight'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["weight"] / (torch.sqrt(v_Wff_moment[jj]["weight"]) + adam_opt_params["eps"])
                Wff[jj]['bias'] += lr['ff'] * np.sqrt(1 - adam_opt_params["beta2"] ** t) / (1 -  adam_opt_params["beta1"] ** t) * m_Wff_moment[jj]["bias"] / (torch.sqrt(v_Wff_moment[jj]["bias"]) + adam_opt_params["eps"])
            
            self.Wff = Wff
            self.m_Wff_moment = m_Wff_moment
            self.v_Wff_moment = v_Wff_moment
        return neurons


class SupervisedPredictiveCoding_wAutoGrad():
    
    def __init__(self, architecture, activation = torch.sigmoid, optimizer_type = "adam", optim_lr = 1e-3, use_stepLR = False, stepLR_step_size = 5*3000, stepLR_gamma = 0.9):
        
        self.architecture = architecture

        self.activation = activation
        self.variances = torch.ones(len(architecture))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = (2 * torch.rand(architecture[idx + 1], architecture[idx]).to(self.device) - 1) * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            Wff.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wff = np.array(Wff)

        self.Wff = Wff

        optim_params = []
        for idx in range(len(self.Wff)):
            for key_ in ["weight", "bias"]:
                optim_params.append(  {'params': self.Wff[idx][key_], 'lr': optim_lr}  )

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(optim_params, maximize = True)
        else:
            self.optimizer = torch.optim.SGD(optim_params, maximize = True)

        if use_stepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = stepLR_step_size, gamma = stepLR_gamma)

    def neurons_zero_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                neurons[idx].grad.zero_()
        return neurons

    def fast_forward(self, x, no_grad = False):
        Wff = self.Wff
        if no_grad:
            with torch.no_grad():
                neurons = []
                for jj in range(len(Wff)):
                    if jj == 0:
                        neurons.append(Wff[jj]['weight'] @ self.activation(x) + Wff[jj]['bias'])
                    else:
                        neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(Wff[jj]['weight'] @ self.activation(x) + Wff[jj]['bias'])
                else:
                    neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        return neurons

    def PC_loss(self, x, neurons):
        F = 0
        Wff = self.Wff
        layers = [x] + neurons
        for jj in range(len(Wff)):
            error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers[jj]) + Wff[jj]['bias'])) / self.variances[jj]
            # print(error.shape, torch.sum(error * error, 0).shape)
            F -= self.variances[jj + 1] * torch.sum(error * error, 0)
        return F

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device

        for jj in range(len(neurons) - 1):
            neurons[jj] = neurons[jj].requires_grad_()
        # pc_loss = self.PC_loss(x, neurons)
        # init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
        # grads = torch.autograd.grad(pc_loss, neurons[:-1], grad_outputs=init_grads, create_graph=False) # dPhi/ds
            
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            pc_loss = self.PC_loss(x, neurons)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(pc_loss, neurons[:-1], grad_outputs=init_grads, create_graph=False) # dPhi/ds
            
            with torch.no_grad():       
                for neuron_iter in range(len(neurons) - 1):
                    # print(torch.norm(grads[neuron_iter]))
                    neurons[neuron_iter] = neurons[neuron_iter] + neural_lr * grads[neuron_iter]
                    neurons[neuron_iter].requires_grad = True

        return neurons

    def batch_step(self, x, y, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, mode = "train"):

        Wff = self.Wff
        # optimizer = self.optimizer
        neurons = self.fast_forward(x, no_grad = True)

        if mode == "train":
            neurons[-1] = y.to(torch.float)

        
        neurons = self.run_neural_dynamics( x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        neuros = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        pc_loss = self.PC_loss(x, neurons).mean()
        pc_loss.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()