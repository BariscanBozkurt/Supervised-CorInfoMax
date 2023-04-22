import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
from torch_utils import *
import math

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

            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
            
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

    def PC_loss(self, x, neurons):

        F = 0
        Wff = self.Wff
        # for idx, (x, y) in tqdm(enumerate(loader)):
        #     x, y = x.to(self.device), y.to(self.device)
        #     x = self.activation_inverse(x.view(x.size(0),-1).T, self.activation_type)
        #     y_one_hot = F.one_hot(y, 10).to(device).T
        #     y_one_hot = 0.94 * y_one_hot + 0.03 * torch.ones(*y_one_hot.shape, device = device)
        #     neurons = self.fast_forward(x)

        #     neurons[-1] = y_one_hot.to(torch.float)
        layers = [x] + neurons
        for jj in range(len(Wff)):
            error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation_func(layers[jj], self.activation_type)[0]+ Wff[jj]['bias'])) / self.variances[jj]
            # print(error.shape, torch.sum(error * error, 0).shape)
            F -= self.variances[jj + 1] * torch.sum(error * error, 0)
        return F

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
        # pc_loss = self.PC_loss(x, neurons).mean()
        layers_after_activation = [list(self.activation_func(layers[jj], self.activation_type)) for jj in range(len(layers) - 1)] + [neurons[-1]]
        error_layers = [(layers[jj+1] - (Wff[jj]['weight'] @ layers_after_activation[jj][0] + Wff[jj]['bias'])) / self.variances[jj + 1] for jj in range(len(layers) - 1)]

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
        
        pc_loss = self.PC_loss(x, neurons).mean().item()

        return neurons, pc_loss

class SupervisedPredictiveCodingNudgedV2_wAutoGrad():
    
    def __init__(self, architecture, activation = F.relu, output_activation = F.softmax, sgd_nesterov = False, sgd_momentum = 0.0,
                 sgd_dampening = 0.0, optimizer_type = "adam", optim_lr = 1e-3, use_stepLR = False, stepLR_step_size = 5*3000, stepLR_gamma = 0.9):
        
        self.architecture = architecture

        self.activation = activation
        self.variances = torch.ones(len(architecture))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = (2 * torch.rand(architecture[idx + 1], architecture[idx]).to(self.device) - 1) * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
            
            # torch.nn.init.xavier_uniform_(weight)
            # bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            # torch.nn.init.xavier_uniform_(bias)
            
            torch.nn.init.kaiming_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
            
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
            self.optimizer = torch.optim.SGD(optim_params, momentum = sgd_momentum, dampening = sgd_dampening, nesterov = sgd_nesterov, maximize = True)

        if use_stepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = stepLR_step_size, gamma = stepLR_gamma)

    def neurons_zero_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def neurons_requires_no_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                # neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def fast_forward(self, x, no_grad = False):
        Wff = self.Wff
        if no_grad:
            with torch.no_grad():
                neurons = []
                for jj in range(len(Wff)):
                    if jj == 0:
                        neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                    else:
                        neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                else:
                    neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        return neurons

    def PC_loss(self, x, y, neurons, lambda_weight = 1e-3):
        mbs  = x.shape[1]
        pc_loss = 0
        Wff = self.Wff
        layers = [x] + neurons
        for jj in range(len(Wff)):
            if jj == 0:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) / self.variances[jj]
            else:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers[jj]) + Wff[jj]['bias'])) / self.variances[jj]
            # print(error.shape, torch.sum(error * error, 0).shape)
            pc_loss -= self.variances[jj + 1] * torch.sum(error * error, 0)
        
        prediction_error = neurons[-1] - y 
        pc_loss -= lambda_weight * torch.sum(prediction_error * prediction_error, 0)

        return pc_loss

    def run_neural_dynamics(self, x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device

        for jj in range(len(neurons)):
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

            pc_loss = self.PC_loss(x, y, neurons, lambda_weight)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(pc_loss, neurons, grad_outputs=init_grads, create_graph=False) # dPhi/ds
            
            with torch.no_grad():       
                for neuron_iter in range(len(neurons)):
                    # print(torch.norm(grads[neuron_iter]))
                    neurons[neuron_iter] = neurons[neuron_iter] + (neural_lr) * grads[neuron_iter]
                    neurons[neuron_iter].requires_grad = True

        return neurons

    def batch_step(self, x, y, lambda_weight, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, mode = "train"):

        Wff = self.Wff
        # optimizer = self.optimizer
        neurons = self.fast_forward(x, no_grad = True)
        
        neurons = self.run_neural_dynamics( x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        pc_loss = (1 / lambda_weight) * self.PC_loss(x, y, neurons).sum()
        pc_loss.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

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

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        pc_loss = self.PC_loss(x, neurons).mean()
        pc_loss.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

class SupervisedPredictiveCodingV2_wAutoGrad():
    
    def __init__(self, architecture, activation = torch.sigmoid, optimizer_type = "adam", optim_lr = 1e-3, 
                 use_stepLR = False, stepLR_step_size = 5*3000, stepLR_gamma = 0.9):
        
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

            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)

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

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        pc_loss = self.PC_loss(x, neurons).mean()
        pc_loss.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

### Models under investigation and development
class CorInfoMax_wAutoGrad():
    
    def __init__(self, architecture, lambda_, epsilon, activation = F.relu, optimizer_type = "sgd", optim_lr = 1e-3, 
                 use_stepLR = False, stepLR_step_size = 5*3000, stepLR_gamma = 0.9):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = (2 * torch.rand(architecture[idx + 1], architecture[idx]).to(self.device) - 1)# * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)

            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)

            Wff.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wff = np.array(Wff)

        self.Wff = Wff

        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            # torch.nn.init.xavier_uniform_(weight)
            torch.nn.init.kaiming_uniform_(weight)
            weight = weight @ weight.T
            # weight = 0.1*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)

        self.B = B

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
                        neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                    else:
                        neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                else:
                    neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        return neurons

    def PC_loss(self, x, neurons):
        F = 0
        Wff = self.Wff
        B = self.B
        epsilon, gam_ = self.epsilon, self.gam_
        layers = [x] + neurons
        for jj in range(len(Wff)):
            if jj == 0:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']))
            else:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers[jj]) + Wff[jj]['bias']))

            lateral_term = epsilon * gam_ * 0.5 * torch.sum((B[jj]['weight'] @ layers[jj + 1]) * layers[jj + 1], 0)
            # print(error.shape, torch.sum(error * error, 0).shape)
            F += lateral_term - torch.sum(error * error, 0)
        return F

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device

        for jj in range(len(neurons) - 1):
            neurons[jj] = neurons[jj].requires_grad_()
            
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
        B = self.B
        lambda_ = self.lambda_
        gam_ = self.gam_
        # optimizer = self.optimizer
        neurons = self.fast_forward(x, no_grad = True)

        if mode == "train":
            neurons[-1] = y.to(torch.float)

        
        neurons = self.run_neural_dynamics( x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = B[jj]['weight'] @ neurons[jj]
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

            self.B = B

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        pc_loss = self.PC_loss(x, neurons).mean()
        pc_loss.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

class CorInfoMaxBiDirectional_wAutoGrad():
    
    def __init__(self, architecture, lambda_, epsilon, activation = F.relu, optimizer_type = "sgd", 
                optim_lr_ff = 1e-3, optim_lr_fb = 1e-3, sgd_nesterov = False, sgd_momentum = 0.0,
                sgd_dampening = 0.0, sgd_weight_decay = 0.0, use_stepLR = False, stepLR_step_size = 5*3000, stepLR_gamma = 0.9):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx + 1], architecture[idx]).to(self.device)# * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)

            # torch.nn.init.xavier_uniform_(weight)
            # torch.nn.init.xavier_uniform_(bias)

            # torch.nn.init.orthogonal_(weight)

            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)

            Wff.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wff = np.array(Wff)

        self.Wff = Wff

        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx], architecture[idx + 1]).to(self.device)
            bias = torch.zeros(architecture[idx], 1).to(self.device)

            # torch.nn.init.xavier_uniform_(weight)
            # torch.nn.init.xavier_uniform_(bias)
            
            # torch.nn.init.orthogonal_(weight)

            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)

            Wfb.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wfb = np.array(Wfb)

        self.Wfb = Wfb

        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)

            # torch.nn.init.xavier_uniform_(weight)

            torch.nn.init.kaiming_uniform_(weight)
            weight = weight @ weight.T

            B.append({'weight': weight})
        B = np.array(B)

        self.B = B

        # Lateral Synapses Initialization
        Bsigma = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)

            torch.nn.init.kaiming_uniform_(weight)
            weight = weight @ weight.T

            Bsigma.append({'weight': weight})
        Bsigma = np.array(Bsigma)

        self.Bsigma = Bsigma

        optim_params = []
        for idx in range(len(self.Wff)):
            for key_ in ["weight", "bias"]:
                # if idx == len(self.Wff) - 1:
                #     optim_params.append(  {'params': self.Wff[idx][key_], 'lr': 4*optim_lr_ff}  )
                # else:
                optim_params.append(  {'params': self.Wff[idx][key_], 'lr': optim_lr_ff}  )

        for idx in range(len(self.Wfb)):
            for key_ in ["weight", "bias"]:
                optim_params.append(  {'params': self.Wfb[idx][key_], 'lr': optim_lr_fb}  )

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(optim_params, maximize = True)
        else:
            self.optimizer = torch.optim.SGD(optim_params, momentum = sgd_momentum, dampening = sgd_dampening,
                                             weight_decay = sgd_weight_decay, nesterov = sgd_nesterov, maximize = True)
        
        if use_stepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = stepLR_step_size, gamma = stepLR_gamma)

    def copy_neurons(self, neurons, requires_grad = False):
        copy = []
        for n in neurons:
            copy.append(torch.empty_like(n).copy_(n.data).requires_grad_(requires_grad))

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
                        neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                    else:
                        neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                else:
                    neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        return neurons

    def PC_loss(self, x, neurons):
        F = 0
        Wff, Wfb = self.Wff, self.Wfb
        B, Bsigma = self.B, self.Bsigma
        epsilon, one_over_epsilon, gam_ = self.epsilon, self.one_over_epsilon, self.gam_
        layers = [x] + neurons
        layers_copy = self.copy_neurons(layers)
        for jj in range(len(Wff)):
            if jj == 0:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers_copy[jj] + Wff[jj]['bias']))
            else:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers_copy[jj]) + Wff[jj]['bias']))

            lateral_term = epsilon * gam_ * 0.5 * torch.sum((B[jj]['weight'] @ layers[jj + 1]) * layers[jj + 1], 0)
            # print(error.shape, torch.sum(error * error, 0).shape)
            # if jj == len(Wff) - 1:
            #     F += 4*(lateral_term - torch.sum(error * error, 0))
            # else:
            F += lateral_term - torch.sum(error * error, 0)

        for jj in range(0, len(Wfb)):
            activated_layer = self.activation(layers[jj])
            backward_error = activated_layer - (Wfb[jj]['weight'] @ layers_copy[jj + 1] + Wfb[jj]['bias'])
            if jj != 0:
                lateral_term = epsilon * gam_ * 0.5 * torch.sum((Bsigma[jj - 1]['weight'] @ activated_layer) * activated_layer, 0)
            else:
                lateral_term = 0
            # if jj == len(Wfb) - 1:
            #     F += 4*(lateral_term - torch.sum(backward_error * backward_error, 0))
            # else:
            F += lateral_term - torch.sum(backward_error * backward_error, 0)

        return F

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device
        self.neural_dynamics_loss_list = []
        for jj in range(len(neurons) - 1):
            neurons[jj] = neurons[jj].requires_grad_()
            
        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            pc_loss = self.PC_loss(x, neurons)
            self.neural_dynamics_loss_list.append(pc_loss.mean().item())
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

        Wff, Wfb = self.Wff, self.Wfb
        B, Bsigma = self.B, self.Bsigma
        lambda_ = self.lambda_
        gam_ = self.gam_

        # optimizer = self.optimizer
        # neurons = self.fast_forward(x, no_grad = True)
        neurons = self.init_neurons(x.size(1))

        if mode == "train":
            neurons[-1] = y.to(torch.float)

        
        neurons = self.run_neural_dynamics( x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = B[jj]['weight'] @ neurons[jj]
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        self.B = B

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = Bsigma[jj]['weight'] @ self.activation(neurons[jj])
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                Bsigma[jj]['weight'] = (1 / lambda_) * (Bsigma[jj]['weight'] - gam_ * B_update)

        self.Bsigma = Bsigma

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        pc_loss = self.PC_loss(x, neurons).mean()
        pc_loss.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

class CorInfoMaxBiDirectional_wAutoGradV2():
    
    def __init__(self, architecture, nc, lambda_, epsilon, activation = F.relu, optimizer_type = "sgd", 
                optim_lr_ff = 1e-3, optim_lr_fb = 1e-3, sgd_nesterov = False, sgd_momentum = 0.0,
                sgd_dampening = 0.0, sgd_weight_decay = 0.0, use_stepLR = False, stepLR_step_size = 5*3000, stepLR_gamma = 0.9):
        
        self.architecture = architecture
        self.nc = nc
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx + 1], architecture[idx]).to(self.device)# * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)

            # torch.nn.init.xavier_uniform_(weight)
            # torch.nn.init.xavier_uniform_(bias)

            # torch.nn.init.orthogonal_(weight)

            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)

            Wff.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wff = np.array(Wff)

        self.Wff = Wff

        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx], architecture[idx + 1]).to(self.device)
            bias = torch.zeros(architecture[idx], 1).to(self.device)

            # torch.nn.init.xavier_uniform_(weight)
            # torch.nn.init.xavier_uniform_(bias)
            
            # torch.nn.init.orthogonal_(weight)

            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)

            Wfb.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wfb = np.array(Wfb)

        self.Wfb = Wfb

        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)

            # torch.nn.init.xavier_uniform_(weight)

            torch.nn.init.kaiming_uniform_(weight)
            weight = weight @ weight.T

            B.append({'weight': weight})
        B = np.array(B)

        self.B = B

        # Lateral Synapses Initialization
        Bsigma = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)

            torch.nn.init.kaiming_uniform_(weight)
            weight = weight @ weight.T

            Bsigma.append({'weight': weight})
        Bsigma = np.array(Bsigma)

        self.Bsigma = Bsigma

        optim_params = []
        for idx in range(len(self.Wff)):
            for key_ in ["weight", "bias"]:
                # if idx == len(self.Wff) - 1:
                #     optim_params.append(  {'params': self.Wff[idx][key_], 'lr': 4*optim_lr_ff}  )
                # else:
                optim_params.append(  {'params': self.Wff[idx][key_], 'lr': optim_lr_ff}  )

        for idx in range(len(self.Wfb)):
            for key_ in ["weight", "bias"]:
                optim_params.append(  {'params': self.Wfb[idx][key_], 'lr': optim_lr_fb}  )

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(optim_params, maximize = True)
        else:
            self.optimizer = torch.optim.SGD(optim_params, momentum = sgd_momentum, dampening = sgd_dampening,
                                             weight_decay = sgd_weight_decay, nesterov = sgd_nesterov, maximize = True)
        
        if use_stepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = stepLR_step_size, gamma = stepLR_gamma)

    def copy_neurons(self, neurons, requires_grad = False):
        copy = []
        for n in neurons:
            copy.append(torch.empty_like(n).copy_(n.data).requires_grad_(requires_grad))

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
                        neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                    else:
                        neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                else:
                    neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        return neurons

    def PC_loss(self, x, neurons):
        F = 0
        Wff, Wfb = self.Wff, self.Wfb
        B, Bsigma = self.B, self.Bsigma
        epsilon, one_over_epsilon, gam_ = self.epsilon, self.one_over_epsilon, self.gam_
        layers = [x] + neurons
        layers_copy = self.copy_neurons(layers)
        for jj in range(len(Wff)):
            if jj == 0:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers_copy[jj] + Wff[jj]['bias']))
            else:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers_copy[jj]) + Wff[jj]['bias']))

            lateral_term = epsilon * gam_ * 0.5 * torch.sum((B[jj]['weight'] @ layers[jj + 1]) * layers[jj + 1], 0)
            F += lateral_term - torch.sum(error * error, 0)

        for jj in range(1, len(Wfb)):
            activated_layer = self.activation(layers[jj])
            backward_error = activated_layer - (Wfb[jj]['weight'] @ layers_copy[jj + 1] + Wfb[jj]['bias'])
            lateral_term = epsilon * gam_ * 0.5 * torch.sum((Bsigma[jj - 1]['weight'] @ activated_layer) * activated_layer, 0)
            F += lateral_term - torch.sum(backward_error * backward_error, 0)

        return F

    def run_neural_dynamics(self, x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device
        self.neural_dynamics_loss_list = []
        neurons[-1][:self.nc, :] = y
        for jj in range(len(neurons)):
            neurons[jj] = neurons[jj].requires_grad_()
            

        for iter_count in range(neural_dynamic_iterations):

            if lr_rule == "constant":
                neural_lr = neural_lr_start
            elif lr_rule == "divide_by_loop_index":
                neural_lr = max(neural_lr_start / (iter_count + 1), neural_lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                neural_lr = max(neural_lr_start / (iter_count * lr_decay_multiplier + 1), neural_lr_stop)

            pc_loss = self.PC_loss(x, neurons)
            self.neural_dynamics_loss_list.append(pc_loss.mean().item())
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(pc_loss, neurons, grad_outputs=init_grads, create_graph=False) # dPhi/ds
            
            with torch.no_grad():       
                for neuron_iter in range(len(neurons)):
                    # print(torch.norm(grads[neuron_iter]))
                    neurons[neuron_iter] = neurons[neuron_iter] + neural_lr * grads[neuron_iter]
                    if neuron_iter == len(neurons) - 1:
                        neurons[-1][:self.nc, :] = y
                    neurons[neuron_iter].requires_grad = True

        return neurons

    def batch_step(self, x, y, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, mode = "train"):

        Wff, Wfb = self.Wff, self.Wfb
        B, Bsigma = self.B, self.Bsigma
        lambda_ = self.lambda_
        gam_ = self.gam_

        # optimizer = self.optimizer
        # neurons = self.fast_forward(x, no_grad = True)
        neurons = self.init_neurons(x.size(1))

        # if mode == "train":
        #     neurons[-1] = y.to(torch.float)

        
        neurons = self.run_neural_dynamics( x, y, neurons, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = B[jj]['weight'] @ neurons[jj]
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        self.B = B

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = Bsigma[jj]['weight'] @ self.activation(neurons[jj])
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                Bsigma[jj]['weight'] = (1 / lambda_) * (Bsigma[jj]['weight'] - gam_ * B_update)

        self.Bsigma = Bsigma

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        pc_loss = self.PC_loss(x, neurons).mean()
        pc_loss.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

class SupervisedPredictiveCodingNudged_wAutoGrad():
    
    def __init__(self, architecture, activation = F.relu, output_activation = F.softmax, sgd_nesterov = False, sgd_momentum = 0.0,
                 sgd_dampening = 0.0, optimizer_type = "adam", optim_lr = 1e-3, use_stepLR = False, stepLR_step_size = 5*3000, stepLR_gamma = 0.9):
        
        self.architecture = architecture

        self.activation = activation
        self.variances = torch.ones(len(architecture))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = (2 * torch.rand(architecture[idx + 1], architecture[idx]).to(self.device) - 1) * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
            
            # torch.nn.init.xavier_uniform_(weight)
            # bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            # torch.nn.init.xavier_uniform_(bias)
            
            torch.nn.init.kaiming_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
            
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
            self.optimizer = torch.optim.SGD(optim_params, momentum = sgd_momentum, dampening = sgd_dampening, nesterov = sgd_nesterov, maximize = True)

        if use_stepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = stepLR_step_size, gamma = stepLR_gamma)

    def neurons_zero_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def neurons_requires_no_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                # neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def fast_forward(self, x, no_grad = False):
        Wff = self.Wff
        if no_grad:
            with torch.no_grad():
                neurons = []
                for jj in range(len(Wff)):
                    if jj == 0:
                        neurons.append(self.activation(Wff[jj]['weight'] @ x + Wff[jj]['bias']))
                    else:
                        neurons.append(self.activation(Wff[jj]['weight'] @ neurons[-1] + Wff[jj]['bias']))
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(self.activation(Wff[jj]['weight'] @ x + Wff[jj]['bias']))
                else:
                    neurons.append(self.activation(Wff[jj]['weight'] @ neurons[-1] + Wff[jj]['bias']))
        return neurons

    def PC_loss(self, x, y, neurons, lambda_weight = 1e-3):
        mbs  = x.shape[1]
        pc_loss = 0
        Wff = self.Wff
        layers = [x] + neurons
        for jj in range(len(Wff)):
            error = (layers[jj + 1] - self.activation(Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) / self.variances[jj]
            # print(error.shape, torch.sum(error * error, 0).shape)
            pc_loss -= self.variances[jj + 1] * torch.sum(error * error, 0)
        
        prediction_error = neurons[-1] - y 
        pc_loss -= lambda_weight * torch.sum(prediction_error * prediction_error, 0)
        # if add_ce_loss:
        #     CE_loss = torch.nn.CrossEntropyLoss(reduction = "none")
        #     y_pred = F.softmax(neurons[-1], 0)
        #     ce_loss = lambda_weight * CE_loss(neurons[-1].T, y.to(torch.float).T)
        #     pc_loss -= ce_loss
        return pc_loss

    def run_neural_dynamics(self, x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device

        for jj in range(len(neurons)):
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

            pc_loss = self.PC_loss(x, y, neurons, lambda_weight)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(pc_loss, neurons, grad_outputs=init_grads, create_graph=False) # dPhi/ds
            
            with torch.no_grad():       
                for neuron_iter in range(len(neurons)):
                    # print(torch.norm(grads[neuron_iter]))
                    neurons[neuron_iter] = neurons[neuron_iter] + (neural_lr) * grads[neuron_iter]
                    neurons[neuron_iter].requires_grad = True

        return neurons

    def batch_step(self, x, y, lambda_weight, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, mode = "train"):

        Wff = self.Wff
        # optimizer = self.optimizer
        neurons = self.fast_forward(x, no_grad = True)
        
        neurons = self.run_neural_dynamics( x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        pc_loss = (1 / lambda_weight) * self.PC_loss(x, y, neurons).sum()
        pc_loss.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

class CorInfoMaxNudged():
    
    def __init__(self, architecture, lambda_, epsilon, activation = F.relu, sgd_nesterov = False, sgd_momentum = 0.0,
                 sgd_dampening = 0.0, optimizer_type = "adam", optim_lr = 1e-3, use_stepLR = False, stepLR_step_size = 5*3000, 
                 stepLR_gamma = 0.9):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon

        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = (2 * torch.rand(architecture[idx + 1], architecture[idx]).to(self.device) - 1) 
            
            # torch.nn.init.xavier_uniform_(weight)
            # bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            # torch.nn.init.xavier_uniform_(bias)
            
            torch.nn.init.kaiming_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
            
            Wff.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wff = np.array(Wff)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            # torch.nn.init.xavier_uniform_(weight)
            torch.nn.init.kaiming_uniform_(weight)
            weight = weight @ weight.T
            # weight = 0.1*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)

        self.Wff = Wff
        self.B = B 

        optim_params = []
        for idx in range(len(self.Wff)):
            for key_ in ["weight", "bias"]:
                optim_params.append(  {'params': self.Wff[idx][key_], 'lr': optim_lr}  )

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(optim_params, maximize = True)
        else:
            self.optimizer = torch.optim.SGD(optim_params, momentum = sgd_momentum, dampening = sgd_dampening, nesterov = sgd_nesterov, maximize = True)

        if use_stepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = stepLR_step_size, gamma = stepLR_gamma)

    def neurons_zero_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def neurons_requires_no_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                # neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def fast_forward(self, x, no_grad = False):
        Wff = self.Wff
        if no_grad:
            with torch.no_grad():
                neurons = []
                for jj in range(len(Wff)):
                    if jj == 0:
                        neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                    else:
                        neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                else:
                    neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        return neurons

    def CorInfo_Cost(self, x, y, neurons, lambda_weight = 1e-3):
        mbs  = x.shape[1]
        corinfo_cost = 0
        Wff = self.Wff
        B = self.B
        epsilon, one_over_epsilon, gam_ = self.epsilon, self.one_over_epsilon, self.gam_
        layers = [x] + neurons
        for jj in range(len(Wff)):
            if jj == 0:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) 
            else:
                error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers[jj]) + Wff[jj]['bias']))

            lateral_term = epsilon * gam_ * 0.5 * torch.sum((B[jj]['weight'] @ layers[jj + 1]) * layers[jj + 1], 0)
            corinfo_cost += lateral_term - torch.sum(error * error, 0)
        
        prediction_error = neurons[-1] - y 
        corinfo_cost -= lambda_weight * torch.sum(prediction_error * prediction_error, 0)

        return corinfo_cost

    def run_neural_dynamics(self, x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device

        for jj in range(len(neurons)):
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

            corinfo_cost = self.CorInfo_Cost(x, y, neurons, lambda_weight)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(corinfo_cost, neurons, grad_outputs=init_grads, create_graph=False) # dPhi/ds
            
            with torch.no_grad():       
                for neuron_iter in range(len(neurons)):
                    # print(torch.norm(grads[neuron_iter]))
                    neurons[neuron_iter] = neurons[neuron_iter] + (neural_lr) * grads[neuron_iter]
                    neurons[neuron_iter].requires_grad = True

        return neurons

    def batch_step(self, x, y, lambda_weight, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, mode = "train"):

        Wff = self.Wff
        B = self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        # optimizer = self.optimizer
        neurons = self.fast_forward(x, no_grad = True)
        
        neurons = self.run_neural_dynamics( x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = B[jj]['weight'] @ neurons[jj]
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

            self.B = B

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        corinfo_cost = self.CorInfo_Cost(x, y, neurons).sum() #* (1 / lambda_weight) 
        corinfo_cost.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

class CorInfoMaxBiDirectionalNudged():
    
    def __init__(self, architecture, lambda_, epsilon, activation = F.relu, sgd_nesterov = False, sgd_momentum = 0.0,
                 sgd_dampening = 0.0, sgd_weight_decay = 0.0, optimizer_type = "sgd", optim_lr_ff = 1e-3, optim_lr_fb = 1e-3, use_stepLR = False, stepLR_step_size = 5*3000, 
                 stepLR_gamma = 0.9):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon

        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            # weight = torch.svd(torch.eye(architecture[idx + 1], architecture[idx]).to(self.device)) [2].T
            weight = torch.eye(architecture[idx + 1], architecture[idx]).to(self.device)
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)

            # torch.nn.init.orthogonal_(weight)
            # torch.nn.init.xavier_uniform_(weight)
            # torch.nn.init.xavier_uniform_(bias)
            
            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
            
            Wff.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wff = np.array(Wff)
        
        # Feedback Synapses Initialization
        Wfb = []
        for idx in range(len(architecture)-1):
            weight = torch.eye(architecture[idx], architecture[idx + 1]).to(self.device)
            bias = torch.zeros(architecture[idx], 1).to(self.device)

            # weight = torch.linalg.pinv(Wff[idx]['weight'].detach())
            # torch.nn.init.xavier_uniform_(weight)
            # torch.nn.init.xavier_uniform_(bias)
            
            torch.nn.init.kaiming_uniform_(weight)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)

            Wfb.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
        Wfb = np.array(Wfb)

        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            # torch.nn.init.xavier_uniform_(weight)
            torch.nn.init.kaiming_uniform_(weight)
            weight = weight @ weight.T
            weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            B.append({'weight': weight})
        B = np.array(B)

        # Lateral Synapses Initialization
        Bsigma = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.kaiming_uniform_(weight)
            weight = weight @ weight.T
            weight = torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            Bsigma.append({'weight': weight})
        Bsigma = np.array(Bsigma)
            

        self.Wff = Wff
        self.Wfb = Wfb
        self.B = B 
        self.Bsigma = Bsigma

        optim_params = []
        for idx in range(len(self.Wff)):
            # for key_ in ["weight", "bias"]:
            for key_ in ["weight", "bias"]:
                if idx == 0:
                    optim_params.append(  {'params': self.Wff[idx][key_], 'lr': optim_lr_ff/100}  )
                else:
                    optim_params.append(  {'params': self.Wff[idx][key_], 'lr': optim_lr_ff}  )

        for idx in range(len(self.Wfb)):
            for key_ in ["weight", "bias"]:
                optim_params.append(  {'params': self.Wfb[idx][key_], 'lr': optim_lr_fb}  )

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(optim_params, maximize = True)
        else:
            self.optimizer = torch.optim.SGD(optim_params, momentum = sgd_momentum, dampening = sgd_dampening, 
                                             weight_decay = sgd_weight_decay, nesterov = sgd_nesterov, maximize = True)
        
        if use_stepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = stepLR_step_size, gamma = stepLR_gamma)

    def copy_neurons(self, neurons, requires_grad = False):
        copy = []
        for n in neurons:
            copy.append(torch.empty_like(n).copy_(n.data).requires_grad_(requires_grad))

        return copy

    def neurons_zero_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def neurons_requires_no_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                # neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

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

    def fast_forward(self, x, no_grad = False):
        Wff = self.Wff
        if no_grad:
            with torch.no_grad():
                neurons = []
                for jj in range(len(Wff)):
                    if jj == 0:
                        neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                    else:
                        neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                else:
                    neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        return neurons

    # def CorInfo_Cost(self, x, y, neurons, lambda_weight = 1e-3):

    #     mbs  = x.shape[1]
    #     corinfo_cost = 0
    #     Wff, Wfb = self.Wff, self.Wfb
    #     B, Bsigma = self.B, self.Bsigma
    #     epsilon, one_over_epsilon, gam_ = self.epsilon, self.one_over_epsilon, self.gam_
    #     layers = [x] + neurons
    #     layers_copy = self.copy_neurons(layers)
    #     for jj in range(len(Wff)):
    #         if jj == 0:
    #             # forward_error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']))
    #             forward_error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers_copy[jj] + Wff[jj]['bias'])) 
    #         else:
    #             # forward_error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers[jj]) + Wff[jj]['bias']))
    #             forward_error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers_copy[jj]) + Wff[jj]['bias']))

    #         lateral_term = epsilon * gam_ * 0.5 * torch.sum((B[jj]['weight'] @ layers[jj + 1]) * layers[jj + 1], 0)
    #         corinfo_cost += (lateral_term - torch.sum(forward_error * forward_error, 0))
        
    #     for jj in range(1, len(Wfb)):
    #         activated_layer = self.activation(layers[jj])
    #         backward_error = activated_layer - (Wfb[jj]['weight'] @ layers_copy[jj + 1] + Wfb[jj]['bias'])
    #         lateral_term = epsilon * gam_ * 0.5 * torch.sum((Bsigma[jj - 1]['weight'] @ activated_layer) * activated_layer, 0)
    #         corinfo_cost += (lateral_term - torch.sum(backward_error * backward_error, 0))

    #     prediction_error = neurons[-1] - y 
    #     corinfo_cost -= lambda_weight * torch.sum(prediction_error * prediction_error, 0)

    #     return corinfo_cost

    def CorInfo_Cost(self, x, y, neurons, lambda_weight = 1e-3):

        mbs  = x.shape[1]
        corinfo_cost = 0
        Wff, Wfb = self.Wff, self.Wfb
        B, Bsigma = self.B, self.Bsigma
        epsilon, one_over_epsilon, gam_ = self.epsilon, self.one_over_epsilon, self.gam_
        layers = [x] + neurons
        layers_copy = self.copy_neurons(layers)
        for jj in range(len(Wff)):
            if jj == 0:
                # forward_error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias']))
                forward_error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers_copy[jj] + Wff[jj]['bias'])) 
            else:
                # forward_error = (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers[jj]) + Wff[jj]['bias']))
                forward_error = (layers[jj + 1] - (Wff[jj]['weight'] @ layers_copy[jj] + Wff[jj]['bias']))

            lateral_term = gam_ * 0.5 * torch.sum((B[jj]['weight'] @ layers[jj + 1]) * layers[jj + 1], 0)
            corinfo_cost += (lateral_term - one_over_epsilon * torch.sum(forward_error * forward_error, 0))
        
        for jj in range(1, len(Wfb)):
            backward_error = layers[jj] - (Wfb[jj]['weight'] @ layers_copy[jj + 1] + Wfb[jj]['bias'])
            lateral_term = gam_ * 0.5 * torch.sum((B[jj - 1]['weight'] @ layers[jj]) * layers[jj], 0)
            corinfo_cost += (lateral_term - one_over_epsilon * torch.sum(backward_error * backward_error, 0))

        prediction_error = neurons[-1] - y 
        corinfo_cost -= lambda_weight * torch.sum(prediction_error * prediction_error, 0)

        return corinfo_cost

    def run_neural_dynamics(self, x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device

        for jj in range(len(neurons)):
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

            corinfo_cost = self.CorInfo_Cost(x, y, neurons, lambda_weight)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(corinfo_cost, neurons, grad_outputs=init_grads, create_graph=False) # dPhi/ds
            
            with torch.no_grad():       
                for neuron_iter in range(len(neurons)):
                    # print(torch.norm(grads[neuron_iter]))
                    neurons[neuron_iter] = self.activation(neurons[neuron_iter] + (neural_lr) * grads[neuron_iter])
                    neurons[neuron_iter].requires_grad = True

        return neurons

    def batch_step(self, x, y, lambda_weight, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, mode = "train"):

        Wff, Wfb = self.Wff, self.Wfb
        B, Bsigma = self.B, self.Bsigma
        lambda_ = self.lambda_
        gam_ = self.gam_

        # optimizer = self.optimizer
        neurons = self.fast_forward(x, no_grad = True)
        # neurons = self.init_neurons(x.size(1), device = self.device)

        neurons = self.run_neural_dynamics( x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = B[jj]['weight'] @ neurons[jj]
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

        self.B = B

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = Bsigma[jj]['weight'] @ self.activation(neurons[jj])
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                Bsigma[jj]['weight'] = (1 / lambda_) * (Bsigma[jj]['weight'] - gam_ * B_update)

        self.Bsigma = Bsigma

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        corinfo_cost = self.CorInfo_Cost(x, y, neurons).sum() #* (1 / lambda_weight) 
        corinfo_cost.backward()
        self.optimizer.step()

        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

class CorInfoMaxNudgedV1():
    
    def __init__(self, architecture, lambda_, epsilon, activation = F.relu, sgd_nesterov = False, sgd_momentum = 0.0,
                 sgd_dampening = 0.0, optimizer_type = "adam", optim_lr = 1e-3, use_stepLR = False, stepLR_step_size = 5*3000, 
                 stepLR_gamma = 0.9):
        
        self.architecture = architecture
        self.lambda_ = lambda_
        self.gam_ = (1 - lambda_) / lambda_
        self.epsilon = epsilon
        self.one_over_epsilon = 1 / epsilon

        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_stepLR = use_stepLR
        # Feedforward Synapses Initialization
        Wff = []
        for idx in range(len(architecture)-1):
            weight = (2 * torch.rand(architecture[idx + 1], architecture[idx]).to(self.device) - 1) 
            
            # torch.nn.init.xavier_uniform_(weight)
            # bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            # torch.nn.init.xavier_uniform_(bias)
            
            torch.nn.init.kaiming_uniform_(weight)
            bias = torch.zeros(architecture[idx + 1], 1).to(self.device)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
            
            Wff.append({'weight': weight.requires_grad_(), 'bias': bias.requires_grad_()})
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

        self.Wff = Wff
        self.B = B 

        optim_params = []
        for idx in range(len(self.Wff)):
            for key_ in ["weight", "bias"]:
                optim_params.append(  {'params': self.Wff[idx][key_], 'lr': optim_lr}  )

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(optim_params, maximize = True)
        else:
            self.optimizer = torch.optim.SGD(optim_params, momentum = sgd_momentum, dampening = sgd_dampening, nesterov = sgd_nesterov, maximize = True)

        if use_stepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = stepLR_step_size, gamma = stepLR_gamma)

    def neurons_zero_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def neurons_requires_no_grad(self, neurons):
        for idx in range(len(neurons)):
            if neurons[idx].grad is not None:
                # neurons[idx].grad.zero_()
                neurons[idx].requires_grad_(False)
        return neurons

    def fast_forward(self, x, no_grad = False):
        Wff = self.Wff
        if no_grad:
            with torch.no_grad():
                neurons = []
                for jj in range(len(Wff)):
                    if jj == 0:
                        neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                    else:
                        neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        else:
            neurons = []
            for jj in range(len(Wff)):
                if jj == 0:
                    neurons.append(Wff[jj]['weight'] @ x + Wff[jj]['bias'])
                else:
                    neurons.append(Wff[jj]['weight'] @ self.activation(neurons[-1]) + Wff[jj]['bias'])
        return neurons

    def CorInfo_Cost(self, x, y, neurons, lambda_weight = 1e-3):
        mbs  = x.shape[1]
        corinfo_cost = 0
        Wff = self.Wff
        B = self.B
        epsilon, one_over_epsilon, gam_ = self.epsilon, self.one_over_epsilon, self.gam_
        layers = [x] + neurons
        for jj in range(len(Wff)):
            if jj == 0:
                error = one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ layers[jj] + Wff[jj]['bias'])) 
            else:
                error = one_over_epsilon * (layers[jj + 1] - (Wff[jj]['weight'] @ self.activation(layers[jj]) + Wff[jj]['bias']))

            lateral_term = gam_ * 0.5 * torch.sum((B[jj]['weight'] @ layers[jj + 1]) * layers[jj + 1], 0)
            corinfo_cost += lateral_term - torch.sum(error * error, 0)
        
        prediction_error = neurons[-1] - y 
        corinfo_cost -= lambda_weight * torch.sum(prediction_error * prediction_error, 0)

        return corinfo_cost

    def run_neural_dynamics(self, x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = "constant", lr_decay_multiplier = 0.1, 
                            neural_dynamic_iterations = 10):

        mbs = x.size(1)
        device = x.device

        for jj in range(len(neurons)):
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

            corinfo_cost = self.CorInfo_Cost(x, y, neurons, lambda_weight)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(corinfo_cost, neurons, grad_outputs=init_grads, create_graph=False) # dPhi/ds
            
            with torch.no_grad():       
                for neuron_iter in range(len(neurons)):
                    # print(torch.norm(grads[neuron_iter]))
                    neurons[neuron_iter] = neurons[neuron_iter] + (neural_lr) * grads[neuron_iter]
                    neurons[neuron_iter].requires_grad = True

        return neurons

    def batch_step(self, x, y, lambda_weight, neural_lr_start, neural_lr_stop, neural_lr_rule = "constant", 
                   neural_lr_decay_multiplier = 0.1, neural_dynamic_iterations = 10, mode = "train"):

        Wff = self.Wff
        B = self.B
        lambda_ = self.lambda_
        gam_ = self.gam_

        # optimizer = self.optimizer
        neurons = self.fast_forward(x, no_grad = True)
        
        neurons = self.run_neural_dynamics( x, y, neurons, lambda_weight, neural_lr_start, neural_lr_stop, lr_rule = neural_lr_rule,
                                            lr_decay_multiplier = neural_lr_decay_multiplier, 
                                            neural_dynamic_iterations = neural_dynamic_iterations)

        with torch.no_grad():
            ### Lateral Weight Updates
            for jj in range(len(B)):
                z = B[jj]['weight'] @ neurons[jj]
                B_update = torch.mean(outer_prod_broadcasting(z.T, z.T), axis = 0)
                B[jj]['weight'] = (1 / lambda_) * (B[jj]['weight'] - gam_ * B_update)

            self.B = B

        neurons = self.neurons_zero_grad(neurons)
        self.optimizer.zero_grad()
        corinfo_cost = (1 / lambda_weight) * self.CorInfo_Cost(x, y, neurons).sum()
        corinfo_cost.backward()
        self.optimizer.step()
        # optimizer = self.optimizer
        if self.use_stepLR:
            self.scheduler.step()

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
            # weight = (2 * torch.rand(architecture[idx + 1], architecture[idx], requires_grad = False).to(self.device) - 1) * (4 * np.sqrt(6 / (architecture[idx + 1] + architecture[idx])))
            bias = torch.zeros(architecture[idx + 1], 1, requires_grad = False).to(self.device)
            Wff.append({'weight': weight, 'bias': bias})
        Wff = np.array(Wff)
        
        # Lateral Synapses Initialization
        B = []
        for idx in range(len(architecture)-1):
            weight = torch.randn(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
            torch.nn.init.xavier_uniform_(weight)
            weight = weight @ weight.T
            weight = 0.1*torch.eye(architecture[idx + 1], architecture[idx + 1], requires_grad = False).to(self.device)
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
