import torch
import numpy as np
import torchvision
import torch.nn.functional as F

import math

from itertools import repeat
from torch.nn.parameter import Parameter

class DiagLinear(torch.nn.Module):
    """ 
    Custom Diagonal Linear Layer 
    Returns x @ D.T where D is a diagonal matrix
    """
    def __init__(self, size_out: int, bias: bool = False, device = None, dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.size_out = size_out

        weight = torch.Tensor(torch.ones(size_out,1)) # Save only the diagonal of the matrix for memory purposes
        self.weight = torch.nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        if bias:
            bias = torch.Tensor(size_out)
            self.bias = torch.nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.weight.t() * x
        if self.bias is not None:
            out = torch.add(out, self.bias)  # w times x + b
        return out

def torch2numpy(x):
    return x.detach().cpu().numpy()
    
# Activation functions
def my_sigmoid(x):
    return 1/(1+torch.exp(-4*(x-0.5)))

def hard_sigmoid(x):
    return (1+F.hardtanh(2*x-1))*0.5

def ctrd_hard_sig(x):
    return (F.hardtanh(2*x))*0.5

def my_hard_sig(x):
    return (1+F.hardtanh(x-1))*0.5

def outer_prod_broadcasting(A, B):
    """Broadcasting trick"""
    return A[...,None]*B[:,None]
# Some helper functions
def grad_or_zero(x):
    if x.grad is None:
        return torch.zeros_like(x).to(x.device)
    else:
        return x.grad

def neurons_zero_grad(neurons):
    for idx in range(len(neurons)):
        if neurons[idx].grad is not None:
            neurons[idx].grad.zero_()

def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy

def my_init(scale):
    def my_scaled_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)
        if isinstance(m, torch.nn.Linear):
            # torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            torch.nn.init.xavier_uniform_(m.weight)
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(0)
    return my_scaled_init
        
def evaluateEP(model, loader, T, device, printing = True):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        neurons = model(x, y, neurons, T) # dynamics for T time steps

        # if not model.softmax:
        #     pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this case prediction is done directly on the last (output) layer of neurons
        # else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
        #     pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

        pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this case prediction is done directly on the last (output) layer of neurons
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluateCorInfoMax(model, loader, neural_lr, T, device, printing = True):
    # Evaluate the model on a dataloader with T steps for the dynamics
    #model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        h, y_hat = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        h, y_hat = model.run_neural_dynamics(x, h, y_hat, 0, neural_lr = neural_lr, 
                                             neural_dynamic_iterations = T, beta = 0) 
        
        pred = torch.argmax(y_hat, dim=0).squeeze()  # in this case prediction is done directly on the last (output) layer of neurons
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc


