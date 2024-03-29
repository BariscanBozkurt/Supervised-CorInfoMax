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
    
def torch_off_diag(X):
    return X - torch.diag(torch.diag(X))

def torch_make_off_diag_nonpositive(X):
    X_diag = torch.diag(torch.diag(X))
    X_off_diag = X - X_diag
    
    return X_diag - torch.relu(-X_off_diag)

# Activation functions
def activation_func(x, type_ = "linear"):
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
    
def activation_inverse(x, type_ = "linear"):
    if type_ == "linear":
        f_x = x 
    elif type_ == "tanh":
        ones_vec = torch.ones(*x.shape, device = x.device)
        f_x = 0.5 * torch.log((ones_vec + x) / (ones_vec - x))
    elif type_ == "sigmoid":
        ones_vec = torch.ones(*x.shape, device = x.device)
        # f_x = torch.log(x / (ones_vec - x))
        f_x = torch.log(x / (ones_vec - x))
    elif type_ == "exp":
        f_x = torch.log(x)
    else: # Use linear inverse
        f_x = x 
    return f_x

def my_sigmoid(x):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/
    return 1/(1+torch.exp(-4*(x-0.5)))

def hard_sigmoid(x):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/
    return (1+F.hardtanh(2*x-1))*0.5

def ctrd_hard_sig(x):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/
    return (F.hardtanh(2*x))*0.5

def my_hard_sig(x):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/
    return (1+F.hardtanh(x-1))*0.5

# Some helper functions
def outer_prod_broadcasting(A, B):
    """Broadcasting trick"""
    return A[...,None]*B[:,None]

def grad_or_zero(x):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/
    if x.grad is None:
        return torch.zeros_like(x).to(x.device)
    else:
        return x.grad

def neurons_zero_grad(neurons):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/
    for idx in range(len(neurons)):
        if neurons[idx].grad is not None:
            neurons[idx].grad.zero_()

def copy(neurons):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy

def my_init(scale):
    # Source : https://github.com/Laborieux-Axel/Equilibrium-Propagation/blob/93660ed6c5b0ec07978b674a69c169ce32e8cd5f/model_utils.py
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
        
def angle_between_two_matrices(A, B):
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

# Model Evaluation Functions
def evaluateEP(model, loader, T, neural_lr, device, printing = True):
    # Evaluate the Equilibrium Propagation type model on a dataloader with T steps for the dynamics for classification task
    model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        neurons = model(x, y, neurons, T, neural_lr) 

        pred = torch.argmax(neurons[-1], dim=1).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluatePC(model, loader, device, apply_activation_inverse = True, activation_type = "sigmoid", printing = True):
    # Evaluate Predictive Coding Model on Classification Task
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        if apply_activation_inverse:
            x = activation_inverse(x.view(x.size(0),-1).T, activation_type).to(device)
        else:
            x = x.view(x.size(0),-1).T.to(device)

        y = y.to(device)
        
        neurons = model.fast_forward(x)

        pred = torch.argmax(neurons[-1], dim=0).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluateClassification(model, loader, device, printing = True):
    # Evaluate Artificial Neural Network on Classification Task
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        y_hat = model(x)
        
        pred = torch.argmax(y_hat, dim=1).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print('Accuracy :\t', acc)   
    return acc

def evaluateContrastiveCorInfoMax(model, loader, neural_lr_start, neural_lr_stop, neural_lr_rule, neural_lr_decay_multiplier,
                                  T, device, printing = True):
    # Evaluate the Contrastive CorInfoMax model on a dataloader with T steps for the dynamics for the classification task
    correct = 0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons = model.run_neural_dynamics(x, 0, neurons, neural_lr_start, neural_lr_stop, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        pred = torch.argmax(neurons[-1], dim=0).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluateContrastiveCorInfoMaxHopfield(model, loader, hopfield_g, neural_lr_start, neural_lr_stop, 
                                          neural_lr_rule, neural_lr_decay_multiplier,
                                          T, device, printing = True):
    # Evaluate the Contrastive CorInfoMax Hopfield model on a dataloader with T steps for the dynamics for the classification task
    correct = 0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons, _, _ = model.run_neural_dynamics_hopfield(x, 0, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        pred = torch.argmax(neurons[-1], dim=0).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluateContrastiveCorInfoMaxHopfieldSparse(model, loader, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list,
                                                neural_lr_rule, neural_lr_decay_multiplier,
                                                T, device, printing = True):
    # Evaluate the Contrastive CorInfoMax Hopfield model on a dataloader with T steps for the dynamics for the classification task
    correct = 0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons, _, _ = model.run_neural_dynamics_hopfield(x, 0, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        pred = torch.argmax(neurons[-1], dim=0).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def topk_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    Modified from: https://github.com/EPFL-LCN/pub-illing2021-neurips/blob/b66061eddaec9d9f41213c3640d3f0961d13cc7b/vision/CLAPPVision/utils/utils.py
    output shape = (number of classes, batch size)
    target size = (batch size)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 0, True, True)
        
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k_list = []
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k_list.append(correct_k.item())
            res.append(correct_k.mul(1.0 / batch_size).item())
        return np.array(res), np.array(correct_k_list)

def evaluateContrastiveCorInfoMaxHopfield_topk( model, loader, hopfield_g, neural_lr_start, neural_lr_stop, 
                                                neural_lr_rule, neural_lr_decay_multiplier,
                                                T, device, topk = (1,), printing = True):
    # Evaluate the Contrastive CorInfoMax Hopfield model on a dataloader with T steps for the dynamics for the classification task
    correct = np.zeros(len(topk))
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons, _, _ = model.run_neural_dynamics_hopfield(x, 0, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        correct += topk_accuracy(neurons[-1], y, topk)[1]

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluateContrastiveCorInfoMaxHopfieldSparse_topk(model, loader, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr,
                                                     neural_lr_rule, neural_lr_decay_multiplier,
                                                     T, device, topk = (1,), printing = True):
    # Evaluate the Contrastive CorInfoMax Hopfield model on a dataloader with T steps for the dynamics for the classification task
    correct = np.zeros(len(topk))
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons, _, _ = model.run_neural_dynamics_hopfield(x, 0, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        # pred = torch.argmax(neurons[-1], dim=0).squeeze()  
        correct += topk_accuracy(neurons[-1], y, topk)[1]

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluateClassification_topk(model, loader, device, topk = (1,), printing = True):
    # Evaluate Artificial Neural Network on Classification Task for topk accuracies
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        y_hat = model(x)
        
        pred = torch.argmax(y_hat, dim=1)#.squeeze()  
        correct += topk_accuracy(y_hat.T, y, topk)[1]

    acc = correct/len(loader.dataset) 
    if printing:
        print('Accuracy :\t', acc)   
    return acc

def columnwise_sparsity(x, threshold = 0.01):
    return (x < threshold).sum(0) / x.shape[0]

def evaluateContrastiveCorInfoMaxHopfieldV2(model, loader, hopfield_g, neural_lr_start, neural_lr_stop,
                                                neural_lr_rule, neural_lr_decay_multiplier,
                                                T, device, printing = True, check_sparsity = True):
    """
    It is the same as evaluateContrastiveCorInfoMaxHopfield, except that it also checks the
    sparsity of the representations in the layers.
    """
    # Evaluate the Contrastive CorInfoMax Hopfield model on a dataloader with T steps for the dynamics for the classification task
    correct = 0
    phase = 'Train' if loader.dataset.train else 'Test'
    if check_sparsity:
        layer_sparsity = [[] for _ in range(len(model.architecture) - 1)]
        
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons, _, _ = model.run_neural_dynamics_hopfield(x, 0, neurons, hopfield_g, neural_lr_start, neural_lr_stop, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        if check_sparsity:
            for jj in range(len(neurons)):
                layer_sparsity[jj].extend(list(torch2numpy(columnwise_sparsity(neurons[jj]))))
                
        correct += topk_accuracy(neurons[-1], y, topk)[1]

    acc = correct/len(loader.dataset) 
    if check_sparsity:
        overall_layer_sparsity = np.array(layer_sparsity).mean(1)
    if printing:
        print(phase+' accuracy :\t', acc)   
        print("Sparsity for layers: ", overall_layer_sparsity)
    
    if check_sparsity:
        return acc, overall_layer_sparsity
    else:
        return acc

def evaluateContrastiveCorInfoMaxHopfieldSparseV2(model, loader, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list,
                                                neural_lr_rule, neural_lr_decay_multiplier,
                                                T, device, printing = True, check_sparsity = True):
    """
    It is the same as evaluateContrastiveCorInfoMaxHopfieldSparse, except that it also checks the
    sparsity of the representations in the layers.
    """
    # Evaluate the Contrastive CorInfoMax Hopfield model on a dataloader with T steps for the dynamics for the classification task
    correct = 0
    phase = 'Train' if loader.dataset.train else 'Test'
    if check_sparsity:
        layer_sparsity = [[] for _ in range(len(model.architecture) - 1)]
        
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons, _, _ = model.run_neural_dynamics_hopfield(x, 0, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        if check_sparsity:
            for jj in range(len(neurons)):
                layer_sparsity[jj].extend(list(torch2numpy(columnwise_sparsity(neurons[jj]))))
                
        pred = torch.argmax(neurons[-1], dim=0).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if check_sparsity:
        overall_layer_sparsity = np.array(layer_sparsity).mean(1)
    if printing:
        print(phase+' accuracy :\t', acc)   
        print("Sparsity for layers: ", overall_layer_sparsity)
    
    if check_sparsity:
        return acc, overall_layer_sparsity
    else:
        return acc

def evaluateContrastiveCorInfoMaxHopfieldSparseV2_topk(model, loader, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list,
                                                neural_lr_rule, neural_lr_decay_multiplier,
                                                T, device, topk = (1,), printing = True, check_sparsity = True):
    """
    It is the same as evaluateContrastiveCorInfoMaxHopfieldSparse_topk, except that it also checks the
    sparsity of the representations in the layers.
    """
    # Evaluate the Contrastive CorInfoMax Hopfield model on a dataloader with T steps for the dynamics for the classification task
    correct = 0
    phase = 'Train' if loader.dataset.train else 'Test'
    if check_sparsity:
        layer_sparsity = [[] for _ in range(len(model.architecture) - 1)]
        
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        neurons = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        neurons, _, _ = model.run_neural_dynamics_hopfield(x, 0, neurons, hopfield_g, neural_lr_start, neural_lr_stop, STlambda_lr_list, neural_lr_rule, neural_lr_decay_multiplier, T, beta = 0) 
        
        if check_sparsity:
            for jj in range(len(neurons)):
                layer_sparsity[jj].extend(list(torch2numpy(columnwise_sparsity(neurons[jj]))))
                
        correct += topk_accuracy(neurons[-1], y, topk)[1]

    acc = correct/len(loader.dataset) 
    if check_sparsity:
        overall_layer_sparsity = np.array(layer_sparsity).mean(1)
    if printing:
        print(phase+' accuracy :\t', acc)   
        print("Sparsity for layers: ", overall_layer_sparsity)
    
    if check_sparsity:
        return acc, overall_layer_sparsity
    else:
        return acc


### Deprecated
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
        
        pred = torch.argmax(y_hat, dim=0).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc

def evaluateCorInfoMaxV2(model, loader, neural_lr_start, neural_lr_stop, neural_lr_rule, neural_lr_decay_multiplier,
                         T, device, printing = True):
    # Evaluate the model on a dataloader with T steps for the dynamics
    #model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x = x.view(x.size(0),-1).to(device).T
        y = y.to(device)
        
        h, y_hat = model.init_neurons(x.size(1), device = model.device)
        
        # dynamics for T time steps
        h, y_hat = model.run_neural_dynamics(x, h, y_hat, 0, neural_lr_start = neural_lr_start, neural_lr_stop = neural_lr_stop,
                                             neural_dynamic_iterations = T, beta = 0, lr_rule = neural_lr_rule, 
                                             lr_decay_multiplier = neural_lr_decay_multiplier, mode = "testing") 
        
        pred = torch.argmax(y_hat, dim=0).squeeze()  
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    if printing:
        print(phase+' accuracy :\t', acc)   
    return acc


