import torch
import torchvision
import torch.nn.functional as F
from torch_utils import *

class MLP(torch.nn.Module):
    """
    Multi Layer Perceptron
    """
    def __init__(self, architecture, activation = F.relu, final_layer_activation = True):
        super(MLP, self).__init__()
        
        self.activation = activation
        self.architecture = architecture 
        self.nc = self.architecture[-1]
        self.final_layer_activation = final_layer_activation
        
        self.linear_layers = torch.nn.ModuleList()
        for idx in range(len(architecture)-1):
            m = torch.nn.Linear(architecture[idx], architecture[idx+1], bias=True)
            self.linear_layers.append(m)
            
    def forward(self, x):
        x = x.view(x.size(0),-1) # flattening the input
        for idx in range(len(self.architecture)-2):
            x = self.activation(self.linear_layers[idx](x))
            
        if self.final_layer_activation:
            x = self.activation(self.linear_layers[-1](x))
        else:
            x = self.linear_layers[-1](x)
            
        return x