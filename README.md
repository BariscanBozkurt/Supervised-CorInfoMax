# Correlative Information Maximization: A Biologically Plausible Approach to Supervised Deep Neural Networks without Weight Symmetry

This repo contains the codes to reproduce the experiments presented in the paper "Correlative Information Maximization: A Biologically Plausible Approach to Supervised Deep Neural Networks without Weight Symmetry". 

TLDR: We propose correlative information maximization as a normative supervised learning principle that allows derivation of biologically-plausible networks of pyramidal neurons and resolves the weight symmetry problem. 

All the codes are written in Python 3.8 and utilizes Pytorch tensors to be able to run the codes in GPU.

## A segment of a CorInfoMax based neural network

![Sample Network Figures](./Figures/CorInfoMaxNN_.png) 

# Python Version and Dependencies

* Python Version : Python 3.8.8

* pip version : 21.0.1

* Required Python Packages : Specified in requirements.txt file.

* Platform Info : OS: Linux (x86_64-pc-linux-gnu) CPU: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz

# Folder Contents

## src
Python Script         |  Explanation
:--------------------:|:-------------------------:
ContrastiveModels.py   | This python file includes our proposed framework Correlative Information-based neural network. ContrastiveCorInfoMaxHopfield and ContrastiveCorInfoMaxHopfieldSparse are the two methods that we proposed in our paper. Also, it includes other contrastive methods Equilibrium Propagation [1] and Contrastive Similarity Matching [2]. 
ExplicitModels.py       | The explicit methods Predictive Coding [3] and Predictive Coding-Nudge [4] algorihtms are included in this python script.
torch_utils.py          | Some Pytorch utilization functions such as activation functions and evaluation functions.
visualization.py        | Some visuzalization helper functions.

## Notebook_Examples

## Simulations

The image classification experiments are included inside the folder "Simulations/Classification". The subfolder for simulations are named accordingly, i.e., "Simulations/Classification/CorInfoMax" folder includes the simulations for our proposed method in the main text. Similarly, "Simulations/Classification/CorInfoMaxSparse" folder includes the simulations of our framework that is presented in Appendix E. The folder "Simulations/Classification/AnalyzeSimulations" contains the jupyter notebook files to generate the plots and tables presented in our paper. Below, we outline recipe to reproduce the experiments in our paper:


## References (for the other algorithms included in this code repo)

[1] Benjamin Scellier and Yoshua Bengio. Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. Frontiers in Computational Neuroscience, 11, 2017.

[2] Shanshan Qin, Nayantara Mudur, and Cengiz Pehlevan. Contrastive similarity matching for supervised learning. Neural computation, 33(5):1300–1328, 2021.

[3] James CR Whittington and Rafal Bogacz. An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity. Neural computation, 29(5):1229–1262, 2017.

[4] Beren Millidge, Yuhang Song, Tommaso Salvatori, Thomas Lukasiewicz, and Rafal Bogacz. Backpropagation at the infinitesimal inference limit of energy-based models: Unifying predictive coding, equilibrium propagation, and contrastive hebbian learning. In The Eleventh International Conference on Learning Representations, 2023.
