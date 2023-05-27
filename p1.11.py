#27-05-2023
"""Activation Function"""

"""They apply a non-linear transformation and decide whether a neuron should be activated or not."""

"""Without activation fn our network is basically just a stacked linear regression model 
   which is not suitable for complex tasks. 
   After each layer we typically use an activation function.
TYPES-  
   Step function - not used in practice
   Sigmoid       - Typically in the last layer of binary classification problem
   TanH          - Hidden layers (-1 and +1)
   ReLU          - Most popular choice in most of networks. 
   Leaky ReLU    - Improved version of ReLU. Tries to solve the vanishing gradient problem
   Softmax       - Good in last layer of multi class classification problems.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F # all activation functions available here

#option 1 - Create nn modules

class NeuralNet(nn.Module) :
    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    
#option 2 - USe activation finctions diectly in forward pass

class NeuralNet(nn.Module) :
    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
