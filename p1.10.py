#26-05-2023
"""Softmax and Cross-entropy Loss"""

"""NOTE: SOFTMAX - It squashes the uouput to be between 0 and 1 so we get probabilities"""

"""            
 __________  Scores-logits  _________ probabilities
|          | -----2.0----->|         |--------> 0.7
|  Linear  | -----1.0----->| Softmax |--------> 0.2  y_pred
|__________| -----0.1----->|_________|--------> 0.1

S(yi) = e^yi / Σ e^yi

""" 

import torch
import torch.nn as nn
import numpy as np

"""Numpy Implementation"""

def softmax(x) :
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("Softmax numpy : ",outputs)

"""PyTorch implementation"""

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print("Softmax Pytorch : ",outputs)


"""NOTE: Cross-Entropy Loss - This measures the performance of our classification model
   whose output is a probability between 0 and 1. Loss increases as the predicted probabilty
   diverges from the actual label.
"""
"""
            - 1  
D(Y^, Y) =  ----- . Σ Yi . log(Y^i)
              N
Our Y must be One-Hot encoded.
"""

"""Numpy Implementation"""

def cross_entropy(actual, predicted) :
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# Y must be one hot encoded
# if class 0 : [1,0,0]
# if class 1 : [0,1,0]
# if class 2 : [0,0,1]
Y = np.array([1, 0, 0])

#y_pred has probabilities
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
print(f"Loss1 numpy : {l1:.4f}")
print(f"Loss2 numpy : {l2:.4f}")
#Loss1 numpy : 0.3567
#Loss2 numpy : 2.3026

"""Pytorch Implementation"""

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# n_samples * n_classes
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)
print(f"Loss1 Pytorch : {l1:.4f}")
print(f"Loss2 Pytorch : {l2:.4f}")
#Loss1 Pytorch : 0.4170
#Loss2 Pytorch : 1.8406

