#17-05-2023
"""Gradient Descent Using Autograd (Optimization)"""

"""NOTE : Manually implementing Gradient Descent"""

import numpy as np

# f = w * x, teake w = 2;

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction calculation - 
def forward(x) :
    return w * x

# loss calculation
#loss  = MSE (mean squared error)
def loss(y, y_predicted) :
    return ((y_predicted - y) ** 2).mean()

# gradient calculation
# MSE = 1/N * (w * x - y) ** 2
# dj/dw = 1/n * 2x (w*x - y)

def gradient(x, y, y_predicted) :
    return np.dot(2*x, y_predicted - y).mean()


print(f"Prediction before training : f(5) = {forward(5):.3f}")

# Training 

learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters) :
    # prediction  = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient
    dw = gradient(X, Y, y_pred)

    #update weights
    w -= learning_rate*dw

    if epoch % 1 == 0 :
        print(f'epoch {epoch+1} : w = {w:.3f}, loss = {l:.5f}')

print(f"Prediction after training : f(5) = {forward(5):.3f}")

"""OUTPUT
epoch 4 : w = 1.949, loss = 0.12288
epoch 5 : w = 1.980, loss = 0.01966
epoch 6 : w = 1.992, loss = 0.00315
epoch 7 : w = 1.997, loss = 0.00050
epoch 8 : w = 1.999, loss = 0.00008
epoch 9 : w = 1.999, loss = 0.00001
epoch 10 : w = 2.000, loss = 0.00000
Prediction after training : f(5) = 9.999

Loss is decreased after every step.
"""



"""NOTE : Pytorch Implementation using Autograd"""

import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction calculation - 
def forward(x) :
    return w * x

# loss calculation
#loss  = MSE (mean squared error)
def loss(y, y_predicted) :
    return ((y_predicted - y) ** 2).mean()


print(f"Prediction before training : f(5) = {forward(5):.3f}")

# Training 

learning_rate = 0.01
n_iters = 1000

for epoch in range(n_iters) :
    # prediction  = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backward pass
    l.backward() # dl/dw

    #update weights
    with torch.no_grad() :
        w -= learning_rate * w.grad

    w.grad.zero_()

    if epoch % 100 == 0 :
        print(f'epoch {epoch+1} : w = {w:.3f}, loss = {l:.5f}')

print(f"Prediction after training : f(5) = {forward(5):.3f}")

"""OUTPUT
Prediction before training : f(5) = 0.000
epoch 1 : w = 0.300, loss = 30.00000
epoch 101 : w = 2.000, loss = 0.00000
epoch 201 : w = 2.000, loss = 0.00000
epoch 301 : w = 2.000, loss = 0.00000
epoch 401 : w = 2.000, loss = 0.00000
epoch 501 : w = 2.000, loss = 0.00000
epoch 601 : w = 2.000, loss = 0.00000
epoch 701 : w = 2.000, loss = 0.00000
epoch 801 : w = 2.000, loss = 0.00000
epoch 901 : w = 2.000, loss = 0.00000
Prediction after training : f(5) = 10.000

"""