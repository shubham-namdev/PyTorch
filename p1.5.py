# 18-05-2023
"""Training Pipeline - (Model  /  loss  / optimizer)"""

""" GENERAl TRAINING PIPELINE -
1 - Desing Model - (input, output size, forward pass)
2 - Construct loss and optimizer
3 - Training loop -
    - forward pass : compute prediction
    - backward pass : gradients
    - update weights
"""


"""NOTE : Replace Manual Loss and optimization steps with pytorch modules and functions"""

import torch
import torch.nn as nn # Neural Network Module

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction calculation - 
def forward(x) :
    return w * x

#print(f"Prediction before training : f(5) = {forward(5):.3f}")


learning_rate = 0.01
n_iters = 1000

# loss calculation
# loss  = MSE (mean squared error)

loss = nn.MSELoss()
optimizer  = torch.optim.SGD([w], lr=learning_rate) # Stochastic Gradient Descent

# Training 

for epoch in range(n_iters) :
    # prediction  = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backward pass
    l.backward() # dl/dw

    #update weights
    optimizer.step()

    #zero gradient
    optimizer.zero_grad()

    if epoch % 100 == 0 :
        print(f'epoch {epoch+1} : w = {w:.3f}, loss = {l:.5f}')

#print(f"Prediction after training : f(5) = {forward(5):.3f}")




"""NOTE : Replace Manual Forward Pass with Pytorch Modes"""

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

"""Number of rows = number of samples; for each row we have features """

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features


model = nn.Linear(input_size, output_size)

X_test = torch.tensor([5], dtype=torch.float32)

print(f"Prediction before training : f(5) = {model(X_test).item():.3f}")



learning_rate = 0.01
n_iters = 1000

# loss calculation
# loss  = MSE (mean squared error)

loss = nn.MSELoss()
optimizer  = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Descent

# Training 

for epoch in range(n_iters) :
    # prediction  = forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradient = backward pass
    l.backward() # dl/dw

    #update weights
    optimizer.step()

    #zero gradient
    optimizer.zero_grad()

    if epoch % 10000 == 0 :
        [w,b] = model.parameters()
        print(f'epoch {epoch+1} : w = {w[0][0].item():.3f}, loss = {l:.5f}')

print(f"Prediction after training : f(5) = {model(X_test).item():.3f}")

