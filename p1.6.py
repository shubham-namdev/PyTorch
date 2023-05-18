#18-05-2023
"""Linear Regression - Implementation and Plotting"""

""" Training Pipeline Steps - 
#1 - Desing Modes (input, output size, forward pass)
#2 - Construct loss and optimizer
#3 - Training Loop-
    - forward pass : compute prediction and loss
    - backward pass : gradients
    - update weights
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# NOTE: STEP (0) : Prepare Data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # reshape the Y tensor
n_samples, n_features = X.shape


# NOTE: STEP (1) : Model Design
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


# NOTE: STEP (2) : Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# NOTE: STEP (3) : Training Loop
n_epochs = 100
for epoch in range(n_epochs) :
    #forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    #backward  pass
    loss.backward()

    #update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0 :
        print(f"Epoch : {epoch+1}, loss = {loss.item():.4f}")


#plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()


