# 19-05-2023
"""Logistic Regression - implementation"""

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 0 - Prepare data

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#torch tensor conversion

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#reshaping dataset

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


"""#1 - MODEL DESIGN """
# f = wx + b, sigmoid fn at the end
class LogisticRegression(nn.Module) :
    def __init__(self, n_input_features) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x) :
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)
         
"""#2 - LOSS AND OPTIMIZER"""
learning_rate = 0.01
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""#3 TRAINING LOOP"""
n_epochs = 100

for epoch in range(n_epochs) :
    #forward pass ans loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    #backward pass
    loss.backward()

    #update
    optimizer.step()

    #zero grad
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0 :
        print(f"epoch : {epoch+1}, loss : {loss.item():.4f}")


with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    accuracy = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy : {accuracy:.4f}")


"""Plotting"""



predicted = model(X_train).detach().numpy()
plt.plot(X_train.numpy(), y_train.numpy(), 'ro')
plt.plot(X_train.numpy(), predicted, 'b')
plt.show()