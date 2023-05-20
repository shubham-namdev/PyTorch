# 20-05-2023
"""Dataset and Dataloader"""

"""NOTE : Terms -
epoch = 1 forward and backward pass of all training samples.
batch_size = number of training samples in one forward and backward pass
number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size = 20 ---> 100/20 = 5 iterations for 1 epoch

"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


"""Creating our own Dataset"""

class WineDataset(Dataset) :
    def __init__(self) -> None:
        #data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self) :
        # len(dataset)
        return self.n_samples


dataset = WineDataset()

first_data = dataset[0]

features, labels = first_data

#print(features, labels)


"""Using Dataloaer"""
dataset = WineDataset()

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

dataiter = iter(dataloader)
data = next(dataiter)

features, labels = data

#print(features, labels)


"""Training Loop - Dummy"""

n_epochs = 2
total_samples = len(dataset) #178
n_iter = math.ceil(total_samples/4) #45

for epoch in range(n_epochs) :
    for i, (inputs, labels) in enumerate(dataloader) :
        #forward, backward, update
        if (i+1) % 5 == 0 :
            print(f"epoch {epoch+1}/{n_epochs} , step {i+1}/{n_iter}, inputs {inputs.shape}")

