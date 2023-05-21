#21-05-2023
"""Dataset Transform"""

from typing import Any
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


"""Creating our own Dataset"""

class WineDataset(Dataset) :
    def __init__(self, transform  = None) -> None:
        #data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        
        # we do not convert to tensors here 
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform :
            sample = self.transform(sample)
        return sample

    def __len__(self) :
        # len(dataset)
        return self.n_samples
    
"""Creating custom transforms"""

class ToTensor :
    def __call__(self, sample) -> Any:
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform :
    def __init__(self, factor) -> None:
        self.factor = factor
    
    def __call__(self, sample) -> Any:
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


dataset = WineDataset(transform=ToTensor())

first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=composed)

first_data = dataset[0]
features, labels = first_data

print(type(features), type(labels))