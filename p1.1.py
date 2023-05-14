#12-05-2023
"""Pytorch 1.1 -  Basics of Tensor"""

import torch
import numpy as np

# creating a tensor

a = torch.empty(5)
a = torch.rand(5, 5) # number of parameters determine the dimentions
a = torch.ones(5) 
a = torch.zeros(5)


# operations  - add , sub, mul, div
# perform element wise operations

a = torch.rand(5)
b = torch.rand(5)

c = torch.add(a, b) #similar to c = a + b
c = torch.sub(a, b)
c = torch.mul(a, b)
c = torch.div(a, b)


# in place changing

a.add_(b) #similarly with every other operations mul_, div_, sub_

# converting tensor to numpy array

a = torch.ones(5)
b = a.numpy()

# converting numpy array to tensor

a = np.ones(5)
b = torch.from_numpy(a)

"""Here while converting the data from tensor to numpy or vice versa they both point to dame location
   that means if we change one the other will also change. This only happens when tensor is on CPU 
"""

# moving cpu tensor to gpu and vice versa

if torch.cuda.is_available() :
    device =  torch.device("cuda")
    x = torch.ones(5, device=device) # making directly on cuda
    y = torch.ones(5) # moving to cuda device
    y = y.to(device)
    z = x + y
    z = z.to("cpu") # moving to cpu

"""NOTE : Numpy can only handle CPU tensors so in GPU we cannot convert tensor to numpy array"""


"""Tells pytorch to calculate the gradient for the tensor later in optimization steps
   whenever you have a variable in your model that needs to be optimized we have to set requires_grad = True
   as we need gradient.
"""

x = torch.ones(5, requires_grad=True)