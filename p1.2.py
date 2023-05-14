#14-05-2023
"""Pytorch 1.2 -  Calculating Gradient"""

import torch

"""Pytorch provides autograd function to calculate gradient. 
   Gradient is helpful for model optimization.
"""

x = torch.randn(3, requires_grad=True) #tensor creation - we have to specify req_grad = True else produces error
y = x + 2

"""NOTE : The above operation will create a computational graph that has for each operation there is a node
   for inputs and outputs, here x and 2 are inputs and y is the output, with the use of BackPropagation
   gradients are calculated in the backward pass 

            ------------> Forward pass
           x
             \
              \_________         |
               |       |         |
               |   +   | ----- y |
               |_______|         |   grad_fn()
              /                  |
             /                   v
            2   
            <------------- Backward pass            
"""

print(y) # tensor([0.9440, 1.8206, 3.2123], grad_fn=<AddBackward0>)

z = y*y*2
print(z) # tensor([ 8.6626,  3.9800, 24.0829], grad_fn=<MulBackward0>)

z = z.mean()
print(z) # print(z) # tensor([ 8.6626,  3.9800, 24.0829], grad_fn=<MulBackward0>)

z.backward() #calculating gradient of z with respect to x

print(x.grad) # gradients are stored in x.grad attribute
#tensor([2.3687, 0.5868, 3.7967])

"""NOTE : In the background it creates Vector Jacovian Product to get the gradient.
   Where we have the Jacovian Matrix with partial derivatives then we multiply it with
   the gradient vector to get the final gradient. AKA Chain Rule
"""
"""NOTE : Since z here has only one value, i.e., it is scalar that's why we dont need to put any
    argument her e for backward function. If we have multiple values in z then we have to use
    a gradient argument. In most cases the last operation is some operation that will create a 
    scalar value.
"""

v = torch.tensor([0.1, 1.0, 0.001], dtype= torch.float32)
z.backward(v)

"""Preventing Pytorch from tracking the gradient"""

#Method 1:
x.requires_grad_(False)

#Method 2:
x.detach()

#Method 3:
with torch.no_grad() : 
    y = x + 2
    print(y) # tensor([2.2763, 3.2154, 2.4171])


"""NOTE : Whenever we call the backward() function, the gradient for the tensor is accumulated into 
   the .grad attribute, the values will be summed up. So before doing the next iteration in our optimization
   steps we must empty the gradient.
"""
x.grad.zero_() # empty the gradient
