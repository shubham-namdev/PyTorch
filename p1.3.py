#16-05-2023
"""1.3 - Backpropagation"""

"""NOTE 1 : Chain Rule"""

"""
Suppose we have two functions a(x) ans b(y). x is input for first function and the output is y.
We use this output y as the input for the second function ans z is the final output.

                 _______          _______
                |       |        |       |
        x-->----|   A   |---y--->|   B   |----->z
                |_______|        |_______|

Now, we want to minimize our z. We want the derivative of z w.r.t., x --> dz/dx
By using chain rule - 
        dz/dx = dz/dy * dy/dx

"""

"""NOTE 2 : Computational Graph"""

"""
For each operation we do with our tensors, pytorch will create a graph for us, where at each node,
we apply one operation or one function.
                 _______        
        x------>|       |  
                |F = x*y|------> z
        y------>|_______|

At each nodes we can calculate the local gradients and use these to calculate the final gradient using chain rule.

For the above example we can calculate two gradients -
 1.   dz    dx * y
      --  = ------  = y
      dx      dx
 2.   dz    dx * y 
      --  = ------  = x
      dy      dy

Why we want Local Gradients ? 
Because our graph has many operations and at the very end, we calculate a loss function that we want to minimize, 
so we have to calculate the gradient of this loss w.r.t. our parameter x in the begining, so if at this position
we know the derivative of loss with respect to z, we can get the final gradient, then the gradient of loss w.r.t. x is -

     dLoss     dLoss     dz  (local gradient) 
     -----  =  -----  *  --
      dx        dz       dx

"""

"""NOTE : The whole concept of Backpropagation consists of three steps -
 1. Forward Pass - Compute Loss
 2. Compute local gradients
 3. Backward Pass : Compute   dLoss     using chain rule.
                             --------
                             dWeights (params)
"""

"""EXAMPLE :
We model our output  with a linear combination of weights ans input. Here,
 
    y^ =  y predictex =  w * x
  loss =  (y^ - y)^2  = (wx - y)^2   (squared error. should be mean squared error but for simplicity we are using squared error)

Minimize Loss -  dLoss / dw

NOTE - Process -

                    --------------------------> FORWARD PASS
                 _______          _______         _______  
        x------>|       |   y^   |       |   s   |       |
                |   *   |------->|   -   |------>|   ^2  |----- Loss
        w------>|_______|        |_______|       |_______|  
                                /
                               / 
                              y  
     BACKWARD PASS  <-----------------------------                                     

STEP 1 : Forward pass - calculate loss

STEP 2 : Calculate Local Gradients -
    Here local gradients -  dy^     ds     dLoss
                            ---  ,  --- ,  -----
                            dw      dy^     ds
STEP 3 : Backward Pass -
Using chain rule - 

    dLoss          dLoss          dLoss
    -----  <-----  -----  <-----  -----                    
     dw             dy^             ds
"""

"""
Suppose -  x = 1, y = 2, w = 1

so,  y^  =  w * x  =  1 * 1  =  1 
     s   =  y^ - y =  1 - 2  = -1
    loss =  s^2    = (-1) ^2 =  1

Local gradients =>
1.     dLoss      ds^2
       -----  =  ----  =  2s  = -2 
        ds        ds

2.     ds       dy^ - y
       ---   =  -------  =  1
       dy^        dy^

3.     dy^       dwx
       ---   =   ----  =  x = 1
       dw         dw

Backward Pass =>

    dLoss     dLoss      ds
    -----  =  -----  *   ---   =  2 * s * 1  = -1
     dy^       ds        dy^
      
    dLoss     dLoss      dy^
    -----  =  -----  *   ---   =  -2 * x = -2
     dw        dy^       dw

So, FINAL LOSS = -2 
      
"""

import torch


x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# Forward pass & compute loss
y_cap = w * x
loss = (y_cap - y) ** 2

print(loss)  # tensor(1., grad_fn=<PowBackward0>)

loss.backward()
print(w.grad) # tensor(-2.)


"""
Update our weights
Netx forward and backward passes

"""