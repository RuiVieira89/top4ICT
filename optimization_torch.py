# https://github.com/githubharald/analyze_ada_hessian/blob/master/src/optimize.py

import torch 

from function_opt import function

def function(X):
	return X[0]**2#*X[1]


params = torch.randn(3)
params.requires_grad_()

# Do gradient descent
n_optim_steps = int(1e4)
optimizer = torch.optim.SGD([params], 1e-2)

path = params.detach().numpy()

for ii in range(n_optim_steps):
    optimizer.zero_grad()
    loss = function(params)

    print(f'Step # {ii}, loss: {loss.item()}, X={params}')
    loss.backward()
    # Access gradient if necessary
    grad = params.grad.data
    optimizer.step()

    path.append(grad.clone().detach().numpy())


def square_x_then_multiply_y(x, y):
    return x**2 * y

def square(x):
    return x**2 

