import numpy as np
import torch

from types import SimpleNamespace


def posterior_mode_gradient_descent(x_init: float,
                                    y: torch.tensor,
                                    obj: callable,
                                    params: SimpleNamespace) -> list:
    
    print(f"Initializing x at {x_init}.")
    
    # Initialize empty history arrays
    x_history    = torch.zeros(params.n_iterations)
    loss_history = torch.zeros(params.n_iterations)
    
    # Turn x into a Torch tensor which is differentiable
    x = torch.tensor(x_init, requires_grad=True)
    
    # Calculate loss at initialization
    loss = obj(x, y, params)
    
    # Add initialization values to history (j=0)
    x_history[0]    = x.item()
    loss_history[0] = loss
    
    # Gradient descent algorithm (for j+1...n_iterations)
    for j in range(params.n_iterations-1):
        obj_x = obj(x, y, params)  # Compute loss
        obj_x.backward()   # Compute gradient of tensor
        
        with torch.no_grad():
            x -= (params.kappa * x.grad)   # Step in direction of gradient
            x.grad.zero_()          # Zero out the gradients
        
        # Recalculate loss
        loss = obj(x, y, params)
        
        # Append to history
        x_history[j+1] = x.item()
        loss_history[j+1] = loss

    print(f"Final value of x: {np.round(x.item(), 3)}")
    return x_history, loss_history