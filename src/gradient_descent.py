import numpy as np
import torch


def posterior_mode_gradient_descent(kappa: float, 
                                    n_iterations: int,
                                    x_init: float,
                                    y: torch.tensor,
                                    obj: callable) -> list:
    
    print(f"Initializing x at {x}.")
    
    # Initialize empty history arrays
    x_history    = torch.zeros(n_iterations)
    loss_history = torch.zeros(n_iterations)
    
    # Turn x into a Torch tensor which is differentiable
    x = torch.tensor(x_init, requires_grad=True)
    
    # Calculate loss at initialization
    loss = obj(x, y)
    
    # Add initialization values to history (j=0)
    x_history[0]    = x.item()
    loss_history[0] = loss
    
    # Gradient descent algorithm (for j+1...n_iterations)
    for j in range(n_iterations-1):
        obj_x = obj(x, y)  # Compute loss
        obj_x.backward()   # Compute gradient of tensor
        
        with torch.no_grad():
            x -= (kappa * x.grad)   # Step in direction of gradient
            x.grad.zero_()          # Zero out the gradients
        
        # Recalculate loss
        loss = obj(x, y)
        
        # Append to history
        x_history[j+1] = x.item()
        loss_history[j+1] = loss

    print(f"Final value of x: {np.round(x.item(), 3)}")
    history = {"x": x_history, "loss": loss_history}
    return history