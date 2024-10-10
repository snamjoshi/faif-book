import torch

from torch.distributions import Normal
from typing import Union

def mle_objective(x: float, y: Union[float, torch.tensor], generating_function: callable) -> torch.tensor:
    
    # Parameters
    beta_0 = 3      # Linear generating function intercept
    beta_1 = 2      # Linear generating function slope
    var_y  = 0.5    # Likelihood variance
    
    # Linear genearting function
    mu_y   = generating_function(beta_0=beta_0, beta_1=beta_1, x=x)
    
    # Calculate log-likelihood over samples    
    log_likelihood = Normal(loc=mu_y, scale=torch.sqrt(var_y)).log_prob(y).sum(axis=0)
    
    return -log_likelihood
    
def map_objective(x: float, y: Union[float, torch.tensor], generating_function: callable) -> torch.tensor:
        
    # Parameters
    beta_0 = 3      # Linear generating function intercept
    beta_1 = 2      # Linear generating function slope
    var_y  = 0.5    # Likelihood variance
    m_x    = 2      # Prior mean
    s_x    = 0.25   # Prior variance
    
    # Linear generating function
    mu_y   = generating_function(beta_0=beta_0, beta_1=beta_1, x=x)
    
    # Calculate log-likelihood over samples    
    log_likelihood = Normal(loc=mu_y, scale=torch.sqrt(var_y)).log_prob(y).sum(axis=0)
    
    # Calculate log-prior
    log_prior = Normal(loc=m_x, scale=s_x).log_prob(x)
    
    return -(log_likelihood + log_prior)