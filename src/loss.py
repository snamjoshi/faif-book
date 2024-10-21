import numpy as np
import torch

from torch.distributions import Normal
from types import SimpleNamespace
from typing import Union

def linear_mle_objective(x: float, y: Union[float, torch.tensor], params: SimpleNamespace) -> torch.tensor:
    
    # Linear genearting function
    mu_y = params.beta_0 + params.beta_1 * x
    
    # Calculate log-likelihood over samples    
    log_likelihood = Normal(loc=mu_y, scale=params.std_y).log_prob(y).sum(axis=0)
    
    return -log_likelihood
    
def linear_map_objective(x: float, y: Union[float, torch.tensor], params: SimpleNamespace) -> torch.tensor:
    
    # Linear generating function
    mu_y = params.beta_0 + params.beta_1 * x
    
    # Calculate log-likelihood over samples    
    log_likelihood = Normal(loc=mu_y, scale=params.std_y).log_prob(y).sum(axis=0)
    
    # Calculate log-prior
    log_prior = Normal(loc=params.m_x, scale=params.s_x).log_prob(x)
    
    return -(log_likelihood + log_prior)

def univariate_vfe(e_x, e_y, p_x, p_y):
    return 0.5 * (p_y * e_y**2 + p_x * e_x**2 + np.log(p_y**-1 * p_x**-1)) 