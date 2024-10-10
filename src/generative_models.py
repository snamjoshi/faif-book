import numpy as np

from scipy.stats import norm, uniform
from typing import Union

def invert_model_exact(log_model: callable):
    posterior = np.exp(log_model - np.max(log_model))   # log-sum-exp
    return posterior

# def gaussian(y: Union[float, np.ndarray], x: Union[float, np.ndarray], 
#              mu_y: callable, std_y: float, 
#              m_x: float, s_x: float):
    
#     likelihood = norm.pdf(y, loc=mu_y, scale=std_y)
#     prior      = norm.pdf(x, loc=m_x, scale=s_x)
    
#     return likelihood, prior

def gaussian_likelihood(y: Union[float, np.ndarray],
                        mu_y: callable, std_y: float) -> np.ndarray:
    return norm.pdf(y, loc=mu_y, scale=std_y)

def gaussian_prior(x: Union[float, np.ndarray],
                   m_x: float, s_x: float) -> np.ndarray:
    return norm.pdf(x, loc=m_x, scale=s_x)

def uniform_prior(x: Union[float, np.ndarray], 
                  ax: float, bx: float) -> np.ndarray:
    return uniform.pdf(x, loc=ax, scale=bx-ax)