import numpy as np

from scipy.stats import norm
from typing import Union

def invert_generative_model():
    ...

def linear_gaussian(y: float, x: Union[float, np.ndarray], 
                    mu_y: callable, std_y: float, 
                    m_x: float, s_x: float):
    likelihood = norm.pdf(y, loc=mu_y, scale=std_y)
    prior      = norm.pdf(x, loc=m_x, scale=s_x)
    return likelihood, prior