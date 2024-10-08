import numpy as np

from typing import Union

from src.maths import quadratic


def linear(x_star: Union[float, np.ndarray],
           intercept: float, 
           slope: float) -> Union[float, np.ndarray]:
    return slope * x_star + intercept


def nonlinear_quadratic(x_star: Union[float, np.ndarray], 
                        intercept: float, 
                        slope: float) -> Union[float, np.ndarray]:
    return slope * quadratic(x_star) + intercept



    
