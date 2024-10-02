import numpy as np

from typing import Union

def linear(x_star: Union[float, np.ndarray],
           intercept: float, slope: float) -> Union[float, np.ndarray]:
    return slope * x_star + intercept