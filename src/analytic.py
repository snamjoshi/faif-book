import numpy as np

from typing import Union

def mle_hidden_state(y: float, beta_0: float, beta_1: float) -> Union[float, np.ndarray]:
    return (np.mean(y) - beta_0) / beta_1

def map_hidden_state(y: float, beta_0: float, beta_1: float, m_x: float) -> Union[float, np.ndarray]:
    return (beta_1 * (np.mean(y) - beta_0) + m_x) / (beta_1**2 + 1)

def mle_beta_1(x: float, y: float) -> float:
    return np.cov(x, y)[0][1] / np.var(x, ddof=1)

def mle_beta_0(x: float, y: float, beta_1: float) -> float:
    return np.mean(y) - beta_1 * np.mean(x)
