import numpy as np

EPS = np.finfo(float).eps

def std_to_precision(std: float) -> float:
    return 1 / std**2

def quadratic(x: float) -> float:
    return x**2

def loge(array: np.ndarray) -> np.ndarray:
    array[array == 0] = EPS
    return np.log(array)

def euler_step(x, transition_function, t, dt):
    return x[t] + dt * transition_function(x[t])