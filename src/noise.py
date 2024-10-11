import numpy as np

def zero_centered_unit_normal():
    return np.random.normal(loc=0, scale=1)

def zero_centered_normal(scale):
    return np.random.normal(loc=0, scale=scale)