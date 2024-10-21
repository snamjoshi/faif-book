import numpy as np

def zero_centered_unit_normal():
    return np.random.normal(loc=0, scale=1)

def zero_centered_normal(scale):
    return np.random.normal(loc=0, scale=scale)

def generate_white_noise(sigma: np.ndarray, T: int, dt: float, seed: bool=False) -> np.array:
    """ Generates white noise
    
    Based on Matlab code from Hijne 2020, pp. 55-56.

    Args:
        sigma (torch.tensor)  : variance of the white noise [1 x n]
        T (int)               : Number of time points for white noise array
        dt (float)            : Sampling time for white noise array
        seed (bool, optional) : Use random seed? Defaults to False.

    Returns:
        np.array: Returns zero-mean white noise with variance sigma [n x T * (1 / dt)]
    """
    
    n = sigma.shape[1]
    N = len(np.arange(0, T, dt))

    if seed:
        np.random.seed(seed)
        
    omega = np.sqrt(sigma).T * np.random.randn(n, N)   # White noise signal
    return omega

def white_noise(sigmas: list, T: int, dt: float, seeds: list):
    # Generates multiple white noise arrays
    
    assert len(sigmas) == len(seeds)
    n_vars = len(sigmas)
    
    # TODO: Create empty array of the right size
    # [var, n, T * (1 / dt)]
    
    for var in range(n_vars):
        ...
    