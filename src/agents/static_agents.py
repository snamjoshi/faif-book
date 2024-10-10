import numpy as np

import src.generating_functions as GeneratingFunctions

from scipy.stats import norm, uniform
from types import SimpleNamespace
from typing import Union


# TODO: Refactor exact agents so they inherit from an ExactAgent base class
# TODO: Change distributions to log space
class ExactLinearAgent:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
        # Model components
        self.likelihood = None
        self.prior = None
        self.gen_model = None
        self.evidence = None
        self.posterior = None
        
    def generative_model(self, y: float, generating_function: callable):
        self.likelihood = norm.pdf(y, loc=generating_function, scale=self.params.std_y)
        self.prior      = norm.pdf(self.params.x_range, loc=self.params.m_x, scale=self.params.s_x)
        return self.likelihood * self.prior
        
    def infer_state(self, y: float):
        
        generating_function = GeneratingFunctions.linear(
            x_star=self.params.x_range, 
            intercept=self.params.beta_0, 
            slope=self.params.beta_1) 
    
        self.gen_model = self.generative_model(y, generating_function)
        self.evidence = np.sum(self.gen_model, axis=0)
        self.posterior = self.gen_model / self.evidence

class ExactNonlinearAgent:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
        # Model components
        self.likelihood = None
        self.prior = None
        self.gen_model = None
        self.evidence = None
        self.posterior = None
    
    def generative_model(self, y: float, generating_function: callable):
        self.likelihood = norm.pdf(y, loc=generating_function, scale=self.params.std_y)
        self.prior      = norm.pdf(self.params.x_range, loc=self.params.m_x, scale=self.params.s_x)
        return self.likelihood * self.prior
        
    def infer_state(self, y: float):
        
        generating_function = GeneratingFunctions.nonlinear_quadratic(
            x_star=self.params.x_range, 
            intercept=self.params.beta_0, 
            slope=self.params.beta_1) 
    
        self.gen_model = self.generative_model(y, generating_function)
        self.evidence = np.sum(self.gen_model, axis=0)
        self.posterior = self.gen_model / self.evidence
        
class ExactLinearUniformPriorAgent:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
        # Model components
        self.likelihood = None
        self.prior = None
        self.gen_model = None
        self.evidence = None
        self.posterior = None
    
    def generative_model(self, y: float, generating_function: callable):
        self.likelihood = norm.pdf(y, loc=generating_function, scale=self.params.std_y)
        self.prior      = uniform.pdf(self.params.x_range, loc=self.params.ax, scale=self.params.bx-self.params.ax)
        return self.likelihood * self.prior
        
    def infer_state(self, y: float):
        
        generating_function = GeneratingFunctions.linear(
            x_star=self.params.x_range, 
            intercept=self.params.beta_0, 
            slope=self.params.beta_1) 
    
        self.gen_model = self.generative_model(y, generating_function)
        self.evidence = np.sum(self.gen_model, axis=0)
        self.posterior = self.gen_model / self.evidence

class LinearMaximumLikelihoodAgent:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
        self.posterior_mode = None
        
    def infer_state(self, y: float) -> Union[float, np.ndarray]:
        self.posterior_mode = (np.mean(y) - self.params.beta_0) / self.params.beta_1
        
class LinearMaximumAprioriAgent:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
        self.posterior_mode = None
        
    def infer_state(self, y: float) -> Union[float, np.ndarray]:
        self.posterior_mode = (self.params.beta_1 * (np.mean(y) - self.params.beta_0) + self.params.m_x) / (self.params.beta_1**2 + 1)