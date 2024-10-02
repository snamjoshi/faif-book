import numpy as np

import src.generating_functions as GeneratingFunctions

from scipy.stats import norm
from types import SimpleNamespace


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
        