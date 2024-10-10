import numpy as np

from types import SimpleNamespace

import src.generating_functions as GeneratingFunctions
import src.noise as Noise 

from src.maths import quadratic


class StaticLinearEnvironment:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
    def build(self, x_star: float):
        
        # Bind state to class
        self.x_star = x_star
        
        # Define noise
        self.noise = Noise.zero_centered_normal(scale=self.params.y_star_std)
        
        # Define generating function
        self.generating_function = GeneratingFunctions.linear(
            x_star=x_star, 
            intercept=self.params.beta_0_star, 
            slope=self.params.beta_1_star)
        
    def store_history(self):
        keys = ["x_star", "noise", "generating_function"]
        values = [self.x_star, self.noise, self.generating_function]
        
    def generate(self):
        return self.generating_function + self.noise

class StaticNonlinearEnvironment:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
    def noise(self) -> float:
        return np.random.normal(loc=0, scale=self.params.y_star_std)
    
    def phi(self, x_star: float) -> float:
        return x_star**2
    
    def generating_function(self, x_star: float) -> float:
        return self.params.beta_1_star * quadratic(x_star) + self.params.beta_0_star
    
    def generate(self, x_star: float) -> float:
        return self.generating_function(x_star) + self.noise()
    