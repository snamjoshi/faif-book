# import numpy as np

from types import SimpleNamespace

import src.generating_functions as GeneratingFunctions
import src.noise as Noise 

# from src.maths import quadratic


class StaticEnvironment:
    def __init__(self, params: dict, nonlinear: bool = False) -> None:
        self.params = SimpleNamespace(**params)
        self.nonlinear = nonlinear
        
    def build(self, x_star: float):
        
        # Bind state to class
        self.x_star = x_star
        
        # Define noise
        self.noise = Noise.zero_centered_normal(scale=self.params.y_star_std)
        
        # Define generating function
        if self.nonlinear:
            self.generating_function = GeneratingFunctions.nonlinear_quadratic(
                x_star=x_star, 
                intercept=self.params.beta_0_star, 
                slope=self.params.beta_1_star)
        else:
            self.generating_function = GeneratingFunctions.linear(
                x_star=x_star, 
                intercept=self.params.beta_0_star, 
                slope=self.params.beta_1_star)
        
    def store_history(self):
        keys = ["x_star", "noise", "generating_function"]
        values = [self.x_star, self.noise, self.generating_function]
        
    def generate(self):
        return self.generating_function + self.noise