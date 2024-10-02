import src.generating_functions as GeneratingFunctions
import src.noise as Noise 

from types import SimpleNamespace


class StaticLinearEnvironment:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
    def generate(self, x_star: float):
        noise               = Noise.zero_centered_normal(scale=self.params.y_star_std)
        generating_function = GeneratingFunctions.linear(
            x_star=x_star, 
            intercept=self.params.beta_0_star, 
            slope=self.params.beta_1_star) 
        
        return generating_function + noise
