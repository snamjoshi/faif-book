import numpy as np

import src.generating_functions as GeneratingFunctions
import src.generative_models as GenerativeModels

from scipy.stats import norm, uniform
from types import SimpleNamespace
from typing import Union

from src.generative_models import invert_model_exact
from src.gradient_descent import posterior_mode_gradient_descent
from src.history import History


# TODO: Refactor exact agents so they inherit from an ExactAgent base class
# TODO: Choice of generating function as an option (if-else statements)
# Note that the other notebooks in ch2 may break with these changes. Rerun and check.

# TODO: Put pydantic base model up here; switch statement for validating params depending on the uniform or not uniform case
        
class ExactAgent:
    def __init__(self, params: dict, uniform_prior: bool = False, nonlinear: bool = False) -> None:
        self.params = SimpleNamespace(**params)
        self.uniform_prior = uniform_prior
        self.nonlinear = nonlinear
        self.history = History()
        
    def build(self, y: Union[float, np.ndarray]):
        
        # Bind data to class
        self.y = y
        
        # Define generating function
        
        if self.nonlinear:
            self.generating_function = GeneratingFunctions.nonlinear_quadratic(
                x_star=self.params.x_range,
                intercept=self.params.beta_0,
                slope=self.params.beta_1
            )
        else:
            self.generating_function = GeneratingFunctions.linear(
                x_star=self.params.x_range, 
                intercept=self.params.beta_0, 
                slope=self.params.beta_1)
        
        self.likelihood = GenerativeModels.gaussian_likelihood(y=y, mu_y=self.generating_function, std_y=self.params.std_y)
        
        if self.uniform_prior:
            self.prior = GenerativeModels.uniform_prior(x=self.params.x_range, ax=self.params.ax, bx=self.params.bx)
        else:
            self.prior = GenerativeModels.gaussian_prior(x=self.params.x_range, m_x=self.params.m_x, s_x=self.params.s_x)
        
        # Define generative model and evidence
        self.model = self.likelihood * self.prior
        self.evidence = np.sum(self.model, axis=0)
        
        # Define log generative model for single and multiple samples
        if isinstance(self.y, np.ndarray):
            self.log_model = np.log(self.likelihood).sum(axis=0) + np.log(self.prior)
        elif isinstance(self.y, float):
            self.log_model = np.log(self.likelihood) + np.log(self.prior)
        else:
            raise TypeError("y must either be a float (single sample) or np.ndarray (array of samples).")
        
    def infer_state(self):
        self.posterior = invert_model_exact(self.log_model)
        
    def store_history(self):

        keys = ["likelihood", "prior", "model", "evidence", "posterior", "generating_function", "y"]
        values = [self.likelihood, self.prior, self.model, self.evidence, self.posterior, self.generating_function, self.y]
        
        self.history.store_multiple(keys, values)
        
    def get_history(self):
        return SimpleNamespace(**self.history.history)
        

# class ExactNonlinearAgent:
#     def __init__(self, params: dict) -> None:
#         self.params = SimpleNamespace(**params)
        
#         # Model components
#         self.likelihood = None
#         self.prior = None
#         self.gen_model = None
#         self.evidence = None
#         self.posterior = None
    
#     def generative_model(self, y: float, generating_function: callable):
#         self.likelihood = norm.pdf(y, loc=generating_function, scale=self.params.std_y)
#         self.prior      = norm.pdf(self.params.x_range, loc=self.params.m_x, scale=self.params.s_x)
#         return self.likelihood * self.prior
        
#     def infer_state(self, y: float):
        
#         generating_function = GeneratingFunctions.nonlinear_quadratic(
#             x_star=self.params.x_range, 
#             intercept=self.params.beta_0, 
#             slope=self.params.beta_1) 
    
#         self.gen_model = self.generative_model(y, generating_function)
#         self.evidence = np.sum(self.gen_model, axis=0)
#         self.posterior = self.gen_model / self.evidence
        
# class ExactLinearUniformPriorAgent:
#     def __init__(self, params: dict) -> None:
#         self.params = SimpleNamespace(**params)
        
#         # Model components
#         self.likelihood = None
#         self.prior = None
#         self.gen_model = None
#         self.evidence = None
#         self.posterior = None
    
#     def generative_model(self, y: float, generating_function: callable):
#         self.likelihood = norm.pdf(y, loc=generating_function, scale=self.params.std_y)
#         self.prior      = uniform.pdf(self.params.x_range, loc=self.params.ax, scale=self.params.bx-self.params.ax)
#         return self.likelihood * self.prior
        
#     def infer_state(self, y: float):
        
#         generating_function = GeneratingFunctions.linear(
#             x_star=self.params.x_range, 
#             intercept=self.params.beta_0, 
#             slope=self.params.beta_1) 
    
#         self.gen_model = self.generative_model(y, generating_function)
#         self.evidence = np.sum(self.gen_model, axis=0)
#         self.posterior = self.gen_model / self.evidence


# TODO: Refactor into a single agent with a choice of MLE of MAP
# TODO: Separate out the inference functions into the maths.py file

