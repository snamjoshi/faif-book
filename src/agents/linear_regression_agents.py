import numpy as np

import src.generating_functions as GeneratingFunctions

from scipy.stats import norm
from types import SimpleNamespace

from src.analytic import mle_theta
from src.generative_models import invert_model_exact
from src.maths import loge
from src.utils import build_data_matrix


class UnivariateLinearRegressionAgent:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
    def mle_beta_1(self, x: float, y: float) -> float:
        return np.cov(x, y)[0][1] / np.var(x, ddof=1)

    def mle_beta_0(self, x: float, y: float, beta_1: float) -> float:
        return np.mean(y) - beta_1 * np.mean(x)
    
    def generating_function(self) -> None:
        return self.beta_1 * self.params.x_range + self.beta_0
    
    def learn_parameters(self, x_star: float, y: float) -> None:
        self.beta_1 = self.mle_beta_1(x_star, y)
        self.beta_0 = self.mle_beta_0(x_star, y, self.beta_1)
        
    def generative_model(self, y: float) -> None:
        self.likelihood = norm.pdf(y, loc=self.generating_function(), scale=self.params.std_y)
        
    def infer_state(self, y: float) -> None:
        self.generative_model(y)
        self.evidence  = np.sum(self.likelihood, axis=0)
        self.posterior = self.likelihood / self.evidence
        
        
class MultivariateLinearRegressionAgent:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
    def learn_parameters(self, X: np.ndarray, y: np.ndarray) -> None:    
        self.X = build_data_matrix(X)
        self.theta = mle_theta(self.X, y)
        
    def build(self) -> None:
        self.generating_function = GeneratingFunctions.linear_multivariate
        mu_y = self.generating_function(X=self.X, theta=self.theta)
        
        self.likelihood = norm(loc=mu_y, scale=self.params.std_y)
    
    def infer_state(self, y: np.ndarray):
        self.model = self.likelihood.pdf(y)
        self.log_model = loge(self.likelihood).sum(axis=0)
        self.evidence = np.sum(self.model, axis=0)
        
        self.posterior = invert_model_exact(self.log_model)
        
    def predict(self, X_new):
        mu_y = self.generating_function(X=X_new, theta=self.theta)
        return norm.rvs(loc=mu_y, scale=self.params.std_y)
        
        
        
agent.learn_parameters(X, y)
agent.build()
agent.infer_state(y)
agent.predict(X_new)