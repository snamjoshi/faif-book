import numpy as np

import src.analytic as Analytic
import src.loss as Loss

from types import SimpleNamespace

from src.maths import euler_step
from src.utils import dynamic_grid



# TODO: Abstract prediction errors. Should be one function that has all the different types of PEs? And you select which set you want in a dict or something?

class UnivariateGeneralizedFilter:
    def __init__(self, y: float, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
        # Transition and generating functions
        self.gm = self.params.gm
        self.fm = self.params.fm
        
        # Set up grid
        self.t = dynamic_grid(self.params.bins, self.params.dt)
        self.T = len(self.t)
        
        # VFE function and gradient w.r.t x evaluated at mu_x
        self.F      = Loss.univariate_vfe
        self.grad_F = Analytic.grad_F_mu_x 
        
        # Initialize
        self._initialize_history()
        self._initialize_variables(y)
    
    def _initialize_history(self):
        
        self.mu_x = np.zeros(self.T)     # Expectation (belief) about x  [C]
        self.mu_y = np.zeros(self.T)     # Expectation (belief) about y  [D]
        self.e_x  = np.zeros(self.T)     # State prediction error        [C]
        self.e_y  = np.zeros(self.T)     # Sensory prediction error      [D]
        self.vfe  = np.zeros(self.T)     # Variational free energy
        
    def _initialize_variables(self, y):
        
        # Convert variances to precisions
        self.p_y     = 1 / self.params.var_y
        self.p_x     = 1 / self.params.var_x
        
        # Means
        self.mu_x[0] = self.params.mu_x_init
        self.mu_y[0] = self.gm(mu_x=self.mu_x[0])
        
        # Prediction errors
        self.e_y[0]  = y[0]         - self.gm(mu_x=self.mu_x[0])
        self.e_x[0]  = self.mu_x[0] - self.fm(mu_x=self.mu_x[0])
        
        # Variational free energy
        self.vfe[0]  = self.F(e_x=self.e_x[0], e_y=self.e_y[0],
                              p_x=self.p_x   , p_y=self.p_y)
        
    def _flow(self, mu_x: np.ndarray):
        return self.params.kappa * self.grad_F(mu_x=mu_x,
                                               e_x=self.e_x, e_y=self.e_y,
                                               p_x=self.p_x, p_y=self.p_y,
                                               dg_dmu=self.params.dg_dmu, 
                                               df_dmu=self.params.df_dmu)
        
    def step(self, y: float, t: int):
        
        # Euler's method gradient flow of x evaluated at mu_x
        self.mu_x[t+1] = euler_step(x=self.mu_x, 
                                    transition_function=self._flow, 
                                    t=self.t, dt=self.params.dt)
        
        # Expected observation update given expected state update
        self.mu_y[t+1] = self.gm(mu_x=self.mu_x[t+1])
        
        # Sensory and state prediction error updates
        self.e_y[t+1]  = y[t+1]         - self.gm(mu_x=self.mu_x[t+1])
        self.e_x[t+1]  = self.mu_x[t+1] - self.fm(mu_x=self.mu_x[t+1])
        
        # VFE update
        self.vfe[t+1]  = self.F(e_x=self.e_x[t+1], e_y=self.e_y[t+1], 
                                p_x=self.p_x, p_y=self.p_y)
        
        
# def euler_step(x, transition_function, t, dt):
#     return x[t] + dt * transition_function(x[t])

