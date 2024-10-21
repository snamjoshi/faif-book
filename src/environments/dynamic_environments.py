import numpy as np
import src.noise as Noise 

from types import SimpleNamespace

from src.utils import dynamic_grid, split_seed


# TODO: Noise should be with each step, not generated all at once...

class UnivariateDynamicEnvironment:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
        # Transition and generating functions
        self.ge = self.params.ge
        self.fe = self.params.fe
        
        # Set up grid
        self.t = dynamic_grid(self.params.bins, self.params.dt)
        self.T = len(self.t)
        
        # Initialize
        self._initialize_history()
        self._initialize_noise()
        self._initialize_variables()
        
    def _initialize_noise(self):
        
        seeds = split_seed(seed=self.params.seed)
        
        sigmas = [np.array([[self.params.var_x_star]]),
                  np.array([[self.params.var_y_star]])]
        
        self.omega_star = Noise.white_noise(
            sigmas=sigmas, T=self.T, dt=self.params.dt, seeds=seeds
        )
        
        self.omega_x_star = self.omega_star[0]
        self.omega_y_star = self.omega_star[1]
    
    def _initialize_history(self):
        self.x = np.zeros((self.T, self.params.C))
        self.y = np.zeros((self.T, self.params.D))
        
    def _initialize_variables(self):
        self.x[0] = self.params.x_star_init + self.omega_x_star[0]
        self.y[0] = self.ge(x=self.x[0])    + self.omega_y_star[0]
        
    def step(self, t):
        # TODO: Euler step vs others
        self.x[t+1] = euler_step(x=self.x, fe=self.fe, t=t) + self.omega_x_star[0]
        self.y[t+1] = self.ge(x=self.x[t+1]) + self.omega_y_star[0]
        
    
# Params
# C
# D
# fe
# ge
# x_star
# dt
# bins
# seed
# var_x_star
# var_y_star

# self.generating_function = GeneratingFunctions.nonlinear_quadratic(
#                 x_star=self.params.x_range,
#                 intercept=self.params.beta_0,
#                 slope=self.params.beta_1
#             )

# Noise.zero_centered_normal(scale=self.params.

# Split seed