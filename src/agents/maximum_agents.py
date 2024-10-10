from types import SimpleNamespace

from src.analytic import map_hidden_state, mle_hidden_state
from src.gradient_descent import posterior_mode_gradient_descent
from src.history import History
from src.loss import linear_mle_objective, linear_map_objective

class LinearMaximumAgent:
    def __init__(self, params: dict, prior: bool = False) -> None:
        self.params = SimpleNamespace(**params)
        self.prior = prior
        self.history = History()
    
    def infer_state(self, y: float) -> None:
        
        if self.prior:
            self.posterior_mode = map_hidden_state(y=y, beta_0=self.params.beta_0, beta_1=self.params.beta_1, m_x=self.params.m_x)
        else:
            self.posterior_mode = mle_hidden_state(y=y, beta_0=self.params.beta_0, beta_1=self.params.beta_1)
            
    def store_history(self) -> None:
        self.history.store(key="posterior_mode", value=self.posterior_mode)
        
    def get_history(self) -> None:
        return SimpleNamespace(**self.history.history)
        
class LinearGradientDescentAgent:
    def __init__(self, params: dict, prior: bool = False) -> None:
        self.params = SimpleNamespace(**params)
        self.history = History()
        
        if prior:
            self.objective = linear_map_objective
        else:
            self.objective = linear_mle_objective
        
    def infer_state(self, y: float) -> None:
        
        self.x_history, self.loss_history = posterior_mode_gradient_descent(
            x_init=self.params.x_init,
            y=y,
            obj=self.objective,
            params=self.params
        )
    
    def store_history(self) -> None:
        
        keys = ["x_history", "loss_history"]
        values = [self.x_history, self.loss_history]
        
        self.history.store_multiple(keys, values)
        
    def get_history(self) -> None:
        return SimpleNamespace(**self.history.history)