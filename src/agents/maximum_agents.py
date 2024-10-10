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
        
class LinearGradientDescentAgent:
    def __init__(self, params: dict) -> None:
        self.params = SimpleNamespace(**params)
        
    def infer_states(self, y: float):
        history = posterior_mode_gradient_descent(
            kappa=self.params.kappa,
            n_iterations=self.params.n_iterations,
            x_init=self.params.x,
            obj=self.params.objective
        )
        
        return history