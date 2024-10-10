from enum import Enum

class EnvironmentRegistry(Enum):
    static_linear = "static_linear"
    static_nonlinear = "static_nonlinear"

class AgentRegistry(Enum):
    exact_linear = "exact_linear"
    exact_linear_flat_prior = "exact_linear_flat_prior"
    exact_nonlinear = "exact_nonlinear"
    exact_nonlinear_flat_prior = "exact_nonlinear_flat_prior"
    linear_mle_agent = "linear_mle_agent"
    linear_map_agent = "linear_map_agent"
    linear_gradient_descent_mle_agent = "linear_gradient_descent_mle_agent"
    linear_gradient_descent_map_agent = "linear_gradient_descent_map_agent"
    