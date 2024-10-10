from src.environments.static_environments import *
from src.agents.exact_agents import *
from src.agents.maximum_agents import *

# TODO: List all available models via enum

def create_environment(name: str, params: dict):
    if   name == "static_linear":
        return StaticEnvironment(params=params)
    
    elif name == "static_nonlinear":
        return StaticEnvironment(params=params, nonlinear=True)
    
    else:
        raise KeyError(f"{name} not supported. Available environments: TODO")
    
def create_agent(name: str, params: dict):
    if   name == "exact_linear":
        return ExactAgent(params=params)
    
    elif name == "exact_linear_flat_prior":
        return ExactAgent(params=params, uniform_prior=True)
    
    elif name == "exact_nonlinear":
        return ExactAgent(params=params, nonlinear=True)
    
    elif name == "exact_nonlinear_flat_prior":
        return ExactAgent(params=params, uniform_prior=True, nonlinear=True)
    
    elif name == "linear_mle_agent":
        return LinearMaximumAgent(params=params)
    
    elif name == "linear_map_agent":
        return LinearMaximumAgent(params=params, prior=True)
    
    elif name == "linear_gradient_descent_mle_agent":
        return LinearGradientDescentAgent(params=params)
    
    elif name == "linear_gradient_descent_map_agent":
        return LinearGradientDescentAgent(params=params, prior=True)
    
    else:
        raise KeyError(f"{name} not supported. Available agents: TODO")
    
def validate_parameters():
    ...
