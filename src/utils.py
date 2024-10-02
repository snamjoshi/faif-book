from src.environments.static_environments import *
from src.agents.static_agents import *

def create_environment(name: str, params: dict):
    if name == "linear_regression":
        raise KeyError("Linear regression not yet supported")
    elif name == "static_linear":
        return StaticLinearEnvironment(params=params)
    else:
        raise KeyError(f"{name} not supported. Available models: TODO")
    
def create_agent(name: str, params: dict):
    if name == "exact_linear":
        return ExactLinearAgent(params=params)
    else:
        raise KeyError(f"{name} not supported. Available models: TODO")
