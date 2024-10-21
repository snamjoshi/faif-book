from src.agents.exact_agents import *
from src.agents.maximum_agents import *
from src.environments.static_environments import *
from src.registry import AgentRegistry, EnvironmentRegistry

AGENT_REGISTRY       = [agent_name.value for agent_name in AgentRegistry]
ENVIRONMENT_REGISTRY = [env_name.value for env_name in EnvironmentRegistry]


def create_environment(name: str, params: dict):
    if   name == EnvironmentRegistry.static_linear.value:
        return StaticEnvironment(params=params)
    
    elif name == EnvironmentRegistry.static_nonlinear.value:
        return StaticEnvironment(params=params, nonlinear=True)
    
    else:
        raise KeyError(f"Environment name '{name}' not supported. Available environments: {ENVIRONMENT_REGISTRY}")
    
def create_agent(name: str, params: dict):
    if   name == AgentRegistry.exact_linear.value:
        return ExactAgent(params=params)
    
    elif name == AgentRegistry.exact_linear_flat_prior.value:
        return ExactAgent(params=params, uniform_prior=True)
    
    elif name == AgentRegistry.exact_nonlinear.value:
        return ExactAgent(params=params, nonlinear=True)
    
    elif name == AgentRegistry.exact_nonlinear_flat_prior.value:
        return ExactAgent(params=params, uniform_prior=True, nonlinear=True)
    
    elif name == AgentRegistry.linear_mle_agent.value:
        return LinearMaximumAgent(params=params)
    
    elif name == AgentRegistry.linear_map_agent.value:
        return LinearMaximumAgent(params=params, prior=True)
    
    elif name == AgentRegistry.linear_gradient_descent_mle_agent.value:
        return LinearGradientDescentAgent(params=params)
    
    elif name == AgentRegistry.linear_gradient_descent_map_agent.value:
        return LinearGradientDescentAgent(params=params, prior=True)
    
    else:
        raise KeyError(f"Agent name '{name}' not supported. Available agents: {AGENT_REGISTRY}")
    
def validate_parameters():
    ...

def build_data_matrix(X: np.ndarray):
    
    # Check if already a data matrix
    
    ...
    
def assert_data_matrix():
    # Checks if matrix is a data matrix based on the number of input features
    ...
    
def dynamic_grid(bins, dt):
    return np.arange(0, bins, dt)

def split_seed(seed, n, max_seed=100000):
    np.random.seed(seed)
    return np.random.randint(low=1, high=max_seed, size=n)