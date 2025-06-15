from enum import Enum

class ActionSelection(Enum):
    """
    Enum for the different types of action selection policies the agent can use.
    """
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB = "ucb"