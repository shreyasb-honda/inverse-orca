"""
Base class for all policies
"""

from typing import Dict
from gymnasium import Env

class Policy:
    """
    An abstract base class for all policies for an agent
    Each child should implement the configure() and predict()
    functions
    """

    def __init__(self, time_step:float = 0.25) -> None:
        self.last_state = None
        self.time_step = time_step
        self.env = None

        # The config dictionary
        self.config = None

    def configure(self, config: Dict):
        """
        Configures the parameters of the policy from a toml file
        :param config - the toml configuration filepath for the policy
        """
        raise NotImplementedError

    def set_env(self, env: Env):
        """
        Associates a policy with a gymnasium environment
        :param env - the environment in which this policy acts
        """
        self.env = env

    def predict(self, observation):
        """
        The action choice function for the policy
        This function takes in the observation at the current time step and
        selects and returns an action for the agent that is using this policy
        :param observation - the observation returned by the environment 
        """
        raise NotImplementedError
