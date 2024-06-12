
class Policy:

    def __init__(self, time_step:float = 0.25) -> None:
        self.last_state = None
        self.time_step = time_step
        self.env = None
    
    def configure(self, config):
        raise NotImplementedError
    
    def set_env(self, env):
        self.env = env
    
    def predict(self, observation):
        raise NotImplementedError
    
    # @staticmethod
    # def reach_destination(obs):
    #     raise NotImplementedError
