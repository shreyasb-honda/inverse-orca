import social_force
from policy.policy import Policy


class SocialForce(Policy):

    def __init__(self, time_step: float = 0.25) -> None:
        super().__init__(time_step)
