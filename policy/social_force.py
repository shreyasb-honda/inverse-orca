"""
The policy for the human based on the social forces model
"""

# TODO: The social forces model sometimes assumes that the human has reached its goal
# TODO: when it is close enough to the goal, but not quite at it.
# TODO: This messes with my goal attainment checker, which only returns true when the human has
# TODO: indeed crossed the finish line. This messes with performance metrics like acceleration,
# TODO: jerk, path irregularity, etc which need a done flag to stop adding new observations.

import logging
import numpy as np
import pysocialforce as psf
from policy.policy import Policy
logging.disable(logging.ERROR)


class SocialForce(Policy):
    """
    Implements the social force based policy for the humans    
    """

    def __init__(self, time_step: float = 0.25) -> None:
        super().__init__(time_step)
        self.config_file = None

    def configure(self, config: str):
        self.config_file = config

    def predict(self, observation):
        human_pos = observation['human pos']
        human_vel = observation['human vel']

        # TODO - set the human goal differently?
        human_goal = (0., human_pos[1])

        robot_pos = observation['robot pos']
        robot_vel = observation['robot vel']

        # TODO - set the robot goal differently?
        robot_goal = (0., robot_pos[1])

        initial_state = np.array(
            [
                [*human_pos, *human_vel, *human_goal],
                [*robot_pos, *robot_vel, *robot_goal]
            ]
        )

        s = psf.Simulator(
            initial_state,
            config_file=self.config_file
        )

        s.step(1)
        states, _ = s.get_states()

        vel = np.array([states[1, 0, 2], states[1, 0, 3]])
        if np.nan in vel:
            vel = np.zeros_like(vel)

        if np.linalg.norm(vel) > 0:
            vel /= np.linalg.norm(vel)

        return tuple(vel)
