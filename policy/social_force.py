import logging
from configparser import RawConfigParser
import numpy as np
import pysocialforce as psf
from policy.policy import Policy
logging.disable(logging.ERROR)


class SocialForce(Policy):
    """
    Implements the social force based policy for the humans    
    """

    def configure(self, config: RawConfigParser):
        pass

    def predict(self, observation):
        human_pos = observation['human pos']
        human_vel = observation['human vel']

        # TODO - set the human goal differently?
        human_goal = (0., human_pos[1])

        robot_pos = observation['robot pos']
        robot_vel = observation['robot vel']

        # TODO - set the robot goal differently?
        robot_goal = (15., robot_pos[1])

        initial_state = np.array(
            [
                [*human_pos, *human_vel, *human_goal],
                [*robot_pos, *robot_vel, *robot_goal]
            ]
        )

        s = psf.Simulator(
            initial_state
        )

        s.step(1)
        states, _ = s.get_states()

        vel = np.array([states[1, 0, 2], states[1, 0, 3]])
        vel /= np.linalg.norm(vel)

        return tuple(vel)
