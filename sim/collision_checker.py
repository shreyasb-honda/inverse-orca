"""
Code for checking collisions between agents at every time step
"""

from numpy.linalg import norm

class CollisionChecker:
    """
    Checks for collisions between the human and the robot at each time step
    """

    def __init__(self, robot_radius, human_radius) -> None:
        self.collision = False
        self.robot_radius = robot_radius
        self.human_radius = human_radius

    def is_collision(self, observation):
        """
        Checks for collision between the human and the robot
        """

        human_pos = observation['human pos']
        robot_pos = observation['robot pos']

        dist = norm(human_pos - robot_pos)

        self.collision =  dist <= (self.human_radius + self.robot_radius)

        return self.collision


def check_collision(obs):
    """
    Checks for collision between the human and the robot
    """
    human_pos = obs['human pos']
    human_radius = obs['human rad']
    robot_pos = obs['robot pos']
    robot_radius = obs['robot rad']

    dist = norm(human_pos - robot_pos)

    return dist <= (human_radius + robot_radius)
