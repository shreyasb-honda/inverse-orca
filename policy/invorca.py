"""
The inverse ORCA policy
"""

from configparser import RawConfigParser
import numpy as np
import rvo2
from policy.policy import Policy
from policy.utils.get_velocity import InverseORCA
from policy.utils.overlap_detection import Point, Circle, VelocityObstacle

class InvOrca(Policy):
    """
    Implements the inverse ORCA algorithm
    """

    def __init__(self, time_step: float = 0.25) -> None:
        super().__init__(time_step)
        self.time_horizon = None
        self.radius = None
        self.max_speed = None
        self.collision_responsibility = None
        self.desired_velocity = None

        self.d_virtual_goal = None
        self.y_virtual_goal = None

        self.vo = None
        self.robot_vel = None
        self.u = None
        self.invorca = None

        # ORCA after stopping criterion achieved
        self.sim = None
        self.max_neighbors = None
        self.orca_time_horizon = None
        self.neighbor_dist = None
        self.orca_time_horizon_obst = None

    def configure(self, config: RawConfigParser):
        self.time_horizon = config.getfloat('invorca', 'time_horizon')
        self.radius = config.getfloat('invorca', 'radius')
        self.collision_responsibility = config.getfloat('invorca', 'collision_responsibility')
        self.max_neighbors = config.getfloat('invorca', 'max_neighbors')
        self.orca_time_horizon = config.getfloat('invorca', 'orca_time_horizon')
        self.neighbor_dist = config.getfloat('invorca', 'neighbor_dist')
        self.orca_time_horizon_obst = config.getfloat('invorca', 
                                                      'orca_time_horizon_obst')

    def set_max_speed(self, max_speed: float):
        """
        Sets the max speed for the agent using this policy
        :param max_speed - the maximum allowed speed for the agent using this policy
        """
        self.max_speed = max_speed

    def set_virtual_goal_params(self, d_virtual_goal: float, y_virtual_goal: float):
        """
        Sets the parameters for the virtual goal line
        :param d_virtual_goal - the x-distance of the virtual goal line from the human
        :param y_virtual_goal - the y-coordinate of the virtual goal line
        """
        self.d_virtual_goal = d_virtual_goal
        self.y_virtual_goal = y_virtual_goal

    def set_desired_velocity(self, velocity: Point):
        """
        Sets the desired velocity for the human for the policy
        :param velocity - the velocity desired for the human at the next time step
        """
        self.desired_velocity = velocity

    def predict(self, observation):
        robot_pos = np.array(observation['robot pos'])
        human_pos = np.array(observation['human pos'])
        human_vel = tuple(observation['human vel'])
        robot_vel = tuple(observation['robot vel'])

        if self.stopping_criterion(observation):
            params = (self.neighbor_dist, self.max_neighbors, 
                      self.orca_time_horizon, self.orca_time_horizon_obst)

            # Create a simulator instance
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius,
                                           self.max_speed,
                                           collisionResponsibility=self.collision_responsibility)

            # Add the robot
            self.sim.addAgent(tuple(robot_pos), *params, self.radius,
                          self.max_speed, robot_vel,
                          collisionResponsibility=self.collision_responsibility)

            # Add the human
            self.sim.addAgent(tuple(human_pos), *params, self.radius,
                          self.max_speed, human_vel,
                          collisionResponsibility=self.collision_responsibility)

            # Set the preferred velocity of the robot to be goal-directed maximum
            # self.sim.setAgentPrefVelocity(0, (self.max_speed-0.1, 0.1))
            self.sim.setAgentPrefVelocity(0, (self.max_speed, 0.))

            # Set the preferred velocity of the human to be their current velocity
            self.sim.setAgentPrefVelocity(1, human_vel)

            # Perform a step
            self.sim.doStep()

            # Get the action for the robot
            action = self.sim.getAgentVelocity(0)

            return action

        center = (robot_pos - human_pos) / self.time_horizon
        # TODO: assuming equal radii for the two agents
        radius = 2 * self.radius / self.time_horizon  
        cutoff_circle = Circle(tuple(center), radius)
        self.vo = VelocityObstacle(cutoff_circle)

        self.invorca = InverseORCA(self.vo, vr_max=self.max_speed,
                                   collision_responsibility=self.collision_responsibility)

        self.robot_vel, self.u = self.invorca.compute_velocity(human_vel, self.desired_velocity)

        return self.robot_vel

    def stopping_criterion(self, observation):
        """
        Function to determine whether to stop the inverse ORCA policy and switch to another policy
        :param observation - the observation at the current time step
        """

        robot_pos = observation['robot pos']
        human_pos = observation['human pos']

        goal_reached = human_pos[1] <= self.y_virtual_goal
        crossed_human = robot_pos[0] >= human_pos[0] - self.d_virtual_goal

        if goal_reached or crossed_human:
            return True

        return False
