"""
The Optimal Reciprocal Collision Avoidance Policy
"""

# TODO: currently we are assuming that the robot is agent 0 and the human is agent 1
# TODO: the policy returns the action for agent 1

from typing import Dict
import rvo2
from policy.policy import Policy

class Orca(Policy):
    """
    Uses the official implementation of ORCA algorithm to generate velocity for
    this agent given the current observation
    """

    def __init__(self, time_step: float = 0.25) -> None:
        super().__init__()
        self.time_step = time_step
        self.neighbor_dist = None
        self.max_neighbors = None
        self.time_horizon = None
        self.time_horizon_obst = None
        self.radius = None
        self.max_speed = None
        self.sim = None
        self.collision_responsibility = None

    def configure(self, config: Dict):
        self.config = config['orca']
        self.neighbor_dist = self.config['neighbor_dist']
        self.max_neighbors = self.config['max_neighbors']
        self.time_horizon = self.config['time_horizon']
        self.time_horizon_obst = self.config['time_horizon_obst']
        self.radius = self.config['radius']

    def set_max_speed(self, max_speed: float):
        """
        Sets the max speed for the agent using this policy
        :param max_speed - the maximum allowed speed for the agent using this policy
        """
        self.max_speed = max_speed

    def set_collision_responsiblity(self, collision_responsibility):
        """
        Sets the collision responsibility that the agent using this policy assumes
        :param collision_responsibility - the fraction of collision avoidance the agent using
                                        this policy does
        """
        self.collision_responsibility = collision_responsibility

    def predict(self, observation):

        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius,
                                       self.max_speed,
                                       collisionResponsibility=self.collision_responsibility)

        # Add the robot
        robot_pos = tuple(observation['robot pos'])
        robot_vel = tuple(observation['robot vel'])
        robot_radius = observation['robot rad']
        self.sim.addAgent(robot_pos, *params, robot_radius,
                          self.max_speed, robot_vel,
                          collisionResponsibility=self.collision_responsibility)
        # TODO: Assuming that the preferred speed is the max speed

        # Add the human
        human_pos = tuple(observation['human pos'])
        human_vel = tuple(observation['human vel'])
        human_radius = observation['human rad']
        self.sim.addAgent(human_pos, *params, human_radius,
                          self.max_speed, human_vel,
                          collisionResponsibility=self.collision_responsibility)
        # TODO: Assuming that the preferred speed is the max speed

        # Set the preferred velocity to the current velocity for the robot
        self.sim.setAgentPrefVelocity(0, robot_vel)

        # Set the preferred velocity to be the goal-directed velocity for the human
        self.sim.setAgentPrefVelocity(1, (-self.max_speed, 0))

        self.sim.doStep()
        action = self.sim.getAgentVelocity(1)

        return action
