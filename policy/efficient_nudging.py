"""
Switches to inverse ORCA only after reaching the efficient nudging region
"""

from typing import Dict
import numpy as np
import rvo2
from policy.invorca import InverseOrca
# from policy.utils.overlap_detection import Tangent


class EfficientNudge(InverseOrca):
    """
    A policy that uses ORCA to get close to the human and
    then switches to inverse ORCA to lead the human towards
    the virtual goal
    """

    def __init__(self, time_step: float = 0.25) -> None:
        super().__init__(time_step)
        self.nudge_radius = None
        self.boundary = None
        self.nudge_circle = None
        self.inverse_started = None
        self.dist_along = None
        self.dist_perp = None

    def configure(self, config: Dict):
        super().configure(config)
        self.config = config
        self.nudge_radius = config['efficient_nudge']['radius']
        self.dist_along = config['efficient_nudge']['dist_along']
        self.dist_perp = config['efficient_nudge']['dist_perp']
        self.inverse_started = False

    def predict(self, observation, direction: int = 1):

        if not self.inverse_started:
            self.check_conditions(observation)

        if self.inverse_started:
            v_invorca = super().predict(observation, direction)
            return v_invorca

        robot_pos = observation['robot pos']
        human_pos = observation['human pos']
        human_vel = tuple(observation['human vel'])
        robot_vel = tuple(observation['robot vel'])

        human_radius = observation['human rad']
        robot_radius = observation['robot rad']

        params = (self.neighbor_dist, self.max_neighbors,
                    self.orca_time_horizon, self.orca_time_horizon_obst)

        # Create a simulator instance
        self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius,
                                        self.max_speed,
                                        collisionResponsibility=self.collision_responsibility)

        # Add the robot
        self.sim.addAgent(tuple(robot_pos), *params, robot_radius,
                        self.max_speed, robot_vel,
                        collisionResponsibility=self.collision_responsibility)

        # Add the human
        self.sim.addAgent(tuple(human_pos), *params, human_radius,
                        self.max_speed, human_vel,
                        collisionResponsibility=self.collision_responsibility)

        # Set the preferred velocity of the robot to be goal-directed maximum
        self.sim.setAgentPrefVelocity(0, (direction * self.max_speed, 0.))

        # Set the preferred velocity of the human to be their current velocity
        self.sim.setAgentPrefVelocity(1, human_vel)

        # Perform a step
        self.sim.doStep()

        # Get the action for the robot
        action = self.sim.getAgentVelocity(0)
        self.robot_vel = action

        return action

    def check_conditions(self, observation):
        """
        Checks the conditions for starting inverse ORCA
        """
        # boundary_condition = False
        human_vel = observation['human vel']
        human_pos = observation['human pos']
        robot_pos = observation['robot pos']
        boundary_condition = robot_pos[0] <= human_pos[0]
        self.boundary = human_vel - np.array(self.desired_velocity)
        # boundary_length = np.linalg.norm(self.boundary)
        # if boundary_length < 1e-4:
        #     # If the current velocity is the same as desired velocity
        #     boundary_condition = True
        # else:
        #     self.boundary /= boundary_length
        #     normal = (-self.boundary[1], self.boundary[0])
        #     point = tuple(observation['human pos'])
        #     boundary_line = Tangent(point, normal)
        #     side1 = boundary_line.side(tuple(observation['robot pos']))
        #     side2 = boundary_line.side(observation['human_pos'] + human_vel)
        #     boundary_condition = side1 == side2

        # Distance between the robot position and the human position is less than the nudge radius
        circle_condition = (human_pos - robot_pos).T @ (human_pos - robot_pos) - self.nudge_radius ** 2 < 0

        self.inverse_started = boundary_condition and circle_condition
