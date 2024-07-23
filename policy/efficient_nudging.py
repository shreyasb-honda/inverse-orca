"""
Switches to inverse ORCA only after reaching the efficient nudging region
"""

from typing import Dict
import numpy as np
import rvo2
from policy.invorca import InverseOrca
from policy.utils.overlap_detection import Tangent


class NaiveEfficientNudge(InverseOrca):
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

    def get_robot_v_pref(self, observation, direction):
        """
        Returns the robot's preferred velocity
        """
        return direction * self.max_speed, 0.


    def predict(self, observation, direction: int = 1):

        if not self.inverse_started:
            self.check_conditions(observation)

        # self.inverse started can change in the above statement. So, we cannot use an else here
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
        self.sim.setAgentPrefVelocity(0, self.get_robot_v_pref(observation, direction))

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
        human_vel = observation['human vel']
        human_pos = observation['human pos']
        robot_pos = observation['robot pos']
        boundary_condition = robot_pos[0] <= human_pos[0]
        self.boundary = human_vel - np.array(self.desired_velocity)

        # Distance between the robot position and the human position is less than the nudge radius
        dist_sq = (human_pos - robot_pos).T @ (human_pos - robot_pos)
        circle_condition = dist_sq - self.nudge_radius ** 2 < 0

        self.inverse_started = boundary_condition and circle_condition


class EfficientNudge(NaiveEfficientNudge):
    """
    Chooses a point that satisfies the conditions for starting
    inverse ORCA and sets the agent preferred velocity based on this point
    """

    def choose_point(self, observation, direction):
        """
        Return a point in the efficient nudge region
        """
        human_vel = observation['human vel']
        human_pos = observation['human pos']
        robot_pos = observation['robot pos']
        # Boundary will have been computed before this call is made
        # Travel some along the boundary
        chosen_point = human_pos + self.boundary * self.dist_along
        # Travel some perpendicular to the boundary
        normal = np.array([-self.boundary[1], self.boundary[1]])
        point_1 = chosen_point + self.dist_perp * normal
        point_2 = chosen_point - self.dist_perp * normal
        point = tuple(human_pos)
        boundary_line = Tangent(point, tuple(normal))
        side_1 = boundary_line.side(tuple(point_1))
        side_2 = boundary_line.side(tuple(point_2))
        side_3 = boundary_line.side(tuple(human_pos + human_vel))

        if side_1 == side_3:
            chosen_point = point_1
        elif side_2 == side_3:
            chosen_point = point_2

        chosen_point[0] = robot_pos[0] + direction * self.max_speed

        return chosen_point

    def get_robot_v_pref(self, observation, direction):
        # Steps : 1. Choose a point near in the nudge-efficient region
        #         2. Compute velocity towards it
        chosen_point = self.choose_point(observation, direction)
        vr_pref = chosen_point - observation['robot pos']
        factor = min(1., self.max_speed/ np.linalg.norm(vr_pref))
        vr_pref = tuple(factor * vr_pref)
        return vr_pref

    def check_conditions(self, observation):
        boundary_condition = False
        human_vel = observation['human vel']
        human_pos = observation['human pos']
        robot_pos = observation['robot pos']
        boundary_condition = robot_pos[0] <= human_pos[0]
        self.boundary = human_vel - np.array(self.desired_velocity)
        boundary_length = np.linalg.norm(self.boundary)
        if boundary_length < 1e-4:
            # If the current velocity is the same as desired velocity
            boundary_condition = False
        else:
            self.boundary /= boundary_length
            point = tuple(observation['human pos'])
            boundary_normal = Tangent(point, tuple(self.boundary))
            # tangent.side checks which side the point lies to the normal to the line
            side1 = boundary_normal.side(tuple(robot_pos))
            side2 = boundary_normal.side(tuple(human_pos + human_vel))
            boundary_condition = side1 == side2
            # if boundary_condition:
            #     print(side1)

        # Distance between the robot position and the human position is less than the nudge radius
        dist_sq = (human_pos - robot_pos).T @ (human_pos - robot_pos)
        circle_condition = dist_sq - self.nudge_radius ** 2 < 0
        y_condition = robot_pos[1] >= human_pos[1] + 0.3

        self.inverse_started = boundary_condition and circle_condition and y_condition


class SmoothEfficientNudge(InverseOrca):
    """
    A policy that smoothly transitions between ORCA and 
    inverse ORCA near the efficient nudging region
    """
    def __init__(self, time_step: float = 0.25) -> None:
        super().__init__(time_step)
        self.nudge_radius = None
        self.boundary = None
        self.nudge_circle = None
        self.inverse_started = None
        self.dist_along = None
        self.dist_perp = None
        self.exp_factor = None
        self.smoothing_radius = None
        self.smoothing_radius_sq = None
        self.time_count = 0
        self.first_step_close = False

    def configure(self, config: Dict):
        super().configure(config)
        self.config = config
        self.nudge_radius = config['efficient_nudge']['radius']
        self.dist_along = config['efficient_nudge']['dist_along']
        self.dist_perp = config['efficient_nudge']['dist_perp']
        self.inverse_started = False
        self.exp_factor = config['smooth_nudge']['exp_factor']
        self.smoothing_radius = config['smooth_nudge']['smoothing_radius']
        self.smoothing_radius_sq = self.smoothing_radius ** 2

    def choose_point(self, observation):
        """
        Return a point in the efficient nudge region
        """
        human_vel = observation['human vel']
        human_pos = observation['human pos']
        # robot_pos = observation['robot pos']
        # Boundary will have been computed before this call is made
        # Travel some along the boundary
        chosen_point = human_pos + self.boundary * self.dist_along
        # Travel some perpendicular to the boundary
        normal = np.array([-self.boundary[1], self.boundary[1]])
        point_1 = chosen_point + self.dist_perp * normal
        point_2 = chosen_point - self.dist_perp * normal
        point = tuple(human_pos)
        boundary_line = Tangent(point, tuple(normal))
        side_1 = boundary_line.side(tuple(point_1))
        side_2 = boundary_line.side(tuple(point_2))
        side_3 = boundary_line.side(tuple(human_pos + human_vel))

        if side_1 == side_3:
            chosen_point = point_1
        elif side_2 == side_3:
            chosen_point = point_2

        # chosen_point[0] = robot_pos[0] + direction * self.max_speed

        return chosen_point

    def get_robot_v_pref(self, observation, human_direction):
        """
        Returns the robot's preferred velocity
        :param observation - the current observation
        :param human_direction - the sign of x-velocity of the human
        """
        # Steps : 1. Choose a point near in the nudge-efficient region
        #         2. Compute velocity towards it
        robot_pos = observation['robot pos']
        chosen_point = self.choose_point(observation)
        chosen_point[0] = robot_pos[0] + human_direction * self.max_speed
        vr_pref = chosen_point - robot_pos
        factor = min(1., self.max_speed/ np.linalg.norm(vr_pref))
        vr_pref = tuple(factor * vr_pref)
        return vr_pref


    def predict(self, observation, direction: int = 1):

        # Compute the inverse ORCA velocity
        invorca_vel = super().predict(observation, direction)

        if not self.inverse_started:
            self.check_conditions(observation)

        # self.inverse started can change in the above statement. So, we cannot use an else here
        if self.inverse_started:
            return invorca_vel

        # Compute the ORCA velocity
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
        human_direction = np.sign(human_vel[0])
        self.sim.setAgentPrefVelocity(0, self.get_robot_v_pref(observation, human_direction))

        # Set the preferred velocity of the human to be their current velocity
        self.sim.setAgentPrefVelocity(1, human_vel)

        # Perform a step
        self.sim.doStep()

        # Get the action for the robot
        orca_vel = self.sim.getAgentVelocity(0)

        weight = self.compute_invorca_weight(observation)
        # print(weight)
        if invorca_vel is not None:
            action = (1 - weight) * np.array(orca_vel) + weight * np.array(invorca_vel)
        else:
            action = np.array(orca_vel)

        self.time_count += 1

        return tuple(action)

    def compute_invorca_weight(self, observation):
        """
        Computes the exponentially decreasing weight for the ORCA velocity
        """
        chosen_point = self.choose_point(observation)
        robot_pos = observation['robot pos']
        dist_sq = (chosen_point - robot_pos).T @ (chosen_point - robot_pos)

        # If outside the smoothing circle, then only use ORCA
        if dist_sq > self.smoothing_radius_sq:
            return 0.0

        if self.first_step_close or self.time_count == 0:
            self.first_step_close = True
            return 0.0

        # return np.exp(-1 * self.exp_factor * dist_sq)
        return 1.0 - dist_sq / self.smoothing_radius_sq

    def check_conditions(self, observation):
        """
        Checks whether the robot has entered the efficient nudge 
        region
        """
        stop_inverse = self.stopping_criterion(observation)
        boundary_condition = False
        human_vel = observation['human vel']
        human_pos = observation['human pos']
        robot_pos = observation['robot pos']
        boundary_condition = robot_pos[0] <= human_pos[0]
        self.boundary = human_vel - np.array(self.desired_velocity)
        boundary_length = np.linalg.norm(self.boundary)
        if boundary_length < 1e-4:
            # If the current velocity is the same as desired velocity
            boundary_condition = False
        else:
            self.boundary /= boundary_length
            point = tuple(observation['human pos'])
            boundary_normal = Tangent(point, tuple(self.boundary))
            # tangent.side checks which side the point lies to the normal to the line
            side1 = boundary_normal.side(tuple(robot_pos))
            side2 = boundary_normal.side(tuple(human_pos + human_vel))
            boundary_condition = side1 == side2
            # if boundary_condition:
            #     print(side1)

        # Distance between the robot position and the human position is less than the nudge radius
        dist_sq = (human_pos - robot_pos).T @ (human_pos - robot_pos)
        circle_condition = dist_sq - self.nudge_radius ** 2 < 0
        y_condition = robot_pos[1] >= human_pos[1] + 0.3

        self.inverse_started = boundary_condition and circle_condition and y_condition

        # Revert to inverse ORCA if the stopping criterion is satisfied 
        # (which will fall back to ORCA)
        # This will essentially give a goal-directed ORCA velocity
        self.inverse_started = self.inverse_started or stop_inverse
