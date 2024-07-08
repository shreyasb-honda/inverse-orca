"""
Defines the agent classes
"""
from typing import Dict
import numpy as np
from numpy.linalg import norm
from policy.orca import Orca
from policy.invorca import InverseOrca


class Agent:
    """
    A base class for all agents
    """

    def __init__(self, radius: float = 0.3, max_speed: float = 1.0, preferred_speed: float = 1.0,
                 time_step: float = 0.25) -> None:
        self.radius = radius
        self.px = None   # The x-coordinate of the agent
        self.py = None   # The y-coordniate of the agent
        self.vx = None   # The x-velocity of the agent
        self.vy = None   # the y-velocity of the agent
        self.gx = None   # the x-coordinate of the goal of the agent
        self.gy = None   # the y-coordinate of the goal of the agent

        self.max_speed = max_speed
        self.preferred_speed = preferred_speed
        self.preferred_velocity_x = None
        self.preferred_velocity_y = None

        self.time_step = time_step

        self.policy = None

    def get_position(self):
        """
        Returns the current position of this agent as a tuple
        """
        return self.px, self.py

    def get_velocity(self):
        """
        Returns the current velocity of the agent as a tuple
        """
        return self.vx, self.vy

    def get_radius(self):
        """
        Returns the radius of the agent
        """
        return self.radius

    def set_position(self, px: float, py: float):
        """
        Sets the position of the agent at (px, py)
        """
        self.px = px
        self.py = py

    def set_preferred_velocity(self):
        """
        Sets the preferred velocity of the agent. It's policy will try to achieve this
        preferred velocity
        """
        error_msg = "Please set the goal position before calling set_preferred_velocity"
        assert self.gx is not None, error_msg
        assert self.gy is not None, error_msg
        self.preferred_velocity_x = min(abs(self.gx - self.px), self.preferred_speed)
        self.preferred_velocity_x = min(self.preferred_velocity_x, self.max_speed)
        self.preferred_velocity_x *= np.sign(self.gx - self.px)
        self.preferred_velocity_y = 0
        self.set_velocity(self.preferred_velocity_x, self.preferred_velocity_y)

    def set_velocity(self, vx: float, vy: float):
        """
        Set the current velocity of this agent as a tuple (vx, vy)
        """
        error_msg = "Setting a speed higher than the max allowed speed"
        assert norm(np.array([vx, vy])) <= self.max_speed, error_msg
        self.vx = vx
        self.vy = vy

    def set_goal(self, gx: float, gy: float):
        """
        Set the location of the goal of this agent
        """
        self.gx = gx
        self.gy = gy

    def step(self, action, delta_t):
        """
        Child classes should implement. 
        Takes one step in the simulation given the currently chosen action and the time step
        """
        raise NotImplementedError


class Human(Agent):
    """
    A human agent
    """

    def __init__(self, radius: float = 0.3, max_speed: float = 1, preferred_speed: float = 1,
                 time_step: float = 0.25) -> None:
        super().__init__(radius, max_speed, preferred_speed, time_step)
        self.collision_responsibility = None
        self.config = None

    def configure(self, config: Dict):
        """
        Configures the human agent according to the environment
        config file
        :param config - the environment configuration file (inverse_orca/sim/config/env.config)
        """
        self.config = config
        self.time_step = self.config['env']['time_step']
        self.radius = self.config['human']['radius']
        self.max_speed = self.config['human']['max_speed']
        self.collision_responsibility = self.config['human']['collision_responsibility']

    def step(self, action, delta_t):
        self.px += action['human vel'][0] * delta_t
        self.py += action['human vel'][1] * delta_t
        self.vx = action['human vel'][0]
        self.vy = action['human vel'][1]

    def set_policy(self, policy: Orca):
        """
        Configures the policy for the human
        """
        self.policy = policy
        try:
            self.policy.set_max_speed(self.max_speed)
        except AttributeError:
            pass

    def choose_action(self, observation):
        """
        Chooses an action given the current observation
        """
        action = (0., 0.)
        if not self.reached_goal():
            action = self.policy.predict(observation)

        return action

    def reached_goal(self):
        """
        Returns true if the human has crossed its finish line
        """
        return self.px - self.radius < self.gx


class Robot(Agent):
    """
    A robotic agent
    """

    def __init__(self, radius: float = 0.3, max_speed: float = 1, preferred_speed: float = 1,
                 time_step:float = 0.25) -> None:
        super().__init__(radius, max_speed, preferred_speed, time_step)
        self.vh_desired = None
        self.aborted = None
        self.d_virtual_goal = None
        self.y_virtual_goal = None
        self.config = None

    def configure(self, config: Dict):
        """
        Configures this robot
        :param config - the environment configuration file (inverse_orca/sim/config/env.config)
        """
        self.config = config
        self.time_step = self.config['env']['time_step']
        self.radius = self.config['robot']['radius']
        self.max_speed = self.config['robot']['max_speed']
        self.d_virtual_goal = self.config['env']['d_virtual_goal']
        self.y_virtual_goal = self.config['env']['y_virtual_goal']

    def step(self, action, delta_t):
        self.px += action['robot vel'][0] * delta_t
        self.py += action['robot vel'][1] * delta_t
        self.vx = action['robot vel'][0]
        self.vy = action['robot vel'][1]

    def set_policy(self, policy: InverseOrca):
        """
        Sets the policy for the robot
        """
        self.policy = policy
        self.policy.set_max_speed(self.max_speed)

    def set_virtual_goal_params(self, d_virtual_goal: float, y_virtual_goal: float):
        """
        Sets the (moving) virtual goal line segment in front of the human
        """
        self.d_virtual_goal = d_virtual_goal
        self.y_virtual_goal = y_virtual_goal
        self.policy.set_virtual_goal_params(d_virtual_goal, y_virtual_goal)

    def get_virtual_goal_params(self):
        """
        Returns the parameters for the virtual goal line 
        of the robot for the human
        """
        return self.d_virtual_goal, self.y_virtual_goal

    def set_vh_desired(self, obs):
        """
        Computes the straight-line velocity for the human to get to the virtual goal line,
        assuming that the human moves one time step in the direction of the current velocity
        Sets this velocity as the desired velocity for the inverse orca algorithm
        """
        human_pos = obs['human pos']
        human_speed = norm(obs['human vel'])
        human_vel_y = obs['human vel'][1]

        x = -self.d_virtual_goal
        y = self.y_virtual_goal - human_pos[1]
        vh_direction = np.array([x, y]) + self.time_step * np.array([0., obs['human vel'][1]])
        vh_desired = vh_direction / norm(vh_direction) * human_speed
        vh_desired[1] = min(human_vel_y, vh_desired[1])
        # vh_desired[1] = -1.0
        self.vh_desired = vh_desired
        self.policy.set_desired_velocity(tuple(self.vh_desired))

    def human_reached_goal(self, obs):
        """
        Returns true if the human is on or below the virtual goal y-coordinate
        """
        human_y = obs['human pos'][1]

        if human_y <= self.y_virtual_goal:
            return True

        return False

    def choose_action(self, observation):
        """
        Chooses an action given the current observation
        """
        action = (0., 0.)
        if not self.reached_goal():
            action = self.policy.predict(observation)

        return action

    def reached_goal(self):
        """
        Returns true if the robot has reached its own goal
        """
        return self.px + self.radius <= self.gx
