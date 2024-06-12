from configparser import RawConfigParser
import numpy as np
from policy.policy import Policy
from policy.utils.GetVelocity import InverseORCA
from policy.utils.OverlapDetection import Point, Circle, VelocityObstacle

class InvOrca(Policy):

    def __init__(self) -> None:
        super().__init__()
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

    def configure(self, config: RawConfigParser):
        self.time_horizon = config.getfloat('invorca', 'time_horizon')
        self.radius = config.getfloat('invorca', 'radius')
        # self.max_speed = config.getfloat('invorca', 'max_speed')
        self.collision_responsibility = config.getfloat('invorca', 'collision_responsibility')
    
    def set_max_speed(self, max_speed: float):
        self.max_speed = max_speed

    def set_virtual_goal_params(self, d_virtual_goal: float, y_virtual_goal: float):
        self.d_virtual_goal = d_virtual_goal
        self.y_virtual_goal = y_virtual_goal

    def set_desired_velocity(self, velocity: Point):
        self.desired_velocity = velocity

    def predict(self, observation):
        robot_pos = np.array(observation['robot pos'])
        human_pos = np.array(observation['human pos'])

        if human_pos[1] <= self.y_virtual_goal or robot_pos[0] >= human_pos[0] - self.d_virtual_goal:
            return (self.max_speed, 0)

        center = (robot_pos - human_pos) / self.time_horizon
        radius = 2 * self.radius / self.time_horizon  # TODO: assuming equal radii for the two agents
        cutoff_circle = Circle(tuple(center), radius)
        self.vo = VelocityObstacle(cutoff_circle)

        self.invorca = InverseORCA(self.vo, vB_max=self.max_speed, 
                                   collision_responsibility=self.collision_responsibility)
        
        human_vel = tuple(observation['human vel'])
        self.robot_vel, self.u = self.invorca.compute_velocity(human_vel, self.desired_velocity)

        return self.robot_vel
