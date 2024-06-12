# TODO: currently we are assuming that the robot is agent 0 and the human is agent 1
# TODO: the policy returns the action for agent 1

from configparser import RawConfigParser
import rvo2
from policy.policy import Policy

class Orca(Policy):

    def __init__(self) -> None:
        super().__init__()
        self.name = 'ORCA'
        self.neighbor_dist = None
        self.max_neighbors = None
        self.time_horizon = None
        self.time_horizon_obst = None
        self.radius = None
        self.max_speed = None
        self.sim = None

    def configure(self, config: RawConfigParser):
        self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        self.max_neighbors = config.getint('orca', 'max_neighbors')
        self.time_horizon = config.getfloat('orca', 'time_horizon')
        self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        self.radius = config.getfloat('orca', 'radius')
        # self.max_speed = config.getfloat('orca', 'max_speed')
    
    def set_max_speed(self, max_speed: float):
        self.max_speed = max_speed


    def predict(self, observation):
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)

        # Add the robot
        robot_pos = tuple(observation['robot pos'])
        robot_vel = tuple(observation['robot vel'])
        self.sim.addAgent(robot_pos, *params, self.radius, self.max_speed, robot_vel)
        # TODO: Assuming that the radius is the same for all agents
        # TODO: Assuming that the preferred speed is the max speed

        # Add the human
        human_pos = tuple(observation['human pos'])
        human_vel = tuple(observation['human vel'])
        self.sim.addAgent(human_pos, *params, self.radius, self.max_speed, human_vel)
        # TODO: Assuming that the radius is the same for all agents
        # TODO: Assuming that the preferred speed is the max speed

        # Set the preferred velocity to the current velocity
        self.sim.setAgentPrefVelocity(0, robot_vel)
        self.sim.setAgentPrefVelocity(1, (-self.max_speed, 0))
        # self.sim.setAgentPrefVelocity(1, human_vel)

        self.sim.doStep()
        action = self.sim.getAgentVelocity(1)
        # print("ORCA action", action)

        return action
