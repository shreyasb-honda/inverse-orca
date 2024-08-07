"""
File defines the human comfort based and robot performance based metrics
for the robot's influencing behavior
"""

import numpy as np
from policy.utils.overlap_detection import Point

class PerformanceMetric:
    """
    Base class for all performance metrics
    """

    def __init__(self, name) -> None:
        self.name = name

    def add(self, observation):
        """
        Computes/Updates the metric based on the current observation
        """
        raise NotImplementedError

    def get_metric(self):
        """
        Returns the value of the metric(s) based on the current data
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the metric to its initial state
        """
        raise NotImplementedError


class CumulativeAcceleration(PerformanceMetric):
    """
    Class to keep track of the cumulative acceleration 
    experienced by an agent
    """

    def __init__(self, time_step: float = 0.25,
                 agent: str = 'human') -> None:
        super().__init__(f'Cumulative Acceleration {agent}')
        self.agent = agent
        self.time_step = time_step
        self.cumulative_acc = 0
        self.velocities = None
        self.accelerations = None
        self.done = False

    def agent_done(self, done):
        """
        Sets whether the agent has reached their goal
        """
        self.done = done

    def add(self, observation):
        """
        Uses the observation to compute acceleration and
        adds it to the list and cumulative acceleration values
        :param observation - the observation to the agent
        :param agent - the name of the agent [human or robot]
        """
        vel = observation[self.agent + ' vel']
        if self.velocities is None:
            self.velocities = [vel]
            self.accelerations = []
        elif not self.done:
            # Only accumulate acceleration if the agent has not reached its goal
            acc = (vel - self.velocities[-1]) / self.time_step
            self.accelerations.append(acc)
            self.cumulative_acc += np.linalg.norm(acc)

    def get_metric(self):
        """
        Returns the value of the cumulative acceleration experienced
        by this agent
        """
        return self.cumulative_acc

    def reset(self):
        return CumulativeAcceleration(self.time_step, self.agent)


class ClosestDistance(PerformanceMetric):
    """
    Class keeps track of the minimum distance between the human and the robot
    """

    def __init__(self) -> None:
        super().__init__('Closest Distance')
        self.min_dist = 1e6
        self.distances = []

    def add(self, observation):
        """
        Uses the observation to compute the metric
        :param observation - the observation of the agent
        """
        human_pos = observation['human pos']
        robot_pos = observation['robot pos']
        dist = np.linalg.norm(human_pos - robot_pos)
        self.distances.append(dist)
        if dist < self.min_dist:
            self.min_dist = dist

    def get_metric(self):
        """
        Returns the value of the metric
        """
        return self.min_dist

    def reset(self):
        return ClosestDistance()


class AverageAcceleration(CumulativeAcceleration):
    """
    Metric to measure the average acceleration over a trajectory
    """
    def __init__(self, time_step: float = 0.25, agent: str = 'human') -> None:
        super().__init__(time_step, agent)
        self.name = f'Average acceleration {agent}'

    def get_metric(self):
        return super().get_metric() / len(self.accelerations)

    def reset(self):
        return AverageAcceleration(self.time_step, self.agent)


class ClosenessToGoal(PerformanceMetric):
    """
    Measures how close the human got to the virtual goal
    line
    """

    def __init__(self, y_virtual_goal) -> None:
        super().__init__('Closeness to goal')
        self.min_dist = 1e5
        self.x_coordinate_at_goal = None
        self.y_goal = y_virtual_goal
        self.reached = False

    def add(self, observation):
        """
        Uses the observation to compute the metric
        """
        human_pos = observation['human pos']
        dist = human_pos[1] - self.y_goal
        if not self.reached:
            self.min_dist = min(self.min_dist, dist)
            if self.min_dist <= 0:
                self.reached = True
                self.min_dist = 0
                self.x_coordinate_at_goal = human_pos[0]

    def get_metric(self):
        """
        Returns the value of the metric - 
            the minimum distance to goal (0 if goal reached)
            the x-coordinate when the goal was first reached (None if goal not reached)
            whether the goal was reached or not
        """
        return self.min_dist, self.x_coordinate_at_goal, self.reached

    def reset(self):
        return ClosenessToGoal(self.y_goal)


class TimeToReachGoal(PerformanceMetric):
    """
    Measures the time it takes the agent to reach its goal
    """
    def __init__(self, time_step: float, robot_goal_x: float, human_goal_x: float) -> None:
        super().__init__('Time to reach goal')
        self.time_step = time_step
        self.human_time = 0
        self.robot_time = 0
        self.human_reached = False
        self.robot_reached = False
        self.robot_goal_x = robot_goal_x
        self.human_goal_x = human_goal_x

    def add(self, observation):
        """
        Updates the time if the agent has not reached their goal
        :param observation - the observation output by the environment
        """
        human_pos = observation['human pos']
        human_radius = observation['human rad']
        self.human_reached = human_pos[0] - human_radius <= self.human_goal_x

        if not self.human_reached:
            self.human_time += self.time_step

        robot_pos = observation['robot pos']
        robot_radius = observation['robot rad']
        self.robot_reached = robot_pos[0] + robot_radius >= self.robot_goal_x
        if not self.robot_reached:
            self.robot_time += self.time_step

    def get_metric(self):
        """
        Returns the time taken by the agents to reach their goals
        """
        return self.human_time, self.robot_time

    def reset(self):
        return TimeToReachGoal(self.time_step, self.robot_goal_x,
                               self.human_goal_x)


class PathEfficiency(PerformanceMetric):
    """
    Computes the ratio between the actual path length to 
    the optimal path length (straight line distance between the 
    initial position and goal)
    """
    def __init__(self, agent: str, hallway_length: float) -> None:
        super().__init__(f'Path efficiency {agent}')
        self.dist_covered = 0
        self.last_pos = None
        self.agent = agent
        self.opt_dist = None
        self.hallway_length = hallway_length

    def add(self, observation):
        agent_pos = observation[f'{self.agent} pos']
        if self.last_pos is None:
            key = f'{self.agent} rad'
            self.last_pos = agent_pos
            if self.agent == 'human':
                self.opt_dist = self.last_pos[0] - observation[key]
            elif self.agent == 'robot':
                self.opt_dist = self.hallway_length - self.last_pos[0] - observation[key]

        self.dist_covered += np.linalg.norm(agent_pos - self.last_pos)
        self.last_pos = agent_pos

    def get_metric(self):
        return self.dist_covered / self.opt_dist

    def reset(self):
        return PathEfficiency(self.agent, self.hallway_length)


class CumulativeJerk(CumulativeAcceleration):
    """
    Tracks the cumulative jerk experienced by the agent during their 
    trajectory
    """

    def __init__(self, time_step: float = 0.25, agent: str = 'human') -> None:
        super().__init__(time_step, agent)
        self.name = 'Cumulative jerk'
        self.cumulative_jerk = 0
        self.jerks = []

    def add(self, observation):
        super().add(observation)
        if len(self.accelerations) >= 2:
            jerk = (self.accelerations[-1] - self.accelerations[-2]) / self.time_step
            self.jerks.append(jerk)
            self.cumulative_jerk += np.linalg.norm(jerk)

    def get_metric(self):
        return self.cumulative_jerk

    def reset(self):
        return CumulativeJerk(self.time_step, self.agent)


class AverageJerk(CumulativeJerk):
    """
    Computes the average jerk experienced by an agent during their
    motion
    """
    def __init__(self, time_step: float = 0.25, agent: str = 'human') -> None:
        super().__init__(time_step, agent)
        self.name = f'Average jerk {agent}'

    def get_metric(self):
        return super().get_metric() / len(self.jerks)

    def reset(self):
        return AverageJerk(self.time_step, self.agent)


class PathIrregularity(PerformanceMetric):
    """
    Averages the absolute angle between the agent's trajectory
    and their goal-directed vector
    """

    def __init__(self, goal: Point, agent: str) -> None:
        super().__init__(f'Path irregularity {agent}')
        self.goal = goal
        self.agent = agent
        self.done = False
        self.metric = 0
        self.angles = []

    def add(self, observation):

        # Only add the observation if the agent is not done
        if not self.done:
            pos = observation[f'{self.agent} pos']
            goal_heading = (self.goal[0] - pos[0], 0.)
            goal_heading /= np.linalg.norm(goal_heading)
            heading = observation[f'{self.agent} vel']

            # Only add if the speed is greater than zero
            if np.linalg.norm(heading) == 0:
                return

            heading /= np.linalg.norm(heading)
            dot = np.dot(heading, goal_heading)
            angle = np.arccos(dot)
            # print(self.agent, goal_heading, heading, dot, angle)
            # input()
            self.angles.append(angle)
            self.metric += angle

    def agent_done(self, done):
        """
        Sets whether the agent has reached their goal
        """
        self.done = done

    def get_metric(self):
        return self.metric / len(self.angles)

    def reset(self):
        return PathIrregularity(self.goal, self.agent)
