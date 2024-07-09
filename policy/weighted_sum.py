"""
Implements a policy that computes the velocity as 
a weighted sum of goal directed velocity for the robot
and the velocity output of the Inverse ORCA algorithm
"""
from typing import Dict
import numpy as np
from policy.invorca import InverseOrca
from policy.utils.overlap_detection import Point

class WeightedSum(InverseOrca):
    """
    A policy that adds a weighted goal-directed velocity component
    to the velocity output by the inverse ORCA algorithm
    """

    def __init__(self, time_step: float = 0.25) -> None:
        super().__init__(time_step)
        self.goal_weight = None

    def configure(self, config: Dict):
        super().configure(config)
        self.goal_weight = config['weighted_sum']['goal_weight']

    def predict(self, observation, direction: int):
        v_invorca =  super().predict(observation)
        v_goal_directed = direction * np.array([self.max_speed, 0.])
        v_sum = v_goal_directed

        if v_invorca is not None:
            v_sum = np.array(v_invorca) + self.goal_weight * v_goal_directed
        if np.linalg.norm(v_sum) > self.max_speed:
            v_sum = v_sum / np.linalg.norm(v_sum) * self.max_speed

        return tuple(v_sum)
