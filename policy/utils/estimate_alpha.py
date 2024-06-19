"""
Given the preferred velocity, the estimated new velocity, and the actual new velocity, 
estimate the value of the collision avoidance responsibility (alpha) of the agent. 
This currently assumes that the robot knows the preferred velocity of the human
"""
import numpy as np
from numpy.linalg import norm
from policy.utils.overlap_detection import Point

def estimate_alpha(v_pref: Point, vA_new: Point, vA_new_exp: Point,
                   alpha_hat: float, u: Point | None = None):
    vA_new = tuple(np.round(vA_new, 4))
    vA_new_exp = tuple(np.round(vA_new_exp, 4))
    v_pref = tuple(np.round(v_pref, 4))

    if vA_new == v_pref == vA_new_exp:
    # Case A.1 or B.1: we do not get any new information about alpha
        # print("Case A.1 or Case B.1")
        return alpha_hat

    if vA_new == v_pref != vA_new_exp:
        # Case A.2: our estimate of alpha is higher than the actual
        # Reduce it
        # print("Case A.2")
        reduction = norm(np.array(vA_new_exp) - np.array(v_pref)) / norm(u)
        return alpha_hat - reduction

    if vA_new != v_pref == vA_new_exp:
        # Case B.2: our estimate of alpha is lower than the actual
        # Increase it
        # print("Case B.2")
        increase = norm(np.array(vA_new) - np.array(v_pref)) / norm(u)
        return alpha_hat + increase

    # Case A.3 or B.3: None of them are equal
    # print("Case A.3 or B.3")
    d1 = norm(np.array(vA_new) - np.array(v_pref)) / norm(u)
    d2 = norm(np.array(vA_new_exp) - np.array(v_pref)) / norm(u)


    return alpha_hat + d1 - d2
