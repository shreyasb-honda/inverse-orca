"""
Test the estimation of the collision avoidance responsibility. 
Set relative_position, vA, v_pref, vB, alpha_hat to some values
Set some true alpha
Create two PyRVOSimulators, one with alpha_hat and the other with alpha
do a step of the simulation. get vA_new_exp with alpha_hat and vA_new with alpha.
Supply these values to the estimator and check the result
"""
import rvo2
from policy.utils.overlap_detection import Point
from policy.utils.estimate_alpha import estimate_alpha

TAU = 6
RADIUS_A = 0.3
RADIUS_B = 0.3
VB_MAX = 1.0

def test(relative_position: Point, vA: Point, v_pref: Point,
         vB: Point, alpha_hat: float, alpha: float):

    params = (10., 10, TAU, 2.0)
    sim1 = rvo2.PyRVOSimulator(0.25, *params, RADIUS_B, VB_MAX)
    sim1.addAgent(relative_position, *params, RADIUS_B, VB_MAX, vB, collisionResponsibility=1.0)
    sim1.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, vA, collisionResponsibility=alpha_hat)
    sim1.setAgentPrefVelocity(0, vB)
    sim1.setAgentPrefVelocity(1, v_pref)

    sim1.doStep()

    vA_new_exp = sim1.getAgentVelocity(1)

    params = (10., 10, TAU, 2.0)
    sim2 = rvo2.PyRVOSimulator(0.25, *params, RADIUS_B, VB_MAX)
    sim2.addAgent(relative_position, *params, RADIUS_B, VB_MAX, vB, collisionResponsibility=1.0)
    sim2.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, vA, collisionResponsibility=alpha)
    sim2.setAgentPrefVelocity(0, vB)
    sim2.setAgentPrefVelocity(1, v_pref)

    sim2.doStep()

    vA_new = sim2.getAgentVelocity(1)

    print(f"vA_new {vA_new}")
    print(f"vA_new_exp {vA_new_exp}")

    alpha_hat_new = estimate_alpha(v_pref, vA_new, vA_new_exp, alpha_hat)

    print("Old estimate of alpha:", alpha_hat)
    print("New estimate of alpha:", alpha_hat_new)
    print("True value of alpha:  ", alpha)

def test_case_A_1():
    relative_position = (1., 1.)
    v_pref = (-1.0, 0)
    alpha_hat = 0.9
    alpha = 0.8

    vA = (0.5, 0)
    vB = (0, -0.4)

    test(relative_position, vA, v_pref, vB, alpha_hat, alpha)


def test_case_A_2():
    pass

def test_case_A_3():
    pass

def test_case_B_1():
    pass

def test_case_B_2():
    pass

def test_case_B_3():
    pass
