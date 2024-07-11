"""
Test the estimation of the collision avoidance responsibility. 
Set relative_position, va, v_pref, vb, alpha_hat to some values
Set some true alpha
Create two PyRVOSimulators, one with alpha_hat and the other with alpha
do a step of the simulation. get va_new_exp with alpha_hat and va_new with alpha.
Supply these values to the estimator and check the result
"""

# TODO: Things seem to break when va itself is feasible, but v_pref is not

import rvo2
import numpy as np
import matplotlib.pyplot as plt
from policy.utils.overlap_detection import Point, VelocityObstacle, Circle
from policy.utils.estimate_alpha import estimate_alpha
from policy.utils.get_velocity import OptimalInfluence

TAU = 6
RADIUS_A = 0.3
RADIUS_B = 0.3
VB_MAX = 1.0

def plot(ax: plt.Axes, va: Point, u: Point, alpha_hat: float,
         alpha: float, v_pref: Point, va_exp: Point, va_new: Point):
    """
    Plots the velocity obstacle and relevant points in velocity space
    """

    circ = plt.Circle((0, 0), VB_MAX, edgecolor='red', fill=False)
    ax.add_patch(circ)

    # Scatter all the velocities
    dot_size = 25
    ax.scatter(va[0], va[1], s=dot_size, c='black', label='va')
    ax.scatter(v_pref[0], v_pref[1], s=dot_size, c='green', label='v_pref')
    ax.scatter(va_exp[0], va_exp[1], s=dot_size, c='purple', label='va_exp')
    ax.scatter(va_new[0], va_new[1], s=dot_size, c='grey', label='va_new')

    # Draw the expected ORCA line
    line_length = 30
    u_perp = np.array([-u[1], u[0]])
    u = np.array(u)

    point1 = tuple(np.array(va) + alpha_hat * u_perp)
    point2 = point1 + line_length * u
    point1 = point1 - line_length * u
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]],
            ls='--', lw=2, c='dodgerblue', label='expected')

    # ratio = alpha / alpha_hat
    point1 = tuple(np.array(va) + alpha * u_perp)
    point2 = point1 + line_length * u
    point1 = point1 - line_length * u
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]],
            ls='--', lw=2, c='gold', label='actual')

    ax.axhline(color='black')
    ax.axvline(color='black')

    ax.legend(bbox_to_anchor=(1.05, 0.5))
    ax.set_xlim([-1.1 * VB_MAX, 1.1 * VB_MAX])
    ax.set_ylim([-1.1 * VB_MAX, 1.1 * VB_MAX])


    return ax


def test(relative_position: Point, va: Point, v_pref: Point,
         vb: Point, alpha_hat: float, alpha: float):
    """
    The main function to run the test with the given data
    """

    params = (10., 10, TAU, 2.0)
    sim1 = rvo2.PyRVOSimulator(0.25, *params, RADIUS_B, VB_MAX)
    sim1.addAgent(relative_position, *params, RADIUS_B, VB_MAX, vb, collisionResponsibility=1.0)
    sim1.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, va, collisionResponsibility=alpha_hat)
    sim1.setAgentPrefVelocity(0, vb)
    sim1.setAgentPrefVelocity(1, v_pref)

    sim1.doStep()

    # A factor of alpha_hat is already multiplied when ORCA computes the orca_point
    # Therefore, this is not the unmodified u. It is actually alpha_hat * u
    orca_line = sim1.getAgentORCALine(1, 0)
    orca_point = (orca_line[0], orca_line[1])
    orca_direction = (orca_line[2], orca_line[3])
    u_mag = np.linalg.norm(np.array(orca_point) - np.array(va)) / alpha_hat
    u = tuple(u_mag * np.array(orca_direction))
    # print(u)

    va_new_exp = sim1.getAgentVelocity(1)

    params = (10., 10, TAU, 2.0)
    sim2 = rvo2.PyRVOSimulator(0.25, *params, RADIUS_B, VB_MAX)
    sim2.addAgent(relative_position, *params, RADIUS_B, VB_MAX, vb, collisionResponsibility=1.0)
    sim2.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, va, collisionResponsibility=alpha)
    sim2.setAgentPrefVelocity(0, vb)
    sim2.setAgentPrefVelocity(1, v_pref)

    sim2.doStep()
    orca_line = sim2.getAgentORCALine(1, 0)
    orca_point = (orca_line[0], orca_line[1])
    orca_direction = (orca_line[2], orca_line[3])
    u_mag = np.linalg.norm(np.array(orca_point) - np.array(va)) / alpha
    u = tuple(u_mag * np.array(orca_direction))
    # print(u)

    va_new = sim2.getAgentVelocity(1)

    fig, ax = plt.subplots(figsize=(9, 6), layout='tight')
    ax.set_aspect('equal')
    ax = plot(ax, va, u, alpha_hat, alpha, v_pref, va_new_exp, va_new)

    # cutoff_center = tuple(relative_position[0] / TAU, relative_position[1] / TAU)
    # cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    # cutoff_circle = Circle(cutoff_center, cutoff_radius)
    # vo = VelocityObstacle(cutoff_circle)
    # invorca = OptimalInfluence(vo, collision_responsibility=alpha_hat)


    print(f"va_new {va_new}")
    print(f"va_new_exp {va_new_exp}")

    alpha_hat_new = estimate_alpha(v_pref, va_new, va_new_exp, alpha_hat, u)

    print("Old estimate of alpha:", alpha_hat)
    print("New estimate of alpha:", alpha_hat_new)
    print("True value of alpha:  ", alpha)

    return ax


def test_case_a_1():
    """
    Estimate of alpha is more than the true alpha. 
    We do not get any new information from the observed new
    velocity of the human
    """
    relative_position = (1., 1.)
    v_pref = (-1.0, 0)
    alpha_hat = 0.9
    alpha = 0.5

    v_a = (0.5, 0)
    v_b = (0, -0.8)

    test(relative_position, v_a, v_pref, v_b, alpha_hat, alpha)


def test_case_a_2():
    """
    Estimate of alpha is more than the true alpha. 
    We get information from the observed new velocity of the human
    that lets us upper bound the value of alpha
    """
    relative_position = (1., 1.)
    v_pref = (-0.8, -0.27)
    alpha_hat = 0.9
    alpha = 0.5

    v_a = (-0.5, 0)
    v_b = (-1.0, -0.4)

    test(relative_position, v_a, v_pref, v_b, alpha_hat, alpha)


def test_case_a_3():
    """
    Estimate of alpha is more than the true alpha. 
    We get information from the observed new velocity of the human
    that lets us pinpoint the true value of alpha
    """

    relative_position = (1., 1.)
    v_pref = (-0.5, 0)
    alpha_hat = 0.9
    alpha = 0.5

    v_a = (0.5, 0)
    v_b = (0, -0.4)

    test(relative_position, v_a, v_pref, v_b, alpha_hat, alpha)


def test_case_b_1():
    """
    Estimate of alpha is less than the true alpha.
    We do not get any new information from the human's 
    observed velocity
    """
    relative_position = (1., 1.)
    v_pref = (-1.0, 0)
    alpha_hat = 0.5
    alpha = 0.9

    v_a = (0.5, 0)
    v_b = (0, -0.8)

    test(relative_position, v_a, v_pref, v_b, alpha_hat, alpha)


def test_case_b_2():
    """
    Estimate of alpha is less than the true alpha. 
    We get new information from the human's observed velocity 
    that lets us put a lower bound on the value of alpha
    """
    relative_position = (1., 1.)
    v_pref = (-0.42, -0.1)
    alpha_hat = 0.5
    alpha = 0.9

    v_a = (0.5, 0.5)
    v_b = (-1.0, -0.4)

    test(relative_position, v_a, v_pref, v_b, alpha_hat, alpha)


def test_case_b_3():
    """
    Estimate of alpha is less than the true value.
    We get new information from the human's observed velocity 
    that lets us pinpoint the true value of alpha
    """
    relative_position = (1., 1.)
    v_pref = (-0.5, -0.2)
    alpha_hat = 0.5
    alpha = 0.9

    v_a = (0.5, 0)
    v_b = (0, -0.4)

    test(relative_position, v_a, v_pref, v_b, alpha_hat, alpha)

# TODO: See why this test case is failing
#       Here, va is feasible for the given vb
#       v_pref is also feasible
#       Not sure why ORCA is not selecting v_pref as the new velocity
#       ORCA seems to have flipped the direction of u and then projected, for some reason
#       Not sure why this is happening
def test_case_v_a_feasible_1():
    """
    In this case, although the current velocity of the human is 
    feasible (not in the VO), the ORCA algorithm tries to find the 
    ORCA set and does something unexpected. It chooses a velocity 
    that is close to the desired velocity and close to the VO.
    In this case, our method of estimating alpha fails
    """
    relative_position = (1., 1.)
    v_pref = (-0.42, -0.1)
    alpha_hat = 0.5
    alpha = 0.9

    v_a = (0.1, 0)
    # va = (-0.42, -0.1)
    v_b = (-0.3, -0.1)

    ax = test(relative_position, v_a, v_pref, v_b, alpha_hat, alpha)

    cutoff_center = (relative_position[0] / TAU, relative_position[1] / TAU)
    cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    vo = VelocityObstacle(cutoff_circle)
    vo.plot(ax)
    relative_velocity = np.array(v_a) - np.array(v_b)
    ax.scatter(relative_velocity[0], relative_velocity[1], color='red', s=25, label='relvel')
    ax.legend(bbox_to_anchor=(1.05, 0.5))


def test_case_v_a_feasible_2():
    """
    In this case, although the current velocity of the human is 
    feasible (not in the VO), the ORCA algorithm tries to find the 
    ORCA set and does something unexpected. It chooses a velocity 
    that is close to the desired velocity and close to the VO.
    In this case, our method of estimating alpha fails
    """
    relative_position = (1., 1.)
    v_pref = (0.18, 0.75)
    alpha_hat = 0.5
    alpha = 0.9

    va = (0.1, 0.8)
    # va = (-0.42, -0.1)
    vb = (-0.1, -0.1)

    ax = test(relative_position, va, v_pref, vb, alpha_hat, alpha)

    cutoff_center = (relative_position[0] / TAU, relative_position[1] / TAU)
    cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    vo = VelocityObstacle(cutoff_circle)
    vo.plot(ax)
    relative_velocity = np.array(va) - np.array(vb)
    ax.scatter(relative_velocity[0], relative_velocity[1], color='red', s=25, label='relvel')
    ax.legend(bbox_to_anchor=(1.05, 0.5))
