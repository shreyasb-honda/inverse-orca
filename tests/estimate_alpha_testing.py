"""
Test the estimation of the collision avoidance responsibility. 
Set relative_position, vA, v_pref, vB, alpha_hat to some values
Set some true alpha
Create two PyRVOSimulators, one with alpha_hat and the other with alpha
do a step of the simulation. get vA_new_exp with alpha_hat and vA_new with alpha.
Supply these values to the estimator and check the result
"""

# TODO: Things seem to break when vA itself is feasible, but v_pref is not

import rvo2
import numpy as np
import matplotlib.pyplot as plt
from policy.utils.overlap_detection import Point, VelocityObstacle, Circle
from policy.utils.estimate_alpha import estimate_alpha
from policy.utils.get_velocity import InverseORCA

TAU = 6
RADIUS_A = 0.3
RADIUS_B = 0.3
VB_MAX = 1.0

def plot(ax: plt.Axes, vA: Point, u: Point, alpha_hat: float,
         alpha: float, v_pref: Point, vA_exp: Point, vA_new: Point):

    circ = plt.Circle((0, 0), VB_MAX, edgecolor='red', fill=False)
    ax.add_patch(circ)

    # Scatter all the velocities
    dot_size = 25
    ax.scatter(vA[0], vA[1], s=dot_size, c='black', label='vA')
    ax.scatter(v_pref[0], v_pref[1], s=dot_size, c='green', label='v_pref')
    ax.scatter(vA_exp[0], vA_exp[1], s=dot_size, c='purple', label='vA_exp')
    ax.scatter(vA_new[0], vA_new[1], s=dot_size, c='grey', label='vA_new')

    # Draw the expected ORCA line
    line_length = 30
    u_perp = np.array([-u[1], u[0]])
    u = np.array(u)

    point1 = tuple(np.array(vA) + u_perp)
    point2 = point1 + line_length * u
    point1 = point1 - line_length * u
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]],
            ls='--', lw=2, c='dodgerblue', label='expected')

    ratio = alpha / alpha_hat
    point1 = tuple(np.array(vA) + ratio * u_perp)
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


def test(relative_position: Point, vA: Point, v_pref: Point,
         vB: Point, alpha_hat: float, alpha: float):

    params = (10., 10, TAU, 2.0)
    sim1 = rvo2.PyRVOSimulator(0.25, *params, RADIUS_B, VB_MAX)
    sim1.addAgent(relative_position, *params, RADIUS_B, VB_MAX, vB, collisionResponsibility=1.0)
    sim1.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, vA, collisionResponsibility=alpha_hat)
    sim1.setAgentPrefVelocity(0, vB)
    sim1.setAgentPrefVelocity(1, v_pref)

    sim1.doStep()
    orca_line = sim1.getAgentORCALine(1, 0)
    orca_point = (orca_line[0], orca_line[1])
    orca_direction = (orca_line[2], orca_line[3])
    u_mag = np.linalg.norm(np.array(orca_point) - np.array(vA))
    u = tuple(u_mag * np.array(orca_direction))

    vA_new_exp = sim1.getAgentVelocity(1)

    params = (10., 10, TAU, 2.0)
    sim2 = rvo2.PyRVOSimulator(0.25, *params, RADIUS_B, VB_MAX)
    sim2.addAgent(relative_position, *params, RADIUS_B, VB_MAX, vB, collisionResponsibility=1.0)
    sim2.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, vA, collisionResponsibility=alpha)
    sim2.setAgentPrefVelocity(0, vB)
    sim2.setAgentPrefVelocity(1, v_pref)

    sim2.doStep()

    vA_new = sim2.getAgentVelocity(1)

    fig, ax = plt.subplots(figsize=(9, 6), layout='tight')
    ax.set_aspect('equal')
    ax = plot(ax, vA, u, alpha_hat, alpha, v_pref, vA_new_exp, vA_new)

    # cutoff_center = tuple(relative_position[0] / TAU, relative_position[1] / TAU)
    # cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    # cutoff_circle = Circle(cutoff_center, cutoff_radius)
    # vo = VelocityObstacle(cutoff_circle)
    # invorca = InverseORCA(vo, collision_responsibility=alpha_hat)


    print(f"vA_new {vA_new}")
    print(f"vA_new_exp {vA_new_exp}")

    alpha_hat_new = estimate_alpha(v_pref, vA_new, vA_new_exp, alpha_hat, u)

    print("Old estimate of alpha:", alpha_hat)
    print("New estimate of alpha:", alpha_hat_new)
    print("True value of alpha:  ", alpha)

    return ax

def test_case_A_1():
    relative_position = (1., 1.)
    v_pref = (-1.0, 0)
    alpha_hat = 0.9
    alpha = 0.7

    vA = (0.5, 0)
    vB = (0, -0.8)

    test(relative_position, vA, v_pref, vB, alpha_hat, alpha)

def test_case_A_2():
    relative_position = (1., 1.)
    v_pref = (-0.8, -0.27)
    alpha_hat = 0.9
    alpha = 0.7

    vA = (-0.5, 0)
    vB = (-1.0, -0.4)

    test(relative_position, vA, v_pref, vB, alpha_hat, alpha)

def test_case_A_3():
    relative_position = (1., 1.)
    v_pref = (-0.5, 0)
    alpha_hat = 0.9
    alpha = 0.7

    vA = (0.5, 0)
    vB = (0, -0.4)

    test(relative_position, vA, v_pref, vB, alpha_hat, alpha)


def test_case_B_1():
    relative_position = (1., 1.)
    v_pref = (-1.0, 0)
    alpha_hat = 0.7
    alpha = 0.9

    vA = (0.5, 0)
    vB = (0, -0.8)

    test(relative_position, vA, v_pref, vB, alpha_hat, alpha)

def test_case_B_2():
    relative_position = (1., 1.)
    v_pref = (-0.42, -0.1)
    alpha_hat = 0.5
    alpha = 0.9

    vA = (0.5, 0.5)
    vB = (-1.0, -0.4)

    test(relative_position, vA, v_pref, vB, alpha_hat, alpha)


def test_case_B_3():
    relative_position = (1., 1.)
    v_pref = (-0.5, -0.2)
    alpha_hat = 0.7
    alpha = 0.9

    vA = (0.5, 0)
    vB = (0, -0.4)

    test(relative_position, vA, v_pref, vB, alpha_hat, alpha)

# TODO: See why this test case is failing
#       Here, vA is feasible for the given vB
#       v_pref is also feasible
#       Not sure why ORCA is not selecting v_pref as the new velocity
#       ORCA seems to have flipped the direction of u and then projected, for some reason
#       Not sure why this is happening
def test_case_vA_feasible_1():
    relative_position = (1., 1.)
    v_pref = (-0.42, -0.1)
    alpha_hat = 0.5
    alpha = 0.9

    vA = (0.1, 0)
    # vA = (-0.42, -0.1)
    vB = (-0.3, -0.1)

    ax = test(relative_position, vA, v_pref, vB, alpha_hat, alpha)

    cutoff_center = (relative_position[0] / TAU, relative_position[1] / TAU)
    cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    vo = VelocityObstacle(cutoff_circle)
    vo.plot(ax)
    relative_velocity = np.array(vA) - np.array(vB)
    rel_vel_pref = np.array(v_pref) - np.array(vB)
    ax.scatter(relative_velocity[0], relative_velocity[1], color='red', s=25, label='relvel')
    ax.scatter(rel_vel_pref[0], rel_vel_pref[1], color='darkred', s=25, label='relv_pref')
    ax.legend(bbox_to_anchor=(1.05, 0.5))

def test_case_vA_feasible_2():
    relative_position = (1., 1.)
    v_pref = (0.18, 0.75)
    alpha_hat = 0.5
    alpha = 0.9

    vA = (0.1, 0.8)
    # vA = (-0.42, -0.1)
    vB = (-0.1, -0.1)

    ax = test(relative_position, vA, v_pref, vB, alpha_hat, alpha)

    cutoff_center = (relative_position[0] / TAU, relative_position[1] / TAU)
    cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    vo = VelocityObstacle(cutoff_circle)
    vo.plot(ax)
    relative_velocity = np.array(vA) - np.array(vB)
    rel_vel_pref = np.array(v_pref) - np.array(vB)
    ax.scatter(relative_velocity[0], relative_velocity[1], color='red', s=25, label='relvel')
    ax.scatter(rel_vel_pref[0], rel_vel_pref[1], color='darkred', s=25, label='relv_pref')
    ax.legend(bbox_to_anchor=(1.05, 0.5))
