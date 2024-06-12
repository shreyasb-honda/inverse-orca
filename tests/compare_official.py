"""
Compares our expected results with the official ORCA implementation
"""
import rvo2
import matplotlib.pyplot as plt
import numpy as np
from policy.utils.GetVelocity import InverseORCA
from policy.utils.OverlapDetection import Circle, VelocityObstacle, Point


TAU = 5.0
RADIUS_A = 0.3
RADIUS_B = 0.3
VB_MAX = 1.0
EPSILON = 1e-5
COLLISION_RESPONSIBILITY = 1.0


def test(relative_position: Point, vA: Point, vA_d: Point, 
         collision_responsibility: float = 0.5):
    """
    Generates and plots the optimal velocity for the robot to influence the human's velocity
        1. If there is a solution, compute and plot it
        2. If there is no solution, get the velocity that takes the relative velocity closer to 
            the velocity obstacle 
    """

    cutoff_center = tuple(np.array(relative_position) / TAU)
    cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    vo = VelocityObstacle(cutoff_circle)
    invorca = InverseORCA(vo, vB_max=VB_MAX, epsilon=EPSILON, 
                          collision_responsibility=collision_responsibility)
    invorca.compute_velocity(vA, vA_d)

    fig, ax = plt.subplots(layout='tight', figsize=(9, 9))
    ax.set_aspect('equal')
    ax.axhline(color='black')
    ax.axvline(color='black')

    # Plot the velocity obstacle
    ax = vo.plot(ax)

    # Plot the relative velocity circle
    velocity_circle = Circle(vA, VB_MAX)
    ax = velocity_circle.plot(ax)

    # Plot the desired velocity for A
    ax.scatter(vA_d[0], vA_d[1], color='tab:purple', label='vA_d')

    # If solution exists, plot vB, vAB, vA_new
    # If solution does not exist, plot vB, vAB (vA_new = vA)
    ax.scatter(invorca.vB[0], invorca.vB[1], color='lime', label='vB')
    ax.scatter(vA[0] - invorca.vB[0], vA[1] - invorca.vB[1], color='tab:orange', label='vAB')

    if invorca.solution_exists:
        ax.scatter(invorca.vA_new[0], invorca.vA_new[1], color='teal', label='vA_new')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    return ax, invorca


def overlap_with_solution_1():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take vA as close as vA_d as possible
        For this testcase, the projection would be on the cutoff circle"""

    # Constants
    # relative_position = (-1., -1.)
    # vA = (0.1, 0.2)
    # vA_d = (0.3, 0.4)  # Possible to get there completely
    # vA_d = (0.8, 0.9)  # Not possible to get there in one step

    vA = (-1.0, 0.)
    vA_d = (-0.983, -0.156)
    relative_position = (-1.866 * 5, 0.161 * 5)
    ax, invorca = test(relative_position, vA, vA_d, COLLISION_RESPONSIBILITY)


    params = (10., 10, TAU, 2.0)
    sim = rvo2.PyRVOSimulator(1.0, *params, RADIUS_B, VB_MAX)
    sim.addAgent(relative_position, *params, RADIUS_B, VB_MAX, invorca.vB)
    sim.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, invorca.vA)

    vA_0 = sim.getAgentVelocity(1)

    sim.setAgentPrefVelocity(0, invorca.vB)
    sim.setAgentPrefVelocity(1, invorca.vA)
    sim.doStep()
    vA_new_official = sim.getAgentVelocity(1)
    vA_new_ours = invorca.vA_new

    print(f'vA (official) {vA_0}')
    print(f"vA_new (official) {vA_new_official}")
    print(f"vA_new (ours) {vA_new_ours}")

    print(f"vB (official) {sim.getAgentVelocity(0)}")
    print(f"vB (ours) {invorca.vB}")


    return ax


def overlap_with_solution_2():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take vA as close as vA_d as possible
        For this testcase, the projection would be on the left leg"""

    # Constants
    relative_position = (-1., 1.)
    vA = (-2.3, 0.5)    
    cutoff_center = tuple(np.array(relative_position) / TAU)
    cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    vo = VelocityObstacle(cutoff_circle)
    left_normal = np.array(vo.left_tangent.normal)
    vA_d = (-2.5, 0.3)  # Not along the normal
    # vA_d = np.array(vA) - 0.3 * left_normal  # Along the normal
    ax, invorca = test(relative_position, vA, vA_d, COLLISION_RESPONSIBILITY)


    params = (10., 10, TAU, 2.0)
    sim = rvo2.PyRVOSimulator(1.0, *params, RADIUS_B, VB_MAX)
    sim.addAgent(relative_position, *params, RADIUS_B, VB_MAX, invorca.vB)
    sim.addAgent((0., 0.), *params, RADIUS_A, 5., invorca.vA)

    vA_0 = sim.getAgentVelocity(1)

    sim.setAgentPrefVelocity(0, invorca.vB)
    sim.setAgentPrefVelocity(1, invorca.vA)
    sim.doStep()
    vA_new_official = sim.getAgentVelocity(1)
    vA_new_ours = invorca.vA_new

    print(f'vA (official) {vA_0}')
    print(f"vA_new (official) {vA_new_official}")
    print(f"vA_new (ours) {vA_new_ours}")

    print(f"vB (official) {sim.getAgentVelocity(0)}")
    print(f"vB (ours) {invorca.vB}")

    return ax


def overlap_with_solution_3():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take vA as close as vA_d as possible
        For this testcase, the projection would be on the right leg"""

    # Constants
    relative_position = (1., 1.)
    vA = (2.3, 0.5)    
    cutoff_center = tuple(np.array(relative_position) / TAU)
    cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    vo = VelocityObstacle(cutoff_circle)
    right_normal = np.array(vo.right_tangent.normal)
    vA_d = (2.5, 0.3)  # Not along the normal
    # vA_d = np.array(vA) + 0.3 * right_normal  # Along the normal

    ax, invorca = test(relative_position, vA, vA_d, COLLISION_RESPONSIBILITY)

    params = (10., 10, TAU, 2.0)
    sim = rvo2.PyRVOSimulator(1.0, *params, RADIUS_B, VB_MAX)
    sim.addAgent(relative_position, *params, RADIUS_B, VB_MAX, invorca.vB)
    sim.addAgent((0., 0.), *params, RADIUS_A, 5., invorca.vA)

    vA_0 = sim.getAgentVelocity(1)

    sim.setAgentPrefVelocity(0, invorca.vB)
    sim.setAgentPrefVelocity(1, invorca.vA)
    sim.doStep()
    vA_new_official = sim.getAgentVelocity(1)
    vA_new_ours = invorca.vA_new

    print(f'vA (official) {vA_0}')
    print(f"vA_new (official) {vA_new_official}")
    print(f"vA_new (ours) {vA_new_ours}")

    print(f"vB (official) {sim.getAgentVelocity(0)}")
    print(f"vB (ours) {invorca.vB}")

    return ax

