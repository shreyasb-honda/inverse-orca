"""
Tests to see if we get the desired velocity for the robot to influence the human's velocity
"""
import os
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from policy.utils.overlap_detection import Circle, VelocityObstacle, Point
from policy.utils.get_velocity import OptimalInfluence
sns.set_theme(context='talk', style='dark')


TAU = 2.0
RADIUS_A = 0.3
RADIUS_B = 0.3
VB_MAX = 1.0
EPSILON = 1e-5
COLLISION_RESPONSIBILITY = 0.5


HOME = expanduser('~')
OUT_DIRECTORY = os.path.join(HOME, 'OneDrive', 'Documents', 'Notes', 'Plots', 'influence-testing')

def test(relative_position: Point, v_a: Point, v_a_d: Point, 
         alpha: float = 0.5):
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
    invorca = OptimalInfluence(vo, vr_max=VB_MAX, epsilon=EPSILON,
                               collision_responsibility=alpha)
    invorca.compute_velocity(v_a, v_a_d)

    fig, ax = plt.subplots(layout='tight', figsize=(9,9))
    ax.set_aspect('equal')
    ax.axhline(color='black')
    ax.axvline(color='black')

    # Plot the velocity obstacle
    ax = vo.plot(ax)

    # Plot the relative velocity circle
    velocity_circle = Circle(v_a, VB_MAX)
    ax = velocity_circle.plot(ax)

    # Plot the desired velocity for A
    ax.scatter(v_a_d[0], v_a_d[1], color='tab:purple', label='vA_d')

    # If solution exists, plot vB, vAB, vA_new
    # If solution does not exist, plot vB, vAB (vA_new = vA)
    ax.scatter(invorca.vr[0], invorca.vr[1], color='lime', label='vB')
    ax.scatter(v_a[0] - invorca.vr[0], v_a[1] - invorca.vr[1], color='tab:orange', label='vAB')

    if invorca.solution_exists:
        ax.scatter(invorca.vh_new[0], invorca.vh_new[1], color='teal', label='vA_new')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    return ax


def no_overlap_1():
    """Case where there is no overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to be the one that takes B closer to the
        velocity obstacle
        For this testcase, the direction would be closer to the cutoff circle"""

    # Constants
    relative_position = (1., 1.)
    # No overlap - in third quadrant. Therefore, the u selected should be directed cutoff circle
    v_a = (-1.2, -0.0)
    v_a_d = (-0.6, -0.7)
    ax = test(relative_position, v_a, v_a_d)
    plt.savefig(os.path.join(OUT_DIRECTORY, 'no_overlap_1.png'))

    return ax


def no_overlap_2():
    """Case where there is no overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to be the one that takes B closer to the
        velocity obstacle
        For this testcase, the direction would be closer to the left leg"""

    # Constants
    relative_position = (-1., 1.)
    # No overlap - in third quadrant.
    # Therefore, the u selected should be directed towards the origin
    v_a = (-3.5, 0.)
    v_a_d = (0., 0.)
    ax = test(relative_position, v_a, v_a_d)
    plt.savefig(os.path.join(OUT_DIRECTORY, 'no_overlap_2.png'))

    return ax


def no_overlap_3():
    """Case where there is no overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to be the one that takes B closer to the
        velocity obstacle
        For this testcase, the direction would be closer to the right leg"""

    # Constants
    relative_position = (1., -1.)
    # No overlap
    v_a = (-0.5, -2.5)
    v_a_d = (0., 0.)
    ax = test(relative_position, v_a, v_a_d)
    plt.savefig(os.path.join(OUT_DIRECTORY, 'no_overlap_3.png'))

    return ax


def overlap_with_solution_1():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take vA as close as vA_d as possible
        For this testcase, the projection would be on the cutoff circle"""

    # Constants
    relative_position = (-1., -1.)
    v_a = (0.1, 0.2)
    v_a_d = (0.3, 0.4)  # Possible to get there completely
    # vA_d = (0.8, 0.9)  # Not possible to get there in one step
    ax = test(relative_position, v_a, v_a_d, COLLISION_RESPONSIBILITY)
    plt.savefig(os.path.join(OUT_DIRECTORY, 'overlap_with_solution_1.png'))

    return ax


def overlap_with_solution_2():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take vA as close as vA_d as possible
        For this testcase, the projection would be on the left leg"""

    # Constants
    relative_position = (-1., 1.)
    v_a = (-2.3, 0.5)
    # cutoff_center = tuple(np.array(relative_position) / TAU)
    # cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    # cutoff_circle = Circle(cutoff_center, cutoff_radius)
    # vo = VelocityObstacle(cutoff_circle)
    # left_normal = np.array(vo.left_tangent.normal)
    v_a_d = (-2.5, 0.3)  # Not along the normal
    # vA_d = np.array(vA) - 0.3 * left_normal  # Along the normal

    ax = test(relative_position, v_a, v_a_d, COLLISION_RESPONSIBILITY)
    plt.savefig(os.path.join(OUT_DIRECTORY, 'overlap_with_solution_2.png'))

    return ax


def overlap_with_solution_3():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take vA as close as vA_d as possible
        For this testcase, the projection would be on the right leg"""

    # Constants
    relative_position = (1., 1.)
    v_a = (2.3, 0.5)
    # cutoff_center = tuple(np.array(relative_position) / TAU)
    # cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    # cutoff_circle = Circle(cutoff_center, cutoff_radius)
    # vo = VelocityObstacle(cutoff_circle)
    # right_normal = np.array(vo.right_tangent.normal)
    v_a_d = (2.5, 0.3)  # Not along the normal
    # vA_d = np.array(vA) + 0.3 * right_normal  # Along the normal

    ax = test(relative_position, v_a, v_a_d, COLLISION_RESPONSIBILITY)
    plt.savefig(os.path.join(OUT_DIRECTORY, 'overlap_with_solution_3.png'))

    return ax


def overlap_without_solution_1():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle, but there is not solution since the line at distance d does not intersect the 
    possible relative velocities circle:
        For this testcase, the direction would take us closer to the right leg"""

    # Constants
    relative_position = (1., 1.)
    v_a = (2.3, -0.1)
    # cutoff_center = tuple(np.array(relative_position) / TAU)
    # cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    # cutoff_circle = Circle(cutoff_center, cutoff_radius)
    # vo = VelocityObstacle(cutoff_circle)
    # right_normal = np.array(vo.right_tangent.normal)
    v_a_d = (2.5, -0.5)  # Not along the normal
    # vA_d = np.array(vA) + 0.3 * right_normal  # Along the normal

    ax = test(relative_position, v_a, v_a_d, COLLISION_RESPONSIBILITY)
    plt.savefig(os.path.join(OUT_DIRECTORY, 'overlap_without_solution_1.png'))

    return ax


def overlap_without_solution_2():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle, but there is not solution since the line at distance d does not intersect the 
    possible relative velocities circle:
        For this testcase, the direction would take us closer to the left leg"""

    # Constants
    relative_position = (1., 1.)
    v_a = (0., 1.5)
    cutoff_center = tuple(np.array(relative_position) / TAU)
    cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    cutoff_circle = Circle(cutoff_center, cutoff_radius)
    vo = VelocityObstacle(cutoff_circle)
    # right_normal = np.array(vo.right_tangent.normal)
    v_a_d = (-0.5, 1.9)  # Not along the normal
    # vA_d = np.array(vA) + 0.3 * right_normal  # Along the normal
    ax = test(relative_position, v_a, v_a_d, COLLISION_RESPONSIBILITY)
    plt.savefig(os.path.join(OUT_DIRECTORY, 'overlap_without_solution_2.png'))

    return ax
