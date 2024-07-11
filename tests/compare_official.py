"""
Compares our expected results with the official ORCA implementation
"""
import rvo2
import matplotlib.pyplot as plt
import numpy as np
from policy.utils.get_velocity import OptimalInfluence
from policy.utils.overlap_detection import Circle, VelocityObstacle, Point


TAU = 5.0
RADIUS_A = 0.3
RADIUS_B = 0.3
VB_MAX = 2.0
EPSILON = 1e-3
COLLISION_RESPONSIBILITY = 0.5


def test(relative_position: Point, va: Point, va_d: Point, 
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
    invorca = OptimalInfluence(vo, vr_max=VB_MAX, epsilon=EPSILON, 
                               collision_responsibility=collision_responsibility)
    invorca.compute_velocity(va, va_d)

    fig, ax = plt.subplots(layout='tight', figsize=(9, 9))
    ax.set_aspect('equal')
    ax.axhline(color='black')
    ax.axvline(color='black')

    # Plot the velocity obstacle
    ax = vo.plot(ax)

    # Plot the relative velocity circle
    velocity_circle = Circle(va, VB_MAX)
    ax = velocity_circle.plot(ax)

    # Plot the desired velocity for A
    ax.scatter(va_d[0], va_d[1], color='tab:purple', label='va_d')

    # If solution exists, plot vB, vaB, va_new
    # If solution does not exist, plot vB, vaB (va_new = va)
    ax.scatter(invorca.vr[0], invorca.vr[1], color='lime', label='vb')
    ax.scatter(va[0] - invorca.vr[0], va[1] - invorca.vr[1], color='tab:orange', label='vab')

    if invorca.solution_exists:
        ax.scatter(invorca.vh_new[0], invorca.vh_new[1], color='teal', label='va_new')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    return ax, invorca


def overlap_with_solution_1():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take va as close as va_d as possible
        For this testcase, the projection would be on the cutoff circle"""

    # Constants
    # relative_position = (-1., -1.)
    # va = (0.1, 0.2)
    # va_d = (0.3, 0.4)  # Possible to get there completely
    # va_d = (0.8, 0.9)  # Not possible to get there in one step

    va = (-1.0, 0.)
    va_d = (-0.983, -0.156)
    relative_position = (-1.866 * 5, 0.161 * 5)
    ax, invorca = test(relative_position, va, va_d, COLLISION_RESPONSIBILITY)


    params = (10., 10, TAU, 2.0)
    sim = rvo2.PyRVOSimulator(1.0, *params, RADIUS_B, VB_MAX)
    sim.addAgent(relative_position, *params, RADIUS_B, VB_MAX, invorca.vr, COLLISION_RESPONSIBILITY)
    sim.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, invorca.vh, COLLISION_RESPONSIBILITY)

    va_0 = sim.getAgentVelocity(1)

    sim.setAgentPrefVelocity(0, invorca.vr)
    sim.setAgentPrefVelocity(1, invorca.vh)
    sim.doStep()
    va_new_official = sim.getAgentVelocity(1)
    va_new_ours = invorca.vh_new

    print(f'va (official) {va_0}')
    print(f"va_new (official) {va_new_official}")
    print(f"va_new (ours) {va_new_ours}")

    # print(f"vB (official) {sim.getAgentVelocity(0)}")
    # print(f"vB (ours) {invorca.vB}")


    return ax


def overlap_with_solution_2():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take va as close as va_d as possible
        For this testcase, the projection would be on the left leg"""

    # Constants
    relative_position = (-1., 1.)
    va = (-2.3, 0.5)
    va_d = (-2.5, 0.3)  # Not along the normal
    # cutoff_center = tuple(np.array(relative_position) / TAU)
    # cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    # cutoff_circle = Circle(cutoff_center, cutoff_radius)
    # vo = VelocityObstacle(cutoff_circle)
    # left_normal = np.array(vo.left_tangent.normal)
    # va_d = np.array(va) - 0.3 * left_normal  # Along the normal
    ax, invorca = test(relative_position, va, va_d, COLLISION_RESPONSIBILITY)


    params = (10., 10, TAU, 2.0)
    sim = rvo2.PyRVOSimulator(1.0, *params, RADIUS_B, VB_MAX)
    sim.addAgent(relative_position, *params, RADIUS_B, VB_MAX, invorca.vr, COLLISION_RESPONSIBILITY)
    sim.addAgent((0., 0.), *params, RADIUS_A, 5., invorca.vh, COLLISION_RESPONSIBILITY)

    va_0 = sim.getAgentVelocity(1)

    sim.setAgentPrefVelocity(0, invorca.vr)
    sim.setAgentPrefVelocity(1, invorca.vh)
    sim.doStep()
    va_new_official = sim.getAgentVelocity(1)
    va_new_ours = invorca.vh_new

    print(f'va (official) {va_0}')
    print(f"va_new (official) {va_new_official}")
    print(f"va_new (ours) {va_new_ours}")

    # print(f"vB (official) {sim.getAgentVelocity(0)}")
    # print(f"vB (ours) {invorca.vB}")

    return ax


def overlap_with_solution_3():
    """Case where there is an overlap between the velocity circle and the 
    velocity obstacle:
        In this case, we expect the direction to take va as close as va_d as possible
        For this testcase, the projection would be on the right leg"""

    # Constants
    relative_position = (1., 1.)
    va = (2.3, 0.5)
    va_d = (2.5, 0.3)  # Not along the normal
    # cutoff_center = tuple(np.array(relative_position) / TAU)
    # cutoff_radius = (RADIUS_A + RADIUS_B) / TAU
    # cutoff_circle = Circle(cutoff_center, cutoff_radius)
    # vo = VelocityObstacle(cutoff_circle)
    # right_normal = np.array(vo.right_tangent.normal)
    # va_d = np.array(va) + 0.3 * right_normal  # Along the normal

    ax, invorca = test(relative_position, va, va_d, COLLISION_RESPONSIBILITY)

    params = (10., 10, TAU, 2.0)
    sim = rvo2.PyRVOSimulator(1.0, *params, RADIUS_B, VB_MAX)
    sim.addAgent(relative_position, *params, RADIUS_B, VB_MAX, invorca.vr, COLLISION_RESPONSIBILITY)
    sim.addAgent((0., 0.), *params, RADIUS_A, 5., invorca.vh, COLLISION_RESPONSIBILITY)

    va_0 = sim.getAgentVelocity(1)

    sim.setAgentPrefVelocity(0, invorca.vr)
    sim.setAgentPrefVelocity(1, invorca.vh)
    sim.doStep()
    va_new_official = sim.getAgentVelocity(1)
    va_new_ours = invorca.vh_new

    print(f'va (official) {va_0}')
    print(f"va_new (official) {va_new_official}")
    print(f"va_new (ours) {va_new_ours}")

    # print(f"vB (official) {sim.getAgentVelocity(0)}")
    # print(f"vB (ours) {invorca.vB}")

    return ax


def different_pref_velocity_1():
    """
    In this case, the human's preferred velocity is different from its current velocity
    We want to check if this affects the output of the official ORCA implementation
    """

    # Constants
    relative_position = (-1., -1.)
    va = (0.1, 0.2)
    va_d = (0.3, 0.4)  # Possible to get there completely
    # va_d = (0.8, 0.9)  # Not possible to get there in one step

    # va = (-0.95, 0.1)
    # va_d = (-0.983, -0.156)
    # relative_position = (-1.866 * 5, 0.161 * 5)
    ax, invorca = test(relative_position, va, va_d, COLLISION_RESPONSIBILITY)


    params = (10., 10, TAU, 2.0)
    sim = rvo2.PyRVOSimulator(1.0, *params, RADIUS_B, VB_MAX)
    sim.addAgent(relative_position, *params, RADIUS_B, VB_MAX, invorca.vr, COLLISION_RESPONSIBILITY)
    sim.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, invorca.vh, COLLISION_RESPONSIBILITY)

    va_0 = sim.getAgentVelocity(1)

    sim.setAgentPrefVelocity(0, invorca.vr)
    sim.setAgentPrefVelocity(1, (1.0, 0))   # Goal directed
    # sim.setAgentPrefVelocity(1, invorca.va)   # Same as current
    sim.doStep()
    va_new_official = sim.getAgentVelocity(1)
    va_new_ours = invorca.vh_new

    print(f'va (official) {va_0}')
    print(f"va_new (official) {va_new_official}")
    print(f"va_new (ours) {va_new_ours}")

    print(f"vB (official) {sim.getAgentVelocity(0)}")
    print(f"vB (ours) {invorca.vr}")

    return ax


def test_random(num_runs: int = 100, seed: int | None = None):
    """
    Generates and tests a random test case
    Not currently working as expected since it is difficult to ensure that
    a random test case has a solution
    """
    rng = np.random.default_rng(seed=seed)
    num_failures = 0
    i = 0

    while i < num_runs:
        # something in the interval of [-5, 5] x [-5, 5]
        relative_position = 10 * rng.random(2) - 5
        while np.linalg.norm(relative_position) < RADIUS_A + RADIUS_B + 0.5:
            relative_position = 10 * rng.random(2) - 5   
            # To ensure that there is no collision to begin with

        # something in the interval of [-1, 1] x [-1, 1]
        v_a = 2 * rng.random(2) - 1
        v_a /= np.linalg.norm(v_a)    # normalize it

        # Convert to tuples
        v_a = tuple(v_a)

        # something in [0, 1.0)
        collision_responsibility = rng.random()

        time_horizon = rng.integers(2, 10)

        cutoff_center = tuple(np.array(relative_position) / time_horizon)
        cutoff_radius = (RADIUS_A + RADIUS_B) / time_horizon
        cutoff_circle = Circle(cutoff_center, cutoff_radius)
        vo = VelocityObstacle(cutoff_circle)
        invorca = OptimalInfluence(vo, vr_max=VB_MAX, epsilon=EPSILON,
                                   collision_responsibility=collision_responsibility)

        dot1 = abs(np.dot(v_a, invorca.vo.right_tangent.normal))
        dot2 = abs(np.dot(v_a, invorca.vo.left_tangent.normal))
        if dot1 < dot2:
            va_d = v_a + 0.4 * rng.random(2) * np.array(invorca.vo.right_tangent.normal)
        else:
            va_d = v_a + 0.4 * rng.random(2) * np.array(invorca.vo.left_tangent.normal)

        va_d = tuple(va_d)

        v_b, _ = invorca.compute_velocity(v_a, va_d)
        if v_b is None:
            i -= 1
            continue

        params = (20., 10, time_horizon, 2.0)
        sim = rvo2.PyRVOSimulator(1.0, *params, RADIUS_B, VB_MAX, 
                                  collisionResponsibility=collision_responsibility)
        sim.addAgent(tuple(relative_position), *params, RADIUS_B, VB_MAX, invorca.vr, 
                     collision_responsibility)
        sim.addAgent((0., 0.), *params, RADIUS_A, VB_MAX, invorca.vh, collision_responsibility)

        sim.setAgentPrefVelocity(0, invorca.vr)
        sim.setAgentPrefVelocity(1, invorca.vh)
        sim.doStep()
        v_a_new_official = sim.getAgentVelocity(1)
        v_a_new_ours = invorca.vh_new

        diff = np.array(v_a_new_official) - np.array(v_a_new_ours)
        if np.linalg.norm(diff) > 1e-3:
            num_failures += 1

        i += 1

    print(f"(num_failures/num_runs) = ({num_failures}/{num_runs})")
