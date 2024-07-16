"""
Compares the outputs of the policy for mirrored inputs.
The outputs should also be mirrored. However, it seems that some
function in the pipeline is not preserving this mirroredness property
"""
import os
import logging
import numpy as np
from numpy.linalg import norm
import rvo2
logging.disable(logging.ERROR)
import pysocialforce as psf
import matplotlib.pyplot as plt
from policy.utils.get_velocity import OptimalInfluence, VelocityObstacle, Circle


AGENT_RADIUS = 0.3
MAX_RADIUS = 3
TIME_HORIZON = 6
NUM_SAMPLES = 10
MAX_SPEED = 1.5

# Original inputs
HUMAN_POS = (0., 0.)
HUMAN_HEADING = (1., 0.)
DESIRED_HEADING = np.array([1., 1.])
DESIRED_HEADING = tuple(DESIRED_HEADING / norm(DESIRED_HEADING))
HUMAN_CIRCLE = Circle(HUMAN_POS, AGENT_RADIUS)

# Mirrored inputs
HUMAN_POS_MIRRORED = (0., 0.)
HUMAN_HEADING_MIRRORED = (1., 0.)
DESIRED_HEADING_MIRRORED = np.array([1., -1.])
DESIRED_HEADING_MIRRORED = tuple(DESIRED_HEADING_MIRRORED / norm(DESIRED_HEADING_MIRRORED))
HUMAN_CIRCLE_MIRRORED = Circle(HUMAN_POS_MIRRORED, AGENT_RADIUS)

HUMAN_POLICY = 'orca'
# HUMAN_POLICY = 'social_force'

VMIN = np.dot(HUMAN_HEADING, DESIRED_HEADING)

def get_velocities(robot_pos, human_pos, heading, desired_heading):
    """
    Returns the human's and the robot's velocities
    """
    # Get the robot's velocity using inverse ORCA
    center = ((robot_pos[0] - human_pos[0]) / TIME_HORIZON,
              (robot_pos[1]- human_pos[1]) / TIME_HORIZON)

    cutoff_circle = Circle(center, 2 * AGENT_RADIUS / TIME_HORIZON)
    vo = VelocityObstacle(cutoff_circle)
    fig, ax = plt.subplots()
    vo.plot(ax)
    opt = OptimalInfluence(vo, vr_max=MAX_SPEED, collision_responsibility=1.0)
    vr, u = opt.compute_velocity(heading, desired_heading)
    print("Projection magnitude:", norm(u))

    velocity_circle = Circle(heading, MAX_SPEED)
    velocity_circle.plot(ax)

    # More to plot
    #   - The normal(s)
    #   - The vector from vh to vh_d

    current_to_desired = np.array(desired_heading) - np.array(heading)
    current_to_desired /= norm(current_to_desired)

    ax.arrow(*vo.left_tangent.point, *current_to_desired, color='blue', lw=2)
    ax.arrow(*vo.right_tangent.point, *current_to_desired, color='blue', lw=2)

    ax.arrow(*vo.left_tangent.point, *vo.left_tangent.normal, color='orange', lw=2)

    ax.arrow(*vo.right_tangent.point, *vo.right_tangent.normal, color='orange', lw=2)

    ax.set_aspect('equal')

    # Compute the human' velocity according to its policy
    if HUMAN_POLICY == 'orca':
        params = (20, 10, 6, 10)

        # Create a simulator instance
        sim = rvo2.PyRVOSimulator(0.25, *params, AGENT_RADIUS,
                                  MAX_SPEED, collisionResponsibility=1.0)

        # Add the robot
        sim.addAgent(robot_pos, *params, AGENT_RADIUS,
                     MAX_SPEED, vr,
                     collisionResponsibility=1.0)

        # Add the human
        sim.addAgent(human_pos, *params, AGENT_RADIUS,
                    1.0, heading,
                    collisionResponsibility=1.0)

        # Set the preferred velocity of the robot to be goal-directed maximum
        sim.setAgentPrefVelocity(0, (MAX_SPEED, 0.))

        # Set the preferred velocity of the human to be their current velocity
        sim.setAgentPrefVelocity(1, heading)

        # Perform a step
        sim.doStep()
        vh = sim.getAgentVelocity(1)

    elif HUMAN_POLICY == 'social_force':
        human_goal = heading
        robot_goal = (0., robot_pos[1])

        initial_state = np.array(
            [
                [*human_pos, *heading, *human_goal],
                [*robot_pos, *vr, *robot_goal]
            ]
        )

        config_file = os.path.join('sim', 'config', 'policy.toml')
        s = psf.Simulator(
            initial_state,
            config_file=config_file
        )

        s.step(1)
        states, _ = s.get_states()

        vh = np.array([states[1, 0, 2], states[1, 0, 3]])

    return vh, vr

def main():
    """
    The main function
    """
    # rng = np.random.default_rng(seed=None)
    # i = 0

    # while i < NUM_SAMPLES:
    #     theta = 2 * np.pi * rng.random()
    #     radius = MAX_RADIUS * rng.random()
    #     robot_pos = (radius * np.cos(theta), radius * np.sin(theta))
    #     robot_circle = Circle(robot_pos, AGENT_RADIUS)

    #     if robot_circle.circle_overlap(HUMAN_CIRCLE):
    #         # i -= 1
    #         continue

    #     vh, vr = get_velocities(robot_pos, HUMAN_POS,
    #                             HUMAN_HEADING,
    #                             DESIRED_HEADING)

    #     robot_pos_mirrored = (-robot_pos[0], robot_pos[1])
    #     vh_mirrored, vr_mirrored = get_velocities(robot_pos_mirrored,
    #                                               HUMAN_POS_MIRRORED,
    #                                               HUMAN_HEADING_MIRRORED,
    #                                               DESIRED_HEADING_MIRRORED)

    #     print(f"{'original:':<15}",
    #           f"Robot velocity: {vr[0]:.2f} {vr[1]:.2f},",
    #           f"Human velocity: {vh[0]:.2f} {vh[1]:.2f}")
    #     print(f"{'mirrored:':<15}",
    #           f"Robot velocity: {vr_mirrored[0]:.2f} {vr_mirrored[1]:.2f},",
    #           f"Human velocity: {vh_mirrored[0]:.2f} {vh_mirrored[1]:.2f}")

    #     if round(vr[0], 2) != -round(vr_mirrored[0], 2):
    #         print(robot_pos)
    #         return

    #     i += 1

    # robot_pos = (-0.617, 0.361)
    # robot_pos = (-0.686, 0.442)

    # robot_pos = (0.665, -1.183)
    robot_pos = (0.667, -1.790)

    # print("Original: ")
    vh, vr = get_velocities(robot_pos, HUMAN_POS,
                            HUMAN_HEADING,
                            DESIRED_HEADING)
    print("Robot velocity:", vr)
    print("Human velocity:", vh)
    print("Human speed:", norm(vh))
    dot = np.dot(vh, DESIRED_HEADING) / norm(vh)
    print("Cos of angle:", dot)

    # print()
    # print()
    # print("Mirrored")

    # robot_pos_mirrored = (robot_pos[0], -robot_pos[1])
    # vh_mirrored, vr_mirrored = get_velocities(robot_pos_mirrored,
    #                                             HUMAN_POS_MIRRORED,
    #                                             HUMAN_HEADING_MIRRORED,
    #                                             DESIRED_HEADING_MIRRORED)
    # print(vh_mirrored)
    # print(vr_mirrored)
    # dot = np.dot(vh_mirrored, DESIRED_HEADING) / norm(vh_mirrored)
    # print(dot)

    plt.show()


if __name__ == "__main__":
    main()
