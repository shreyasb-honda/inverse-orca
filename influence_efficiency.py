"""
Gets an idea of the optimal influence position of the robot w.r.t the human
"""
import os
import logging
import numpy as np
from numpy.linalg import norm
import rvo2
import pysocialforce as psf
import matplotlib.pyplot as plt
from policy.utils.get_velocity import OptimalInfluence, VelocityObstacle, Circle
logging.disable(logging.ERROR)

HUMAN_POS = (0., 0.)
HUMAN_HEADING = (-1., 0.)
DESIRED_HEADING = np.array([-2., 1.])
DESIRED_HEADING = tuple(DESIRED_HEADING / norm(DESIRED_HEADING))
AGENT_RADIUS = 0.3
MAX_RADIUS = 3
TIME_HORIZON = 6
NUM_SAMPLES = int(1e4)
MAX_SPEED = 1.5
HUMAN_CIRCLE = Circle(HUMAN_POS, AGENT_RADIUS)

HUMAN_POLICY = 'orca'
# HUMAN_POLICY = 'social_force'


def main():
    """
    The main function
    """
    rng = np.random.default_rng(seed=123)
    i = 0

    data = np.zeros((NUM_SAMPLES, 10))
    while i < NUM_SAMPLES:
        theta = 2 * np.pi * rng.random()
        radius = MAX_RADIUS * rng.random()
        robot_pos = (radius * np.cos(theta), radius * np.sin(theta))
        robot_circle = Circle(robot_pos, AGENT_RADIUS)

        if robot_circle.circle_overlap(HUMAN_CIRCLE):
            # i -= 1
            continue

        # Get the robot's velocity using inverse ORCA
        center = ((robot_pos[0] - HUMAN_POS[0]) / TIME_HORIZON,
                  (robot_pos[1]- HUMAN_POS[1]) / TIME_HORIZON)

        cutoff_circle = Circle(center, 2 * AGENT_RADIUS / TIME_HORIZON)
        vo = VelocityObstacle(cutoff_circle)
        opt = OptimalInfluence(vo, vr_max=MAX_SPEED, collision_responsibility=1.0)
        vr, u = opt.compute_velocity(HUMAN_HEADING, DESIRED_HEADING)

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
            sim.addAgent(HUMAN_POS, *params, AGENT_RADIUS,
                        1.0, HUMAN_HEADING,
                        collisionResponsibility=1.0)

            # Set the preferred velocity of the robot to be goal-directed maximum
            sim.setAgentPrefVelocity(0, (MAX_SPEED, 0.))

            # Set the preferred velocity of the human to be their current velocity
            sim.setAgentPrefVelocity(1, HUMAN_HEADING)

            # Perform a step
            sim.doStep()
            vh = sim.getAgentVelocity(1)

        elif HUMAN_POLICY == 'social_force':
            human_goal = HUMAN_HEADING
            robot_goal = (0., robot_pos[1])

            initial_state = np.array(
                [
                    [*HUMAN_POS, *HUMAN_HEADING, *human_goal],
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

        dot = np.dot(vh, DESIRED_HEADING)
        # Save the data
        data[i, 0] = theta
        data[i, 1] = radius
        data[i, 2] = robot_pos[0]
        data[i, 3] = robot_pos[1]
        data[i, 4] = dot / norm(vh)
        data[i, 5] = vr[0]
        data[i, 6] = vr[1]
        data[i, 7] = vh[0]
        data[i, 8] = vh[1]
        data[i, 9] = dot

        i += 1

    print(data[:, 4].max())
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=data[:, 2], y=data[:, 3], c=data[:, 4], cmap='viridis', vmin=0.0, vmax=1.0)

    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             HUMAN_HEADING[0], HUMAN_HEADING[1],
             color='red', lw=1.5, label=r'$v_h$')
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             DESIRED_HEADING[0], DESIRED_HEADING[1],
             color='blue', lw=1.5, label=r'$v_h^d$')
    ax.set_title('Cos of angle between new velocity and desired velocity')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.colorbar(scatter)

    fig, ax = plt.subplots()
    scatter = ax.scatter(x=data[:, 2], y=data[:, 3], c=data[:, 9], cmap='viridis', vmin=0.0, vmax=1.0)
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             HUMAN_HEADING[0], HUMAN_HEADING[1],
             color='red', lw=1.5, label=r'$v_h$')
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             DESIRED_HEADING[0], DESIRED_HEADING[1],
             color='blue', lw=1.5, label=r'$v_h^d$')
    ax.set_title('Dot product between new velocity and desired velocity')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.colorbar(scatter)

    fig, ax = plt.subplots()
    ax.quiver(data[:,2], data[:,3], data[:,5], data[:,6])
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             HUMAN_HEADING[0], HUMAN_HEADING[1],
             color='red', lw=1.5, label=r'$v_h$')
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             DESIRED_HEADING[0], DESIRED_HEADING[1],
             color='blue', lw=1.5, label=r'$v_h^d$')
    ax.set_aspect('equal')
    ax.set_title('Field of robot velocities')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    ax.quiver(data[:,2], data[:,3], data[:,7], data[:,8])
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             HUMAN_HEADING[0], HUMAN_HEADING[1],
             color='red', lw=1.5, label=r'$v_h$')
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             DESIRED_HEADING[0], DESIRED_HEADING[1],
             color='blue', lw=1.5, label=r'$v_h^d$')
    ax.set_aspect('equal')
    ax.set_title('Field of human velocities')
    ax.legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    main()