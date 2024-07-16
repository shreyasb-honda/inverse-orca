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
HUMAN_HEADING = (1., 0.)
DESIRED_HEADING = np.array([2., 1.])
DESIRED_HEADING = tuple(DESIRED_HEADING / norm(DESIRED_HEADING))
AGENT_RADIUS = 0.3
MAX_RADIUS = 3
TIME_HORIZON = 6
NUM_SAMPLES = int(1e4)
MAX_SPEED = 2.0
HUMAN_CIRCLE = Circle(HUMAN_POS, AGENT_RADIUS)

# HUMAN_POLICY = 'orca'
HUMAN_POLICY = 'social_force'

VMIN = np.dot(HUMAN_HEADING, DESIRED_HEADING)


def get_vh_orca(robot_pos, vr):
    """
    Returns the human's velocity using ORCA
    """

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

    return vh


def get_vh_sf(robot_pos, vr):
    """
    Returns the human's velocity using social force model
    """
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

    return vh


def plot_influence(data):
    """
    Plots the cosine of the angle between the human's new velocity 
    and the desired velocity
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=data[:, 2], y=data[:, 3],
                         c=data[:, 4], cmap='viridis', vmin=VMIN, vmax=1.0)

    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             HUMAN_HEADING[0], HUMAN_HEADING[1],
             color='red', lw=1.5, label=r'$v_h$')
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             DESIRED_HEADING[0], DESIRED_HEADING[1],
             color='blue', lw=1.5, label=r'$v_h^d$')
    ax.set_title('Cos of angle between new velocity and desired velocity')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plot_boundary(ax)
    plt.colorbar(scatter)


def plot_dot(data):
    """
    Plots the dot product between the desired velocity and the human's new velocity
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=data[:, 2], y=data[:, 3],
                         c=data[:, 9], cmap='viridis', vmin=VMIN, vmax=1.0)
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


def plot_cos_expected(data):
    """
    Plots the cosine of the angle between the human's new velocity expected by
    inverse orca and the desired velocity
    """

    fig, ax = plt.subplots()
    scatter = ax.scatter(x=data[:, 2], y=data[:, 3],
                         c=data[:, 10], cmap='viridis', vmin=VMIN, vmax=1.0)

    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             HUMAN_HEADING[0], HUMAN_HEADING[1],
             color='red', lw=1.5, label=r'$v_h$')
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
             DESIRED_HEADING[0], DESIRED_HEADING[1],
             color='blue', lw=1.5, label=r'$v_h^d$')
    t_str = 'Cos of angle between \nnew velocity (expected by inverse orca) and desired velocity'
    ax.set_title(t_str)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    plt.colorbar(scatter)


def plot_vel_field(data, agent = 'robot'):
    """
    Plots the velocity fields for the agents
    depending on the robot's position
    """
    fig, ax = plt.subplots()

    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
            HUMAN_HEADING[0], HUMAN_HEADING[1],
            color='red', lw=1.5, label=r'$v_h$')
    ax.arrow(HUMAN_POS[0], HUMAN_POS[1],
            DESIRED_HEADING[0], DESIRED_HEADING[1],
            color='blue', lw=1.5, label=r'$v_h^d$')

    if agent == 'robot':
        ax.quiver(data[:,2], data[:,3], data[:,5], data[:,6])
        ax.set_title('Field of robot velocities')
    elif agent == 'human':
        ax.quiver(data[:,2], data[:,3], data[:,7], data[:,8])
        ax.set_title('Field of human velocities')

    ax.set_aspect('equal')
    ax.legend(loc='upper right')


def plot_boundary(ax: plt.Axes):
    """
    Plots the mathematical boundary
    """
    current_to_desired = np.array(DESIRED_HEADING) - np.array(HUMAN_HEADING)
    current_to_desired /= norm(current_to_desired)
    ax.arrow(0, 0, *(-3 * current_to_desired), color='black', lw=2)


def main():
    """
    The main function
    """
    rng = np.random.default_rng(seed=123)
    i = 0
    data = np.zeros((NUM_SAMPLES, 12))
    while i < NUM_SAMPLES:
        theta = 2 * np.pi * rng.random()
        radius = MAX_RADIUS * rng.random()
        robot_pos = (radius * np.cos(theta), radius * np.sin(theta))
        robot_circle = Circle(robot_pos, AGENT_RADIUS)

        if robot_circle.circle_overlap(HUMAN_CIRCLE):
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
            vh = get_vh_orca(robot_pos, vr)

        elif HUMAN_POLICY == 'social_force':
            vh = get_vh_sf(robot_pos, vr)

        dot = np.dot(vh, DESIRED_HEADING)
        expected_dot = np.dot(opt.vh_new, DESIRED_HEADING)
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
        data[i, 10] = expected_dot / norm(vh)
        data[i, 11] = expected_dot

        i += 1

    plot_influence(data)
    # plot_dot(data)
    # plot_cos_expected(data)
    # plot_vel_field(data, 'robot')
    # plot_vel_field(data, 'human')

    plt.show()


if __name__ == "__main__":
    main()
