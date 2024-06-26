"""
A general runner code for running the simulation
"""

import logging
import os
from configparser import RawConfigParser
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from sim.agent import Robot, Human
from policy.orca import Orca
from policy.invorca import InverseOrca
from policy.social_force import SocialForce
from policy.weighted_sum import WeightedSum
from policy.utils.estimate_alpha import estimate_alpha
logging.disable(logging.ERROR)


def configure_human(policy_name: str, policy_config: RawConfigParser,
                    env_config: RawConfigParser, time_horizon: float | None = None,
                    alpha: float | None = None):
    """
    Configures the human agent for the environment
    """
    human = Human()
    human.configure(env_config)
    human_policy = None
    if policy_name == 'orca':
        human_policy = Orca(human.time_step)
        human_policy.configure(policy_config)
        alpha = env_config.getfloat('human', 'collision_responsibility')
        human_policy.set_collision_responsiblity(alpha)
        if time_horizon is not None:
            human_policy.time_horizon = time_horizon
        if alpha is not None:
            human_policy.set_collision_responsiblity(alpha)

    elif policy_name == 'social_force':
        human_policy = SocialForce(human.time_step)
    else:
        raise ValueError(f"Unknown human policy {policy_name}.",
                          "Use one of 'orca' or 'social_force'.")

    human.set_policy(human_policy)
    return human


def configure_robot(policy_name: str, policy_config: RawConfigParser,
                    env_config: RawConfigParser, time_horizon: float | None = None,
                    max_speed: float | None = None):
    """
    Configures the robot agent for the environment
    """
    robot = Robot()
    robot.configure(env_config)
    robot_policy = None
    if policy_name == 'inverse_orca':
        robot_policy = InverseOrca(robot.time_step)
    elif policy_name == 'weighted_sum':
        robot_policy = WeightedSum(robot.time_step)
    else:
        raise ValueError(f"Unknown policy {policy_name} for the robot.",
                         "Use one of 'inverse_orca' or 'weighted_sum'.")

    robot_policy.configure(policy_config)
    if time_horizon is not None:
        robot_policy.time_horizon = time_horizon
        robot_policy.orca_time_horizon = time_horizon
    if max_speed is not None:
        robot_policy.max_speed = max_speed

    robot_policy.set_virtual_goal_params(*robot.get_virtual_goal_params())
    robot.set_policy(robot_policy)

    return robot


def run_sim(render_mode: str = 'human', save_anim: bool = True, num_runs: int = 1,
            alpha: float | None = None, max_speed_robot: float | None = None,
            time_horizon_robot: int | None = None,
            time_horizon_human: int | None = None,
            out_fname: str | None = None,
            human_policy_str: str = 'orca',
            robot_policy_str: str = 'invorca'):
    """
    A helper function to set up and run the simulation according to the desired parameters 
    supplied to it
    """

    # Configure the environment
    env_config = RawConfigParser()
    env_config.read(os.path.join('.', 'sim', 'config', 'env.config'))
    env = gym.make('HallwayScene-v0')

    # Policy config
    policy_config = RawConfigParser()
    policy_config.read(os.path.join('.', 'sim', 'config', 'policy.config'))

    # Configure the human
    human = configure_human(human_policy_str, policy_config, env_config,
                            time_horizon_human, alpha)

    # Configure the robot
    robot = configure_robot(robot_policy_str, policy_config, env_config,
                            time_horizon_robot, max_speed_robot)

    # Set these in the environment
    env.unwrapped.set_human(human)
    env.unwrapped.set_robot(robot)
    env.unwrapped.configure(env_config, save_anim, render_mode)

    if out_fname is not None:
        env.unwrapped.set_output_filename(out_fname)

    num_failed = 0
    env.reset(seed=100)

    for _ in tqdm(range(num_runs)):
        try:
            obs, _ = env.reset()
            robot.set_vh_desired(obs)
            robot_action = robot.policy.predict(obs)
            obs['robot vel'] = np.array(robot_action)
            human_action = human.get_velocity()

            done = False
            while not done:
                action = {
                    "robot vel": np.array(robot_action),
                    "human vel": np.array(human_action)
                }
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                robot.set_vh_desired(obs)
                # obs["human vel"] = np.array([-1.0, 0.])
                robot_action = robot.choose_action(obs)
                obs['robot vel'] = np.array(robot_action)
                # Update the observation to include the current velocity of the robot
                human_action = human.choose_action(obs)

                # Estimate the value of alpha
                alpha_hat = estimate_alpha((-1.0, 0), human_action, 
                                            robot.policy.invorca.vh_new,
                                            robot.policy.collision_responsibility,
                                            tuple(robot.policy.invorca.u))
                # print(alpha_hat)

            if env.unwrapped.collision:
                print("Collision happened at frame(s)", env.unwrapped.collision_frames)

            env.render()
        except TypeError as err:
            print("TypeError: ", err)
            num_failed += 1
            continue

    return num_failed



def main():
    """
    Just the main function to run the simulation
    """

    parser = ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False,
                        help="Whether to run the simulation in debug mode (default: False)")
    parser.add_argument('--save-anim', type=bool, default=True,
                        help="Whether to save the animation to a file (default: False)")
    parser.add_argument('--render-mode', type=str, default='human',
                        help='The mode in which to render (human or static plot) (default: human)')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='The number of times the simulation should be run (default: 1)')
    parser.add_argument('--tau-robot', type=int, default=6,
                        help='the planning time horizon for the robot (default: 6)')
    parser.add_argument('--tau-human', type=int, default=6,
                        help='the planning time horizon for the human (default: 6)')
    parser.add_argument('--human-policy', type=str, default='orca',
                        help='The human policy [orca or social_force] (default: orca)')
    parser.add_argument('--robot-policy', type=str, default='inverse_orca',
                        help='The robot policy [inverse_orca or weighted_sum] '
                            '(default: inverse_orca)')

    args = parser.parse_args()

    num_failed = run_sim(args.render_mode, args.save_anim, args.num_runs,
                         time_horizon_robot=args.tau_robot,
                         time_horizon_human=args.tau_human,
                         human_policy_str=args.human_policy,
                         robot_policy_str=args.robot_policy)

    print(f"(failed/total) = ({num_failed}/{args.num_runs})")

if __name__ == "__main__":
    main()
