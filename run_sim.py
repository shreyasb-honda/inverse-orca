import logging
logging.disable(logging.ERROR)
import os
from configparser import RawConfigParser
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from sim.agent import Robot, Human
from policy.orca import Orca
from policy.invorca import InvOrca
from policy.social_force import SocialForce
from policy.weighted_sum import WeightedSum
from policy.utils.estimate_alpha import estimate_alpha


def run_sim(render_mode: str = 'human', save_anim: bool = True, num_runs: int = 1,
            alpha: float | None = None, max_speed_robot: float | None = None,
            time_horizon_robot: int | None = None,
            time_horizon_human: int | None = None,
            out_fname: str | None = None,
            human_policy: str = 'orca',
            robot_policy: str = 'invorca'):
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

    # Configure the policy
    if human_policy == 'orca':
        orca = Orca()
        orca.configure(policy_config)
        if time_horizon_human is not None:
            orca.time_horizon = time_horizon_human

    if robot_policy == 'invorca':
        invorca = InvOrca()
        invorca.configure(policy_config)

        if time_horizon_robot is not None:
            invorca.time_horizon = time_horizon_robot
            invorca.orca_time_horizon = time_horizon_robot

    if robot_policy == 'weighted_sum':
        weighted_sum = WeightedSum()
        weighted_sum.configure(policy_config)
        if time_horizon_robot is not None:
            weighted_sum.time_horizon = time_horizon_robot
            weighted_sum.orca_time_horizon = time_horizon_robot


    # Configure the human
    time_step = env_config.getfloat('env', 'time_step')
    radius = env_config.getfloat('human', 'radius')
    max_speed = env_config.getfloat('human', 'max_speed')
    collision_responsibility = env_config.getfloat('human', 'collision_responsibility')
    human = Human(radius, max_speed, max_speed, time_step, collision_responsibility)

    if human_policy == 'orca':
        orca.set_collision_responsiblity(collision_responsibility)
        human.set_policy(orca)

    if human_policy == 'socialforce':
        social_force = SocialForce()
        human.set_policy(social_force)

    # Configure the robot
    radius = env_config.getfloat('robot', 'radius')
    max_speed = env_config.getfloat('robot', 'max_speed')
    d_virtual_goal = env_config.getfloat('env', 'd_virtual_goal')
    y_virtual_goal = env_config.getfloat('env', 'y_virtual_goal')
    if robot_policy == 'invorca':
        invorca.set_virtual_goal_params(d_virtual_goal, y_virtual_goal)
    elif robot_policy == 'weighted_sum':
        weighted_sum.set_virtual_goal_params(d_virtual_goal, y_virtual_goal)
    
    robot = Robot(radius, max_speed, max_speed, time_step)
    robot.set_virtual_goal_params(d_virtual_goal, y_virtual_goal)

    if robot_policy == 'invorca':
        robot.set_policy(invorca)
    elif robot_policy == 'weighted_sum':
        robot.set_policy(weighted_sum)

    env.unwrapped.set_human(human)
    env.unwrapped.set_robot(robot)
    env.unwrapped.configure(env_config, save_anim, render_mode)

    if out_fname is not None:
        env.unwrapped.set_output_filename(out_fname)

    num_failed = 0

    if human_policy == 'orca' and alpha is not None:
        human.collision_responsibility = alpha
        human.policy.set_collision_responsiblity(alpha)

    if max_speed_robot is not None:
        robot.max_speed = max_speed_robot
        robot.policy.set_max_speed(max_speed_robot)

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
                        help='The human policy [orca or socialforce] (default: orca)')
    parser.add_argument('--robot-policy', type=str, default='invorca',
                        help='The robot policy [invorca or weighted_sum] (default: invorca)')

    args = parser.parse_args()

    num_failed = run_sim(args.render_mode, args.save_anim, args.num_runs,
                         time_horizon_robot=args.tau_robot,
                         time_horizon_human=args.tau_human,
                         human_policy=args.human_policy,
                         robot_policy=args.robot_policy)

    print(f"(failed/total) = ({num_failed}/{args.num_runs})")

if __name__ == "__main__":
    main()
