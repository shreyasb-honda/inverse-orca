import os
from configparser import RawConfigParser
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from sim.agent import Robot, Human
from policy.orca import Orca
from policy.invorca import InvOrca


def main():

    parser = ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False,
                        help="Whether to run the simulation in debug mode (default: False)")
    parser.add_argument('--save-anim', type=bool, default=True,
                        help="Whether to save the animation to a file (default: False)")
    parser.add_argument('--render-mode', type=str, default='human',
                        help='The mode in which to render (human or static plot) (default: human)')

    parser.add_argument('--num-runs', type=int, default=1,
                        help='The number of times the simulation should be run (default: 1)')

    args = parser.parse_args()

    # Configure the environment
    env_config = RawConfigParser()
    env_config.read(os.path.join('.', 'sim', 'config', 'env.config'))
    env = gym.make('HallwayScene-v0')

    # Configure the policy
    orca = Orca()
    policy_config = RawConfigParser()
    policy_config.read(os.path.join('.', 'sim', 'config', 'policy.config'))
    orca.configure(policy_config)

    invorca = InvOrca()
    invorca.configure(policy_config)

    # Configure the human
    time_step = env_config.getfloat('env', 'time_step')
    radius = env_config.getfloat('human', 'radius')
    max_speed = env_config.getfloat('human', 'max_speed')
    collision_responsibility = env_config.getfloat('human', 'collision_responsibility')
    human = Human(radius, max_speed, max_speed, time_step, collision_responsibility)
    orca.set_collision_responsiblity(collision_responsibility)
    human.set_policy(orca)

    # Configure the robot
    radius = env_config.getfloat('robot', 'radius')
    max_speed = env_config.getfloat('robot', 'max_speed')
    d_virtual_goal = env_config.getfloat('env', 'd_virtual_goal')
    y_virtual_goal = env_config.getfloat('env', 'y_virtual_goal')
    invorca.set_virtual_goal_params(d_virtual_goal, y_virtual_goal)
    robot = Robot(radius, max_speed, max_speed, time_step)
    robot.set_virtual_goal_params(d_virtual_goal, y_virtual_goal)
    robot.set_policy(invorca)

    env.unwrapped.set_human(human)
    env.unwrapped.set_robot(robot)
    env.unwrapped.configure(env_config, args.save_anim, args.render_mode)

    num_failed = 0
    env.reset(seed=100)

    for _ in tqdm(range(args.num_runs)):
        # try:
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
            robot_action = robot.policy.predict(obs)
            obs['robot vel'] = np.array(robot_action)
            # Update the observation to include the current velocity of the robot
            human_action = human.policy.predict(obs)

        env.render()
        # except TypeError as err:
        #     print("TypeError: ", err)
        #     num_failed += 1
        #     continue

    print(f"(failed/total) = ({num_failed}/{args.num_runs})")

if __name__ == "__main__":
    main()
