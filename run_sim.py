"""
A general runner code for running the simulation
"""
import os
from typing import Dict
from argparse import ArgumentParser
import toml
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from sim.agent import Robot, Human
import sim.performance_metrics as pm
from policy.orca import Orca
from policy.invorca import InverseOrca
from policy.social_force import SocialForce
from policy.weighted_sum import WeightedSum
from policy.utils.estimate_alpha import estimate_alpha


class SimulationRunner:
    """
    A class for running the simulation with the supplied parameters
    """

    def __init__(self) -> None:
        self.render_mode = None
        self.save_anim = None
        self.num_runs = None
        self.alpha = None
        self.vr_max = None
        self.tau_robot = None
        self.tau_human = None
        self.out_fname = None
        self.human_policy_str = None
        self.robot_policy_str = None

        # Configuration dictionaries
        self.config_files = {}
        self.config = {}
        self.env = None
        self.human = None
        self.robot = None

        # Performance metrics
        self.perf_metrics = None

    def configure_from_file(self, sim_config: str,
                            env_config: str,
                            policy_config: str):
        """
        Configures the simulation runner with the supplied parameters
        """

        # Save the config file paths
        self.config_files['env'] = env_config
        self.config_files['sim'] = sim_config
        self.config_files['policy'] = policy_config

        # Load the configuration dictionaries at the file paths
        for key in self.config_files:
            self.config[key] = toml.load(self.config_files[key])

    # def update_config(self, env_data: Dict | None = None,
    #                   policy_data: Dict | None = None, sim_data: Dict | None = None):
    #     """
    #     Updates the configuration from a dictionary
    #     :param env_data - the dict with newer config items for the environment
    #     :param policy_data - the dict with newer config items for the policies
    #     :param sim_data - the dict with newer config items for the sim settings

    #     To use this, first read and store the configuration from the config files
    #                  then update the necessary values in them and pass the updated
    #                 dictionaries here
    #     """
    #     if env_data is not None:
    #         self.config['env'] = env_data

    #     if policy_data is not None:
    #         self.config['policy'] = policy_data

    #     if sim_data is not None:
    #         self.config['sim'] = sim_data

    def configure_human(self):
        """
        Configures the human agent for the environment
        """
        human = Human()
        human.configure(self.config_files['env'])
        human_policy = None
        if self.config['sim']['human_policy'] == 'orca':
            human_policy = Orca(human.time_step)
            human_policy.configure(self.config_files['policy'])
            alpha = self.config['env']['human']['collision_responsibility']
            human_policy.set_collision_responsiblity(alpha)

        elif self.config['sim']['human_policy'] == 'social_force':
            human_policy = SocialForce(human.time_step)
            human_policy.configure(self.config_files['policy'])
        else:
            raise ValueError(f"Unknown human policy {self.config['sim']['human_policy']}.",
                            "Use one of 'orca' or 'social_force'.")

        human.set_policy(human_policy)
        self.human = human

    def configure_robot(self):
        """
        Configures the robot agent for the environment
        """
        robot = Robot()
        robot.configure(self.config_files['env'])
        robot_policy = None
        if self.config['sim']['robot_policy'] == 'inverse_orca':
            robot_policy = InverseOrca(robot.time_step)
        elif self.config['sim']['robot_policy'] == 'weighted_sum':
            robot_policy = WeightedSum(robot.time_step)
        else:
            raise ValueError(f"Unknown policy {self.config['sim']['robot_policy']} for the robot.",
                            "Use one of 'inverse_orca' or 'weighted_sum'.")

        robot_policy.configure(self.config_files['policy'])
        robot_policy.set_virtual_goal_params(*robot.get_virtual_goal_params())
        robot.set_policy(robot_policy)
        self.robot = robot

    def setup_environment(self, seed: int | None = 100):
        """
        Sets up the simulation environment. 
        Only call once
        """
        # Configure the human
        self.configure_human()

        # Configure the robot
        self.configure_robot()

        # Configure the environment
        env = gym.make('HallwayScene-v0')
        # Set these in the environment
        env.unwrapped.set_human(self.human)
        env.unwrapped.set_robot(self.robot)
        env.unwrapped.configure(self.config_files['env'],
                                self.config['sim']['save_anim'],
                                self.config['sim']['render_mode'])
        self.env = env

        if self.config['sim']['out_fname'] != 'temp':
            env.unwrapped.set_output_filename(self.config['sim']['out_fname'])
        self.env.reset(seed=seed)

    def run_sim(self):
        """
        Runs the sim multiple times
        """
        for _ in range(self.config['sim']['num_runs']):
            self.run_sim_once()

    def run_sim_once(self):
        """
        Runs the simulation once
        """

        # Configure the performance metrics
        time_step = self.config['env']['env']['time_step']
        y_virtual_goal = self.config['env']['env']['y_virtual_goal']
        acceleration_metric_human = pm.CumulativeAcceleration(time_step, 'human')
        acceleration_metric_robot = pm.CumulativeAcceleration(time_step, 'robot')
        closest_dist_metric = pm.ClosestDistance()
        closeness_to_goal_metric = pm.ClosenessToGoal(y_virtual_goal)
        time_to_reach_goal_metric = pm.TimeToReachGoal(time_step, self.robot.gx, self.human.gx)
        self.perf_metrics = [acceleration_metric_human, acceleration_metric_robot,
                            closest_dist_metric, closeness_to_goal_metric,
                            time_to_reach_goal_metric]

        obs, _ = self.env.reset()
        self.robot.set_vh_desired(obs)
        robot_action = self.robot.policy.predict(obs)
        obs['robot vel'] = np.array(robot_action)
        human_action = self.human.get_velocity()
        for metric in self.perf_metrics:
            metric.add(obs)

        done = False
        while not done:
            action = {
                "robot vel": np.array(robot_action),
                "human vel": np.array(human_action)
            }
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.robot.set_vh_desired(obs)
            # obs["human vel"] = np.array([-1.0, 0.])
            robot_action = self.robot.choose_action(obs)
            obs['robot vel'] = np.array(robot_action)
            for metric in self.perf_metrics:
                metric.add(obs)
            # Update the observation to include the current velocity of the robot
            human_action = self.human.choose_action(obs)

            # Estimate the value of alpha
            alpha_hat = estimate_alpha((-1.0, 0), human_action,
                                        self.robot.policy.invorca.vh_new,
                                        self.robot.policy.collision_responsibility,
                                        tuple(self.robot.policy.invorca.u))
            # print(alpha_hat)

        if self.env.unwrapped.collision:
            print("Collision happened at frame(s)", self.env.unwrapped.collision_frames)

        self.env.render()

        for metric in self.perf_metrics:
            print(metric.name, metric.get_metric())


def configure_human(policy_name: str, policy_config: str,
                    env_config: str, time_horizon: float | None = None,
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
        alpha = toml.load(env_config)['human']['collision_responsibility']
        human_policy.set_collision_responsiblity(alpha)
        if time_horizon is not None:
            human_policy.time_horizon = time_horizon
        if alpha is not None:
            human_policy.set_collision_responsiblity(alpha)

    elif policy_name == 'social_force':
        human_policy = SocialForce(human.time_step)
        human_policy.configure(os.path.join('sim', 'config', 'policy.toml'))
    else:
        raise ValueError(f"Unknown human policy {policy_name}.",
                          "Use one of 'orca' or 'social_force'.")

    human.set_policy(human_policy)
    return human


def configure_robot(policy_name: str, policy_config: str,
                    env_config: str, time_horizon: float | None = None,
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
            robot_policy_str: str = 'inverse_orca'):
    """
    A helper function to set up and run the simulation according to the desired parameters 
    supplied to it
    """

    # Configure the environment
    env_config = os.path.join('.', 'sim', 'config', 'env.toml')

    # Policy config
    policy_config = os.path.join('.', 'sim', 'config', 'policy.toml')

    # Configure the human
    human = configure_human(human_policy_str, policy_config, env_config,
                            time_horizon_human, alpha)

    # Configure the robot
    robot = configure_robot(robot_policy_str, policy_config, env_config,
                            time_horizon_robot, max_speed_robot)

    env = gym.make('HallwayScene-v0')
    # Set these in the environment
    env.unwrapped.set_human(human)
    env.unwrapped.set_robot(robot)
    env.unwrapped.configure(env_config, save_anim, render_mode)

    if out_fname is not None:
        env.unwrapped.set_output_filename(out_fname)

    num_failed = 0
    env.reset(seed=100)

    # Configure the performance metrics
    acceleration_metric_human = pm.CumulativeAcceleration(robot.time_step, 'human')
    acceleration_metric_robot = pm.CumulativeAcceleration(robot.time_step, 'robot')
    closest_dist_metric = pm.ClosestDistance()
    closeness_to_goal_metric = pm.ClosenessToGoal(robot.y_virtual_goal)
    time_to_reach_goal_metric = pm.TimeToReachGoal(robot.time_step, robot.gx, human.gx)
    perf_metrics = [acceleration_metric_human, acceleration_metric_robot,
                    closest_dist_metric, closeness_to_goal_metric,
                    time_to_reach_goal_metric]

    for _ in tqdm(range(num_runs)):
        try:
            obs, _ = env.reset()
            robot.set_vh_desired(obs)
            robot_action = robot.policy.predict(obs)
            obs['robot vel'] = np.array(robot_action)
            human_action = human.get_velocity()
            for metric in perf_metrics:
                metric.add(obs)

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
                for metric in perf_metrics:
                    metric.add(obs)
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

            for metric in perf_metrics:
                print(metric.name, metric.get_metric())

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
    parser.add_argument('--no-save-anim', action='store_false',
                        help="Add this flag to not save the animation/plot (default: False)")

    parser.add_argument('--render-mode', type=str, default='human',
                        help='The mode in which to render (default: human)',
                        choices=['human', 'static', 'debug'])

    parser.add_argument('--num-runs', type=int, default=1,
                        help='The number of times the simulation should be run (default: 1)')

    parser.add_argument('--tau-robot', type=int, default=6,
                        help='the planning time horizon for the robot (default: 6)')

    parser.add_argument('--tau-human', type=int, default=6,
                        help='the planning time horizon for the human (default: 6)')

    parser.add_argument('--human-policy', default='orca', choices=['orca', 'social_force'],
                        help='The human policy (default: orca)')

    parser.add_argument('--robot-policy', type=str, default='inverse_orca',
                        choices=['inverse_orca', 'weighted_sum'],
                        help='The robot policy (default: inverse_orca)')

    args = parser.parse_args()

    # Initialize the simulation runner
    sim_runner = SimulationRunner()

    # Get the config files
    env_config = os.path.join('sim', 'config', 'env.toml')
    policy_config = os.path.join('sim', 'config', 'policy.toml')
    sim_config = os.path.join('sim', 'config', 'sim.toml')

    # configure the simulation runner
    sim_runner.configure_from_file(sim_config, env_config, policy_config)

    # Update the configurations as needed
    sim_runner.config['sim']['save_anim'] = args.no_save_anim
    sim_runner.config['sim']['render_mode'] = args.render_mode
    sim_runner.config['sim']['num_runs'] = args.num_runs
    sim_runner.config['sim']['robot_policy'] = args.robot_policy
    sim_runner.config['sim']['human_policy'] = args.human_policy

    sim_runner.config['policy']['inverse_orca']['time_horizon'] = args.tau_robot
    sim_runner.config['policy']['orca']['time_horizon'] = args.tau_human

    sim_runner.setup_environment()
    sim_runner.run_sim()

    # num_failed = run_sim(args.render_mode, args.no_save_anim, args.num_runs,
    #                      time_horizon_robot=args.tau_robot,
    #                      time_horizon_human=args.tau_human,
    #                      human_policy_str=args.human_policy,
    #                      robot_policy_str=args.robot_policy)

    # print(f"(failed/total) = ({num_failed}/{args.num_runs})")

if __name__ == "__main__":
    main()
