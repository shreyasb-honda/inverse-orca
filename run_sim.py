"""
A general runner code for running the simulation
"""
import os
import datetime
import uuid
from argparse import ArgumentParser
import pickle
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
from policy.efficient_nudging import NaiveEfficientNudge, EfficientNudge
# from policy.utils.estimate_alpha import estimate_alpha


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
        self.metric_values = None
        self.metrics_avg = None
        self.metrics_std = None
        self.metrics_overall = None

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
        for key, value in self.config_files.items():
            self.config[key] = toml.load(value)

    def configure_human(self):
        """
        Configures the human agent for the environment
        """
        human = Human()
        human.configure(self.config['env'])
        human_policy = None
        if self.config['sim']['human_policy'] == 'orca':
            human_policy = Orca(human.time_step)
            human_policy.configure(self.config['policy'])
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
        robot.configure(self.config['env'])
        robot_policy = None
        if self.config['sim']['robot_policy'] == 'inverse_orca':
            robot_policy = InverseOrca(robot.time_step)
        elif self.config['sim']['robot_policy'] == 'weighted_sum':
            robot_policy = WeightedSum(robot.time_step)
        elif self.config['sim']['robot_policy'] == 'efficient_nudge':
            # robot_policy = NaiveEfficientNudge(robot.time_step)
            robot_policy = EfficientNudge(robot.time_step)
        else:
            raise ValueError(f"Unknown policy {self.config['sim']['robot_policy']} for the robot.",
                            "Use one of 'inverse_orca', 'weighted_sum', or 'efficient_nudge.")

        robot_policy.configure(self.config['policy'])
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
        if self.config['env']['env']['type'] == 'passing':
            env = gym.make('HallwayScene-v0')
        elif self.config['env']['env']['type'] == 'overtaking':
            env = gym.make('OvertakingScene-v0')

        # Set these in the environment
        env.unwrapped.set_human(self.human)
        env.unwrapped.set_robot(self.robot)
        env.unwrapped.configure(self.config['env'],
                                self.config['sim']['save_anim'],
                                self.config['sim']['render_mode'])
        self.env = env

        if self.config['sim']['out_fname'] != 'temp':
            env.unwrapped.set_output_filename(self.config['sim']['out_fname'])
        self.env.reset(seed=seed)

    def save_configs(self, experiment_directory: str):
        """
        Save the configuration file for the experiment
        """
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)
        # Save the configurations (once per set of runs)
        with open(os.path.join(experiment_directory, 'env.toml'), 'w', encoding='utf-8') as f:
            toml.dump(self.config['env'], f)
        with open(os.path.join(experiment_directory, 'policy.toml'), 'w', encoding='utf-8') as f:
            toml.dump(self.config['policy'], f)
        with open(os.path.join(experiment_directory, 'sim.toml'), 'w', encoding='utf-8') as f:
            toml.dump(self.config['sim'], f)

    def add_perf_metrics(self, save: bool, experiment_directory: str, run_name: str):
        """
        Adds the performance metrics to a list
        Saves the list to file if saving si enabled
        """
        if save:
            data_directory = os.path.join(experiment_directory, run_name)
            if not os.path.exists(data_directory):
                os.makedirs(data_directory)
            # Save the observations
            with open(os.path.join(data_directory, 'obs.pkl'), 'wb') as f:
                pickle.dump(self.env.unwrapped.observations, f)
        # Save the performance metrics
        for metric in self.perf_metrics:
            name = metric.name
            value = metric.get_metric()
            if self.metric_values is None:
                self.metric_values = {name: [value]}
            elif name not in self.metric_values:
                self.metric_values[name] = [value]
            else:
                self.metric_values[name].append(value)
        if save:
            with open(os.path.join(data_directory, 'perf.pkl'), 'wb') as f:
                current_metrics = {}
                for key, val in self.metric_values.items():
                    current_metrics[key] = val[-1]
                pickle.dump(current_metrics, f)

    def run_sim(self):
        """
        Runs the sim multiple times
        """
        save_data = self.config['sim']['save_data']
        experiment_directory = None
        run_name = None
        if save_data:
            ts = datetime.datetime.now().strftime("%m-%d %H-%M-%S")
            experiment_directory = os.path.join('data', ts)
            self.save_configs(experiment_directory)

        for _ in tqdm(range(self.config['sim']['num_runs'])):
            self.run_sim_once()
            if save_data:
                run_name = uuid.uuid4().hex

            self.add_perf_metrics(save_data, experiment_directory, run_name)

        self.generate_performance_summary(self.config['sim']['print_summary'])
        if save_data:
            self.metrics_overall = {"mean": self.metrics_avg, "std": self.metrics_std}
            with open(os.path.join(experiment_directory, 'perf_summary.pkl'), 'wb') as f:
                pickle.dump(self.metrics_overall, f)

    def generate_performance_summary(self, _print=True):
        """
        Generates the summary of the performance metrics over multiple simulation runs
        """
        self.metrics_avg = {}
        self.metrics_std = {}
        for key, val in self.metric_values.items():
            if key == 'Closeness to goal':
                metrics_array = np.array(val, dtype=float)
                self.metrics_avg['Minimum y_dist'] = np.average(metrics_array[:, 0])
                self.metrics_avg['x-coordinate at goal'] = np.nanmean(metrics_array[:, 1])
                self.metrics_avg['Virtual goal reached'] = np.average(metrics_array[:, 2])

                self.metrics_std['Minimum y_dist'] = np.std(metrics_array[:, 0])
                self.metrics_std['x-coordinate at goal'] = np.nanstd(metrics_array[:, 1])
                self.metrics_std['Virtual goal reached'] = np.std(metrics_array[:, 2])
            elif key == 'Time to reach goal':
                metrics_array = np.array(val, dtype=float)
                self.metrics_avg['Time to goal human'] = np.average(metrics_array[:, 0])
                self.metrics_avg['Time to goal robot'] = np.average(metrics_array[:, 1])

                self.metrics_std['Time to goal human'] = np.std(metrics_array[:, 0])
                self.metrics_std['Time to goal robot'] = np.std(metrics_array[:, 1])
            else:
                self.metrics_avg[key] = np.average(val)
                self.metrics_std[key] = np.std(val)

        if _print:
            print("Metrics:")
            for key, val in self.metrics_avg.items():
                print(f"{key:<20}: Average {val:.3f}, Std {self.metrics_std[key]: .3f}")

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
        direction = np.sign(self.robot.gx - obs['robot pos'][0])
        robot_action = self.robot.policy.predict(obs, direction)
        obs['robot vel'] = np.array(robot_action)
        human_action = self.human.get_velocity()
        acceleration_metric_human.agent_done(self.human.reached_goal())
        acceleration_metric_robot.agent_done(self.robot.reached_goal(direction))

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
            direction = np.sign(self.robot.gx - obs['robot pos'][0])
            robot_action = self.robot.choose_action(obs, direction)

            # This seems to happen when the desired
            # velocity has been achieved by the human
            if robot_action is None:
                robot_action = tuple(action['robot vel'])

            obs['robot vel'] = np.array(robot_action)
            acceleration_metric_human.agent_done(self.human.reached_goal())
            acceleration_metric_robot.agent_done(self.robot.reached_goal(direction))
            for metric in self.perf_metrics:
                metric.add(obs)
            # Update the observation to include the current velocity of the robot
            human_action = self.human.choose_action(obs)

            # Estimate the value of alpha
            # alpha_hat = estimate_alpha((-1.0, 0), human_action,
            #                             self.robot.policy.invorca.vh_new,
            #                             self.robot.policy.collision_responsibility,
            #                             tuple(self.robot.policy.invorca.u))

        if self.env.unwrapped.collision:
            print("Collision happened at frame(s)", self.env.unwrapped.collision_frames)

        self.env.render()


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
                        choices=['inverse_orca', 'weighted_sum', 'efficient_nudge'],
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

if __name__ == "__main__":
    main()
