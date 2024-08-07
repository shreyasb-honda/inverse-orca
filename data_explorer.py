"""
Classes for reading the stored simulation data
"""
import os
from math import sqrt
import pickle
import toml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sim.renderer import Renderer
sns.set_theme(context='talk')


class DataReader:
    """
    Takes a parent directory and reads the relevant information from it
    """

    def __init__(self, parent_dir: str, indep_var: str) -> None:
        self.parent_dir = parent_dir
        self.data = {}
        self.indep_var = indep_var
        self.env_config = None
        self.policy_config = None
        self.selected_runs = []

    def read_data(self):
        """
        Reads and stores the data
        """
        indep_var = self.indep_var
        experiments = os.listdir(self.parent_dir)  # Will contain one folder for each experiment

        for experiment in experiments:
            # Load the configuration
            env_config = toml.load(os.path.join(self.parent_dir, experiment, 'env.toml'))
            # sim_config = toml.load(os.path.join(self.parent_dir, experiment, 'sim.toml'))
            policy_config = toml.load(os.path.join(self.parent_dir, experiment, 'policy.toml'))
            if indep_var == "alpha":
                alpha = env_config['human']['collision_responsibility']
                perf_file = os.path.join(self.parent_dir, experiment, 'perf_summary.pkl')
                with open(perf_file, 'rb') as f:
                    self.data[alpha] = pickle.load(f)
            elif indep_var == 'max_speed':
                max_speed = env_config['robot']['max_speed']
                perf_file = os.path.join(self.parent_dir, experiment, 'perf_summary.pkl')
                with open(perf_file, 'rb') as f:
                    self.data[max_speed] = pickle.load(f)
            elif indep_var == 'time_horizon':
                time_horizon = policy_config['inverse_orca']['time_horizon']
                perf_file = os.path.join(self.parent_dir, experiment, 'perf_summary.pkl')
                with open(perf_file, 'rb') as f:
                    self.data[time_horizon] = pickle.load(f)

    def get_list(self, metric_name: str):
        """
        Gets the dependent variable as a list
        """
        indep_var = []
        dep_var_mean = []
        dep_var_std = []
        for var, perf_summary in self.data.items():
            indep_var.append(var)
            # print(perf_summary['mean'].keys())
            dep_var_mean.append(perf_summary['mean'][metric_name])
            dep_var_std.append(perf_summary['std'][metric_name])

        indep_var = np.array(indep_var)
        dep_var_mean = np.array(dep_var_mean)
        dep_var_std = np.array(dep_var_std)

        indices = np.argsort(indep_var)
        indep_var = indep_var[indices]
        dep_var_mean = dep_var_mean[indices]
        dep_var_std = dep_var_std[indices]

        return indep_var, dep_var_mean, dep_var_std

    def __plot(self, metric_name: str, ylabel: str):
        fig, ax = plt.subplots()
        indep_var, dep_var_mean, dep_var_std = self.get_list(metric_name)
        width = 0.8 * (indep_var[1] - indep_var[0])
        ax.scatter(indep_var, dep_var_mean)
        yerr = dep_var_std / np.sqrt(dep_var_std.shape[0])
        ax.errorbar(indep_var, dep_var_mean, yerr=yerr)

        # ax.bar(indep_var, dep_var_mean,
        #        yerr=dep_var_std / np.sqrt(dep_var_mean.shape[0]),
        #        width=width, capsize=5.)
        ax.set_title(metric_name)
        ax.set_xlabel(self.indep_var)
        ax.set_ylabel(ylabel)
        # if "Path efficiency" in metric_name:
        #     ax.set_ylim([0.95, 1.1])
        fig.tight_layout()

        return fig, ax

    def get_save_dir(self, policy_combo=None, weight=None, sim_type=None):
        """
        Gets the path to the directory where the plots need to be saved
        """
        dir_names = str(self.parent_dir).split('/')
        if policy_combo is None:
            policy_str = dir_names[2]
            split = policy_str.split('-')
            policy_combo = [split[1], split[-1]]
            if split[-1] == "weighted":
                weight = float(dir_names[3].split('-')[-1])

        save_dir = os.path.join('media', 'effect-plots')

        if sim_type is not None:
            save_dir = os.path.join(save_dir, sim_type)

        save_dir = os.path.join(save_dir, f'human-{policy_combo[0]}-robot-{policy_combo[1]}')
        if weight is not None:
            save_dir = os.path.join(save_dir, f"weight-{weight:.1f}")

        save_dir = os.path.join(save_dir, f"{self.indep_var}_effect")

        return save_dir

    def plot_effect(self, policy_combo=None, weight=None, sim_type=None):
        """
        Plots the summary for a set of experiments
        """
        save_dir = self.get_save_dir(policy_combo, weight, sim_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig, ax = self.__plot('Average acceleration human', r'Acceleration $(/s^2)$')
        filename = os.path.join(save_dir, 'avg-acc-human.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Average acceleration robot', r'Acceleration $(/s^2)$')
        filename = os.path.join(save_dir, 'avg-acc-robot.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Closest Distance', 'Distance')
        filename = os.path.join(save_dir, 'closest-distance.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Minimum y_dist', 'Distance')
        filename = os.path.join(save_dir, 'min-y-distance.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('x-coordinate at goal', 'x-coordinate')
        filename = os.path.join(save_dir, 'x-coord-at-goal.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Virtual goal reached', 'ratio')
        filename = os.path.join(save_dir, 'ratio-virtual-goal-reached.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Time to goal human', r'$s$')
        filename = os.path.join(save_dir, 'time-to-goal-human.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Time to goal robot', r'$s$')
        filename = os.path.join(save_dir, 'time-to-goal-robot.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Path efficiency human', 'ratio')
        filename = os.path.join(save_dir, 'path-efficiency-human.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Path efficiency robot', 'ratio')
        filename = os.path.join(save_dir, 'path-efficiency-robot.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Average jerk human', r'$/s^3$')
        filename = os.path.join(save_dir, 'avg-jerk-human.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Average jerk robot', r'$/s^3$')
        filename = os.path.join(save_dir, 'avg-jerk-robot.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Path irregularity human', 'rad')
        filename = os.path.join(save_dir, 'path-irregularity-human.png')
        plt.savefig(filename)
        plt.close(fig)

        fig, ax = self.__plot('Path irregularity robot', 'rad')
        filename = os.path.join(save_dir, 'path-irregularity-robot.png')
        plt.savefig(filename)
        plt.close(fig)

        # plt.show()

    def print(self):
        """
        Prints the metrics to stdout
        """
        for var, perf_summary in self.data.items():
            print(f"{self.indep_var}: {var:.2f}")
            for key, val in perf_summary['mean'].items():
                val_s = f"{val:.3f},"
                val2_s = f"{perf_summary['std'][key]:.3f}"
                print(f"{key:<35}: Average {val_s:<10} Std {val2_s:<10}")

    def plot_random_trajectory(self, indep_var_val, mode='static'):
        """
        Plots a random trajectory for the given value
        of the independent variable
        :param indep_var_value - the value of the independent variable for which 
                                 a random trajectory is to be plotted
        """
        if self.selected_runs is None:
            self.selected_runs = set()

        experiments = os.listdir(self.parent_dir)
        for experiment in experiments:
            env_config = toml.load(os.path.join(self.parent_dir, experiment, 'env.toml'))
            # sim_config = toml.load(os.path.join(self.parent_dir, experiment, 'sim.toml'))
            policy_config = toml.load(os.path.join(self.parent_dir, experiment, 'policy.toml'))
            if self.indep_var == "alpha":
                val = env_config['human']['collision_responsibility']
            elif self.indep_var == 'max_speed':
                val = env_config['robot']['max_speed']
            elif self.indep_var == 'time_horizon':
                val = policy_config['inverse_orca']['time_horizon']
            if round(val, 1) == indep_var_val:
                chosen_exp = experiment
                break

        # Grab a random folder from the subfolders
        runs = os.listdir(os.path.join(self.parent_dir, chosen_exp))
        chosen_run_idx = np.random.choice(len(runs))
        # while chosen_run_idx in self.selected_runs:
        #     chosen_run_idx = np.random.choice(len(runs))
        # self.selected_runs.add(chosen_run_idx)
        chosen_run = runs[chosen_run_idx]
        while 'toml' in chosen_run or 'pkl' in chosen_run:
            chosen_run_idx = np.random.choice(len(runs))
            chosen_run = runs[chosen_run_idx]

        run_dir = os.path.join(self.parent_dir, chosen_exp, chosen_run)
        with open(os.path.join(run_dir, 'obs.pkl'), 'rb') as f:
            observations = pickle.load(f)

        hallway_length = env_config['env']['hallway_length']
        hallway_width = env_config['env']['hallway_width']
        robot_draw_params = {"radius": observations[0]['robot rad'], "color": "tab:blue"}
        human_draw_params = {"radius": observations[0]['human rad'], "color": 'tab:red'}
        hallway_dimensions = {"length": hallway_length, "width": hallway_width}
        renderer = Renderer(hallway_dimensions, robot_draw_params, human_draw_params,
                            env_config['env']['time_step'])
        d_virtual_goal = env_config['env']['d_virtual_goal']
        y_virtual_goal = env_config['env']['y_virtual_goal']
        renderer.set_goal_params(d_virtual_goal, y_virtual_goal)

        goal_frame = self.get_goal_frame(observations, env_config)
        renderer.set_observations(observations, goal_frame)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_aspect('equal')
        if mode == 'static':
            return renderer.static_plot(fig, ax)
        elif mode == 'human':
            return renderer.animate(fig, ax)

    def get_goal_frame(self, observations, env_config):
        """
        Returns the frame number of the frame in which 
        the human reaches the virtual goal line
        """
        frame = 0
        for obs in observations:
            if obs['human pos'][1] <= env_config['env']['y_virtual_goal']:
                break
            frame += 1
        return frame


def main():
    """
    The main function
    """
    policy_combo = ['orca', 'inverse']
    # policy_combo = ['orca', 'weighted']
    # policy_combo = ['sf', 'inverse']
    # policy_combo = ['sf', 'weighted']
    indep_var = 'alpha'
    # indep_val = 1.0
    # indep_var = 'max_speed'
    # indep_val = 1.5
    # indep_var = 'time_horizon'
    # indep_val = 10
    weight = None
    # weight = 0.5
    parent_dir = os.path.join('data',
                              f'human-{policy_combo[0]}-robot-{policy_combo[1]}')

    # render_mode = 'static'
    render_mode = 'human'
    if weight is not None:
        parent_dir = os.path.join(parent_dir, f'weight-{weight:.1f}')

    parent_dir = os.path.join(parent_dir, f'{indep_var}_effect')

    data_explorer = DataReader(parent_dir=parent_dir, indep_var=indep_var)
    data_explorer.read_data()
    data_explorer.plot_effect(policy_combo, weight)
    # out = data_explorer.plot_random_trajectory(indep_val, render_mode)
    # plt.show()


def time_horizon_effect():
    """
    Plots the data from the independent variation of the planning time
    horizons for the human and the robot
    For each pair of (tau_human, tau_robot), we have one value of the performance metric
    We can show the plots as an image (plt.imshow)
    """
    parent_dir = os.path.join('data', 'human-orca-robot-inverse', 'both_time_horizons')
    experiments = os.listdir(parent_dir)
    data = {}
    metric_names = None
    for experiment in experiments:
        policy_config = toml.load(os.path.join(parent_dir, experiment, 'policy.toml'))
        tau_robot = policy_config['inverse_orca']['time_horizon']
        tau_human = policy_config['orca']['time_horizon']
        perf_file = os.path.join(parent_dir, experiment, 'perf_summary.pkl')
        with open(perf_file, 'rb') as f:
            perf_summary = pickle.load(f)
        avg_perf = perf_summary['mean']
        if metric_names is None:
            metric_names = list(avg_perf.keys())
        data[(tau_human, tau_robot)] = avg_perf

    num_data_points = len(data.keys())
    num_rows = int(sqrt(num_data_points))
    for metric_name in metric_names:
        data_array = np.zeros((num_rows, num_rows))
        for (human_tau, robot_tau), val in data.items():
            idx_1 = (human_tau - 2) // 2
            idx_2 = (robot_tau - 2) // 2
            data_array[idx_1, idx_2] = val[metric_name]

        fig, ax = plt.subplots()
        im = ax.imshow(data_array, origin='lower')
        ax.set_title(metric_name)
        ax.set_xlabel('robot tau')
        ax.set_ylabel('human tau')
        cbar = ax.figure.colorbar(im, ax=ax)

        col_labels = [2*(i+1) for i in range(7)]
        ax.set_xticks(np.arange(data_array.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data_array.shape[0]), labels=col_labels)
        ax.grid(False)

        fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
    # time_horizon_effect()
