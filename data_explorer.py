"""
Classes for reading the stored simulation data
"""
import os
import pickle
import toml
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk')


class DataReader:
    """
    Takes a parent directory and reads the relevant information from it
    """

    def __init__(self, parent_dir: str) -> None:
        self.parent_dir = parent_dir
        self.data = {}
        self.indep_var = None

    def read_data(self, indep_var: str):
        """
        Reads and stores the data
        """
        self.indep_var = indep_var
        experiments = os.listdir(self.parent_dir)  # Will contain one folder for each experiment

        for experiment in experiments:
            # Load the configuration
            env_config = toml.load(os.path.join(self.parent_dir, experiment, 'env.toml'))
            sim_config = toml.load(os.path.join(self.parent_dir, experiment, 'sim.toml'))
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
        dep_var = []
        for var, perf_summary in self.data.items():
            indep_var.append(var)
            dep_var.append(perf_summary['mean'][metric_name])

        return indep_var, dep_var

    def __plot(self, metric_name: str):
        fig, ax = plt.subplots()
        alphas, dep_var = self.get_list(metric_name)
        ax.scatter(alphas, dep_var)
        ax.set_title(metric_name)
        ax.set_xlabel(self.indep_var)

        return fig, ax


    def plot(self):
        """
        Plots the summary for a set of experiments
        """
        fig, ax = self.__plot('Cumulative Acceleration human')
        ax.set_ylabel(r'Acceleration $(/s^2)$')
        fig.tight_layout()

        fig, ax = self.__plot('Cumulative Acceleration robot')
        ax.set_ylabel(r'Acceleration $(/s^2)$')
        fig.tight_layout()

        fig, ax = self.__plot('Closest Distance')
        ax.set_ylabel('Distance')
        fig.tight_layout()

        fig, ax = self.__plot('Minimum y_dist')
        ax.set_ylabel('Distance')
        fig.tight_layout()

        fig, ax = self.__plot('x-coordinate at goal')
        ax.set_ylabel('x-coordinate')
        fig.tight_layout()

        fig, ax = self.__plot('Virtual goal reached')
        ax.set_ylabel('ratio')
        fig.tight_layout()

        fig, ax = self.__plot('Time to goal human')
        ax.set_ylabel(r'$s$')
        fig.tight_layout()

        fig, ax = self.__plot('Time to goal robot')
        ax.set_ylabel(r'$s$')
        fig.tight_layout()

        plt.show()


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

def main():
    """
    The main function
    """
    # parent_dir = os.path.join('data', 'human-orca-robot-inverse', 'alpha_effect')
    # parent_dir = os.path.join('data', 'human-orca-robot-inverse', 'max_speed_effect')
    parent_dir = os.path.join('data', 'human-orca-robot-inverse', 'time_horizon_effect')

    data_explorer = DataReader(parent_dir=parent_dir)

    data_explorer.read_data('time_horizon')
    data_explorer.plot()


if __name__ == "__main__":
    main()
