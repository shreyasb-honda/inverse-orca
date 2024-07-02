"""
Classes for reading the stored simulation data
"""
import os
import pickle
import toml
import matplotlib.pyplot as plt


class DataReader:
    """
    Takes a parent directory and reads the relevant information from it
    """

    def __init__(self, parent_dir: str) -> None:
        self.parent_dir = parent_dir
        self.config = None
        self.perf_summary = None

    def read_data(self):
        """
        Reads and stores the data
        """
        self.config = {}
        self.config['env'] = toml.load(os.path.join(self.parent_dir, 'env.toml'))
        self.config['sim'] = toml.load(os.path.join(self.parent_dir, 'sim.toml'))
        self.config['policy'] = toml.load(os.path.join(self.parent_dir, 'policy.toml'))

        with open(os.path.join(self.parent_dir, 'perf_summary.pkl'), 'rb') as f:
            self.perf_summary = pickle.load(f)

        # print("Metrics:")
        # for key, val in self.perf_summary['mean'].items():
        #     val_s = f"{val:.3f},"
        #     val2_s = f"{self.perf_summary['std'][key]:.3f}"
        #     print(f"{key:<35}: Average {val_s:<10} Std {val2_s:<10}")

def main():
    """
    The main function
    """
    parent_dir = os.path.join('data', 'human-orca-robot-inverse', 'alpha_effect', '07-02 11-35-31')
    data_explorer = DataReader(parent_dir=parent_dir)

    data_explorer.read_data()


if __name__ == "__main__":
    main()
