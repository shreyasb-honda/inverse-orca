"""
Plots and saves random trajectories from the saved simulation data
"""

import os
from os import path
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_explorer import DataReader
from run_effect_plotter import get_exp_dir

def main():
    """
    The main function
    """
    sim_type = 'overtaking'
    render_mode = 'static'
    num_trajectories = 10
    policy_combos = [['orca', 'inverse'],
                     ['orca', 'weighted'],
                     ['sf', 'inverse'],
                     ['sf', 'weighted']]

    # policy_combos = [['sf', 'inverse'],
    #                  ['sf', 'weighted']]

    weights = [0.2, 0.5, 0.8]
    effects = ['alpha', 'time_horizon', 'max_speed']
    exp_dirs = []
    for policy_combo in policy_combos:
        if policy_combo[0] == 'orca':
            if policy_combo[1] == 'inverse':
                for effect in effects:
                    exp_dirs.append(get_exp_dir(policy_combo, effect, sim_type=sim_type))
            else:
                for weight in weights:
                    for effect in effects:
                        exp_dirs.append(get_exp_dir(policy_combo, effect,
                                                    weight, sim_type=sim_type))
        else:
            if policy_combo[1] == 'inverse':
                for effect in effects[1:]:
                    exp_dirs.append(get_exp_dir(policy_combo, effect, sim_type=sim_type))
            else:
                for weight in weights:
                    for effect in effects[1:]:
                        exp_dirs.append(get_exp_dir(policy_combo, effect,
                                                    weight, sim_type=sim_type))
    alpha_values = [round(0.1 * i, 1) for i in range(1, 11)]
    tau_values = [i+1 for i in range(10)]
    max_speed_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                        0.8, 1.0, 1.5, 2.0, 2.5, 3.0]

    for exp_dir in tqdm(exp_dirs):
        effect = str(exp_dir).rsplit('/', maxsplit=1)[-1].split('_')
        if len(effect) < 3:
            effect = effect[0]
        else:
            effect = f'{effect[0]}_{effect[1]}'
        data_reader = DataReader(exp_dir, effect)
        if effect == 'alpha':
            vals = alpha_values
        elif effect == 'time_horizon':
            vals = tau_values
        elif effect == 'max_speed':
            vals = max_speed_values

        for val in vals:
            # print(val)
            if render_mode == 'static':
                for i in range(num_trajectories):
                    fig, ax = data_reader.plot_random_trajectory(val, render_mode)
                    save_dir = data_reader.get_save_dir(sim_type=sim_type)
                    save_dir = path.join(save_dir, 'trajectories')
                    if not path.exists(save_dir):
                        os.makedirs(save_dir)
                    filepath = path.join(save_dir, f'{effect}={val}({i+1}).png')
                    plt.savefig(filepath)
                    plt.close(fig)


if __name__ == '__main__':
    main()
