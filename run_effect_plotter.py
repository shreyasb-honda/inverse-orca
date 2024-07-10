"""
Plots the effect of varying one parameter in the simulation 
on the various performance metrics considered
"""

from os import path
from data_explorer import DataReader

def get_exp_dir(policy_combo, effect, weight=None):
    """
    Returns the experiment directory
    """
    exp_dir = path.join('data',
                        f'human-{policy_combo[0]}-robot-{policy_combo[1]}')
    if weight is not None:
        exp_dir = path.join(exp_dir, f"weight-{weight:.1f}")

    exp_dir = path.join(exp_dir, f"{effect}_effect")

    return exp_dir

def main():
    """
    The main function
    """
    policy_combos = [['orca', 'inverse'],
                     ['orca', 'weighted'],
                     ['sf', 'inverse'],
                     ['sf', 'weighted']]

    weights = [0.2, 0.5, 0.8]

    effects = ['alpha', 'time_horizon', 'max_speed']
    exp_dirs = []
    for policy_combo in policy_combos:
        if policy_combo[0] == 'orca':
            if policy_combo[1] == 'inverse':
                for effect in effects:
                    exp_dirs.append(get_exp_dir(policy_combo, effect))
            else:
                for weight in weights:
                    for effect in effects:
                        exp_dirs.append(get_exp_dir(policy_combo, effect, weight))
        else:
            if policy_combo[1] == 'inverse':
                for effect in effects[1:]:
                    exp_dirs.append(get_exp_dir(policy_combo, effect))
            else:
                for weight in weights:
                    for effect in effects[1:]:
                        exp_dirs.append(get_exp_dir(policy_combo, effect, weight))

    for exp_dir in exp_dirs:
        effect = str(exp_dir).rsplit('/', maxsplit=1)[-1].split('_')
        if len(effect) < 3:
            effect = effect[0]
        else:
            effect = f'{effect[0]}_{effect[1]}'
        data_reader = DataReader(exp_dir, effect)
        data_reader.read_data()
        data_reader.plot_effect()

if __name__ == '__main__':
    main()
