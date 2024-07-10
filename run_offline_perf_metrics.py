"""
Runs the performance metrics on the saved simulation data
"""

from os import path
from offline_perf_metrics import OfflineMetrics
import sim.performance_metrics as pm

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
        print(exp_dir)
        offline_metrics = OfflineMetrics(exp_dir)
        # List of metrics to run
        avg_acc_hum = pm.AverageAcceleration(agent='human')
        offline_metrics.add(avg_acc_hum)
        avg_acc_rob = pm.AverageAcceleration(agent='robot')
        offline_metrics.add(avg_acc_rob)
        closest_distance = pm.ClosestDistance()
        offline_metrics.add(closest_distance)
        y_virtual_goal = 1.0
        close_to_goal = pm.ClosenessToGoal(y_virtual_goal)
        offline_metrics.add(close_to_goal)
        time_to_goal = pm.TimeToReachGoal(0.25, 15., 0.)
        offline_metrics.add(time_to_goal)
        path_eff_hum = pm.PathEfficiency('human', 15.)
        offline_metrics.add(path_eff_hum)
        path_eff_rob = pm.PathEfficiency('robot', 15.)
        offline_metrics.add(path_eff_rob)
        avg_jerk_hum = pm.AverageJerk(agent='human')
        offline_metrics.add(avg_jerk_hum)
        avg_jerk_rob = pm.AverageJerk(agent='robot')
        offline_metrics.add(avg_jerk_rob)
        path_irr_hum = pm.PathIrregularity(goal=(0.,0.), agent='human')
        offline_metrics.add(path_irr_hum)
        path_irr_rob = pm.PathIrregularity(goal=(15., 0.), agent='robot')
        offline_metrics.add(path_irr_rob)

        offline_metrics.run()


if __name__ == "__main__":
    main()
