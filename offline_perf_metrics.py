"""
Runs the performance metrics on the saved simulation data
"""

import os
import pickle
import toml
import numpy as np
import sim.performance_metrics as pm


class OfflineMetrics:
    """
    Run one or multiple performance metrics on the saved simulation data.
    Save the performance results in the simulation run folder under perf.pkl
    Save the performance metrics summary for an experiment under perf_summary.pkl
    in the experiment folder
    """

    def __init__(self, exp_dir: str) -> None:
        """
        :param exp_dir - the directory containing one experiment - changing one parameter 
                         while keeping the others constant (e.g. alpha_effect)
        """
        self.exp_dir = exp_dir
        self.metrics = []
        self.single_sim_performance = {}
        self.performance_lists = {}

    def add(self, metric: pm.PerformanceMetric):
        """
        Adds a performance metric to the list of performance metrics to 
        be computed on the offline data
        """
        self.metrics.append(metric)

    def run(self):
        """
        Runs the offline perfomance metric computation
        """
        assert len(self.metrics) > 0, "Please add at least one performance metric"

        # Get the list of directories in which the variable is fixed
        fixed_val_dirs = os.listdir(self.exp_dir)
        for fixed_val_dir in fixed_val_dirs:
            # Reset the data variable
            self.performance_lists = {}
            dir_path = os.path.join(self.exp_dir, fixed_val_dir)
            summary_file = os.path.join(self.exp_dir, fixed_val_dir,
                                        'perf_summary.pkl')
            env_config = os.path.join(dir_path, 'env.toml')
            env_config = toml.load(env_config)
            policy_config = os.path.join(dir_path, 'policy.toml')
            policy_config = toml.load(policy_config)
            sim_config = os.path.join(dir_path, 'sim.toml')
            sim_config = toml.load(sim_config)
            run_dirs = os.listdir(dir_path)

            # For each run
            for run_dir in run_dirs:
                if '.pkl' in run_dir:
                    continue
                if '.toml' in run_dir:
                    continue
                run_path = os.path.join(dir_path, run_dir)
                # Read the observations for each run
                obs_file = os.path.join(run_path, 'obs.pkl')
                with open(obs_file, 'rb') as f:
                    obs = pickle.load(f)
                # For each metric
                for metric in self.metrics:
                    # Reset it
                    metric = metric.reset()
                    # For a single observation in this run
                    for single_obs in obs:
                        # Add it to the metric calculator
                        acc = 'Average acceleration human'
                        jerk = 'Average jerk human'
                        if metric.name in (acc, jerk):
                            human_done = single_obs['human pos'][0] - single_obs['human rad'] <= 0
                            metric.agent_done(human_done)
                        acc = 'Average acceleration robot'
                        jerk = 'Average jerk robot'
                        if metric.name in (acc, jerk):
                            robot_done = single_obs['robot pos'][0] + single_obs['robot rad'] >= 15
                            metric.agent_done(robot_done)
                        metric.add(single_obs)
                    # Once all observations are added,
                    # Set the value of the performance metric for this run
                    self.single_sim_performance[metric.name] = metric.get_metric()
                    # Add the performance metric for this run
                    if metric.name not in self.performance_lists:
                        self.performance_lists[metric.name] = [metric.get_metric()]
                    else:
                        self.performance_lists[metric.name].append(metric.get_metric())

                    # Save the performance metric data for this run
                    perf_file = os.path.join(run_path, 'perf.pkl')
                    with open(perf_file, 'wb') as f:
                        pickle.dump(self.single_sim_performance, f)

            # Once all runs performance data has been added to the lists,
            # Compute the mean and average of each performance metric
            metrics_avg = {}
            metrics_std = {}
            performance_summary = {}
            for metric in self.metrics:
                metrics_array = np.array(self.performance_lists[metric.name], dtype=float)
                key = metric.name
                if key == 'Closeness to goal':
                    metrics_avg['Minimum y_dist'] = np.average(metrics_array[:, 0])
                    metrics_avg['x-coordinate at goal'] = np.nanmean(metrics_array[:, 1])
                    metrics_avg['Virtual goal reached'] = np.average(metrics_array[:, 2])

                    metrics_std['Minimum y_dist'] = np.std(metrics_array[:, 0])
                    metrics_std['x-coordinate at goal'] = np.nanstd(metrics_array[:, 1])
                    metrics_std['Virtual goal reached'] = np.std(metrics_array[:, 2])
                elif key == 'Time to reach goal':
                    metrics_avg['Time to goal human'] = np.average(metrics_array[:, 0])
                    metrics_avg['Time to goal robot'] = np.average(metrics_array[:, 1])

                    metrics_std['Time to goal human'] = np.std(metrics_array[:, 0])
                    metrics_std['Time to goal robot'] = np.std(metrics_array[:, 1])
                else:
                    metrics_avg[key] = np.average(metrics_array)
                    metrics_std[key] = np.std(metrics_array)

            performance_summary['mean'] = metrics_avg
            performance_summary['std'] = metrics_std
            for k, v in performance_summary['mean'].items():
                print(f"{k:<30} {v:.2f} {performance_summary['std'][k]:.2f}")
            with open(summary_file, 'wb') as f:
                pickle.dump(performance_summary, f)

def main():
    """
    The main function
    """
    # policy_combo = ['orca', 'inverse']
    # policy_combo = ['orca', 'weighted']
    # policy_combo = ['sf', 'inverse']
    policy_combo = ['sf', 'weighted']
    # effect = 'alpha_effect'
    # effect = 'time_horizon_effect'
    effect = 'max_speed_effect'

    weight = 0.8
    exp_dir = os.path.join('data',
                           f'human-{policy_combo[0]}-robot-{policy_combo[1]}')

    if weight is not None:
        exp_dir = os.path.join(exp_dir, f"weight-{weight:.1f}")

    exp_dir = os.path.join(exp_dir, effect)

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
