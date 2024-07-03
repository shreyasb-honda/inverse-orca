"""
Runs the performance metrics on the saved simulation data
"""

import os
import pickle
import toml
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
