"""
Runs the code to check the effect of differing planning horizon
for the robot and the human
"""
import os
from run_sim import SimulationRunner


def main():
    """
    Just the main function
    """

    # Config files
    config_directory = os.path.join('sim', 'config')
    env_config = os.path.join(config_directory, 'env.toml')
    sim_config = os.path.join(config_directory, 'sim.toml')
    policy_config = os.path.join(config_directory, 'policy.toml')
    time_horizons = [i+1 for i in range(10)]

    for tau in time_horizons:
        print("Robot Tau:", tau)
        sim_runner = SimulationRunner()
        # configure the simulation runner
        sim_runner.configure_from_file(sim_config=sim_config,
                                       env_config=env_config,
                                       policy_config=policy_config)
        sim_runner.config['sim']['render_mode'] = None
        sim_runner.config['policy']['inverse_orca']['time_horizon'] = tau
        sim_runner.setup_environment()
        sim_runner.run_sim()

if __name__ == "__main__":
    main()
