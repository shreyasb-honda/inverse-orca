"""
Runs the code to check the effect of the robot's max speed
"""
import os
import matplotlib.pyplot as plt
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

    sim_runner = SimulationRunner()
    # configure the simulation runner
    sim_runner.configure_from_file(sim_config=sim_config,
                                    env_config=env_config,
                                    policy_config=policy_config)
    # sim_runner.config['sim']['render_mode'] = None
    sim_runner.setup_environment()
    sim_runner.run_sim()

    plt.show()

if __name__ == "__main__":
    main()
