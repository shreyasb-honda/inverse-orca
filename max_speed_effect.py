"""
Runs the code to check the effect of the robot's max speed
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
    max_speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]

    for max_speed in max_speeds:
        print("Max speed:", max_speed)
        sim_runner = SimulationRunner()
        # configure the simulation runner
        sim_runner.configure_from_file(sim_config=sim_config,
                                       env_config=env_config,
                                       policy_config=policy_config)
        sim_runner.config['sim']['render_mode'] = None
        sim_runner.config['env']['robot']['max_speed'] = max_speed
        sim_runner.setup_environment()
        sim_runner.run_sim()

if __name__ == "__main__":
    main()
