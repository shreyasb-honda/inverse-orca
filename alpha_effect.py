"""
Runs the code to check the effect of alpha
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

    alpha_start = 0.1
    alpha_end = 1.0
    alpha_step = 0.1
    num_alphas = int((alpha_end - alpha_start) / alpha_step + 1)
    alphas = [alpha_start + alpha_step * i for i in range(num_alphas)]

    for alpha in alphas:
        print("Alpha", alpha)
        sim_runner = SimulationRunner()
        # configure the simulation runner
        sim_runner.configure_from_file(sim_config=sim_config,
                                       env_config=env_config,
                                       policy_config=policy_config)
        sim_runner.config['sim']['render_mode'] = None
        sim_runner.config['env']['human']['collision_responsibility'] = alpha
        sim_runner.setup_environment()
        sim_runner.run_sim()

if __name__ == "__main__":
    main()
