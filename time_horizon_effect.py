"""
Runs the code to check the effect of differing planning horizon
for the robot and the human
"""
import os
from typing import List, Dict
from run_sim import SimulationRunner

def only_robot(time_horizons: List[int], 
               env_config: str, sim_config: str, policy_config: str):

    """
    Run simulations by varying the robot's planning time horizon
    """

    for tau in time_horizons:
        # print("Robot Tau:", tau)
        sim_runner = SimulationRunner()
        # configure the simulation runner
        sim_runner.configure_from_file(sim_config=sim_config,
                                       env_config=env_config,
                                       policy_config=policy_config)
        sim_runner.config['sim']['render_mode'] = None
        sim_runner.config['policy']['inverse_orca']['time_horizon'] = tau
        sim_runner.setup_environment()
        sim_runner.run_sim()

def robot_and_human(time_horizons: Dict[str, List[int]],
                    env_config: str, sim_config: str, policy_config: str):
    """
    Change both the human's and the robot's planning horizons
    """
    list1 = time_horizons['human']
    list2 = time_horizons['robot']

    for tau1 in list1:
        for tau2 in list2:
            sim_runner = SimulationRunner()
            # configure the simulation runner
            sim_runner.configure_from_file(sim_config=sim_config,
                                        env_config=env_config,
                                        policy_config=policy_config)
            sim_runner.config['sim']['render_mode'] = None
            sim_runner.config['policy']['orca']['time_horizon'] = tau1
            sim_runner.config['policy']['inverse_orca']['time_horizon'] = tau2
            sim_runner.setup_environment()
            sim_runner.run_sim()


def main():
    """
    Just the main function
    """
    # Config files
    config_directory = os.path.join('sim', 'config')
    env_config = os.path.join(config_directory, 'env.toml')
    sim_config = os.path.join(config_directory, 'sim.toml')
    policy_config = os.path.join(config_directory, 'policy.toml')

    # Only change the robot's planning time horizon
    time_horizons = [i+1 for i in range(10)]
    only_robot(time_horizons, env_config, sim_config, policy_config)

    # Change the robot's and the human's planning time horizons
    # time_horizons = {'human': [], 'robot': []}
    # tau_list = [2*(i+1) for i in range(7)]
    # for tau in tau_list:
    #     time_horizons['human'].append(tau)
    #     time_horizons['robot'].append(tau)

    # robot_and_human(time_horizons, env_config, sim_config, policy_config)


if __name__ == "__main__":
    main()
