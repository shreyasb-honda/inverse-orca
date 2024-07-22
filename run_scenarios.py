"""
Runs the 6 possible scenarios in simulation
1. Passing scenario, the robot begins above the human
2. Passing scenario, the robot begins below the human
3. Overtaking scenario, the robot begins above and in front of the human
4. Overtaking scenario, the robot begins below and in front of the human
5. Overtaking scenario, the robot begins above and behind the human
6. Overtaking scenario, the robot begins below and behind the human
"""
import os
from argparse import ArgumentParser
from run_sim import SimulationRunner

def main():
    """
    The main function
    """
    parser = ArgumentParser()
    parser.add_argument('--human-policy', default='orca', choices=['orca', 'social_force'],
                        help='The human policy (default: orca)')
    args = parser.parse_args()

    sim_runner = SimulationRunner()
    # Get the config files
    env_config = os.path.join('sim', 'config', 'env.toml')
    policy_config = os.path.join('sim', 'config', 'policy.toml')
    sim_config = os.path.join('sim', 'config', 'sim.toml')

    # configure the simulation runner
    sim_runner.configure_from_file(sim_config, env_config, policy_config)
    cases = {'passing': [0, 1], 'overtaking': [0, 1, 2, 3]}

    # Update the configurations as needed
    sim_runner.config['sim']['human_policy'] = args.human_policy

    for case, case_nums in cases.items():
        for case_num in case_nums:
            sim_runner.config['env']['env']['type'] = case
            sim_runner.config['sim']['out_fname'] = f'{args.human_policy}-{case}-{case_num}'
            sim_runner.setup_environment(case=case_num)
            sim_runner.run_sim()


if __name__ == "__main__":
    main()
