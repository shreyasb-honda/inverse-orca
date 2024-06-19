"""
Runs the code to check the effect of differing planning horizon
for the robot and the human
"""
from run_sim import run_sim


def main():

    alpha = 1.0
    render_mode = "human"
    max_speed = 2.0

    robot_horizons = list(range(2, 9))
    human_horizon = 6

    for robot_horizon in robot_horizons:
        run_sim(render_mode, alpha=alpha,
                max_speed_robot=max_speed,
                time_horizon_robot=robot_horizon,
                time_horizon_human=human_horizon,
                out_fname=f'robot-time-horizon-{robot_horizon}')


if __name__ == "__main__":
    main()
