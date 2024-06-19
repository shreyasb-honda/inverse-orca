"""
Runs the code to check the effect of robot's max speed
"""
from run_sim import run_sim


def main():

    # Alpha from 0.1 to 1.0
    alpha = 1.0
    render_mode = "human"
    max_speeds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for max_speed in max_speeds:
        run_sim(render_mode, alpha=alpha,
                max_speed_robot=max_speed)


if __name__ == "__main__":
    main()
