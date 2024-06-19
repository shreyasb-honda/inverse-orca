"""
Runs the code to check the effect of robot's max speed
"""
from run_sim import run_sim


def main():

    alpha = 0.5
    render_mode = "human"
    max_speeds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for max_speed in max_speeds:
        run_sim(render_mode, alpha=alpha,
                max_speed_robot=max_speed, 
                out_fname=f'max-speed-{max_speed:.2f}')


if __name__ == "__main__":
    main()
