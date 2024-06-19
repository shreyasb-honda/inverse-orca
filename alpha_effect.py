"""
Runs the code to check the effect of alpha
"""
import numpy as np
from run_sim import run_sim


def main():

    # Alpha from 0.1 to 1.0
    alphas = np.linspace(0.1, 1.0, 10)
    render_mode = "human"
    max_speed_robot = 2.0

    for alpha in alphas:
        run_sim(render_mode, alpha=alpha,
                max_speed_robot=max_speed_robot, 
                out_fname=f'alpha-{alpha:.2f}')


if __name__ == "__main__":
    main()
