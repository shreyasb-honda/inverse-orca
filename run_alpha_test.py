"""
Runs the test for estimation of alpha
Currently succeeds whenever there is overlap between
the velocity obstacle and the relative velocities circle
If they do not intersect, then ORCA does something weird and
we get a bad estimate for alpha
"""

import argparse
import matplotlib.pyplot as plt
from tests.estimate_alpha_testing import *

def main():
    """
    the main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-num', type=int, default=0,
                        help='Index of the test to run (default: 0)')
    args = parser.parse_args()

    test_num = args.test_num
    if test_num == 0:
        test_case_a_1()
    elif test_num == 1:
        test_case_a_2()
    elif test_num == 2:
        test_case_a_3()
    elif test_num == 3:
        test_case_b_1()
    elif test_num == 4:
        test_case_b_2()
    elif test_num == 5:
        test_case_b_3()
    elif test_num == 6:
        test_case_v_a_feasible_1()
    elif test_num == 7:
        test_case_v_a_feasible_2()
    else:
        print(f"Invalid test num {test_num}")
        print("Enter a number between 0 and 6 inclusive")

    plt.show()

if __name__ == "__main__":
    main()
