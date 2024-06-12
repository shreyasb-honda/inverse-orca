import argparse
import matplotlib.pyplot as plt
from tests.compare_official import *


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--test-num', type=int, help='Index of the test to run 0-2 (default: 0)', default=0)

    # args = parser.parse_args()

    # if args.test_num == 0:
    #     overlap_with_solution_1()
    # elif args.test_num == 1:
    #     overlap_with_solution_2()
    # elif args.test_num == 2:
    #     overlap_with_solution_3()
    # else:
    #     print("Please enter test number between 0 and 2 (inclusive)...")
    #     return

    overlap_with_solution_1()

    plt.show()

if __name__ == "__main__":
    main()
