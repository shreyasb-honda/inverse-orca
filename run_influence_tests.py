from argparse import ArgumentParser
from tests.influence_testing import *


def main(test_num: int):

    if test_num == 0:
        no_overlap_1()
    elif test_num == 1:
        no_overlap_2()
    elif test_num == 2:
        no_overlap_3()
    elif test_num == 3:
        overlap_with_solution_1()
    elif test_num == 4:
        overlap_with_solution_2()
    elif test_num == 5:
        overlap_with_solution_3()
    elif test_num == 6:
        overlap_without_solution_1()
    elif test_num == 7:
        overlap_without_solution_2()
    elif test_num == 8:
        same_goal_direction()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test-num', type=int, default=0, help="Index of the test to run 0-7 (default: 0)")
    args = parser.parse_args()
    main(args.test_num)
    plt.show()
