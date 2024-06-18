from argparse import ArgumentParser
from tests.overlap_testing import *


def main(test_num: int):

    if test_num == 0:
        first_quadrant()
    if test_num == 1:
        second_quadrant()
    if test_num == 2:
        third_quadrant()
    if test_num == 3:
        fourth_quadrant()
    if test_num == 4:
        line_circle_overlap()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test-num', type=int, default=0, help="Index of the test to run 0-3 (default: 0)")
    args = parser.parse_args()
    main(args.test_num)
