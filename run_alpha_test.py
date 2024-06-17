import argparse
from tests.estimate_alpha_testing import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-num', type=int, default=0,
                        help='Index of the test to run (default: 0)')
    args = parser.parse_args()

    test_num = args.test_num
    if test_num == 0:
        test_case_A_1()
    elif test_num == 1:
        test_case_A_2()
    elif test_num == 2:
        test_case_A_3()
    elif test_num == 3:
        test_case_B_1()
    elif test_num == 4:
        test_case_B_2()
    elif test_num == 5:
        test_case_B_3()
    elif test_num == 6:
        test_case_vA_feasible_1()
    elif test_num == 7:
        test_case_vA_feasible_2()
    else:
        print(f"Invalid test num {test_num}")
        print("Enter a number between 0 and 6 inclusive")

    plt.show()

if __name__ == "__main__":
    main()
