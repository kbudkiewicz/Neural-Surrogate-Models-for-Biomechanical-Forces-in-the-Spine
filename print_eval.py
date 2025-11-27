import os
import argparse
from utils.eval import print_mean_std

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval_file', required=True, type=str)
    parser.add_argument('-k', '--keyword', required=True, type=str, default='relative_error')
    parser.add_argument('-s', '--stack', required=False, action=argparse.BooleanOptionalAction)
    # NOTE: passing --no-stack is equivalent to running the script without the flag
    parser.add_argument('-p', '--path', required=False, type=str)
    args = parser.parse_args()

    if isinstance(args.path, str):
        for path in os.scandir(args.path):
            path_to_eval_file = os.path.join(path, args.eval_file).replace('\\', '/')
            if os.path.isfile(path_to_eval_file):
                print(f'PATH:', path_to_eval_file)
                print_mean_std(path_to_eval_file, args.keyword, args.stack)
    else:
        print_mean_std(args.eval_file, args.keyword, args.stack)
