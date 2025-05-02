######################################################
# check the empty files (or remove them)
# by SZ; 9/25/2023
######################################################

import argparse
import os

def empty_file(path):
    with open(path, 'r') as rf:
        lines = rf.readlines()
    return len(lines) == 0

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, default='../../Results/diffab/forward-diff_struc/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_2023_05_01__17_05_48/')
parser.add_argument('--remove', type=int, default=0)

args = parser.parse_args()

args.remove = bool(args.remove)

### single file
if not os.path.isdir(args.input_path):
    if empty_file(args.input_path):
        print('%s is an empty file.'%args.input_path)
        if args.remove:
            os.remove(args.input_path)
            print('Removed.')
    else:
        print('%s is not an empty file.'%args.input_path)

### directory
else:
    if not args.input_path.endswith('/'):
        args.input_path += '/'

    file_list = os.listdir(args.input_path)
    
    for fil in file_list:
        if (not os.path.isdir(args.input_path + fil)) and empty_file(args.input_path + fil):
            if args.remove:
                print(fil, 'removed')
            else:
                print(fil)














