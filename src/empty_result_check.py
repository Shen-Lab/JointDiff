######################################################
# check whether there are valid results
######################################################

import os
import shutil
import argparse

####################################### main function #######################################

parser = argparse.ArgumentParser()

parser.add_argument('--in_path', type=str, 
    default='../Results/jointDiff_development/jointdiff-x_stru_pred_multinomial_model6-128-64-step100_posi-scale-50.0_micro-posi+mse-dist+mse-distogram-clash_2025_04_21__23_25_21/'
)
parser.add_argument('--threshold', type=int, default = 50)
parser.add_argument('--remove', type=int, default = 1)

args = parser.parse_args()
args.remove = bool(args.remove)

dir_list = [
    d for d in os.listdir(args.in_path)
    if os.path.isdir(os.path.join(args.in_path, d)) \
    and (d.startswith('samples') or d.startswith('motifscaffolding')) 
]

flag = False
out = ''

for name in dir_list:
    path = os.path.join(args.in_path, name)
    file_num = len(os.listdir(path))
    out += '%s: %d' % (name, file_num)

    if file_num >= args.threshold:
        flag = True
    elif args.remove:
        shutil.rmtree(path)
        out += ' removed'

    out += '\n'


if not flag:
    status = 'Failed'
    if args.remove:
        shutil.rmtree(args.in_path)
        status += '(removed)'
else:
    status = 'Succeeded'

out = '%s: %s\n' %(args.in_path, status) + out.strip('\n')
print(out)

















