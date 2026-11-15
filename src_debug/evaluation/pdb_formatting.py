######################################################
# pdb formatting for ProteinMPNN
# by SZ; 2/01/2023
######################################################

import numpy as np
import argparse
from utils_eval import read_pdb_coor, pdb_write

####################################### main function #######################################

parser = argparse.ArgumentParser()

parser.add_argument(
    '--pdb_path', 
    type=str, 
    default='../../Results/originDiff/sample_sele_forAF2/codesign_RememberPadding_2024_01_29__20_57_00/structures/len131_0_2.pdb'
)
parser.add_argument(
    '--out_path', 
    type=str, 
    default='../../Results/originDiff/sample_sele_forAF2/codesign_RememberPadding_2024_01_29__20_57_00/structures_forProteinMPNN/len131_0_2.pdb'
)
parser.add_argument(
    '--print_status', 
    type=int,
    default=1
)

args = parser.parse_args()
args.print_status = bool(args.print_status)

try:
    coor_dict = read_pdb_coor(args.pdb_path)
    pdb_write(coor_dict, args.out_path)

except:
    if args.print_status:
        print('Cannot process with BioPython. Directly do the formatting...')

    with open(args.pdb_path, 'r') as rf, open(args.out_path, 'w') as wf:
        for line in rf:
            if (not line.startswith('ATOM')) or \
            (line[38] == ' ' and line[46] == ' ' and line[54] == ' '):
                wf.write(line)
            else:
                coor = line[30:56].strip(' ').split('.')
                if len(coor) != 4:
                    if args.print_status:
                        print('Error for %s!' % line.strip('\n'))
                    continue

                ###### new coor ######
                coor_new = []
                for i in range(3): # x, y, z
                    inte = coor[0]
                    if len(inte) <= 4:
                        deci = coor[1][:3]
                    else:
                        deci = coor[1][:7 - len(inte)]
                    coor_new.append('%s.%s' % (inte, deci))
                    coor[1] = coor[1][3:]
                    coor.pop(0)

                ###### new line ######
                line_new = line[:30] + '{:>8}{:>8}{:>8}'.format(coor_new[0], coor_new[1], coor_new[2])
                line_new += line[56:]

                ###### record ######
                wf.write(line_new)

if args.print_status:
    print('Done')

