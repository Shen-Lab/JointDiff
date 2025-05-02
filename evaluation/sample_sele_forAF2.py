######################################################
# random select several samples for further analysis
# by SZ; 6/5/2023
######################################################

import numpy as np
import random
import os
import argparse

import utils_eval

####################################### main function #######################################

parser = argparse.ArgumentParser()

parser.add_argument('--sample_path', type=str, default='../../Results/diffab/Uncon_struc/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_2023_05_01__17_05_48/')
parser.add_argument('--out_path', type=str, default='../../Results/diffab/sample_sele_forAF2/')

parser.add_argument('--sele_num', type=int, default=100)

parser.add_argument('--seq_flag', type=int, default=1, help='whether extract the sequences')
parser.add_argument('--AF2_flag', type=int, default=0, help='whether prepare the fasta files for AF2')
parser.add_argument('--AF2_token', type=str, default=None, help='certain index for the prepared AF2 input files')

args = parser.parse_args()

###### tasks and corresponding setting ######

args.AF2_flag = int(args.AF2_flag)
if args.AF2_flag:
    args.seq_flag = True
else:
    args.seq_flag = int(args.seq_flag)

if args.AF2_token is None or args.AF2_token.upper() == 'NONE' or args.AF2_token == '': 
    args.AF2_token = ''
else:
    args.AF2_token += '_'

###### paths ######

if not args.sample_path.endswith('/'):
    args.sample_path += '/'

if not args.out_path.endswith('/'):
    args.out_path += '/'

stru_path = args.out_path + 'structures/'
if not os.path.exists(stru_path):
    os.mkdir(stru_path)

if args.seq_flag:
    seq_path = args.out_path + 'sequences/'
    if not os.path.exists(seq_path):
        os.mkdir(seq_path)

if args.AF2_flag:
    AF2_path = args.out_path + 'for_AF2/'
    if not os.path.exists(AF2_path):
        os.mkdir(AF2_path)

###### sampling ######

pdb_list = [p for p in os.listdir(args.sample_path) if p.endswith('.pdb')]
ori_size = len(pdb_list)

if ori_size <= args.sele_num:
    print('Sample size (%d) is larger than or equal to the original size (%d).'%(args.sele_num, ori_size))
    args.sele_num = ori_size

pdb_sele = random.sample(pdb_list, args.sele_num)

seq_num = 0 
file_idx = 0

for pdb in pdb_sele:
    pdb_file = args.sample_path + pdb
    ### structure
    os.system('cp %s %s'%(pdb_file, stru_path))
    ### sequence
    if args.seq_flag:
        name = pdb.split('.pdb')[0]
        seq_list = utils_eval.seq_extract(pdb_file)
        if len(seq_list) > 1:
            seq = ''.join(seq_list)
        else:
            seq = seq_list[0] 

        with open(seq_path + '%s.fasta'%name, 'w') as wf:
            wf.write('>%s\n'%name)
            wf.write('%s\n'%seq)

        if args.AF2_flag:
            if seq_num % 2 == 0:
                file_idx += 1
                with open(AF2_path + '%ssample_%d.fasta'%(args.AF2_token, file_idx), 'w') as wf:
                    wf.write('>%s\n'%name)
                    wf.write('%s\n'%seq)
            else:
                with open(AF2_path + '%ssample_%d.fasta'%(args.AF2_token, file_idx), 'a') as wf:
                    wf.write('>%s\n'%name)
                    wf.write('%s\n'%seq)

        seq_num += 1





















