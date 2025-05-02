#####################################################
# Randomly select samples for evaluation. 
#####################################################

import os
import math
import numpy as np
import random
import argparse
import time
from tqdm.auto import tqdm

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_path', type=str, default='../../Results/Chroma/seq_gen.fa')
    parser.add_argument('--struc_path', type=str, default='../../Results/Chroma/samples/')
    parser.add_argument('--seq_out', type=str, default='../../Results/Chroma/seq_gen_sele.500.fa')
    parser.add_argument('--struc_out', type=str, default='../../Results/Chroma/struc_gen_sele.500/')

    parser.add_argument('--sele_num', type=int, default=500)
    parser.add_argument('--token', type=str, default='none')

    args = parser.parse_args()

    if args.token == 'none':
        args.token = ''

    ###### sequence load ###### 

    seq_dict = {}

    with open(args.seq_path, 'r') as rf:
        flag = True
        for line in rf:
            if line.startswith('>'):
                name = line.strip('>\n').split(';')[0]
                flag = args.token in name
            elif flag:
                seq_dict[name] = line.strip('\n')

    ###### structure load ######

    pdb_set = set([
        p[:-4] for p in os.listdir(args.struc_path) 
        if p.endswith('.pdb') and args.token in p
    ])

    ###### selection ######

    inter_set = pdb_set.intersection(seq_dict)
    if len(inter_set) > args.sele_num:
        sele_list = random.sample(inter_set, args.sele_num)   
    else:
        sele_list = list(inter_set)  

    print('%d samples selected out of %d.' % (len(sele_list), len(inter_set)))

    ###### copy the files ######

    with open(args.seq_out, 'w') as wf:
        for sample in tqdm(sele_list):

            ### sequence
            wf.write('>%s\n' % sample)
            wf.write('%s\n' % seq_dict[sample])

            ### structure
            ori_path = os.path.join(args.struc_path, '%s.pdb' % sample)
            new_path = os.path.join(args.struc_out, '%s.pdb' % sample)
            os.system('cp %s %s' % (ori_path, new_path))


