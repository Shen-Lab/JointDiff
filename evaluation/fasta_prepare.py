######################################################
# prepare the fasta file based on the features 
# by SZ; 6/5/2023
######################################################

import numpy as np
import os
import argparse
import pickle

from utils_eval import seq_extract 

def dict_save(dictionary,path):
    with open(path,'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0

def dict_load(path):
    with open(path,'rb') as handle:
        result = pickle.load(handle)
    return result

####################################### main function #######################################

parser = argparse.ArgumentParser()

parser.add_argument('--feature_path', type=str, default='../../Results/diffab/Features/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_2023_05_01__17_05_48/')
parser.add_argument('--out_path', type=str, default='../../Results/diffab/Uncon_seq/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_2023_05_01__17_05_48.fasta')
parser.add_argument('--with_pdb', type=str, default=1, help='1 for pdb and 0 for feature dict')

args = parser.parse_args()
args.with_pdb = bool(args.with_pdb)

if args.with_pdb:
    dict_list = [d for d in os.listdir(args.feature_path) if d.endswith('.pdb')]
else:
    dict_list = [d for d in os.listdir(args.feature_path) if d.endswith('.pkl')]

with open(args.out_path, 'w') as wf:
    for d in dict_list:
        dict_path = os.path.join(args.feature_path, d)
        
        if not args.with_pdb:
            name = d.split('.pkl')[0]
            fea_dict = dict_load(dict_path)
            seq = ''.join(fea_dict['seq'])

        else:
            name = d.split('.pdb')[0]
            seq = seq_extract(dict_path)[0]

        ### write down the sequence
        wf.write('>%s\n'%name)
        wf.write('%s\n'%seq)





















