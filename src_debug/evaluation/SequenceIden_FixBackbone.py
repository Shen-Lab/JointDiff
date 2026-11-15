#####################################################################
# Calulate the sequence identity of the fixed-backbone designed 
# sequences and the natural sequences.
#####################################################################

import numpy as np
import os
import pickle
import argparse
from tqdm.auto import tqdm
import math

from utils_eval import dict_save, SeqRecovery, Identity, stat_print

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_path', type=str, 
        default='../../Results/originDiff/SingleModal_seq/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_large-3_2024_01_13__14_21_50/'
    )
    parser.add_argument('--gt_path', type=str, 
        default='../../Data/Processed/CATH_seq/CATH_seq_all.fasta'
    )
    parser.add_argument('--out_path', type=str, 
        default='../../Results/originDiff/fixbackbone_SeqIden/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_large-3_2024_01_13__14_21_50.pkl'
    )
    parser.add_argument('--alignment',
        type=int,
        default=1
    )

    args = parser.parse_args()

    args.alignment = bool(args.alignment) # whether do the alignment with NW algorithm

    ###################### read the natural sequences #########################

    gt_seq_dict = {}

    with open(args.gt_path, 'r') as rf:
        for line in rf:
            if line.startswith('>'):
                name = line.strip('>').split(';')[0]
                name = '-'.join(name.split('_'))
                dset = line.strip('\n').split(';')[1]

                if not dset in gt_seq_dict:
                    gt_seq_dict[dset] = {}

            else:
                seq = line.strip('\n')
                gt_seq_dict[dset][name] = seq

    ##### load the designed sequences and calculate SI ########################

    out_dict = {}
    gen_seq_list = [f for f in os.listdir(args.seq_path) if f.endswith('.fa')]

    for sample in tqdm(gen_seq_list):
        ### name
        dset = sample.split('_')[0]
        name = sample.split('_')[1]
        step = int(sample.split('_step')[-1].split('_')[0])

        ### natural seq
        if dset == 'test' and name in gt_seq_dict['id']:
            dset = 'id'
        elif dset == 'test':
            dset = 'od'

        gt_seq = gt_seq_dict[dset][name]

        if not dset in out_dict:
            out_dict[dset] = {}
        if not name in out_dict[dset]:
            out_dict[dset][name] = []

        ### designed seq
        seq_path = os.path.join(args.seq_path, sample)
        with open(seq_path, 'r') as rf:
            gen_seq = rf.readlines()[1].strip('\n')
        
        ### identity
        if args.alignment:
            iden = Identity(gen_seq, gt_seq)
        else:
            iden = SeqRecovery(gen_seq, gt_seq)
        out_dict[dset][name].append(iden)

    _ = dict_save(out_dict, args.out_path)

    ######################## statistics #######################################

    iden_all = []

    for dset in out_dict:
        iden_dset = []

        for name in out_dict[dset]:
            iden_mean = np.mean(out_dict[dset][name])
            iden_dset.append(iden_mean)
            iden_all.append(iden_mean)

        stat_print(dset, iden_dset) 

    stat_print('Overall', iden_all)

