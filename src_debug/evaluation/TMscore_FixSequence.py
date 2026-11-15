#####################################################################
# Calulate the TMscores of the fixed-sequence designed 
# structures and the natural structures.
#####################################################################

import numpy as np
import os
import pickle
import argparse
from tqdm.auto import tqdm
import math

from utils_eval import dict_load, dict_save, TM_score, stat_print

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--struc_path', type=str, 
        default='../../Results/originDiff/SingleModal_struc/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_large-3_2024_01_13__14_21_50/'
    )
    parser.add_argument('--gt_path', type=str, 
        default='../../Data/Origin/CATH/pdb_all_AtomOnly/'
    )
    parser.add_argument('--dset_map', type=str, 
        default='../../Data/Processed/CATH_sample-dset_map.pkl'
    )
    parser.add_argument('--seq_path', type=str, 
        default='../../Data/Processed/CATH_seq/CATH_seq_all.fasta'
    )
    parser.add_argument('--out_path', type=str, 
        default='../../Results/originDiff/fixseq_TMscore/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_large-3_2024_01_13__14_21_50.pkl'
    )

    args = parser.parse_args()

    ###################### map the sample to the set ##########################

    if os.path.exists(args.dset_map):
        dset_map_dict = dict_load(args.dset_map)

    else:
        dset_map_dict = {}

        with open(args.seq_path, 'r') as rf:
            for line in rf:
                if line.startswith('>'):
                    name = line.strip('>').split(';')[0]
                    name = '-'.join(name.split('_'))
                    dset = line.strip('\n').split(';')[1]
                    dset_map_dict[name] = dset

        _ = dict_save(dset_map_dict, args.dset_map)

    ##### load the designed structures and calculate TM-scores ################

    out_dict = {}
    gen_struc_list = [f for f in os.listdir(args.struc_path) if f.endswith('.pdb')]

    for sample in tqdm(gen_struc_list):
        ### name
        name = sample.split('_')[1]
        dset = dset_map_dict[name]
        step = int(sample.split('_step')[-1].split('_')[0])

        ### natural structure
        name_gt = name.split('-')[0] + '_' + '-'.join(name.split('-')[1:])
        gt_path = os.path.join(args.gt_path, name_gt + '.pdb')

        ### TMscore
        tmscore, rmsd = TM_score(
            os.path.join(args.struc_path, sample), 
            gt_path
        )

        if tmscore is None:
            print('Failed for %s.' % sample)
            continue

        if not dset in out_dict:
            out_dict[dset] = {}
        if not name in out_dict[dset]:
            out_dict[dset][name] = [[], []]
        out_dict[dset][name][0].append(tmscore)
        out_dict[dset][name][1].append(rmsd)

    _ = dict_save(out_dict, args.out_path)

    ######################## statistics #######################################

    for i, name in enumerate(['TM-score', 'RMSD']):
        print(name)
        tmscore_all = []

        for dset in out_dict:
            tmscore_dset = []

            for name in out_dict[dset]:
                tmscore_mean = np.mean(out_dict[dset][name][i])
                tmscore_dset.append(tmscore_mean)
                tmscore_all.append(tmscore_mean)

            stat_print(dset, tmscore_dset) 

        stat_print('Overall', tmscore_all)

        print()

