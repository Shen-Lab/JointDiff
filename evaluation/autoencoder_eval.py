#####################################################################
# Calulate the TMscores and SI for the decoded samples from the 
# autoencoder.
#####################################################################

import numpy as np
import os
import pickle
import argparse
from tqdm.auto import tqdm
import math

from utils_eval import dict_load, dict_save, Identity, TM_score_asym, stat_print

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, 
        default='../../Results/autoencoder/samples/autoencoder_joint-mlp-4-512/files_sample_autoencoder_joint-mlp-4-512_in-seq_test/'
    )
    parser.add_argument('--gt_path_seq', type=str, 
        default='../../Data/Processed/CATH_seq/CATH_seq_test.fasta'
    )
    parser.add_argument('--gt_path_structure', type=str, 
        default='../../Data/Origin/CATH/pdb_all_AtomOnly/'
    )
    parser.add_argument('--out_path', type=str, 
        default='../../Results/autoencoder/ModalityRecovery/autoencoder_joint-mlp-4-512/results_in-seq_test.pkl'
    )

    args = parser.parse_args()

    ###################### load GT samples ##########################
    # * Load the sequence fist as the dataset is detemined based on the 
    #   sequence file.
    # * Check whether the corresponding gt structure and the generated
    #   samples exist.

    seq_dict = {}
    with open(args.gt_path_seq, 'r') as rf:

        flag = False  # whether save this sample
        for i, line in enumerate(rf):
            if i % 2 == 0: 
                ### title line
                title = line.strip('>\n')
                name = title.split(';')[0]
                dset = title.split(';')[1]

                gen_seq_path = os.path.join(args.sample_path, '%s.fa' % name)
                gen_struc_path = os.path.join(args.sample_path, '%s.pdb' % name)
                gt_struc_path = os.path.join(args.gt_path_structure, '%s.pdb' % name)

                if os.path.exists(gen_seq_path) or \
                (os.path.exists(gen_struc_path) and os.path.exists(gt_struc_path)): 
                    flag = True
                    if dset not in seq_dict:
                        seq_dict[dset] = {}
                else:
                    flag = False

            elif flag:
                seq_dict[dset][name] = line.strip('\n')
               
    ###################### evaluation ##########################

    result_dict = {}
    iden_all = []
    tms_all = []
    rmsd_all = []

    for dset in seq_dict:
        print('%d samples for the %s set...' % (len(seq_dict[dset]), dset))
        result_dict[dset] = {}
        iden_dset = []
        tms_dset = []
        rmsd_dset = []

        for name in seq_dict[dset]:

            result_dict[dset][name] = {}

            gen_seq_path = os.path.join(args.sample_path, '%s.fa' % name)
            gen_struc_path = os.path.join(args.sample_path, '%s.pdb' % name)
            gt_struc_path = os.path.join(args.gt_path_structure, '%s.pdb' % name)

            ###### sequence identity ######

            if os.path.exists(gen_seq_path):
                with open(gen_seq_path, 'r') as rf:
                    lines = rf.readlines()
                    seq_gen = lines[1].strip('\n')

                    try:
                        iden = Identity(seq_gen, seq_dict[dset][name]) 
                    except Exception as e:
                        print('%s (seq)' % name, e)
                        iden = None
            else:
                iden = None

            result_dict[dset][name]['SI'] = iden
            if iden is not None:
                iden_dset.append(iden)
                iden_all.append(iden)

            ###### TMscores ######

            if os.path.exists(gen_struc_path):
                try:
                    tmscore, rmsd = TM_score_asym(
                        gen_struc_path,
                        gt_struc_path,
                        with_RMSD = True
                    )
                except Exception as e:
                    print('%s (struc)' % name, e)
                    tmscore = None
                    rmsd = None
            else:
                tmscore = None
                rmsd = None

            result_dict[dset][name]['TM-score'] = tmscore
            result_dict[dset][name]['RMSD'] = rmsd
            if tmscore is not None:
                tms_dset.append(tmscore)
                tms_all.append(tmscore)
                rmsd_dset.append(rmsd)
                rmsd_all.append(rmsd)

        if iden_dset:
            stat_print('Identity', iden_dset)
        if tms_dset:
            stat_print('TM-score', tms_dset)
        if rmsd_dset:
            stat_print('RMSD', rmsd_dset)
        _ = dict_save(result_dict, args.out_path)
        print('************************************************')

    print('Overall')
    if iden_all:
        stat_print('Identity', iden_all)
    if rmsd_all:
        stat_print('TM-score', tms_all)
    if rmsd_all:
        stat_print('RMSD', rmsd_all)

