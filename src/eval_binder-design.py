#####################################################
# transform the dictionaries the pdb files 
#####################################################

import os
import argparse
from tqdm.auto import tqdm
import pickle
import lmdb
import time

import numpy as np
import torch
from utils_infer import inference_pdb_write_multi, dict_load, dict_save
from diffab.utils.protein.constants import ressymb_order

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, 
        default='../data/Protein_MPNN/structures.test.lmdb'
    )
    parser.add_argument('--interface_dict', type=str, 
        default='../data/Protein_MPNN/interface_dict_all.pt'
    )
    parser.add_argument('--in_path', type=str, 
        default='../results_2/codesign_dim-128_step100_lr1.e-4_wd0.0_posiscale10.0_2024_09_23__23_16_17_centra/sample_1.pkl'
    )
    parser.add_argument('--out_path', type=str, 
        default='../results_2/codesign_dim-128_step100_lr1.e-4_wd0.0_posiscale10.0_2024_09_23__23_16_17_centra/pdb_sample_1/'
    )
    parser.add_argument('--result_path', type=str, 
        default='../results_2/codesign_dim-128_step100_lr1.e-4_wd0.0_posiscale10.0_2024_09_23__23_16_17_centra/eval_dict_sample_1.pkl'
    )
    parser.add_argument('--steps', type=int, nargs='*', default=[0])
    parser.add_argument('--sele_num', type=int, default=3000)

    parser.add_argument('--with_scaffold', type=int, default=0)

    args = parser.parse_args()
    
    args.with_scaffold = bool(args.with_scaffold)

    ############################################################################
    # pre-process
    ############################################################################

    sample_dict = dict_load(args.in_path)
    db_conn = lmdb.open(
        args.data_path,
        map_size=32*(1024*1024*1024),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    interface_dict = torch.load(args.interface_dict)
    if 'posiscale' in args.in_path:
        psw = float(args.in_path.split('posiscale')[-1].split('_')[0])
    else:
        psw = 10.

    ############################################################################
    # main process
    ############################################################################

    eval_dict = {}
    start_time = time.time()
    save_num = 0

    sample_list = list(sample_dict.keys())
    sample_list = sample_list[:args.sele_num]

    for sample in tqdm(sample_list):

        ########################################################################
        # sample-wise process 
        ########################################################################

        if sample not in interface_dict:
            continue
        eval_dict[sample] = {}

        ################### load the groundtruth information ###################
        with db_conn.begin() as txn:
            feat_dict = pickle.loads(txn.get(sample.encode()))

        chain_list = feat_dict['chains']
        size_list = []
        gt_seq_dict = {}
        gt_coor_dict = {}

        for chain in chain_list:
            size_list.append(len(feat_dict['feat'][chain]['aa']))
            gt_seq_dict[chain] = ''.join([
                ressymb_order[int(idx)] for idx in feat_dict['feat'][chain]['aa']
            ])
            gt_coor_dict[chain] = feat_dict['feat'][chain]['pos_heavyatom'][:, :4, :]  # (L, 4, 3)

        ###### step-wise process ######

        for step in args.steps:

            if step not in sample_dict[sample]:
                print('Step %d not found for %s!' % (step, sample))
                continue

            sele_flag = True
            eval_dict[sample][step] = {}
            
            ###### process designed feature ######
            coor = sample_dict[sample][step]['coor']  # (L_all, 4, 3)
            seq = sample_dict[sample][step]['seq'] # string of L_all
            fragment_type = sample_dict[sample][step]['fragment_type'] 

            seq_dict = {}
            start = 0
            design_chain = None

            for i, chain in enumerate(chain_list):

                seq_gt = gt_seq_dict[chain]  # string
                coor_gt = gt_coor_dict[chain]  # (L, 4, 3)

                ### target or with scaffold
                if start >= len(fragment_type):
                    print('Fragment size issue (%s and %s)!' % (fragment_type.shape, start))
                    sele_flag = False
                    break
                  
                if args.with_scaffold or fragment_type[start] != 2:
                    size = len(seq_gt)

                ### designed region
                else:
                    size = (fragment_type == 2).sum()
                size_list[i] = size

                end = start + size
                seq_chain = seq[start : end]
                coor_chain = coor[start : end].clone()

                ### whether the designed chain
                design_flag = (fragment_type[start : end] == 2).any()
                if design_flag and design_chain is None:
                    design_chain = chain
                elif design_flag:
                    print('Multiple designed chains detected for %s!' % sample)
                    design_flag = False
                    
                ###### designed chain ######
                if design_flag:                
                    design_start = 1000
                    design_end = 0
                    match_flag = True

                    for chain_sub in interface_dict[sample][chain]:
                        design_start = min(interface_dict[sample][chain_sub][chain][0], design_start)
                        design_end = max(interface_dict[sample][chain_sub][chain][-1], design_end)
                    design_len = max(0.1, design_end - design_start + 1)
                    if design_len < 1:
                        print('No designed region for chain %s of %s!' % (chain, sample))

                    if (not args.with_scaffold and len(seq_chain) != design_len) \
                    or (args.with_scaffold and len(seq_chain) != len(seq_gt)):
                        print('Length not match for chain %s of %s!.' % (chain, sample))
                        match_flag = False

                    seq_gen_all = ''
                    gen_idx = 0
                    same_aa = 0
                    design_coor = []
                    refer_coor = []

                    ###### sequence comparison ######

                    for i, aa in enumerate(seq_gt):
                        if i >= design_start and i <= design_end:
                            # print(i)
                            # print(seq_gt)
                            # print(gen_idx)
                            # print(seq_chain)
                            if gen_idx >= len(seq_chain):
                                break
                            seq_gen_all += seq_chain[gen_idx]
                            design_coor.append(coor_chain[gen_idx].unsqueeze(0))
                            refer_coor.append(coor_gt[i].unsqueeze(0))

                            if seq_chain[gen_idx] == aa:
                                same_aa += 1
                            gen_idx += 1
 
                        elif args.with_scaffold:
                            seq_gen_all += seq_chain[gen_idx]
                            if seq_chain[gen_idx] != aa:
                                match_flag = False
                            gen_idx += 1

                        else:
                            seq_gen_all += '-'

                    if not match_flag:
                        print('Scaffold does not match for chain %s of %s!' % (chain, sample))
                        print(design_start, design_end)
                        print(seq_gt)
                        print(seq_chain)
                        #quit()

                    seq_dict[chain] = (True, seq_gen_all)
                    eval_dict[sample][step]['SI'] = same_aa / design_len
                    
                    if not design_coor:
                        print('Empty design region for chain %s of %s!' % (chain, sample))
                        sele_flag = False
                        continue

                    design_coor = torch.cat(design_coor)
                    refer_coor = torch.cat(refer_coor)

                    rmsd = ((design_coor - refer_coor) ** 2).sum(dim = -1) # (L, 4)
                    rmsd = float((rmsd.mean()).sqrt()) 
                    eval_dict[sample][step]['rmsd'] = rmsd
                 
                ###### other chains ######
                else:
                    if seq_chain != seq_gt:
                        print('Context sequence not match for chain %s of %s!.' % (chain, sample)) 
                    seq_dict[chain] = (False, seq_chain)

                    if coor_gt.shape != coor_chain.shape:
                        print('Context structure not match for chain %s of %s!.' % (chain, sample))
                    elif ((coor_gt - coor_chain) ** 2).mean() > 0.1:
                        print('Structure not aligned for chain %s of %s!' % (chain, sample))

                start = end

            ###### save the outputs ######

            if not sele_flag:
                del eval_dict[sample][step]
                continue

            ### save path
            save_pdb_path = os.path.join(args.out_path, '%s_t%d.pdb' % (sample, step))
            save_seq_path = os.path.join(args.out_path, '%s_t%d.fa' % (sample, step))

            ### seq write
            len_start = 0
            inter_start = None

            with open(save_seq_path, 'w') as wf:
                for i, chain in enumerate(chain_list):
                    design_flag, seq_chain = seq_dict[chain]
                    title = '>chain-%s' % chain

                    if design_flag:
                        ### origin chain
                        wf.write('%s;origin\n' % title)
                        wf.write('%s\n' % gt_seq_dict[chain])

                        ### designed chain
                        title += ';design:%d-%d;recovery=%.4f' % (
                            design_start + 1, design_end + 1, eval_dict[sample][step]['SI']
                        )

                    wf.write('%s\n' % title)
                    wf.write('%s\n' % seq_chain)

            ### pdb write
            inference_pdb_write_multi(
                coor = coor, 
                path = save_pdb_path, 
                seq = sample_dict[sample][step]['seq'], 
                chain_list = chain_list.copy(),
                size_list = size_list.copy()
            )
   
            save_num += 1

    _ = dict_save(eval_dict, args.result_path)

    print('%d pdb files writen in %.4fs.'%(save_num, time.time() - start_time))

    ############################################################################
    # statistics
    ############################################################################

    si_all = {}
    rmsd_all = {}

    for sample in eval_dict:
        for step in eval_dict[sample]:
            if step not in si_all:
                si_all[step] = []
                rmsd_all[step] = []

            if 'SI' in eval_dict[sample][step]:
                si_all[step].append(eval_dict[sample][step]['SI'])

            if 'rmsd' in eval_dict[sample][step]:
                rmsd_all[step].append(eval_dict[sample][step]['rmsd'])

    for step in sorted(si_all.keys()):
        print('Step %d: SI=%.6f, rmsd=%.6f' % (step, np.mean(si_all[step]), np.mean(rmsd_all[step])))



