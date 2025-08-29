#####################################################
# calculate the fitness scores with ESM-1b 
# and energy scores with openmm given the 
# sequences
#####################################################

import os
import math
import numpy as np
import argparse
import time
from tqdm.auto import tqdm
import logging

import utils_eval
import utils_guidance

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_path', type=str, default='../../Results/originDiff/forward-diff_struc/seq/Step5_posiscale10.0.fasta')
    parser.add_argument('--struc_path', type=str, default='../../Results/originDiff/forward-diff_struc/struc/Step5_posiscale10.0/')
    parser.add_argument('--contact_path', type=str, default='../../Data/Processed/CATH_forDiffAb/ContactMap_test_rearanged/')
    parser.add_argument('--out_path', type=str, default='../../Results/originDiff/sanity_check/forward_Step5_posiscale10.0_100-1.pkl')
    parser.add_argument('--log_fail', type=str, default=None)

    parser.add_argument('--fitness_cal', type=int, default=1)
    parser.add_argument('--grad_esm', type=int, default=0)
    parser.add_argument('--energy_sbmopenmm', type=int, default=0)
    parser.add_argument('--energy_openmm', type=int, default=0)
    parser.add_argument('--force_sbmopenmm', type=int, default=0)
    parser.add_argument('--force_openmm', type=int, default=0)
    #parser.add_argument('--energy_cal', type=int, default=1)
    #parser.add_argument('--energy_tool', type=str, default='sbm-openmm')
    parser.add_argument('--attempt', type=int, default=None)

    #parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--job_num', type=int, default=1)
    parser.add_argument('--job_idx', type=int, default=1)

    args = parser.parse_args()

    args.fitness_cal = bool(args.fitness_cal)
    args.grad_esm = bool(args.grad_esm)
    args.energy_sbmopenmm = bool(args.energy_sbmopenmm)
    args.energy_openmm = bool(args.energy_openmm)
    args.force_sbmopenmm = bool(args.force_sbmopenmm)
    args.force_openmm = bool(args.force_openmm)

    if not (args.fitness_cal or args.energy_openmm or args.energy_sbmopenmm or \
            args.grad_esm or args.force_sbmopenmm or args.force_openmm ):
        print('Nothing to calculate.')
        quit()

    if args.contact_path is not None and args.contact_path.upper() == 'NONE':
        args.contact_path = None

    ###########################################################################
    # sample list 
    ###########################################################################

    ###### sequence ######
    if args.fitness_cal or args.grad_esm:
        title_list_seq = []
        seq_dict = {}
        seq_flag = False

        with open(args.seq_path, 'r') as rf:

            for line in rf:
                ### title line
                if line.startswith('>'):
                    title = line.strip('>\n')
                    if (args.attempt is None) or ('attempt%d' % args.attempt in title):
                        seq_flag = True
                        title_list_seq.append(title)
                    else:
                        seq_flag = False

                ### seq_line
                elif seq_flag:
                    seq = line.strip('\n')
                    seq_dict[title] = seq

        ### load ESM
        esm_model = utils_guidance.FitnessGrad()

    ###### structure ######
    if args.energy_sbmopenmm or args.energy_openmm or args.force_sbmopenmm or args.force_openmm:
        title_list_struc = ['.pdb'.join(pdb.split('.pdb')[:-1]) 
                                for pdb in os.listdir(args.struc_path) 
                                if pdb.endswith('.pdb') and ((args.attempt is None) or ('attempt%d' % args.attempt in pdb))]

    ###### prepare the selected samples ######
    if args.fitness_cal and (args.energy_sbmopenmm or args.energy_openmm or args.force_sbmopenmm):
        title_list = list(set(title_list_seq) & set(title_list_struc))

    elif args.fitness_cal or args.grad_esm:
        title_list = title_list_seq

    else:
        title_list = title_list_struc

    if title_list:
        title_list = sorted(title_list)
    num_all = len(title_list)
    interval = math.ceil(num_all / args.job_num)
   
    title_list_sele = title_list[(args.job_idx - 1) * interval : args.job_idx * interval]
    print('%d samples selected out of %d.' % (len(title_list_sele), num_all))

    ###########################################################################
    # Path 
    ###########################################################################

    if os.path.exists(args.out_path):
        out_dict = utils_eval.dict_load(args.out_path)
    else:
        out_dict = {}

    if args.log_fail is not None and args.log_fail.upper() != 'NONE':
        logging.basicConfig(filename=args.log_fail, filemode='w')

    ###########################################################################
    # Fitness and Energy Calculation 
    ###########################################################################

    for idx, title in tqdm(enumerate(title_list_sele)): 

        if not title in out_dict:
            out_dict[title] = {}

        ##################### fitness/grad #########################################
        if (args.fitness_cal and 'fitness' not in out_dict[title]) or \
        (args.grad_esm and 'grad_esm' not in out_dict[title]):
            try:
                batch_seq_list = [seq_dict[title]]
                start = time.time()
                fitness_score, grad = esm_model(seq = batch_seq_list, with_grad = args.grad_esm)
        
                if args.fitness_cal:
                    out_dict[title]['fitness'] = float(fitness_score)

                if args.grad_esm:
                    out_dict[title]['grad_esm'] = grad.numpy()

                out_dict[title]['size'] = len(seq_dict[title])
                out_dict[title]['esm_time'] = time.time() - start

            except Exception as e:
                logging.warning('ESM failed for %s: %s!' % (title, e))   

        #################### energy/force (sbm-openmm) ##################################

        if (args.energy_sbmopenmm and 'energy_sbm' not in out_dict[title]) or \
        (args.force_sbmopenmm and 'force_sbmopenmm' not in out_dict[title]):
            try:
                start = time.time()

                pdb_path = os.path.join(args.struc_path, '%s.pdb' % title)
                sample_id = 'sample-'.join(title.split('sample-')[1:])
                sample_id = sample_id.split('_step')[0]

                if args.contact_path is None:
                    contact_map = True
                else:
                    contact_map = os.path.join(args.contact_path, '%s.contact'%sample_id)

                if isinstance(contact_map, str) and (not os.path.exists(contact_map)):
                    print('Contact map %s was not found!' % contact_map)
                    contact_map = True

                force_dict, energy_dict = utils_guidance.force_and_energy(
                      pdb_path = pdb_path, 
                      temp_dir = './',
                      contact_thre = 12, 
                      with_contact = contact_map, 
                      get_force = args.force_sbmopenmm, 
                      get_energy = args.energy_sbmopenmm,
                      name_tag = None, 
                      sum_result = False, 
                      device = 'cpu',
                      atom_list = ['CA'], 
                )

                if args.energy_sbmopenmm:
                    out_dict[title]['energy_sbm'] = energy_dict
                if args.force_sbmopenmm:
                    out_dict[title]['force_sbmopenmm'] = force_dict
                out_dict[title]['sbm_time'] = time.time() - start

            except Exception as e:
                logging.warning('sbm-openmm failed for %s: %s!' % (title, e))  

        #################### energy (openmm) ##################################

        if (args.energy_openmm and 'energy_openmm' not in out_dict[title])  or \
        (args.force_openmm and 'force_openmm' not in out_dict[title]):

            try:
                start = time.time()
                pdb_path = os.path.join(args.struc_path, '%s.pdb' % title)

                force, energy_score = utils_guidance.force_and_energy_openmm(
                    pdb_path = pdb_path,
                    force_field = ('amber14-all.xml', 'amber14/tip3pfb.xml'),
                    get_force = args.force_openmm,
                    get_energy = args.energy_openmm,
                    device = 'cpu'
                )

                if args.force_openmm:
                    out_dict[title]['force_openmm'] = force
                if args.energy_openmm:
                    out_dict[title]['energy_openmm'] = energy_score

                out_dict[title]['openmm_time'] = time.time() - start

            except Exception as e:
                logging.warning('OpenMM failed for %s: %s!' % (title, e))  

        #################### energy (openmm) ##################################
        if not out_dict[title]:
            del out_dict[title]

        if idx % 100 == 0:
            _ = utils_eval.dict_save(out_dict, args.out_path)

    ###########################################################################
    # save the results
    ###########################################################################

    _ = utils_eval.dict_save(out_dict, args.out_path)

