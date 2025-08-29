#####################################################
# Extract the CA-coordinates matrix of the samples. 
#####################################################

import os
import math
import numpy as np
import argparse
import time
from tqdm.auto import tqdm
import logging

from Bio.PDB import PDBParser
import utils_eval

pdb_parser = PDBParser()

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--struc_path', type=str, default='../../Results/originDiff/forward-diff_struc/struc/Step5_posiscale10.0/')
    parser.add_argument('--out_path', type=str, default='../../Results/originDiff/sanity_check/coor/forward_Step5_posiscale10.0.pkl')

    args = parser.parse_args()

    pdb_list = [p for p in os.listdir(args.struc_path) if p.endswith('.pdb')]

    pdb_dict = {}
    for pdb_file in tqdm(pdb_list):

        pdb = '.'.join(pdb_file.split('sample-')[-1].split('.')[:-1])

        if 'attempt' in pdb:  # forward on natural samples
            name = '_'.join(pdb.split('_')[:2])
            step = int(pdb.split('_step')[1].split('_')[0])
            attempt = int(pdb.split('_attempt')[1].split('_')[0])

        else:  # reverse on the sythesized data
            name = pdb.split('_')[0]
            step = int(pdb.split('_')[1])
            attempt = int(pdb.split('_')[2])

        if name not in pdb_dict:
            pdb_dict[name] = {}
        if attempt not in pdb_dict[name]:
            pdb_dict[name][attempt] = {}

        ###### get coordinates ######
        pdb_path = os.path.join(args.struc_path, pdb_file)
        structure = pdb_parser.get_structure("protein", pdb_path)[0]
        coor_list = []

        for chain in structure:
            for residue in chain:
                for atom in residue:
                    atom_name = atom.get_name()
                    if atom_name == 'CA':
                        coor_list.append(atom.get_coord())

        pdb_dict[name][attempt][step] = np.array(coor_list)

    _ = utils_eval.dict_save(pdb_dict, args.out_path)
