######################################################
# recover the oxygen for pdb backbones given N, CA and C (for folding diff)
# by SZ; 6/14/2023
######################################################

import numpy as np
import argparse
from utils_eval import read_pdb_coor, pdb_write

def recover_O(N, CA, C):
    # Bond lengths (in angstroms): based on https://www.ruppweb.org/Xray/tutorial/protein_structure.htm
    N_CA_length = 1.46
    CA_C_length = 1.52
    N_C_length = 1.33

    # Bond angles (in degrees): based on https://www.sciencedirect.com/topics/medicine-and-dentistry/peptide-bond
    N_CA_C_angle = 110.0
    C_N_CA_angle = 123.0

    # Convert angles to radians
    N_CA_C_angle_rad = np.radians(N_CA_C_angle)
    C_N_CA_angle_rad = np.radians(C_N_CA_angle)

    # Calculate unit vectors along the bonds
    N_CA_vec = (CA - N) / np.linalg.norm(CA - N)
    CA_C_vec = (C - CA) / np.linalg.norm(C - CA)
    N_C_vec = (C - N) / np.linalg.norm(C - N)

    # Calculate the position of O atom
    O = C + (N_CA_vec * N_CA_length) - ((CA_C_vec * CA_C_length) / np.sin(N_CA_C_angle_rad)) + ((N_C_vec * N_C_length) / np.sin(C_N_CA_angle_rad))

    return O

####################################### main function #######################################

parser = argparse.ArgumentParser()

parser.add_argument('--pdb_path', type=str, default='../../Results/foldingdiff/sample_sele/structures/generated_1000.pdb')
parser.add_argument('--out_path', type=str, default='../../Results/foldingdiff/sample_sele/structures_withO/generated_1000.pdb')

args = parser.parse_args()

pdb_dict = read_pdb_coor(args.pdb_path) 

for chain in pdb_dict.keys():
    for resi in pdb_dict[chain]['ordered_idx']:
        flag = True
        if 'O' in pdb_dict[chain]['coor'][resi].keys():
            print('Atom O already exists.')
            continue

        for atom in ['N', 'CA', 'C']:
            if not atom in pdb_dict[chain]['coor'][resi].keys():
                print('Atom %s cannot be found for %s-%s.'%(atom, chain, resi))
                flag = False
                break
        if flag:
            N_coor = pdb_dict[chain]['coor'][resi]['N']
            CA_coor = pdb_dict[chain]['coor'][resi]['CA']
            C_coor = pdb_dict[chain]['coor'][resi]['C']

            pdb_dict[chain]['coor'][resi]['O'] = recover_O(N_coor, CA_coor, C_coor)
        
pdb_write(pdb_dict, args.out_path)

















