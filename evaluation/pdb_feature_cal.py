#####################################################
# calculate the features given the protein pdb files:
#     * dihedral torsion angles (phi, psi and omega)
#     * clash
#     * sequence
# by SZ; 5/27/2023
#####################################################

import numpy as np
import pickle
import argparse

import utils_eval

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='../../Results/diffab/Uncon_struc/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_2023_05_01__17_05_48/len100_0_1.pdb')
    parser.add_argument('--out_path', type=str, default='temp.pkl')

    parser.add_argument('--clash_cutoff', type=float, default=0.63)

    args = parser.parse_args()

    try:
        ### angles and sequences ###   
        result_dict = utils_eval.angles_cal_singlechain(args.in_path)

        ### clash ### 
        #clash_num, atom_num = clash_cal_singlechain(args.in_path)
        clash_num, atom_num = utils_eval.count_clashes(args.in_path, clash_cutoff=args.clash_cutoff)

        result_dict['clash'] = clash_num
        print('%d clashes detected (out of %d atoms).'%(clash_num, atom_num))

        ### save ###
        _ = utils_eval.dict_save(result_dict, args.out_path)

    except Exception as e:
        print('Failed for %s:' % (args.in_path) , e)


 
