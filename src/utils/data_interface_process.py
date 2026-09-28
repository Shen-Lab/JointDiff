################################################################################
# collect the interface region
################################################################################

import os
import argparse
from tqdm.auto import tqdm
import numpy as np
import torch

################################################################################
# functions
################################################################################

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.sqrt(dist)


def distance_cal(resi_coor_1, resi_coor_2):
    """Get the minimum distance of the heavy atoms between 2 residues.

    Args:
        resi_coor_1: (14, 3)
        resi_coor_2: (14, 3)
    """
    #dist_min = float('inf')
   
    #for coor_1 in resi_coor_1:

    #    if (torch.isnan(coor_1)).any():
    #        continue

    #    for coor_2 in resi_coor_2:
    #        if (torch.isnan(coor_1)).any():
    #            continue

    #        dist = torch.norm(coor_1 - coor_2)
    #        dist_min = min(dist, dist_min)

    dist_mat = pairwise_distances(resi_coor_1, resi_coor_2)
    dist_mask = ~ torch.isnan(dist_mat)

    if dist_mask.any():
        return min(dist_mat[dist_mask])
    else:
        return float(float('inf'))


def find_contact_residues(coor_dict, distance_threshold=8.0):

    chain_list = list(coor_dict.keys())
    contact_range = {}
  
    for chain_1 in chain_list:

        ########################## for the target chain ########################
        contact_range[chain_1] = None
        L = coor_dict[chain_1].shape[0]
 
        for chain_2 in chain_list:
            ###################### for the other chain #########################
            if chain_1 == chain_2:
                continue

            ###### left ######
            left_idx = None
            for idx in range(L):
                resi_1 = coor_dict[chain_1][idx]
               
                for resi_2 in coor_dict[chain_2]:
                    if distance_cal(resi_1, resi_2) <= distance_threshold:
                        left_idx = idx
                        break
                if left_idx is not None:
                    break

            if left_idx is None:
                continue
            
            ###### right ######
            right_idx = None
            for idx in range(L-1, left_idx, -1):
                resi_1 = coor_dict[chain_1][idx]
               
                for resi_2 in coor_dict[chain_2]:
                    if distance_cal(resi_1, resi_2) <= distance_threshold:
                        right_idx = idx
                        break
                if right_idx is not None:
                    break

            if right_idx is None:
                right_idx = left_idx

            if contact_range[chain_1] is None:
                contact_range[chain_1] = [left_idx, right_idx]
            else:
                contact_range[chain_1][0] = min(
                    contact_range[chain_1][0], left_idx
                )
                contact_range[chain_1][-1] = max(
                    contact_range[chain_1][-1], right_idx
                )

        ########################## for the target chain ########################
        if contact_range[chain_1] is None:
            del contact_range[chain_1]
        else:
            contact_range[chain_1] = range(
                contact_range[chain_1][0], contact_range[chain_1][-1] + 1
            )

    return contact_range



def find_contact_residues_2(coor_dict, distance_threshold=8.0, with_gpu = False):

    chain_list = list(coor_dict.keys())
    contact_range = {}

    for chain_1 in chain_list:

        ########################## for the target chain ########################
        contact_range[chain_1] = None
        L = coor_dict[chain_1].shape[0]

        coor_all_1 = coor_dict[chain_1].reshape(-1, 3) # (L*14, 3)
        if with_gpu:
            coor_all_1 = coor_all_1.cuda()

        for chain_2 in chain_list:
            ###################### for the other chain #########################
            if chain_1 == chain_2:
                continue

            coor_all_2 = coor_dict[chain_2].reshape(-1, 3) # (L2*14, 3)
            if with_gpu:
                coor_all_2 = coor_all_2.cuda()

            dist_all = pairwise_distances(coor_all_1, coor_all_2) # (L*14, L2*14)
            dist_all[torch.isnan(dist_all)] = float('inf')
            dist_min = torch.min(dist_all.reshape(L, -1), dim = -1).values.cpu() # (L, 1)
            
            ###### left ######
            left_idx = None
            for idx in range(L):
                if dist_min[idx] <= distance_threshold:
                    left_idx = idx
                    break

            if left_idx is None:
                continue
 
            ###### left ######
            right_idx = None
            for idx in range(L-1, left_idx, -1):
                if dist_min[idx] <= distance_threshold:
                    right_idx = idx
                    break

            if right_idx is None:
                right_idx = left_idx

            if contact_range[chain_1] is None:
                contact_range[chain_1] = [left_idx, right_idx]
            else:
                contact_range[chain_1][0] = min(
                    contact_range[chain_1][0], left_idx
                )
                contact_range[chain_1][-1] = max(
                    contact_range[chain_1][-1], right_idx
                )
            
        ########################## for the target chain ########################
        if contact_range[chain_1] is None:
            del contact_range[chain_1]
        else:
            contact_range[chain_1] = range(
                contact_range[chain_1][0], contact_range[chain_1][-1] + 1
            )

    return contact_range


def find_contact_residues_all(coor_dict, distance_threshold=8.0, with_gpu = False):

    chain_list = list(coor_dict.keys())
    contact_dict = {}

    for chain_1 in chain_list:

        ########################## for the target chain ########################
        contact_dict[chain_1] = {}
        L = coor_dict[chain_1].shape[0]

        coor_all_1 = coor_dict[chain_1].reshape(-1, 3) # (L*14, 3)
        if with_gpu:
            coor_all_1 = coor_all_1.cuda()

        for chain_2 in chain_list:
            ###################### for the other chain #########################
            if chain_1 == chain_2:
                continue

            coor_all_2 = coor_dict[chain_2].reshape(-1, 3) # (L2*14, 3)
            if with_gpu:
                coor_all_2 = coor_all_2.cuda()

            ###### distance calculation #######
            dist_all = pairwise_distances(coor_all_1, coor_all_2) # (L*14, L2*14)
            dist_all[torch.isnan(dist_all)] = float('inf')
            dist_min = torch.min(dist_all.reshape(L, -1), dim = -1).values.cpu()  # (L,1)

            ###### contacts ######

            contact_dict[chain_1][chain_2] = []
            for idx in range(L):
                if dist_min[idx] <= distance_threshold:
                    contact_dict[chain_1][chain_2].append(idx)
                
            if not contact_dict[chain_1][chain_2]:
                del contact_dict[chain_1][chain_2]

        ########################## for the target chain ########################
        if contact_dict[chain_1] is None:
            del contact_dict[chain_1]

    return contact_dict


################################################################################
# arguments
################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--sample_list', type=str, default='../data/Protein_MPNN/all_chain_list.txt')
parser.add_argument('--pdb_path', type=str, default='../data/Protein_MPNN/pdb_2021aug02/')
parser.add_argument('--save_path', type=str, default='../data/Protein_MPNN/interface_dict_all.pt')

parser.add_argument('--len_max', type=int, default=800)
parser.add_argument('--with_gpu', type=int, default=1)
parser.add_argument('--range_only', type=int, default=0)

parser.add_argument('--job_num', type=int, default=1)
parser.add_argument('--job_idx', type=int, default=1)

args = parser.parse_args()

if torch.cuda.is_available():
    args.with_gpu = bool(args.with_gpu)
else:
    args.with_gpu = False

args.range_only = bool(args.range_only)

################################################################################
# main process
################################################################################

###### prepare the sample dict ######
sample_dict = {}
pdb_list = []

with open(args.sample_list, 'r') as rf:
    for line in rf:
        path = line.split('\n')[0]
        sample = path.split('/')[-1].split('.pt')[0]
        pdb, chain = sample.split('_')
        if pdb not in sample_dict:
            sample_dict[pdb] = {}
            pdb_list.append(pdb)
        sample_dict[pdb][chain] = path

pdb_num = len(pdb_list)
inter_num = pdb_num // args.job_num
if args.job_num == args.job_idx:
     pdb_sele = pdb_list[inter_num * (args.job_idx - 1):]
else:
     pdb_sele = pdb_list[inter_num * (args.job_idx - 1): inter_num * args.job_idx]

print('%d pdbs selected out of %d.' % (len(pdb_sele), len(pdb_list)))


###### distance calculation ######

contact_range_dict = {}
process_num = 0

for _, pdb in tqdm(enumerate(pdb_sele)):
    if len(sample_dict[pdb]) < 2:
        continue

    ### coordinates collection
    coor_dict = {}
    l_all = 0

    for chain in sample_dict[pdb]:
        path = os.path.join(args.pdb_path, sample_dict[pdb][chain]) 
        info_dict = torch.load(path)
        if 'xyz' not in info_dict:
            continue

        resi_coor = []
        for resi in info_dict['xyz']:
            if torch.isnan(resi).all():
                continue

            resi_coor.append(resi)
      
        if resi_coor:
            coor_dict[chain] = torch.stack(resi_coor)
            l_all += len(resi_coor)
    
    ### distance cal
    if len(coor_dict) < 2 or l_all > args.len_max:
        continue

    # contact_range_dict[pdb] = find_contact_residues(
    #     coor_dict, distance_threshold=8.0, 
    # )
    try:
        if args.range_only:
            contact_range_dict[pdb] = find_contact_residues_2(
                coor_dict, distance_threshold=8.0, with_gpu = args.with_gpu,
            )
        else:
            contact_range_dict[pdb] = find_contact_residues_all(
                coor_dict, distance_threshold=8.0, with_gpu = args.with_gpu, 
            )

        process_num += 1
 
        if process_num % 100 == 0:
            print('%d valid samples processed!' % process_num)
            torch.save(contact_range_dict, args.save_path)

    except Exception as e:
        print(pdb, e)


torch.save(contact_range_dict, args.save_path)
print('%d multimers processed out of %d samples.' % (
    len(contact_range_dict), len(sample_dict)
))






