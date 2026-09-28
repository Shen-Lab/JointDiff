#####################################################
# sample and generate the pdb files 
# by SZ; 5/15/2023
#####################################################

import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn 
import torch.nn.functional as F_ 
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.datasets.binders import ProteinMPNNDataset, FineTuningDataset
from diffab.utils.protein.constants import ressymb_order

from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix, reconstruct_backbone, reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3

import torch.multiprocessing 
torch.multiprocessing.set_sharing_strategy('file_system') 
import time

from utils_infer import inference_pdb_write, dict_save

######################################################################################
# Utility                                                                            #
######################################################################################

def normalize(v):
    """Normalize a batch of vectors"""
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def batch_dot(v1, v2):
    """Batch-wise dot product of two sets of vectors"""
    return torch.sum(v1 * v2, dim=-1, keepdim=True)


def batch_cross(v1, v2):
    """Batch-wise cross product of two sets of vectors"""
    return torch.cross(v1, v2, dim=-1)


def compute_rotation_matrix(v1, v2):
    """Compute the batch-wise rotation matrix that aligns v1 to v2"""
    v1 = normalize(v1)  # Normalize vector1
    v2 = normalize(v2)  # Normalize vector2

    # Compute axis of rotation (cross product)
    axis = batch_cross(v1, v2)

    # Compute angle between v1 and v2 using dot product
    cos_theta = batch_dot(v1, v2).clamp(-1, 1)  # Clamp values for numerical stability
    theta = torch.acos(cos_theta)  # Angle between vectors

    # Skew-symmetric cross-product matrix for each batch
    K = torch.zeros((v1.size(0), 3, 3), device=v1.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    # Identity matrix
    I = torch.eye(3, device=v1.device).unsqueeze(0).repeat(v1.size(0), 1, 1)

    # Rodrigues' rotation formula
    R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)

    return R


def postion_align(pos, fragment, linear_reg = [0.0443, 14.6753]):
    """Based on the antigen and epitopes, estimate the antibody center. Then move the
    the structure center to the origin and rotate the structure to make ag-epi-ab align
    with the x-axis.

    Args:
        pos: backbone atom coordinates; (N, L, 4, 3)
        fragment: type vectors; (N, L)
    """
    pos_CA = pos[:, :, 1, :]  # (N,L,3)
    N, L, _ = pos_CA.shape

    ###### masks ######
    epi_mask = (fragment == 4)
    ag_mask = (fragment == 1) + epi_mask
    binder_len = (fragment == 2).sum(dim=-1)
    center_dist = binder_len * linear_reg[0] + linear_reg[1] # (N,)
    epi_target = torch.zeros(N, 3).to(pos_CA.device)
    epi_target[:,0] = - center_dist

    ###### centers ######
    ag_center = (pos_CA * ag_mask.unsqueeze(-1)).sum(dim=1) / (ag_mask.sum(dim=1) + 1e-8).unsqueeze(-1)
    epi_center = (pos_CA * epi_mask.unsqueeze(-1)).sum(dim=1) / (epi_mask.sum(dim=1) + 1e-8).unsqueeze(-1)

    ###### orientation vectors ######
    vec_ag_epi = epi_center - ag_center # orientation from antigen center to interface
    x = torch.zeros(N, 3).to(pos_CA.device) # x axis
    x[:,0] = 1
    R = compute_rotation_matrix(vec_ag_epi, x)
    pos = torch.bmm(R, pos.reshape(N, -1, 3).transpose(1,2)).transpose(1,2).reshape(N, L, -1, 3)

    ###### translation
    trans_vec = epi_target - epi_center  # (N, 3)
    pos += trans_vec[:,None,None,:]  # (N, L, 4, 3)

    return pos

######################################################################################
# DataLoading                                                                        #
######################################################################################

def load_dataset(args, dset, reset = False):
    if dset in {'train', 'val', 'test'}:
        dataset = ProteinMPNNDataset(
            summary_path = args.summary_path,
            pdb_dir = args.pdb_dir,
            processed_dir = args.processed_dir,
            interface_path = args.interface_path,
            dset = dset,
            reset = reset,
            reso_threshold = args.reso_threshold,
            length_min = args.length_min,
            length_max = args.length_max,
            with_monomer = args.with_monomer,
            load_interface = args.load_interface,
            with_epitope = args.with_epitope,
            with_bindingsite = args.with_bindingsite,
            with_scaffold = args.with_scaffold,
            random_masking = args.random_masking
        )
    else:
        dataset = FineTuningDataset(
            data_path = args.summary_path,
            length_min = args.length_min,
            length_max = args.length_max,
            with_monomer = args.with_monomer,
            with_epitope = args.with_epitope,
            with_bindingsite = args.with_bindingsite,
            with_scaffold = args.with_scaffold,
            random_masking = args.random_masking
        ) 

    return dataset


def seq_recover(aa:torch.Tensor, length:int = None) -> str:
    """Recover sequence from the tensor.

    Args:
        aa: embedded sequence tensor; (L,).
        length: length of the sequence; if None consider the paddings.

    Return:
        seq: recovered sequence string. 
    """

    length = aa.shape[0] if length is None else min(length, aa.shape[0])
    seq = ''
    for i in range(length):
        idx = int(aa[i])
        if idx > 20:
            print('Error! Index %d is larger than 20.'%idx)
            break
        seq += ressymb_order[idx]
    return seq


####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ###### paths ######
    parser.add_argument('--model_path', type=str, 
        default='../logs_epitope/codesign_dim-128_step100_lr1.e-4_wd0.0_posiscale10.0_2024_10_02__10_17_15_withEpi-withSite_LLM/checkpoints/204000.pt'
    )
    parser.add_argument('--result_path', type=str, 
        default='../results/debug/samples.pkl'
    )
    parser.add_argument('--summary_path', type=str,
        default='../data/Protein_MPNN/mpnn_data_info.pkl'
        #default='../data/FineTuning/data_list.pkl'
    )
    parser.add_argument('--pdb_dir', type=str,
        default='../data/Protein_MPNN/pdb_2021aug02/pdb/'
    )
    parser.add_argument('--processed_dir', type=str,
        default='../data/Protein_MPNN/'
    )
    parser.add_argument('--interface_path', type=str,
        default='../data/Protein_MPNN/interface_dict_all.pt'
    )
    ###### devices ######
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    ###### inference setting #####
    ### for experiments
    parser.add_argument('--dset', type=str, default='test')
    #parser.add_argument('--dset', type=str, default='finetune')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_type', type=str, default='sele', help='"sele", "all" or "last"')
    parser.add_argument('--save_steps', type=int, nargs='*', default=[0])
    parser.add_argument('--t_bias', type=int, default=0)
    parser.add_argument('--attempts', type=int, default=1)

    ### for models
    parser.add_argument('--same_chain', type=int, default=0)
    parser.add_argument('--design_centralize', type=int, default=0)
    parser.add_argument('--centralization', type=int, default=1)
    parser.add_argument('--load_interface', type=int, default=1)
    parser.add_argument('--with_epitope', type=int, default=1)
    parser.add_argument('--with_bindingsite', type=int, default=1)
    parser.add_argument('--with_scaffold', type=int, default=0)
    parser.add_argument('--random_masking', type=int, default=0)

    args = parser.parse_args()

    if args.device == 'cuda' and ( not torch.cuda.is_available() ):
        print('GPUs are not available! Use CPU instead.')
        args.device = 'cpu'
        args.multi_gpu = 0

    args.same_chain = bool(args.same_chain)
    args.design_centralize = bool(args.design_centralize)
    args.centralization = bool(args.centralization)
    args.reso_threshold = 3.0 # checkpoint['args'].reso_threshold,
    args.length_min = 20 # checkpoint['args'].length_min,
    args.length_max = 800 #checkpoint['args'].length_max,
    args.with_monomer = False
    args.load_interface = bool(args.load_interface)
    args.with_epitope = bool(args.with_epitope)
    args.with_bindingsite = bool(args.with_bindingsite)
    args.with_scaffold = bool(args.with_scaffold)
    args.random_masking = bool(args.random_masking)

    ###########################################################
    # Model Loading 
    ###########################################################

    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']
    if not config.model.__contains__('chain_feat_version'):
        config.model.chain_feat_version = 'same'

    model = get_model(config.model).to(args.device)
    print('Number of parameters: %d' % count_parameters(model))

    checkpoint = torch.load(args.model_path)
    parameter_dict = {}
    for key in checkpoint['model'].keys():
        key_new = key

        if key.startswith('module'):
            key_new = key[7:]

        key_new = key_new.split('.')
        key_new_last = []
        for token in key_new:
            if token in {'spatial_coef', 'proj_query_point', 'proj_key_point'}:
                token = token + '_intra'
            key_new_last.append(token)
        key_new = '.'.join(key_new_last)
        parameter_dict[key_new] = checkpoint['model'][key]
        
    model.load_state_dict(parameter_dict)

    ### Parallel
    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if args.multi_gpu:
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        print("%d GPUs detected. Applying parallel computation."%(torch.cuda.device_count()))

    #######################################################################
    # Data Loading
    #######################################################################

    dataset = load_dataset(args, args.dset, reset = False)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=False,
        num_workers=args.num_workers
    )

    ###########################################################
    # Sampling
    ###########################################################

    start_time = time.time()
    sample_num = 0
    out_dict = {}

    for batch in tqdm(data_loader):

        ###### centralize the data ######
        if args.design_centralize:
            batch['pos_heavyatom'] = postion_align(
                batch['pos_heavyatom'], batch['fragment_type']
            )

        elif args.centralization:
            mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
            mean = mean / batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1)  # (N, 3)
            batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 15, 3)
            batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

        ############################################################################
        # Forward and loss cal
        #     if args.debug: torch.set_anomaly_enabled(True) 
        ############################################################################

        ###### batch ######
        # aa: sequence; (N, L)
        # pos_heavyatom: atom coordinates; (N, L, 4, 3)
        # generate_flag: 1 for target and 0 for others; (N, L)
        # mask: 1 for valid token and 0 for paddings; (N, L)
        # mask_heavyatom: 1 for valid atom and 0 for others; (N, L)
        # resi_nb: residue index; (N, L)
        # chain_nb: chain index; (N, L)
        # fragment_type: 1 for antigen, 2 for target, 3 for scaffold, 4 for epitope, 0 for padding; (N, L)

        for key in ['aa', 'pos_heavyatom', 'generate_flag', 'mask', 'mask_heavyatom', 'resi_nb', 'chain_nb', 'fragment_type']:
            batch[key] = batch[key].to(args.device)

        batch['generate_flag'] = batch['generate_flag'].bool()
        batch['mask'] = batch['mask'].bool()
        batch['mask_heavyatom'] = batch['mask_heavyatom'].bool()
        if 'name' in batch:
            name_list = batch['name']
        else:
            name_list = ['sample%d' for d in batch['idx']]

        if args.same_chain:
            batch['chain_nb'] *= 0

        try:
            for attp in range(args.attempts):

                ###### inference ######
                if args.multi_gpu:
                    traj_batch = model.module.sample(batch = batch)
                else:
                    traj_batch = model.sample(batch = batch)

                lengths = batch['mask'].sum(dim=-1)

                if attp == 0:
                    name_list_out = name_list
                else:
                    name_list_out = [
                        '%s_att%d' % (name, attp) for name in name_list
                    ]
                for i, name in enumerate(name_list_out):
                    out_dict[name] = {}

                ###### transformation ######
                for t in args.save_steps:

                    R = so3vec_to_rotation(traj_batch[t][0])
                    aa_new = traj_batch[t][2].cpu()   # t: sampling step. 2: Amino acid.
                    bb_coor_batch, mask_atom_new = reconstruct_backbone_partially(
                        pos_ctx = batch['pos_heavyatom'].cpu(),
                        R_new = R.cpu(),
                        t_new = traj_batch[t][1].cpu(),
                        aa = aa_new,
                        chain_nb = batch['chain_nb'].cpu(),
                        res_nb = batch['resi_nb'].cpu(),
                        mask_atoms = batch['mask_heavyatom'].cpu(),
                        mask_recons = batch['generate_flag'].cpu(),
                    )  # (N, L_max, 4, 3), _

                    for i, bb_coor in enumerate(bb_coor_batch):
                        ### sample-wise process
                        seq = seq_recover(aa_new[i], length = int(lengths[i]))
                        out_dict[name_list_out[i]][t] = {
                            'coor_true': batch['pos_heavyatom'][i][:lengths[i]].cpu(),
                            'aa_true': batch['aa'][i][:lengths[i]].cpu(), 
                            'coor': bb_coor[:lengths[i]], 
                            'seq': seq, 
                            'fragment_type': batch['fragment_type'][i][:lengths[i]].cpu()
                        }

                sample_num += 1
                _ = dict_save(out_dict, args.result_path)

        except Exception as e:
            print(e)
                 
    ###### summarizing ######
    

    print('%d samples genrated in %.4fs.'%(sample_num, time.time() - start_time))

