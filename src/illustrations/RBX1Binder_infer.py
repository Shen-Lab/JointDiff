import os
import math
import numpy as np
from tqdm.auto import tqdm

from Bio.PDB import PDBParser
import py3Dmol

import torch

import sys
sys.path.append('./')
sys.path.append('../')

import utils_infer
from utils_infer import (
    dict_load, dict_save, inference_pdb_write, pdb_line_write, RESIDUE_dict
)

from diffab.models import get_model

from diffab.modules.common.geometry import (
    apply_rotation_to_vector, 
    quaternion_1ijk_to_rotation_matrix, 
    reconstruct_backbone, 
    reconstruct_backbone_partially
)
from diffab.modules.common.so3 import (
    so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
)

################################################################################
# Utility Functions
################################################################################

######################## data loading ##########################################

def binder_batch_prepare(
    structure_dict, epitope_dict = {}, binder_size = 200, print_shape = False,
):
    """
    Input:
        target data
    Output:
        processed binder data
        placeholder for the binder
    """
    batch = {}
    feat_list = [
        'aa',
        'pos_heavyatom',
        'mask_heavyatom',
        'fragment_type',  ## flagment token: 1 for antigen, 2 for target, 3 for scaffold, 4 for epitope
        'mask',
        'generate_flag',
        'chain_nb',
        'resi_nb',
    ]
    for feat in feat_list:
        batch[feat] = []
    
    ###################################
    # Target
    ###################################

    ###### chain-wise process ######
    chain_nb = 0
    
    for chain_id in structure_dict:

        if chain_id in epitope_dict:
            epitope_set = epitope_dict[chain_id]
        else:
            epitope_set = set()
        
        ### residue-wise process ###
        res_idx = 0 
        for r_i, res in enumerate(structure_dict[chain_id]):
            ### placeholder
            batch['aa'].append(res['aa'])
            coor = torch.zeros(4, 3)
            pos_mask = torch.zeros(4)
            ### coordinates and masks
            mask = 0
            for _i, atom in enumerate(['N', 'CA', 'C', 'O']):
                if atom in res['coor']:
                    coor[_i] = torch.from_numpy(res['coor'][atom])
                    pos_mask[_i] = 1
                    mask = 1
            batch['pos_heavyatom'].append(coor)
            batch['mask_heavyatom'].append(pos_mask)
            batch['mask'].append(mask)
            ### indexes
            batch['chain_nb'].append(chain_nb)
            batch['resi_nb'].append(res_idx)
            ### feature indicator
            batch['generate_flag'].append(0)
            # epitope
            if res['id'] in epitope_set:
                batch['fragment_type'].append(4)
            # other region of the traget
            else:
                batch['fragment_type'].append(1)
            res_idx += 1

        chain_nb += 1
    
    ###################################
    # Binder
    ###################################

    res_idx = 0
    for _ in range(binder_size):
    
        batch['aa'].append(0)
        batch['pos_heavyatom'].append(torch.zeros(4, 3))
        batch['mask_heavyatom'].append(torch.ones(4))
        batch['mask'].append(1)
        batch['chain_nb'].append(chain_nb)
        batch['resi_nb'].append(res_idx)
        ### 1 for target
        batch['generate_flag'].append(1)
        ### 2 for target, 3 for scaffold
        batch['fragment_type'].append(2)
        res_idx += 1
        
    ###################################
    # final out
    ###################################
    
    for feat in feat_list:  
        if 'heavyatom' in feat:
            batch[feat] = torch.stack(batch[feat])
        else:
            batch[feat] = torch.tensor(batch[feat])
        batch[feat] = batch[feat].unsqueeze(0)
        if print_shape:
            print(feat, batch[feat].shape)

    return batch

######################## data loading ##########################################

def model_loading(model_path):

    checkpoint = torch.load(model_path, weights_only = False)
    print(model_path)
    print(checkpoint.keys())
    
    config = checkpoint['config']
    model = get_model(config.model)
    
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
    
    return model


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

def pdb_write(
    coor, path, seq, chain_nb, chain_list = 'ABCDEFG',
    atom_list = ['N', 'CA', 'C', 'O'],
    print_seq = False,
):
    seq_dict = {}
    with open(path, 'w') as wf:
        a_idx = 0

        for i, resi in enumerate(seq):
            ### residue-wise info
            r_idx = i + 1
            aa = RESIDUE_dict[resi]
            chain = chain_list[int(chain_nb[i])]
            if chain not in seq_dict:
                seq_dict[chain] = ''
            seq_dict[chain] += resi

            for j,vec in enumerate(coor[i]):
                ### atom-wise info
                atom = atom_list[j]
                a_idx += 1
                pdb_line_write(chain, aa, r_idx, atom, a_idx, vec, wf)

    if print_seq:
        for chain in seq_dict:
            print(f'*********** {chain} **************')
            print(seq_dict[chain])


def inference(
    model, batch, device = 'cuda', attempts = 10, 
    result_path = '../results/RBX1Binder/dict/debug.pkl',
    save_pdb = True,
    pdb_dir = '../results/RBX1Binder/pdbs/debug/',
    save_name = 'debug', 
    print_seq = False,
    chain_list = 'AB',
):
    os.makedirs(pdb_dir, exist_ok=True)
    model = model.to(device)

    ###### centralization ######
    mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
    mean = mean / batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1)  # (N, 3)
    batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 15, 3)
    batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

    ###### device & masks ######
    feat_list = ['aa', 'pos_heavyatom', 'generate_flag', 'mask', 'mask_heavyatom', 'resi_nb', 'chain_nb', 'fragment_type']
    for key in feat_list:
        batch[key] = batch[key].to(device)
    batch['generate_flag'] = batch['generate_flag'].bool()
    batch['mask'] = batch['mask'].bool()
    batch['mask_heavyatom'] = batch['mask_heavyatom'].bool()

    out_dict = {}
    sample_idx = 1

    ### inference
    for attp in tqdm(range(attempts)):
        ### inference
        traj_batch = model.sample(batch = batch)

        ### feature transform
        t = 0
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

        ### feature out
        length = batch['mask'].shape[1]
        for i, bb_coor in enumerate(bb_coor_batch):
            ### sample-wise process
            seq = seq_recover(aa_new[i], length = length)
            out_dict[sample_idx] = {
                'coor_true': batch['pos_heavyatom'][i][:length].cpu(),
                'aa_true': batch['aa'][i][:length].cpu(),
                'coor': bb_coor[:length],
                'seq': seq,
                'fragment_type': batch['fragment_type'][i][:length].cpu(),
                #'linker_mask': None if linker_mask is None else linker_mask[i][:length]
            }
            sample_idx += 1

    ###### mv feature and model to CPU ######
    for key in feat_list:
        batch[key] = batch[key].to(device)
    model = model.cpu()

    ### result save
    _ = dict_save(out_dict, result_path)
    if save_pdb:
        for attp in range(1, attempts + 1):
            pdb_write(
                coor = out_dict[attp]['coor'], 
                path = os.path.join(pdb_dir, '%s_attp%d.pdb' % (save_name, attp)), 
                seq = out_dict[attp]['seq'], 
                chain_nb = batch['chain_nb'][0],
                chain_list = chain_list,
                print_seq = print_seq,
            )
    
    return out_dict


################################################################################
# Data
################################################################################

parser = PDBParser(PERMISSIVE=True)

pdb_id = '2LGV'
path = '../data/RBX1_binder/%s.pdb' % pdb_id
structure_all = parser.get_structure(pdb_id, path)

ressymb_order = 'ACDEFGHIKLMNPQRSTVWYX'
RESIDUE_reverse_dict = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
                        'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F',
                        'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V', 'ASX':'B',
                        'GLX':'Z', 'UNK':'X'}
res_type_dict = {}
for i, char in enumerate(ressymb_order):
    res_type_dict[char] = i

structure_dict_all = {}

for model_id in range(3):
    
    structure = structure_all[model_id]
    structure_dict = {}
    
    for chain in structure:
        chain_id = chain.get_id()
        structure_dict[chain_id] = []
    
        for res in chain:
            res_id = res.get_id()
            if res_id[0] != ' ':
                continue
    
            res_idx = (str(res_id[1]) + res_id[2]).strip(' ')
            res_name = res.resname
            if res_name in RESIDUE_reverse_dict:
                res_letter = RESIDUE_reverse_dict[res_name]
            else:
                res_letter = 'X'
                
            res_info = {
                'id': res_idx, 
                'type': res_name, 
                'letter': res_letter,
                'aa': res_type_dict[res_letter],
                'coor': {}
            }
            for atom in ['N', 'CA', 'C', 'O', 'CB']:
                if atom in res.child_dict:
                    res_info['coor'][atom] = res.child_dict[atom].get_coord()
                elif atom != 'CB':
                    print(f'{atom} not found for {chain_id}-{res_idx}.')
                    
            structure_dict[chain_id].append(res_info)

    structure_dict_all[model_id] = structure_dict


for model_id in structure_dict_all:

    print(f'###################### Model {model_id} ######################')
    structure_dict = structure_dict_all[model_id]
    
    for chain in structure_dict:
        print(f'**************** {chain} *****************')
        id_list = [res['id'] for res in structure_dict[chain]]
        print(id_list)

    print()


data_dict = {}

for epitope in [(43,46), (54, 57), (87, 96)]:
    epitope_dict = {'A': {str(i) for i in range(epitope[0], epitope[1] + 1)}}

    for model_id in structure_dict_all:
        structure_dict = structure_dict_all[model_id]

        for binder_size in range(100, 251, 50):
            data_name = f'{pdb_id}-{model_id}_epi{epitope[0]}-{epitope[1]}_L{binder_size}'
            print(data_name)
    
            data_dict[data_name] = binder_batch_prepare(
                structure_dict, 
                epitope_dict = epitope_dict, 
                binder_size = binder_size,
                print_shape = False,
            )

print()
print(f'{len(data_dict)} data versions.')
for feat in data_dict[data_name]:
    print(feat, data_dict[data_name][feat].shape)


################################################################################
# Inference
################################################################################

model_dir = '../checkpoints/'
model_list = [
    'JointDiff-binder_withEpi-withSite_randomMask_withSfold.pt',
    'JointDiff-binder_withEpi-withSite-withMono.pt',
    'JointDiff-x-binder_withEpi-withSite_randomMask_withSfold.pt',
    'JointDiff-x-binder_withEpi-withSite_withDist.pt',
]

device = 'cuda'
attempts = 5
result_dir = "../results/RBX1Binder/"

for version in model_list:
    m_name = '.pt'.join(version.split('.pt')[:-1])
    model_path = os.path.join(model_dir, version)
    model = model_loading(model_path)

    print(f'*************** {m_name} ******************')

    for data_version in data_dict:
        print(f'### {data_version}')

        job_name = f'{m_name}_{data_version}'
        result_path = os.path.join(result_dir, 'dict', '%s.pkl' % job_name)
        if os.path.exists(result_path):
            continue
        pdb_dir = os.path.join(result_dir, 'pdbs', job_name)
        
        _ = inference(
            model = model, 
            batch = data_dict[data_version],
            device = device,
            attempts = attempts,
            result_path = result_path,
            save_pdb = True,
            pdb_dir = pdb_dir,
            save_name = job_name,
            print_seq = False,
        )

    print()
