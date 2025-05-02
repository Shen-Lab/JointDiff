#####################################################
# sample and generate the pdb files 
# by SZ; 5/15/2023
#####################################################

import os
import shutil
import argparse
from Bio.PDB import PDBParser
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn 
import torch.nn.functional as F_ 
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from jointdiff.model import DiffusionSingleChainDesign
from jointdiff.trainer import (
    dict_load, dict_save, inference_pdb_write, count_parameters
)
from jointdiff.modules.diffusion.dpm_full import seq_recover

import torch.multiprocessing 
torch.multiprocessing.set_sharing_strategy('file_system') 
import time

####################################### dataset #############################################

resolution_to_num_atoms = {
    'backbone+CB': 5,  # N, CA, C, O, CB
    'backbone': 4, # by SZ; for single chain; N, CA, C, O
    'full': 15   # 15; N, CA, C, O, CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2, OXT
}

################ for motif-scaffolding ########################################

###### constants ######

ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
}

RESIDUE_dict = {'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'Q':'GLN', 'E':'GLU',
                'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS', 'M':'MET', 'F':'PHE',
                'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP', 'Y':'TYR', 'V':'VAL', 'B':'ASX',
                'Z':'GLX', 'X':'UNK'}

RESIDUE_reverse_dict = {}
for key, value in RESIDUE_dict.items():
    RESIDUE_reverse_dict[value] = key

restype_to_heavyatom_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    '', 'OXT'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    '', 'OXT'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    '', 'OXT'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    '', 'OXT'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    '', 'OXT'],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    '', 'OXT'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    '', 'OXT'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    '', 'OXT'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    '', 'OXT'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    '', 'OXT'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    '', 'OXT'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    '', 'OXT'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'UNK': ['N', 'CA', 'C', 'O',  '',   '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
}

restype_to_heavyatom_order = {}

for resi in restype_to_heavyatom_names:
    restype_to_heavyatom_order[resi] = {}
    for i, atom in enumerate(restype_to_heavyatom_names[resi]):
        if atom == '':
            continue
        restype_to_heavyatom_order[resi][atom] = i

###### pdb load ######

def pdb_info_read(pdb_path, chain_id):
    parser = PDBParser()
    structure = parser.get_structure("protein_name", pdb_path)
                
    model = structure[0]
    chain = model[chain_id]
    info_dict = {}
    resi_start = float('inf')

    for residue in chain:
        resi_idx = residue.get_id()[1]
        resi_name = residue.resname
        if resi_name not in RESIDUE_reverse_dict:
            continue
        resi_start = min(resi_start, resi_idx)
        resi_letter = RESIDUE_reverse_dict[resi_name]
        resi_val = ressymb_to_resindex[resi_letter]
        info_dict[resi_idx] = [resi_letter, resi_val]
        coor_resi = np.zeros((1, 15, 3))
        coor_mask = np.zeros((1, 15))
        
        for atom in residue:
            atom_name = atom.get_name()
            if atom_name not in restype_to_heavyatom_order[resi_name]:
                continue
            atom_idx = restype_to_heavyatom_order[resi_name][atom_name]
            coor_resi[0, atom_idx] = atom.get_coord()
            coor_mask[0, atom_idx] = 1
        info_dict[resi_idx].append(coor_resi)
        info_dict[resi_idx].append(coor_mask)

    ###### summarization ######
    seq = ''
    seq_array = []
    coor_mat = []
    atom_mask = []
    index_list = sorted(info_dict.keys())

    for resi_idx in index_list:
        seq += info_dict[resi_idx][0]
        seq_array.append(info_dict[resi_idx][1])
        coor_mat.append(info_dict[resi_idx][2])
        atom_mask.append(info_dict[resi_idx][3])
    
    return (
        np.vstack(coor_mat),   # (L, 15, 3) 
        np.array(seq_array),   # (L, )
        seq,                   # (L, )
        np.vstack(atom_mask),  # (L, 15)
        resi_start,            # scalar
        index_list             # (L, )
    )

###### scaffold sampling ######

def motif_shape_sele(motif_region, index_list):

    ###### motif region process ######
    motif_region = motif_region.split(',')
    motif_list = []
    
    for size in motif_region:
        if size[0] in '1234567890': 
            ### scaffold region
            continue
            
        ### motif_region
        size = size[1:]  # discard chain ID
        size = size.split('-')
        size = (int(size[0]), int(size[1]))
        motif_list.append(size)

    motif_list = sorted(motif_list)

    ###### mask prepare ######

    mask = [1] * len(index_list) 

    for i, idx in enumerate(index_list):
        if idx >= motif_list[0][0] and i <= motif_list[0][1]:
            mask[i] = 0
        
        if idx > motif_list[0][1]:
            motif_list.pop(0)
        if not motif_list:
            break

    return np.array(mask)

###### data loader ######

class MotifScaffoldingDataset(Dataset):
    def __init__(self, 
        info_path = '../../../../Documents/Data/real_experiment/motif.csv', 
        pdb_path = '../../../../Documents/Data/real_experiment/pdbs',
        info_dict_path = '../../../../Documents/Data/real_experiment/motif_data.pkl',
        with_frag = True, 
        force_cover = False,
    ):
        
        self.with_frag = with_frag

        ###### load the preprocessed data ######
        if os.path.exists(info_dict_path) and (not force_cover):
            self.info_dict = dict_load(info_dict_path)
            self.max_size = self.info_dict['max_size']
            self.name_list = list([
                key for key in self.info_dict.keys() if key != 'max_size'
            ])
            
        ###### process the pdb files ######
        else:
            self.info_dict = {}
            self.name_list = []
            self.max_size = 0
            
            with open(info_path, 'r') as rf:
                for i, line in enumerate(rf):
                    if i == 0:
                        continue

                    ### info
                    line = line.strip('\n').split("\"")
                    line = [token for token in line[0].split(',') + [line[1]] + line[2].split(',') if token != '']
            
                    ### name and path 
                    name = line[1]
                    pdb = name.split('_')[0]
                    pdb_file = os.path.join(pdb_path, '%s.pdb' % pdb)
                
                    if not os.path.exists(pdb_file):
                        print('%s not found!' % pdb_file)
                        continue
    
                    try:
                        ### langth region
                        length = line[3]
                        if '-' in length:
                            length = length.split('-')
                            length_list = list(range(int(length[0]), int(length[1]) + 1))
                        else:
                            length = int(length)
                            length_list = list(range(length, length + 1))
                        self.max_size = max(self.max_size, length_list[-1])
        
                        ### modif region
                        motif_region = line[2]
                        ### chain id
                        for token in motif_region.split(','):
                            if token[0] not in '0123456789':
                                chain_id = token[0]
                                break
                        ### read the pdb
                        (
                            coor_mat,   # (L, 15, 3)
                            seq_array,  # (L, )
                            seq,        # string; (L, )
                            atom_mask,  # (L, 15)
                            resi_start, # scalar
                            index_list, # (L, )
                        ) = pdb_info_read(pdb_file, chain_id)
        
                        ### save 
                        self.name_list.append(name)
                        self.info_dict[name] = {
                            'coor': coor_mat, 
                            'seq_array': seq_array, 
                            'seq': seq, 
                            'atom_mask': atom_mask,
                            'motif_region': motif_region, 
                            'length_set': length_list, 
                            'resi_start': resi_start,
                            'index_list': index_list
                        }
                    except Exception as e:
                        print(name, e)

            self.info_dict['max_size'] = self.max_size
            _ = dict_save(self.info_dict, info_dict_path)

        print('%d entries loaded. max_length=%d' %(self.__len__(), self.max_size))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        """
        Output:
            aa: amino acid sequence; (N, L)
            pos_heavyatom: heavy atom coordinates; float; (N, L, atom_num = 15; 3)
            mask_heavyatom: heavy atom mask, bool; (N, L, atom_num = 15)
            mask_res: residue mask; (N, L)
            mask_gen: design region; (N, L)
            chain_nb: chain idx; 0 for padding; (N, L)
            res_nb: residue idx, start from 1; (N, L)
            fragment_type: 1 for design region, 2 for context (if with_frag), 0 for padding; (N,L)
        """

        name = self.name_list[idx]
        
        ###### info ######
        data_info = self.info_dict[name]

        coor_mat = data_info['coor']
        seq_array = data_info['seq_array']
        seq = data_info['seq']
        atom_mask = data_info['atom_mask']

        # size_sele, _ = motif_shape_sele(data_info['motif_region'], data_info['length_set'])
        #resi_start = data_info['resi_start']
        # [int, tuple, int, ...]

        index_list = data_info['index_list']
        mask_gen = motif_shape_sele(data_info['motif_region'], index_list)

        fragment_type = np.ones(seq_array.shape)
        if self.with_frag:
            fragment_type = fragment_type * 2 - mask_gen

        out = {'name': name}
        out['aa'] = seq_array
        out['pos_heavyatom'] = coor_mat
        out['mask_heavyatom'] = (atom_mask == 1)
        out['mask_gen'] = (mask_gen == 1)
        out['fragment_type'] = fragment_type
        out['length'] = out['aa'].shape[0]
        out['chain_nb'] = np.ones(out['length'])
        out['res_nb'] = np.arange(1, out['length'] + 1)
        out['mask'] = (np.ones(out['length']) == 1)

        length_pad = self.max_size - out['length']
        if length_pad > 0:
            for key in out:
                if key == 'name' or key == 'length':
                    continue
                pad_shape = [length_pad]
                for s in out[key].shape[1:]:
                    pad_shape.append(s)
                mat_pad = np.zeros(pad_shape)

                if len(out[key].shape) == 1:
                    out[key] = np.hstack([out[key], mat_pad])
                else:
                    out[key] = np.vstack([out[key], mat_pad])
               
        return out



class FlexibleMotifScaffoldingDataset(Dataset):
    def __init__(self, 
        info_path = '../../Data/Origin/motif-scaffolding_benchmark/sampled_GFP.pkl',
        with_frag = False
    ):
        self.info_list = dict_load(info_path)
        self.with_frag = with_frag

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        """
        Output:
            aa: amino acid sequence; (N, L)
            pos_heavyatom: heavy atom coordinates; float; (N, L, atom_num = 15; 3)
            mask_heavyatom: heavy atom mask, bool; (N, L, atom_num = 15)
            mask_res: residue mask; (N, L)
            mask_gen: design region; (N, L)
            chain_nb: chain idx; 0 for padding; (N, L)
            res_nb: residue idx, start from 1; (N, L)
            fragment_type: 1 for design region, 2 for context (if with_frag), 0 for padding; (N,L)
        """
        return self.info_list[idx]
        
        
####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ###### paths ######
    parser.add_argument('--model_path', type=str, 
     default='../../Documents/TrainedModels/JointDiff/logs_jointdiff_development/jointdiff-x_joint_multinomial_model6-128-64-step100_posi-scale-50.0_micro-posi+mse_2025_04_07__23_31_18/checkpoints/88000.pt'
    )
    parser.add_argument('--data_path', type=str, 
        default='../../Documents/Data/real_experiment/motif.csv'
    )
    parser.add_argument('--pdb_path', type=str, 
        default='../../Documents/Data/real_experiment/pdbs/'
    )
    parser.add_argument('--info_dict_path', type=str, 
        default='../../Documents/Data/real_experiment/motif_data.pkl'
    )
    parser.add_argument('--flexible_data_path', type=str,
        default='none'
    )
    parser.add_argument('--result_path', type=str,
        default='../Results/debug/'
    )
    ###### devices ######
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    ###### inference setting #####
    parser.add_argument('--attempt', type=int, default=20)
    parser.add_argument('--sample_structure', type=int, default=1)
    parser.add_argument('--sample_sequence', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_type', type=str, default='sele', help='"sele", "all" or "last"')
    parser.add_argument('--save_steps', type=int, nargs='*', default=[0])
    parser.add_argument('--t_bias', type=int, default=-1)

    args = parser.parse_args()

    ###### path ######
    if args.flexible_data_path is None or args.flexible_data_path.upper() == 'NONE':
        args.flexible_data_path = None

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    ###### device ######
    if args.device == 'cuda' and ( not torch.cuda.is_available() ):
        print('GPUs are not available! Use CPU instead.')
        args.device = 'cpu'
        args.multi_gpu = 0

    ###### setting ######
    args.sample_structure = bool(args.sample_structure)
    args.sample_sequence = bool(args.sample_sequence)

    if args.save_type == 'last':
        args.save_steps = [0]

    ###########################################################
    # Model Loading 
    ###########################################################

    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']

    model = DiffusionSingleChainDesign(config.model).to(args.device)
    print('Number of parameters: %d' % count_parameters(model))

    parameter_dict = {}
    for key in checkpoint['model'].keys():
        if key.startswith('module'):
            key_new = key[7:]
            parameter_dict[key_new] = checkpoint['model'][key]
        else:
            parameter_dict[key] = checkpoint['model'][key]
    model.load_state_dict(parameter_dict)

    ###### Parallel ######
    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if args.multi_gpu:
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        print("%d GPUs detected. Applying parallel computation."%(torch.cuda.device_count()))

    ###########################################################
    # Data Loader
    ###########################################################

    ###### other hyperparameter ######

    ### centering
    args.centralize = config.train.get('centralize', True)
    if args.centralize:
        print('With centralization...')

    ### random masking
    args.random_mask = config.train.get('random_mask', False)
    if args.random_mask:
         print('With random masking...')

    ###### dataset ######
    if args.flexible_data_path is None:
        dataset = MotifScaffoldingDataset(
            info_path = args.data_path, 
            pdb_path = args.pdb_path,
            info_dict_path = args.info_dict_path,
            with_frag = args.random_mask, 
            force_cover = False 
        )
    else:
        dataset = FlexibleMotifScaffoldingDataset(
            info_path = args.flexible_data_path, 
            with_frag = args.random_mask,
        )

    ###### data loader ######
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ###########################################################
    # Sampling
    ###########################################################

    start_time = time.time()
    sample_num = 0
 
    ####################### preprocess ########################

    ### inference function
    infer_function = model.module.sample if args.multi_gpu else model.sample

    ####################### sampling ########################
    for attempt_idx in tqdm(range(args.attempt)):
        for idx, batch in enumerate(data_loader):

            batch['aa'] = batch['aa'].to(torch.int64)
            batch['fragment_type'] = batch['fragment_type'].int()
            batch['res_nb'] = batch['res_nb'].int()
            batch['chain_nb'] = batch['chain_nb'].int()
            batch['pos_heavyatom'] = batch['pos_heavyatom'][:, :, :4]  #(N, L, 4, 3)
            batch['mask_heavyatom'] = batch['mask_heavyatom'][:, :, :4]  #(N, L, 4)

            ### centraliztation
            if args.centralize:
                mask_motif = batch['mask_heavyatom'] * (~batch['mask_gen'].bool().unsqueeze(-1))
                mean = (batch['pos_heavyatom'] * mask_motif.unsqueeze(-1)).sum(dim = (1, 2))   # (N, 3)
                mean = mean / mask_motif.sum(dim = (1, 2)).unsqueeze(1)  # (N, 3)
                batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 15, 3)
                batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

            length_list = batch['length']
            seq_motif = [
                seq_recover(batch['aa'][i][:l], length = l) 
                for i, l in enumerate(length_list)
            ]

            for key in batch:
                if key == 'name' or key == 'length':
                    continue
                batch[key] = batch[key].to(args.device)
                if 'mask' in key:
                    batch[key] = batch[key].bool()
                elif key not in {'aa', 'fragment_type', 'res_nb', 'chain_nb'}:
                    batch[key] = batch[key].float()

            ############################## inference ###########################

            #try:
            if True:
                ###### inference ######
                out_dict, traj = infer_function(
                    mask_res = batch['mask'],
                    mask_generate = batch['mask_gen'],
                    batch = batch, 
                    t_bias = args.t_bias, 
                    sample_opt={
                        'sample_structure': args.sample_structure,
                        'sample_sequence': args.sample_sequence
                    }
                )

                ###### save the results ######
                save_steps = args.save_steps if args.save_type != 'all' else traj.keys()

                for t in save_steps:

                    if t not in out_dict:
                        continue
                    batch_size = len(out_dict[t])

                    for i in range(batch_size):

                        name = batch['name'][i]
                        bb_coor = out_dict[t][i]['coor']
                        seq = out_dict[t][i]['seq']
                        
                        ### record pdb files
                        pdb_path = os.path.join(
                            args.result_path, 
                            '%s_%d_%d.pdb'%(name, t, attempt_idx)
                        )
                        inference_pdb_write(
                            coor =bb_coor , 
                            path = pdb_path, 
                            seq = seq
                        )

                        ### record fasta files
                        seq_path = os.path.join(
                            args.result_path, 
                            '%s_%d_%d.fa'%(name, t, attempt_idx)
                        )
                        with open(seq_path, 'w') as wf:
                            wf.write('>%s_%d_%d\n' % (name, t, attempt_idx))
                            wf.write('%s\n' % seq)
                            wf.write('>%s;motif\n' % name)
                            wf.write('%s\n' % seq_motif[i])

                        sample_num += 1
           
            #except Exception as e:
            #    print(e)

    ###### summarizing ######
    print('%d samples genrated in %.4fs.'%(sample_num, time.time() - start_time))

