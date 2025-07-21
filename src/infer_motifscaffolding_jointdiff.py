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

################################################################################
# Dataset 
################################################################################

############################ constants ########################################

resolution_to_num_atoms = {
    'backbone+CB': 5,  # N, CA, C, O, CB
    'backbone': 4, # by SZ; for single chain; N, CA, C, O
    'full': 15   # 15; N, CA, C, O, CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2, OXT
}

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

########################## PDB loading ########################################

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

################################ dataloader ###################################

class MotifScaffoldingDataset(Dataset):
    def __init__(self, 
        info_path = '../../../../Documents/Data/real_experiment/motif.csv', 
        pdb_path = '../../../../Documents/Data/real_experiment/pdbs',
        info_dict_path = '../../../../Documents/Data/real_experiment/motif_data.pkl',
        with_frag = True, 
        force_cover = False,
        length_sampling = False,
    ):
        
        self.with_frag = with_frag
        self.pdb_path = pdb_path
        self.length_sampling = length_sampling

        ####################################################
        # load the preprocessed data
        ####################################################

        if os.path.exists(info_dict_path) and (not force_cover):
            self.info_dict = dict_load(info_dict_path)
            self.max_size = self.info_dict['max_size']
            self.name_list = list([
                key for key in self.info_dict.keys() if key != 'max_size'
            ])
            
        ####################################################
        # process the pdb_files based on the information
        ####################################################
        else:
            self.info_dict = {}
            self.name_list = []
            self.max_size = 0
           
            ###### infomation file ######
            with open(info_path, 'r') as rf:
                for i, line in enumerate(rf):
                    if i == 0:
                        continue

                    try:
                    #if True:
                        sample_dict = self.motifscaffolding_dataprocess(line)
                        if sample_dict is None:
                            continue
                        name = sample_dict['name']
                        self.info_dict[name] = sample_dict
                        self.max_size = max(sample_dict['length_max'], self.max_size)
                        self.name_list.append(name)

                    except Exception as e:
                        print(line, e)

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
        
        ####################################################
        # preprocess
        ####################################################

        data_info = self.info_dict[name]

        ###### motifs ######
        motif_coor = data_info['motif_coor']
        motif_seq = data_info['motif_seq_array']
        motif_atom_mask = data_info['motif_atom_mask']
        motif_size = data_info['motif_size']

        ###### scaffolds #####
  
        ### sample the length of scaffolds
        if self.length_sampling:
            length_range = data_info['length_range']
            length_all = random.randint(length_range[0], length_range[1])
            scaffold_all = length_all - motif_size

            scaffold_len_range_list = data_info['scaffold_length']
            scaffold_len_list = [sl_range[0] for sl_range in scaffold_len_range_list]
            l_scaffold = sum(scaffold_len_list)
            l_scaffold_max = sum([sl_range[1] for sl_range in scaffold_len_range_list])

            if l_scaffold > scaffold_all or l_scaffold_max < scaffold_all:
                raise ValueError(f"Impossible task for {name}!")

            elif l_scaffold < scaffold_all:
                diff = scaffold_all - l_scaffold
                ### check the tolerance of each piece
                tolerance_list = []
                for i, sl_range in enumerate(scaffold_len_range_list):
                    tolerance = sl_range[1] - scaffold_len_list[i]
                    if tolerance > 0:
                        tolerance_list.append([i, tolerance])
                ### add residues
                toler_p_num = len(tolerance_list)
                for _ in range(diff):
                    idx_local = random.randint(0, toler_p_num-1)
                    idx = tolerance_list[idx_local][0]
                    scaffold_len_list[idx] += 1
                    tolerance_list[idx_local][1] -= 1
                    if tolerance_list[idx_local][1] == 0:
                        tolerance_list.pop(idx_local)
                        toler_p_num -= 1

        ### use the original length
        else:
            scaffold_len_list = data_info['scaffold_length_true']
            length_all = data_info['length']

        ###### construct the freatures ######
        atom_num, dim = motif_coor[0].shape[-2:]
        motif_num = len(motif_coor)
        coor_out = []
        seq_out = []
        atom_mask_out = []
        mask_gen = []

        for i in range(motif_num):
            ### scaffold
            l = scaffold_len_list[i]
            if l > 0:
                mask_gen += [1] * l 
                # 0 for scaffold coordinates
                coor_out.append(np.zeros((l, atom_num, dim)))
                # X (20) for scaffold tokens
                seq_out.append(np.ones(l) * 20)
                # 1 for scaffold atom mask (assume all atoms known)
                atom_mask_out.append(np.ones((l, atom_num)))

            ### motif
            mask_gen += [0] * motif_coor[i].shape[0] 
            coor_out.append(motif_coor[i])
            seq_out.append(motif_seq[i])
            atom_mask_out.append(motif_atom_mask[i])

        ### right ter scaffold
        if len(scaffold_len_list) > motif_num:
            l = scaffold_len_list[motif_num]
            if l > 0:
                mask_gen += [1] * l 
                # 0 for scaffold coordinates
                coor_out.append(np.zeros((l, atom_num, dim)))
                # X (20) for scaffold tokens
                seq_out.append(np.ones(l) * 20)
                # 1 for scaffold atom mask (assume all atoms known)
                atom_mask_out.append(np.ones((l, atom_num)))

        ###### summarize the features ######
        #print([coor.shape for coor in coor_out])
        coor_out = np.vstack(coor_out)
        seq_out = np.hstack(seq_out)
        atom_mask_out = np.vstack(atom_mask_out)
        mask_gen = np.array(mask_gen)

        fragment_type = np.ones(seq_out.shape)
        if self.with_frag:
            fragment_type = fragment_type * 2 - mask_gen

        out = {'name': name, 'length': length_all}
        out['aa'] = seq_out
        out['pos_heavyatom'] = coor_out
        out['mask_heavyatom'] = (atom_mask_out == 1)
        out['mask_gen'] = (mask_gen == 1)
        out['fragment_type'] = fragment_type
        out['length'] = out['aa'].shape[0]
        out['chain_nb'] = np.ones(length_all)
        out['res_nb'] = np.arange(1, length_all + 1)
        out['mask'] = (np.ones(length_all) == 1)

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


    def motifscaffolding_dataprocess(self, line):
    
        ###### info ######
        # e.g. 
        # "0,1PRW,\"5-20,A16-35,10-25,A52-71,5-20\",60-105\n"
        line = line.strip('\n').split("\"")
        # ["0,1PRW,", "5-20,A16-35,10-25,A52-71,5-20", ",60-105"]
        #  pdb_id      motif & scaffolding               length
        line = [token for token in line[0].split(',') + [line[1]] + line[2].split(',') if token != '']
        # ["0", "1PRW", "5-20,A16-35,10-25,A52-71,5-20", "60-105"]
        #  idx pdb_id      motif & scaffolding            length

        ###### name and path ######
        name = line[1]
        pdb = name.split('_')[0]
        pdb_file = os.path.join(self.pdb_path, '%s.pdb' % pdb)
    
        if not os.path.exists(pdb_file):
            print('%s not found!' % pdb_file)
            return None
    
        # motif info: e.g. "5-20,A16-35,10-25,A52-71,5-20"
        motif_info = line[2].split(',')
        if len(motif_info) == 0:
            raise ValueError(f'Empty motif info for {name}!')


        ###### read the pdb ######

        ### chain id
        for token in motif_info:
            if token[0] not in '0123456789':
                chain_id = token[0]
                break

        ### pdb loading 
        (
            coor_mat,   # (L, 15, 3)
            seq_array,  # (L, )
            seq,        # string; (L, )
            atom_mask,  # (L, 15)
            resi_start, # scalar
            index_list, # (L, )
        ) = pdb_info_read(pdb_file, chain_id)
    
        ###### overall length ######
        L = coor_mat.shape[0]
        l_range = tuple([int(val) for val in line[-1].split('-')])

        ###### motif and scaffolding region ######
        motif_region = []
        scaffold_length = []

        ### start
        if motif_info[0][0] in '0123456789':
            scaffold_length.append(tuple([int(l) for l in motif_info[0].split('-')]))
            motif_info.pop(0)
        else:
            scaffold_length.append((0,0))

        ### other pieces
        scaffold_idx = 1
        for reg in motif_info:
            ### scaffold length
            if reg[0] in '0123456789':
                scaffold_length.insert(scaffold_idx, tuple([int(l) for l in reg.split('-')]))
                scaffold_idx += 1
            ### motif
            else:
                motif_region.append(tuple([int(l) for l in reg[1:].split('-')]))
        motif_region = sorted(motif_region)
        motif_region_copy = motif_region.copy()

        ###### motif ###### 
        motif_coor = []
        motif_seq = []
        motif_atom_mask = []
        scaffold_length_true = []
        sca_len = 0
        motif_flag = False
        motif_size = 0

        # print(name)
        # print(motif_region)
        # print(index_list)

        for i, idx in enumerate(index_list):

            ### motif stop
            if motif_region and idx > motif_region[0][1]:
                motif_flag = False
                motif_region.pop(0)

                motif_coor.append(np.vstack(motif_coor_piece[0]))
                motif_seq.append(np.array(motif_coor_piece[1]))
                motif_atom_mask.append(np.vstack(motif_coor_piece[2]))
                motif_coor_piece = [[], [], []]

            ### motif region
            elif motif_region and idx >= motif_region[0][0]:
                if not motif_flag:
                    scaffold_length_true.append(sca_len)
                    sca_len = 0
                    motif_coor_piece = [[], [], []]
                motif_flag = True

            if motif_flag:
                motif_coor_piece[0].append(coor_mat[i:i+1]) # (1, 15, 3) 
                motif_coor_piece[1].append(seq_array[i]) # (1,)
                motif_coor_piece[2].append(atom_mask[i]) # (15,)
                motif_size += 1
            else:
                sca_len += 1

        if motif_coor_piece[0]: # last piece is motif
            motif_coor.append(np.vstack(motif_coor_piece[0]))
            motif_seq.append(np.array(motif_coor_piece[1]))
            motif_atom_mask.append(np.vstack(motif_coor_piece[2]))
        else: # last piece is scaffold
            scaffold_length_true.append(sca_len)
    
        return {
            ### original 
            'coor': coor_mat,
            'seq_array': seq_array,
            'seq': seq,
            'atom_mask': atom_mask,
            'index_list': index_list,
            ### motifs
            'motif_region': motif_region,        # List of tuple
            'motif_coor': motif_coor,            # List of array
            'motif_seq_array': motif_seq,        # List of array
            'motif_atom_mask': motif_atom_mask,  # List of array
            'motif_size': motif_size,
            ### scaffold 
            'scaffold_length_true': scaffold_length_true, # List of int
            'scaffold_length': scaffold_length,  # List of tuple; length range for sampling
            ### others
            'name': name,
            'length': L,
            'length_range': l_range,
            'length_max': max(L, l_range[-1]),
            'resi_start': resi_start,
        }


############### dataloader for self-defined datasets ##########################

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
    parser.add_argument('--sample_length', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_type', type=str, default='sele', help='"sele", "all" or "last"')
    parser.add_argument('--save_steps', type=int, nargs='*', default=[0])
    parser.add_argument('--t_bias', type=int, default=-1)
    parser.add_argument('--seq_sample_method', type=str, default='multinomial')

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
    args.sample_length = bool(args.sample_length)

    if args.save_type == 'last':
        args.save_steps = [0]

    ###########################################################
    # Model Loading 
    ###########################################################

    checkpoint = torch.load(args.model_path, map_location = args.device)
    config = checkpoint['config']

    ###### define the model
    model = DiffusionSingleChainDesign(config.model).to(args.device)

    ###### parameters prepare ######
    parameter_dict = model.state_dict()
    parameter_set = set(parameter_dict.keys())
    ### map the parameters
    for key in checkpoint['model'].keys():
        if key.startswith('module'):
            key_new = key[7:]
        else:
            key_new = key

        if key_new in parameter_dict:
            parameter_dict[key_new] = checkpoint['model'][key]
            parameter_set.remove(key_new)
        else:
            print('parameter %s not needed.' % key_new)

    ### load the dictionary
    model.load_state_dict(parameter_dict)
    print('Model loaded from %s.' % args.model_path)
    print('Number of parameters: %d' % count_parameters(model))

    ### unloaded parameters
    for name in parameter_set:
        print('%s not loaded.' % name)
    print('**********************************************************')

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
            force_cover = False,
            length_sampling = args.sample_length  
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

            try:
            #if True:
                ###### inference ######
                out_dict, traj = infer_function(
                    mask_res = batch['mask'],
                    mask_generate = batch['mask_gen'],
                    batch = batch, 
                    t_bias = args.t_bias,
                    seq_sample_method = args.seq_sample_method,
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
           
            except Exception as e:
                print(e)

    ###### summarizing ######
    print('%d samples genrated in %.4fs.'%(sample_num, time.time() - start_time))

