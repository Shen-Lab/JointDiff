# diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'
import os
import shutil
import argparse
import pickle
from Bio.PDB import PDBParser, is_aa

from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.stats import spearmanr

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jointdiff.modules.utils.train import *
from jointdiff.modules.utils.misc import *
from jointdiff.modules.data.data_utils import *
from jointdiff.model import ConfidenceNet
from jointdiff.modules.data.constants import ressymb_order
# diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'
from jointdiff.trainer import dict_save, RESIDUE_reverse_dict

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')  
torch.autograd.set_detect_anomaly(True)

char_to_idx = {}
for i, char in enumerate(ressymb_order):
    char_to_idx[char] = i

######################################################################################
# DataLoader                                                                         #
######################################################################################

class confidence_dataloader_infer(Dataset):
    def __init__(self, pdb_dir):
        super().__init__()
        self.name_list = [p[:-4] for p in os.listdir(pdb_dir) if p.endswith('.pdb')]
        if not self.name_list:
            raise Exception('No pdb found in %s!' % pdb_dir)

        self.data_dict = {}
        self.max_len = 0
        for name in self.name_list:
            pdb_path = os.path.join(pdb_dir, '%s.pdb' % name)
            self.data_dict[name] = self.pdb_load(pdb_path)
            self.max_len = max(self.max_len, self.data_dict[name]['length'])

        ### add padding
        for name in self.name_list:
            self.data_dict[name] = self.add_padding(self.data_dict[name])


    def pdb_load(self, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]

        backbone_atoms = ['N', 'CA', 'C', 'O']
        coords = []
        sequence = []
        atom_mask = []

        for chain in model:
            for residue in chain:
                if not is_aa(residue):
                    continue
                residue_coords = []
                atom_mask_resi = []
                missing_atom = False

                for atom_name in backbone_atoms:
                    if atom_name in residue:
                        residue_coords.append(residue[atom_name].get_coord())
                        atom_mask_resi.append(1)
                    else:
                        residue_coords.append(np.zeros(3))
                        atom_mask_resi.append(0)

                coords.append(residue_coords)
                sequence.append(residue.get_resname())
                atom_mask.append(atom_mask_resi)

        coords = np.array(coords)
        atom_mask = np.array(atom_mask)
        length = len(sequence)
        sequence = np.array([
            char_to_idx[RESIDUE_reverse_dict[res]]
            for res in sequence
        ])

        return {
            'pos_heavyatom': coords,
            'aa': sequence,
            'mask_heavyatom': (atom_mask == 1),
            'mask': np.sum(atom_mask, axis = -1) > 0,
            'res_nb': np.arange(1, length+1),
            'length': length,
        }


    def add_padding(self, data_dict):
        pad_len = self.max_len - data_dict['length']
        for key in data_dict:
            if key == 'length':
                continue
            elif key == 'pos_heavyatom':
                pad_width = ((0, pad_len), (0, 0), (0, 0))
            elif key == 'mask_heavyatom':
                pad_width = ((0, pad_len), (0, 0))
            else:
                pad_width = ((0, pad_len))
            data_dict[key] = np.pad(
                data_dict[key] , pad_width, mode='constant'
            )
        return data_dict 


    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        name = self.name_list[index]
        data = self.data_dict[name]
        data['idx'] = index
        return data


######################################################################################
# Evaluation Fucntions                                                               #
######################################################################################
    
def inference(model, dataloader, args):

    model.eval()
    thre_dict = {
        'consist-seq': args.consist_seq_thre, 
        'consist-stru': args.consist_stru_thre, 
        'foldability': args.foldability_thre, 
        'designability': args.designability_thre
    }

    ############################################################################
    # batch-wise inference 
    ############################################################################

    pred_all = []
    label_all = []

    for i, batch in tqdm(enumerate(dataloader)):

        #if i >= 100: # for debugging
        #    break

        # for key in batch:
        #     print(key, batch[key].shape)
        #     print(batch[key])

        batch = recursive_to(batch, args.device)
        batch['chain_nb'] = batch['mask']
        batch['pos_heavyatom'] = batch['pos_heavyatom'][:, :, :args.num_atoms, :] 
        batch['mask_heavyatom'] = batch['mask_heavyatom'][:, :, :args.num_atoms] 

        ###### centralize the data ######
        if args.centralize:
            mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
            mean = mean / batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1)  # (N, 3)
            batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 4, 3)
            batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

        ############################################################################
        # Inference
        ############################################################################

        pred = model(batch)  # (N, 4)
        pred = pred.detach().cpu()

        batch = recursive_to(batch, 'cpu')
        torch.cuda.empty_cache()

        pred_all.append(pred)

    ### predictions
    pred_all = torch.cat(pred_all, dim = 0)  # (N, 4)

    if args.label_norm:
        for i, metric in enumerate(
            ['consist-seq', 'consist-stru', 'foldability', 'designability']
        ):
            pred_vec = pred_all[:, i]
            ### denormalization
            if args.label_norm:
                min_val, max_val = args.min_max_dict[metric]
                pred_all[:, i] = (pred_vec - 1) * (max_val - min_val) / 2 + max_val

    print(pred_all.mean(0))

    return pred_all.numpy()


######################################################################################
# Main Function                                                                      #
######################################################################################

def main(args):

    #######################################################################
    # load the Model
    #######################################################################

    print('Loading from checkpoint: %s' % args.ckpt_path)
    ckpt = torch.load(args.ckpt_path, map_location=args.device) #, weights_only = True)
    model_args = ckpt['args']
    model_args.consist_seq_thre = args.consist_seq_thre
    model_args.consist_stru_thre = args.consist_stru_thre
    model_args.foldability_thre = args.foldability_thre
    model_args.designability_thre = args.designability_thre

    model = ConfidenceNet(
        res_feat_dim = model_args.res_feat_dim,
        pair_feat_dim = model_args.pair_feat_dim,
        num_layers = model_args.num_layers,
        num_atoms = model_args.num_atoms,
        max_relpos = model_args.max_relpos,
        binary = model_args.binary
    ).to(args.device)
    print('Number of parameters: %d' % count_parameters(model))

    ckpt['model'] = {''.join(key.split('module.')[:]) : ckpt['model'][key]
        for key in ckpt['model']
    }
    model.load_state_dict(ckpt['model'])

    ################## Parallel  ##################
    if args.multi_gpu:
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        print('%d GPUs detected. Applying parallel training and a batch size of %d.' % 
            (torch.cuda.device_count(), args.batch_size)
        )

    elif args.device == 'cuda':
        print('Applying single GPU training with a batch size of %d.' % 
            (args.batch_size)
        )

    else:
        print('Applying CPU training with a batch size of %d.' %
            (args.batch_size)
        )

    #######################################################################
    # Data Loading
    #######################################################################

    test_dataset = confidence_dataloader_infer(args.pdb_dir)
    test_dataloader = DataLoader(
        test_dataset, batch_size = args.batch_size,
        shuffle=False, 
        num_workers = args.num_workers
    )
    print('%d test samples loaded.'  % (len(test_dataset)))

    #######################################################################
    # Evaluation
    #######################################################################
    
    model_args.min_max_dict = {
        'consist-seq': args.consist_seq_range,
        'consist-stru': args.consist_stru_range,
        'foldability': args.foldability_range,
        'designability': args.designability_range,
    }
    pred = inference(model, test_dataloader, model_args)
    if args.result_path is not None:
        np.save(args.result_path, pred)


######################################################################################
# Running the Script                                                                 #
######################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##################### paths and name ######################################
    parser.add_argument('--pdb_dir', type=str,
        default='../../Documents/Data/PDB_forDebug/'
    )
    parser.add_argument('--ckpt_path', type=str,
        default='../../Documents/TrainedModels/JointDiff/logs_confidence_development/confidence_128-64-6_posiscale50.0000_regression_/checkpoints/checkpoint_35.pt'
    )
    parser.add_argument('--result_path', type=str, default=None)

    ########################### setting #######################################
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    ###### data ######
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    ###### for label process ######
    ### label threshold
    parser.add_argument('--consist_seq_thre', type=float, default=0.3)
    parser.add_argument('--consist_stru_thre', type=float, default=2.0)
    parser.add_argument('--foldability_thre', type=float, default=0.3)
    parser.add_argument('--designability_thre', type=float, default=2.0)
    ### label normalization
    parser.add_argument(
        '--consist_seq_range', type=float, nargs='*', default=[0.004364, 0.800595]
    )
    parser.add_argument(
        '--consist_stru_range', type=float, nargs='*', default=[0.0, 6.67]
    )
    parser.add_argument(
        '--foldability_range', type=float, nargs='*', default=[0.010417, 0.807143]
    )
    parser.add_argument(
        '--designability_range', type=float, nargs='*', default=[0.20375, 5.76]
    )
    ###### setting ######
    parser.add_argument('--debug', type=int, default=0)

    args = parser.parse_args()

    ###### arguments setting ######
    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if args.result_path is None or args.result_path.upper() == 'NONE':
        args.result_path = None
    args.debug = bool(args.debug)

    ###### running ######
    main(args)
