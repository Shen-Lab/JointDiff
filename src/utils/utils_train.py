import argparse
import os
import pickle
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torch.distributed as dist

###############################################################################
# Constants and Auxciliary Functions
###############################################################################

mpnn_restypes = 'ACDEFGHIKLMNPQRSTVWYX'
esm_restypes = 'ARNDCQEGHILKMFPSTWYVX'


def dict_save(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0


def dict_load(path):
    with open(path, 'rb') as handle:
        result = pickle.load(handle)
    return result


def add_right_padding(tensor_ori, dim=[0], pad_length=[1], val=0):
    dim_num = len(tensor_ori.shape)
    pad = [0] * (dim_num * 2)
    for i, d in enumerate(dim):
        pad[2*(dim_num - d)-1] = pad_length[i]
    return F.pad(tensor_ori, tuple(pad), 'constant', val)


###############################################################################
# DataLoader of Autoencoders
###############################################################################

class AutoencoderDataset(Dataset):
    def __init__(self,
        args,
        key_list=[
            'aatype',
            'seq_mask',
            'pseudo_beta',
            'pseudo_beta_mask',
            'backbone_rigid_tensor',
            'backbone_rigid_mask',
            'rigidgroups_gt_frames',
            'rigidgroups_alt_gt_frames',
            'rigidgroups_gt_exists',
            'atom14_gt_positions',
            'atom14_alt_gt_positions',
            'atom14_gt_exists',
            'atom14_atom_is_ambiguous',
            'atom14_alt_gt_exists',
        ]
    ):

        self.voxel_size = len(args.esm_restypes)

        data_info_all = dict_load(args.data_path)
        entry_list = dict_load(args.entry_list_path)
        if args.debug_num is not None:
            entry_list = entry_list[:args.debug_num]

        self.data = []
        self.name_list = []
        self.padded_length = 0
        discard_num = 0
        sample_idx = 0

        for entry in entry_list:
            if entry not in data_info_all:
                discard_num += 1
                continue

            length = data_info_all[entry]['aatype'].shape[0]

            if args.max_length is None or length <= args.max_length:
                data_info = {key: data_info_all[entry][key]
                             for key in key_list}
                self.padded_length = max(self.padded_length, length)
                data_info['length'] = length
                data_info['residx'] = torch.arange(length) + 1
                data_info['sample_idx'] = torch.tensor(sample_idx)
                sample_idx += 1

                self.data.append(data_info)
                self.name_list.append(entry)

            else:
                discard_num += 1

        if args.max_length is not None:
            self.max_length = args.max_length
            self.padded_length = args.max_length
        else:
            self.max_length = self.padded_length

        print('%d entries loaded. %d entries discarded.' %
              (self.__len__(), discard_num))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        aatype: (L,),
        seq_mask: (L,),
        residx: (L,),
        pseudo_beta: (L, 3),
        pseudo_beta_mask: (L,),
        backbone_rigid_tensor: (L, 4, 4),
        backbone_rigid_mask: (L,),
        rigidgroups_gt_frames: (L, 8, 4, 4),
        rigidgroups_alt_gt_frames: (L, 8, 4, 4),
        rigidgroups_gt_exists: (L, 8),
        atom14_gt_positions: (L, 14, 3),
        atom14_alt_gt_positions: (L, 14, 3),
        atom14_gt_exists: (L, 14),
        atom14_atom_is_ambiguous: (L, 14),
        atom14_alt_gt_exists: (L, 14),
        """

        data_info = self.data[idx]
        pad_length = self.padded_length - data_info['length']

        for key in data_info:
            if key not in {'sample_idx', 'name', 'length'}:
                if key == 'aatype':
                    pad_val = 0
                else:
                    pad_val = 0
                data_info[key] = add_right_padding(
                    data_info[key], dim=[0], pad_length=[pad_length], val=pad_val
                )

        return data_info


class LatentAutoencoderDataset_dynamic(Dataset):
    def __init__(self, args):
        """Dataset for the latent AE on sequence and structure embedding."""

        ######################### Settings #####################################

        self.args = args
        self.with_ori_data = args.with_ori_data
        self.with_pair_feat = args.with_pair_feat

        if args.__contains__('align'):
            self.align = args.align
        else:
            self.align = 'left'

        self.seq_emb_path = args.seq_emb_path
        self.stru_emb_path = args.stru_emb_path
        length_dict = dict_load(args.protein_length_dict)

        ############################# filter entries ###########################

        ### original entry list
        entry_list = dict_load(args.entry_list_path)
        if args.debug_num is not None:
            entry_list = entry_list[:args.debug_num]

        ### original data (sequence, structure)
        if self.with_ori_data:
            self.ori_data_dict = dict_load(args.ori_data_path)
        else:
            self.ori_data_dict = None

        ### for selected entries
        self.name_list = []
        discard_num = 0

        for entry in entry_list:
            entry = entry[:4] + '_' + entry[5:]
            seq_emb_file = os.path.join(args.seq_emb_path, '%s.pt' % entry)
            struc_emb_file = os.path.join(args.stru_emb_path, '%s.pt' % entry)

            if not (os.path.exists(seq_emb_file) and os.path.exists(struc_emb_file)):
                discard_num += 1
                continue
            elif args.with_ori_data and (entry not in self.ori_data_dict):
                discard_num += 1
                continue
            elif entry not in length_dict:
                discard_num += 1
                continue
            else:
                self.name_list.append(entry)

        ################################ masks ################################
        self.mask_dict = {}
        length_list = []
        for entry in self.name_list:
            length = length_dict[entry]
            self.mask_dict[entry] = np.ones(length)
            length_list.append(length)

        if args.max_length is None:
            self.max_length = max(length_list)
        else:
            self.max_length = args.max_length

        self.seq_max_length = self.max_length
        if self.with_pair_feat:
            self.struc_max_length = self.max_length
        else:  
            self.struc_max_length = self.max_length + 2

        print('%d entries loaded. %d entries discarded.' %
            (self.__len__(), discard_num)
        )

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        """
        node: (L, dim),
        pair: (L, L, dim),
        mask: (L,),
        """
        entry = self.name_list[idx]
        data_out = {}

        ######################## structure feature ##############################
        struc_emb = torch.load(os.path.join(self.stru_emb_path, '%s.pt' % entry))
        if self.with_pair_feat:
            pad_length = self.struc_max_length - struc_emb['node_feat'].shape[0]
        else:
            pad_length = self.struc_max_length - struc_emb.shape[0]

        if self.align == 'left':
            left_pad = 0
            right_pad = pad_length
        elif self.align == 'right':
            left_pad = pad_length
            right_pad = 0
        else:
            left_pad = math.ceil(pad_length / 2)
            right_pad = pad_length - right_path

        ### for ProteinMPNN
        if self.with_pair_feat:
            data_out['struc_feat'] = np.pad(
                struc_emb['node_feat'], ((left_pad, right_pad), (0,0))
            )  # (L_max, dim)
            data_out['struc_pair_feat'] = np.pad(
                struc_emb['pair_feat'],
                ((left_pad, right_pad), (left_pad, right_pad), (0,0))
            )  # (L_max, dim)

        ### for ESM-IF
        else:
            data_out['struc_feat'] = np.pad(
                struc_emb, ((left_pad, right_pad), (0,0))
            )  # (L_max, dim)

        ######################## sequence feature ##############################
        seq_emb = torch.load(os.path.join(self.seq_emb_path, '%s.pt' % entry))
        pad_length = self.seq_max_length - seq_emb['node_feat'].shape[0]
        if self.align == 'left':
            left_pad = 0 
            right_pad = pad_length 
        elif self.align == 'right':
            left_pad = pad_length
            right_pad = 0
        else:
            left_pad = math.ceil(pad_length / 2)
            right_pad = pad_length - right_path

        data_out['seq_feat'] = np.pad(
            seq_emb['node_feat'], ((left_pad, right_pad), (0,0))
        )  # (L_max, dim)

        if self.with_pair_feat:
            data_out['seq_pair_feat'] = np.pad(
                seq_emb['pair_feat'], 
                ((left_pad, right_pad), (left_pad, right_pad), (0,0))
            )  # (L_max, dim)

        ########################## mask and others ##############################

        data_out['length'] = self.mask_dict[entry].shape[0]
        pad_length = self.seq_max_length - data_out['length']
        if self.align == 'left':
            left_pad = 0
            right_pad = pad_length
        elif self.align == 'right':
            left_pad = pad_length
            right_pad = 0
        else:
            left_pad = math.ceil(pad_length / 2)
            right_pad = pad_length - right_path

        data_out['seq_mask'] = np.pad(
            self.mask_dict[entry], ((left_pad, right_pad))
        )  # (L_max,)
        data_out['residx'] = np.pad(
            np.arange(1, data_out['length']+1), ((left_pad, right_pad))
        )
        data_out['sample_idx'] = idx
        data_out['name'] = entry

        ######################## original feature ###############################
     
        if self.with_ori_data:
            for key in self.ori_data_dict[entry]:
                data_out[key] = add_right_padding(
                    self.ori_data_dict[entry][key], 
                    dim=[0], pad_length=[pad_length], val=0
                )

        return data_out


###############################################################################
# DataLoader of Latent Diffusion Models
###############################################################################

class DiffusionDataset(Dataset):
    def __init__(self, args, key_list=['node', 'pair']):
        """Dataset of the latent diffusion model.

        Laod the samples at the beginning (require more than 180G of the CPU 
        space).
        """

        ######################### Settings #####################################

        entry_list = dict_load(args.entry_list_path)
        if args.debug_num is not None:
            entry_list = entry_list[:args.debug_num]

        self.key_list = key_list
        self.data = []
        self.name_list = []
        self.padded_length = 0
        discard_num = 0
        sample_idx = 0

        ######################### Load the data ###############################

        for entry in entry_list:
            entry = entry[:4] + '_' + entry[5:]
            sample_path = os.path.join(args.data_path, '%s.pkl' % entry)
            
            if os.path.exists(sample_path):
                sample_dict = dict_load(sample_path)
                length = sample_dict['node'].shape[0]
            else:
                discard_num += 1
                continue

            if args.max_length is None or length <= args.max_length:
                data_info = {key: sample_dict[key]
                    for key in key_list
                }
                self.padded_length = max(self.padded_length, length)
                data_info['sample_idx'] = torch.tensor(sample_idx)
                data_info['name'] = entry
                data_info['length'] = torch.tensor(length)
                sample_idx += 1

                self.data.append(data_info)
                self.name_list.append(entry)

            else:
                discard_num += 1

            if args.max_length is not None:
                self.max_length = args.max_length
                self.padded_length = args.max_length
            else:
                self.max_length = self.padded_length

        print('%d entries loaded. %d entries discarded.' %
            (self.__len__(), discard_num)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        node: (L, dim),
        pair: (L, L, dim),
        mask: (L,),
        """
        data_info = self.data[idx]
        pad_length = self.padded_length - data_info['length']

        ###### mask ######
        data_info['mask'] = torch.ones(
            data_info['node'].shape[0],
        )  # (L,)

        ###### padding ######
        data_info['node'] = F.pad(
            torch.from_numpy(data_info['node']), 
            (0, 0, 0, pad_length), 'constant', 0
        )  # (L_max, dim) 

        data_info['mask'] = F.pad(
            data_info['mask'], (0, pad_length), 'constant', 0
        )  # (L_max,) 

        if 'pair' in data_info:
            data_info['pair'] = F.pad(
                torch.from_numpy(data_info['pair']), 
                (0, 0, 0, pad_length, 0, pad_length), 'constant', 0
            )  # (L_max, L_max, dim)  

        return data_info


class DiffusionDataset_dynamic(Dataset):
    def __init__(self, 
        args, key_list=['node', 'pair'], with_ori_feat = False,
        key_list_ori=[
            'aatype',
            'seq_mask',
            'pseudo_beta',
            'pseudo_beta_mask',
            'backbone_rigid_tensor',
            'backbone_rigid_mask',
            'rigidgroups_gt_frames',
            'rigidgroups_alt_gt_frames',
            'rigidgroups_gt_exists',
            'atom14_gt_positions',
            'atom14_alt_gt_positions',
            'atom14_gt_exists',
            'atom14_atom_is_ambiguous',
            'atom14_alt_gt_exists',
        ]
    ):
        """Dataset of the latent diffusion model.

        Dynamically load the samples in each iteration.
        """

        ######################### Settings #####################################

        self.args = args
        self.key_list = key_list
        self.with_mask = (
            args.__contains__('with_mask') \
            and args.__contains__('protein_length_dict') \
            and args.__contains__('kernel_size') \
            and args.with_mask
        )
        self.with_ori_feat = with_ori_feat
        if self.with_mask or self.with_ori_feat:
            length_dict = dict_load(args.protein_length_dict)

        ########################## filter the entries #########################
        entry_list = dict_load(args.entry_list_path)
        if args.debug_num is not None:
            entry_list = entry_list[:args.debug_num]

        self.name_list = []
        discard_num = 0

        for entry in entry_list:
            entry = entry[:4] + '_' + entry[5:]
            sample_path = os.path.join(args.data_path, '%s.pkl' % entry)

            if os.path.exists(sample_path) \
            and ((not self.with_mask) or (self.with_mask and entry in length_dict)):
                self.name_list.append(entry) 
            else:
                discard_num += 1
                continue

        ######################## prepare masks #################################
        if self.with_mask:
            self.mask_dict = {}
            for entry in self.name_list:
                length = length_dict[entry]
                length_emb = math.ceil((length - 1) / (args.kernel_size - 1))
                self.mask_dict[entry] = np.ones(length_emb)

        ###################### original features ################################
        if self.with_ori_feat:

            if not self.args.__contains__('max_length_ori'):
                self.args.max_length_ori = self.args.max_length * 2

            data_info_ori = dict_load(args.ori_data_path)
            self.data_info_ori = {}
            for entry in self.name_list:
                self.data_info_ori[entry] = {
                    key:data_info_ori[entry][key] for key in key_list_ori
                }
                length = length_dict[entry]
                self.data_info_ori[entry]['length_ori'] = length
                self.data_info_ori[entry]['residx'] = torch.arange(length) + 1

        print('%d entries loaded. %d entries discarded.' %
            (self.__len__(), discard_num)
        )

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        """
        node: (L, dim),
        pair: (L, L, dim),
        mask: (L,),
        """
        name = self.name_list[idx]

        ###################### latent embedding for diffusion ##################
        sample_path = os.path.join(self.args.data_path, '%s.pkl' % name)
        data_info = dict_load(sample_path)

        ###### node feature ######
        if 'node' in data_info and data_info['node'].shape[0] < self.args.max_length:
            pad_length = self.args.max_length - data_info['node'].shape[0]
            data_info['node'] = np.pad(
                data_info['node'], 
                ((0, pad_length), (0,0))
            )
        
        ###### pair feature ######
        if 'pair' in data_info and data_info['pair'].shape[0] < self.args.max_length:
            pad_length = self.args.max_length - data_info['pair'].shape[0]
            data_info['pair'] = np.pad(
                data_info['pair'],
                ((0,pad_length), (0,pad_length), (0,0))
            )

        ###### latent mask ######
        if self.with_mask:
            mask = self.mask_dict[name]  # (m,)
            pad_length = self.args.max_length - mask.shape[0]
            data_info['mask'] = np.pad(
                mask, (0, pad_length)
            )
            if 'pair' in data_info and data_info['pair'] is not None:
                pair_mask = np.matmul(
                    mask.reshape(-1,1), mask.reshape(1,-1)
                )  # (m,m)
                data_info['pair_mask'] = np.pad(
                    pair_mask, (0, pad_length)
                )  # (L,L)

        ############################ original features ########################

        if self.with_ori_feat:
            feat_ori = self.data_info_ori[name]
            pad_length_ori = self.args.max_length_ori - feat_ori['length_ori']
            for key in feat_ori:
                if key not in {'sample_idx', 'name', 'length_ori'}:
                    pad_val = 0
                    feat_ori[key] = add_right_padding(
                        feat_ori[key], dim=[0], pad_length=[pad_length_ori], val=pad_val
                    )
            data_info.update(feat_ori)

        ######################### name and index ##############################

        data_info['sample_idx'] = idx
        data_info['name'] = name

        return data_info


###############################################################################
# Model check
###############################################################################

def model_size_check(model):
    param_size = 0
    param_size_train = 0
    for param in model.parameters():
        element_size = param.nelement() * param.element_size()
        param_size += element_size
        if param.requires_grad:
            param_size_train += element_size

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model parameters: {}'.format(param_size))
    print('model parameters for training: {}'.format(param_size_train))
    print('buffer parameters: {}'.format(buffer_size))
    print('model size: {:.3f}MB'.format(size_all_mb))


def check_parameters_on_device(model):
    """
    Check which parameters and gradients are on the GPU.

    Args:
        model: PyTorch model.

    Returns:
        None
    """
    print("Parameters on GPU:")
    for name, param in model.named_parameters():
        if param.is_cuda:
            element_size = param.nelement() * param.element_size()
            print(name, element_size)
    print('###############################################################')

    print("Gradients on GPU:")
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.is_cuda:
            element_size = param.nelement() * param.element_size()
            print(name, element_size)
    print('###############################################################')


def setup(rank:int, world_size:int, version:str = 'ddp') -> None:
    """Set up the environment for parallel training.

    Args:
        rank: rank of the current branch.
        world_size: number of the branches.
        version: ddp or fsdp. 
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    if version == 'ddp':
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    elif version == 'fsdb':
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        raise NameError('No parallel version named %s!' % version)


def cleanup():
    dist.destroy_process_group()


def save_state(args, model, optimizer, Loss_all_dict, epo, name):
    save_dict = {
        'args': args,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epo,
    }
    if args.parallel_method == 'dp':
        save_dict['model_state_dict'] = model.module.state_dict()
    else:
        save_dict['model_state_dict'] = model.state_dict()
    ### save ###
    _ = dict_save(Loss_all_dict, args.save_path + '/loss.pkl')
    torch.save(save_dict, args.save_path + '/%s.pt' % name)


###############################################################################
# for debugging
###############################################################################


def main(args, data_module=AutoencoderDataset):
    dataset = data_module(args)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             num_workers=args.num_workers)

    for i, batch in enumerate(data_loader):
        print(i, batch['atom14_gt_positions'].shape)
        if i == 0:
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    print(key, batch[key].shape)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # for dataset
    argparser.add_argument('--data_path', type=str, default='../../Data/Processed/CATH_forLatentDiff/Latent_AE_data.pkl',
                           help='path for preprocessed data')
    argparser.add_argument('--entry_list_path', type=str, default='../../Data/Processed/CATH_forLatentDiff/test_data_list.pkl',
                           help='path for the entry list')
    argparser.add_argument('--max_length', type=int,
                           default=200, help='maximum length of the samples')
    argparser.add_argument('--esm_restypes', type=str,
                           default='ARNDCQEGHILKMFPSTWYVX', help='ordered voxel set of ESMFold')
    argparser.add_argument('--mpnn_restypes', type=str,
                           default='ACDEFGHIKLMNPQRSTVWYX', help='ordered voxel set of proteinMPNN')

    argparser.add_argument('--debug_num', type=int, default=200)
    # for data loader
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--shuffle', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=1)

    args = argparser.parse_args()

    args.shuffle = bool(args.shuffle)

    main(args)
