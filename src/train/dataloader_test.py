import os
import shutil
import argparse
from tqdm.auto import tqdm
from easydict import EasyDict

import sys
sys.path.append('./')
sys.path.append('../')

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn  
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.multiprocessing

from jointdiff.trainer import get_dataset
from jointdiff.modules.data.data_utils import PaddingCollate
from jointdiff.modules.utils.misc import (
    get_new_log_dir, get_logger, inf_iterator,
)

######################################################################################
# Arguments                                                                          #
######################################################################################

def arguments():

    parser = argparse.ArgumentParser()

    ############################# paths ######################################
    parser.add_argument('--train_summary', type=str,
        #default='../data/cath_summary_all.tsv'
        default='../data/ProteinMPNN/mpnn_data_info.pkl'
    )
    parser.add_argument('--train_pdb', type=str,
        #default='../../../documents/Data/Origin/CATH/pdb_all'
        default=None
    )
    parser.add_argument('--train_processed', type=str,
        #default='../data/'  # CATH
        default='../data/ProteinMPNN/'  # ProteinMPNN
    )
    parser.add_argument('--interface_path', type=str,
        default='../data/ProteinMPNN/interface_dict_all.pt'  # ProteinMPNN only
    )
    parser.add_argument('--val_summary', type=str,
        default='../data/cath_summary_all.tsv'
    )
    parser.add_argument('--val_pdb', type=str,
        default='../../../documents/Data/Origin/CATH/pdb_all'
    )
    parser.add_argument('--val_processed', type=str,
        #default='../data/'  # CATH
        default='../data/ProteinMPNN/'  # ProteinMPNN
    )

    ########################## dataloader setting ##############################
    #parser.add_argument('--data_version', type=str, default='monomer')
    parser.add_argument('--data_version', type=str, default='multimer')
    #parser.add_argument('--with_scaffold', type=int, default=1)
    #parser.add_argument('--random_masking', type=int, default=0)

    ############################ set up ########################################
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_iters', type=int, default=30)

    ########################### arguments summary ##############################
    args = parser.parse_args()

    ###### arguments summarization ######
    args_dict = {
        ###### data ######
        'data_version': args.data_version,
        'dataset': {
            'train': {
                'summary_path': args.train_summary,
                'pdb_dir': args.train_pdb,
                'processed_dir': args.train_processed,
            },
        },
        ###### dataloder ######
        'dataloader': {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'max_iters': args.max_iters,
        },
    }
    if args.data_version == 'monomer':
        args_dict['dataset']['train']['split'] = 'train'
    else:
        args_dict['dataset']['train']['dset'] = 'train'
        args_dict['dataset']['train']['interface_path'] = args.interface_path

    ### validation dataset
    if args.data_version == 'monomer' \
    and args.val_summary is not None and args.val_summary.upper() != 'NONE' \
    and args.val_pdb is not None and args.val_pdb.upper() != 'NONE' \
    and args.val_processed is not None and args.val_processed.upper() != 'NONE':
        args_dict['dataset']['val'] = {
            'summary_path': args.val_summary,
            'pdb_dir': args.val_pdb,
            'processed_dir': args.val_processed,
            'split': 'val',
        }

    return EasyDict(args_dict)


######################################################################################
# Main Function                                                                      #
######################################################################################

def main(config):

    #######################################################################
    # Data Loading
    #######################################################################
    print('Loading dataset...')

    ###### training set ######
    train_dataset = get_dataset(
        config.dataset.train, version = config.data_version
    )
    train_iterator = inf_iterator(DataLoader(
        train_dataset,
        batch_size = config.dataloader.batch_size,
        collate_fn = PaddingCollate(),
        shuffle = True,
        num_workers = config.dataloader.num_workers,
    ))

    ###### validation set ######
    if config.dataset.__contains__('val'):
        val_dataset = get_dataset(
            config.dataset.val, verison = config.data_version
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.dataloader.batch_size,
            collate_fn=PaddingCollate(), 
            shuffle=False, 
            num_workers=config.dataloader.num_workers,
        )
        print('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))
    else:
        val_loader = None
        print('Train %d | No validation set' % (len(train_dataset)))

    #######################################################################
    # dataloading
    #######################################################################

    for it in tqdm(range(config.dataloader.max_iters)):
        batch = next(train_iterator)
        
    ###### validation & save the checkpoints ######
    print()
    print("Features...")
    for key in batch:
        if isinstance(batch[key], list):
            print(key, len(batch[key]), batch[key][0])
        else:
            print(key, batch[key].shape)


######################################################################################
# Running the Script                                                                 #
######################################################################################

if __name__ == '__main__':
    config = arguments()
    main(config)

