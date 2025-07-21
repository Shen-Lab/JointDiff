#####################################################
# continue training from a saved checkpoints 
#####################################################

import os
import shutil
import argparse
import math
import random

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn 
import torch.nn.functional as F_ 
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import torch.multiprocessing 
torch.multiprocessing.set_sharing_strategy('file_system') 
import time

from jointdiff.model import DiffusionSingleChainDesign
from jointdiff.modules.utils.train import get_optimizer, get_scheduler
from jointdiff.modules.utils.misc import (
    BlackHole, seed_all,
    get_new_log_dir, get_logger, inf_iterator,
)
from jointdiff.modules.data.data_utils import PaddingCollate
from jointdiff.trainer import (
    train, validate, load_config, get_dataset, count_parameters
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

######################################################################################
# Arguments                                                                          #
######################################################################################

def arguments():

    parser = argparse.ArgumentParser()
    
    ############################ Model ##########################################
    parser.add_argument('--model_path', type=str, 
        #default='../../Documents/TrainedModels/JointDiff/logs_jointdiff_development/jointdiff-x_joint_multinomial_model6-128-64-step100_posi-scale-50.0_micro-posi+fape-dist+l1-clashfape_dist-l1_clash_2025_06_20__23_44_35/checkpoints/128000.pt'
        default='../../Documents/TrainedModels/JointDiff/logs_debug/jointdiff-x_joint_multinomial_model6-128-64-step100_posi-scale-50.0_micro-posi+fape-dist+l1-clashfape_dist-l1_clash_2025_06_20__23_44_35/checkpoints/128000.pt'
    )

    ############################ Data ##########################################
    parser.add_argument('--train_summary', type=str,
        default='../Data/examples/summary_debug.tsv'
    )
    parser.add_argument('--train_pdb', type=str,
        default='../Data/examples/pdbs/'
    )
    parser.add_argument('--train_processed', type=str,
        default='../Data/examples/'
    )
    parser.add_argument('--val_summary', type=str,
        default='../Data/examples/summary_debug.tsv'
    )
    parser.add_argument('--val_pdb', type=str,
        default='../Data/examples/pdbs/'
    )
    parser.add_argument('--val_processed', type=str,
        default='../Data/examples/'
    )

    ############################ training setting ###############################
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--default_hp', type=int, default=1,
        help="whether use the default hyper-parameters saved in the checkpoints"        
    ) 
    ### device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    ### dataloader
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    ### training & val
    parser.add_argument('--max_iters', type=int, default=1000000)
    parser.add_argument('--val_freq', type=int, default=1000)

    args = parser.parse_args()

    args.debug = bool(args.debug)

    if args.device == 'cuda' and ( not torch.cuda.is_available() ):
        print('GPUs are not available! Use CPU instead.')
        args.device = 'cpu'
        args.multi_gpu = 0

    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
        args.batch_size *= torch.cuda.device_count()
    else:
        args.multi_gpu = False

    return args

######################################################################################
# Arguments                                                                          #
######################################################################################

def main(args):

    checkpoint = torch.load(args.model_path, map_location = args.device)
    config = checkpoint['config']
    seed_all(config.train.seed)
    config.train.device = args.device

    ###########################################################
    # Hyper-parameters
    ###########################################################

    ###### use new data ######
    if args.debug or (not args.default_hp):
        config.dataset.train.summary_path = args.train_summary
        config.dataset.train.pdb_dir = args.train_pdb
        config.dataset.train.processed_dir = args.train_processed

        if args.val_summary is not None and args.val_summary.upper() != 'NONE' \
        and args.val_pdb is not None and args.val_pdb.upper() != 'NONE' \
        and args.val_processed is not None and args.val_processed.upper() != 'NONE':
            config['dataset']['val'] = {
                'summary_path': args.val_summary,
                'pdb_dir': args.val_pdb,
                'processed_dir': args.val_processed,
                'split': 'val',
            }

    if not args.default_hp:
        config.dataloader.batch_size = args.batch_size
        config.dataloader.num_workers = args.num_workers

        config.train.max_iters = args.max_iters
        config.train.val_freq = args.val_freq

    ###########################################################
    # Logger
    ###########################################################

    if args.debug:
        print('Debugging...')
        logger = get_logger('train', None)
        writer = BlackHole()

    else:
        log_dir = os.path.dirname(
            os.path.dirname(args.model_path)
        )
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)

    logger.info(config)

    ###########################################################
    # Model Loading 
    ###########################################################

    ####################### Model #############################
    logger.info('Building model...')

    ###### define the model ######
    model = DiffusionSingleChainDesign(
        config.model
    ).to(config.train.device)

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
            logger.info('parameter %s not needed.' % key_new)

    ### load the dictionary
    model.load_state_dict(parameter_dict)
    logger.info('Model loaded from %s.' % args.model_path)
    logger.info('Number of parameters: %d' % count_parameters(model))

    ### unloaded parameters
    for name in parameter_set:
        logger.info('%s not loaded.' % name)
    logger.info('**********************************************************')

    ###### parallel ######
    if args.multi_gpu:
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        logger.info("%d GPUs detected. Applying parallel computation."%(torch.cuda.device_count()))

    ############ optimizer and scheduler ######################
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
 
    logger.info('Resuming optimizer states...')
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info('Resuming scheduler states...')
    scheduler.load_state_dict(checkpoint['scheduler'])

    ###########################################################
    # Data
    ###########################################################

    logger.info('Loading dataset...')

    ###### training set ######
    train_dataset = get_dataset(config.dataset.train)
    train_iterator = inf_iterator(DataLoader(
        train_dataset,
        batch_size = config.dataloader.batch_size,
        collate_fn = PaddingCollate(),
        shuffle = True,
        num_workers = config.dataloader.num_workers,
    ))

    ###### validation set ######
    if config.dataset.__contains__('val'):
        val_dataset = get_dataset(config.dataset.val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.dataloader.batch_size,
            collate_fn=PaddingCollate(),
            shuffle=False,
            num_workers=config.dataloader.num_workers,
        )
        logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))
    else:
        val_loader = None
        logger.info('Train %d | No validation set' % (len(train_dataset)))

    ###########################################################
    # Training
    ###########################################################

    it_first = checkpoint['iteration'] + 1
    logger.info('Start from Iteration %d...' % it_first)

    try:
        for it in range(it_first, config.train.max_iters + 1):
            ###### train ######
            train(
                model, optimizer, scheduler, train_iterator,
                logger, writer, config, it,
            )

            ###### validation & save the checkpoints ######
            if args.debug:
                continue

            if it % config.train.val_freq == 0 or it == it_first:
                ### save the model
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    },
                    ckpt_path
                )

                ### validation 
                if val_loader is None:
                    continue

                try:
                    avg_val_loss = validate(
                        model, scheduler, val_loader, logger, writer, config, it,
                    )
                except Exception as e:
                    avg_val_loss = torch.nan
                    logger.info('Validation error: %s' % e)

    except KeyboardInterrupt:
        logger.info('Terminating...')


######################################################################################
# Arguments                                                                          #
######################################################################################

if __name__ == '__main__':
    args = arguments()
    main(args)
