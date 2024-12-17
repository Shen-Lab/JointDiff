# diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'
import pandas as pd
from diffab.utils.protein.constants import ressymb_order
import torch.multiprocessing  # SZ
from diffab.utils.train import *
from diffab.utils.data import *
from diffab.utils.misc import *
from diffab.models import get_model
from diffab.datasets import get_dataset
from diffab.datasets.binders import ProteinMPNNDataset
import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn  # SZ
import torch.nn.functional as F  # SZ
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')  # SZ


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
    return ProteinMPNNDataset(
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


######################################################################################
# Train and Validation                                                               #
######################################################################################

def train(
    model, 
    optimizer, 
    scheduler, 
    train_iterator, 
    logger, 
    writer, 
    args, 
    config, 
    it,
):
    """
    engergy_guide: None or energy oracle
    fitness_guide: None or fitness oracle or str
    """

    time_start = current_milli_time()
    model.train()

    ############################################################################
    # Prepare data 
    ############################################################################

    batch = recursive_to(next(train_iterator), args.device)

    ###### centralize the data ######
    if args.design_centralize:
        batch['pos_heavyatom'] = postion_align(
            batch['pos_heavyatom'], batch['fragment_type']
        )

    elif args.centralize:
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

    batch['generate_flag'] = batch['generate_flag'].bool()
    batch['mask'] = batch['mask'].bool()
    batch['mask_heavyatom'] = batch['mask_heavyatom'].bool()

    # print(batch['aa'].shape)

    #print('Loss cal...')

    loss_dict = model(batch = batch)
    #print(loss_dict)

    for key in loss_dict:
        loss_dict[key] = loss_dict[key].mean()
    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
    loss_dict['overall'] = loss

    if torch.isnan(loss):
        print('nan detected!')
        return None

    time_forward_end = current_milli_time()

    ############################################################################
    # Backpropogate 
    ############################################################################

    loss.backward()
    orig_grad_norm = clip_grad_norm_(
        model.parameters(), config.train.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    time_backward_end = current_milli_time()

    ############################################################################
    # Record 
    ############################################################################

    log_losses(loss_dict, it, 'train', logger, writer, others={
        'grad': orig_grad_norm,
        'lr': optimizer.param_groups[0]['lr'],
        'time_forward': (time_forward_end - time_start) / 1000,
        'time_backward': (time_backward_end - time_forward_end) / 1000,
    })

    if not torch.isfinite(loss):
        logger.error('NaN or Inf detected.')
        torch.save({
            'config': config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': it,
            'batch': recursive_to(batch, 'cpu'),
        }, os.path.join(args.log_dir, 'checkpoint_nan_%d.pt' % it))
        raise KeyboardInterrupt()

    torch.cuda.empty_cache()


def validate(model, scheduler, val_loader,
             logger, writer, args, config, it):

    loss_tape = ValidationLossTape()
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
            # Prepare data
            batch = recursive_to(batch, args.device)
            # Forward
            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss

            loss_tape.update(loss_dict, 1)

            torch.cuda.empty_cache()

    avg_loss = loss_tape.log(it, logger, writer, 'val')
    # Trigger scheduler
    if config.train.scheduler.type == 'plateau':
        scheduler.step(avg_loss)
    else:
        scheduler.step()
    return avg_loss

######################################################################################
# Main Function                                                                      #
######################################################################################


def main(args):
    #######################################################################
    # Configuration and Logger
    #######################################################################
    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            args.log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            args.log_dir = get_new_log_dir(
                args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(args.log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        logger = get_logger('train', args.log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(args.log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(
            args.log_dir)
        if not os.path.exists(os.path.join(args.log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(
                args.log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    #######################################################################
    # Model and Optimizer
    #######################################################################

    ################## Model ########################
    logger.info('Building model...')

    config.model.chain_feat_version = args.chain_feat_version
    config.model.with_LLM = args.with_LLM
    config.model.with_dist = args.with_dist
    config.model.dist_clamp = args.dist_clamp
    config.model.with_rmsd = args.with_rmsd
    config.model.with_center_loss = args.with_center_loss
    config.model.with_clash_loss = args.with_clash_loss
    config.model.with_af3_relpos = args.with_af3_relpos
    config.model.cross_attn = args.cross_attn

    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    ################# Optimizer & scheduler #################
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()

    ################# Resume #################
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration'] + 1
        ckpt['model'] = {''.join(key.split('module.')[:]) : ckpt['model'][key]
                            for key in ckpt['model']}
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        it_first = 1
        logger.info('Starting from scratch...')

    ################## Parallel (by SZ) ##################
    if args.multi_gpu:
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        logger.info('%d GPUs detected. Applying parallel training and a batch size of %d.' % 
            (torch.cuda.device_count(), args.batch_size)
        )
    elif args.device == 'cuda':
        logger.info('Applying single GPU training with a batch size of %d.' % 
            (args.batch_size)
        )
    else:
        logger.info('Applying CPU training with a batch size of %d.' %
            (args.batch_size)
        )

    #######################################################################
    # Data Loading
    #######################################################################

    logger.info('Loading dataset...')
    train_dataset = load_dataset(args, 'train', reset = False)
    val_dataset = load_dataset(args, 'val', reset = False)

    train_iterator = inf_iterator(DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=True,
        num_workers=args.num_workers
    ))
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        collate_fn=PaddingCollate(), 
        shuffle=False, 
        num_workers=args.num_workers
    )
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    #######################################################################
    # Training
    #######################################################################
    try:
        for it in range(it_first, config.train.max_iters + 1):

            ######################### train ###############################
            train(
                model, optimizer, scheduler, train_iterator, 
                logger, writer, args, config, it,
            )

            ######################### validation ##########################
            if (it % config.train.val_freq == 0 or it == it_first) and (not args.debug):
                try:
                    avg_val_loss = validate(
                        model, scheduler, val_loader, logger, writer, args, config, it
                    )
                except Exception as e:
                    avg_val_loss = torch.nan
                    logger.info('Validation error: %s' % e)

                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)

    except KeyboardInterrupt:
        logger.info('Terminating...')

######################################################################################
# Running the Script                                                                 #
######################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##################### paths and name ######################################
    parser.add_argument('--config',
        type=str, default='configs/codesign_dim-256_step100_lr1.e-4_wd0.0_posiscale50.0.yml'
    )
    parser.add_argument('--logdir', type=str,
        default='../logs/debug/'
    )
    parser.add_argument('--summary_path', type=str,
        default='../data/Protein_MPNN/mpnn_data_info.pkl'
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
    parser.add_argument('--tag', type=str, default='',
        help='tag of the saved files'
    )
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)

    ########################### setting #######################################
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--multi_gpu', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--design_centralize', type=int, default=0)
    parser.add_argument('--centralize', type=int, default=0)
    parser.add_argument('--reso_threshold', type=float, default=3.0)
    parser.add_argument('--length_min', type=int, default=20)
    parser.add_argument('--length_max', type=int, default=800)
    parser.add_argument('--with_monomer', type=int, default=0)
    parser.add_argument('--load_interface', type=int, default=1)
    parser.add_argument('--with_epitope', type=int, default=1)
    parser.add_argument('--with_bindingsite', type=int, default=1)
    parser.add_argument('--with_scaffold', type=int, default=0)
    parser.add_argument('--chain_feat_version', type=str, default='same')
    parser.add_argument('--with_LLM', type=int, default=0)
    parser.add_argument('--random_masking', type=int, default=0)
    parser.add_argument('--with_dist', type=int, default=0)
    parser.add_argument('--dist_clamp', type=float, default=None)
    parser.add_argument('--with_rmsd', type=int, default=0)
    parser.add_argument('--with_center_loss', type=int, default=0)
    parser.add_argument('--with_clash_loss', type=int, default=0)
    parser.add_argument('--with_af3_relpos', type=int, default=0)
    parser.add_argument('--cross_attn', type=int, default=0)

    args = parser.parse_args()

    ########################## arguments setting ##############################
    args.multi_gpu = bool(args.multi_gpu)
    if args.device != 'cuda' or torch.cuda.device_count() <= 1:
        args.multi_gpu = False

    args.design_centralize = bool(args.design_centralize)
    args.centralize = bool(args.centralize)
    args.with_monomer = bool(args.with_monomer)
    args.load_interface = bool(args.load_interface)
    args.with_epitope = bool(args.with_epitope)
    args.with_bindingsite = bool(args.with_bindingsite)
    args.with_scaffold = bool(args.with_scaffold)
    args.with_LLM = bool(args.with_LLM)
    args.random_masking = bool(args.random_masking)
    args.with_dist = bool(args.with_dist)
    args.with_rmsd = bool(args.with_rmsd)
    args.with_center_loss = bool(args.with_center_loss)
    args.with_clash_loss = bool(args.with_clash_loss)
    args.with_af3_relpos = bool(args.with_af3_relpos)
    args.cross_attn = bool(args.cross_attn)

    ################################### running ###############################
    main(args)
