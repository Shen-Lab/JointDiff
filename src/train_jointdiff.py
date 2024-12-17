# diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'
try:
    from utils.utils_guidance import Guidance_cal
except Exception as e:
    print(e)

from diffab.utils.protein.constants import ressymb_order
import torch.multiprocessing  # SZ
from diffab.utils.train import *
from diffab.utils.data import *
from diffab.utils.misc import *
from diffab.models import get_model
from diffab.datasets import get_dataset
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

from networks_proteinMPNN import ProteinMPNN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')  # SZ
torch.autograd.set_detect_anomaly(True)

######################################################################################
# Utility Functions                                                                  #
######################################################################################

def add_random_masking(
    batch, random_mask = False,
    sele_ratio = 0.7, min_mask_ratio = 0.3, max_mask_ratio = 1.
):
    """Random masking part of the protein and do the inpainting.

    Args:
        batch: original data.
        random_mask: whether apply random masking.
        sele_ratio: ratio of the samples for partial design.
        min_mask_ratio: minimum masking ratio.
        max_mask_ratio: maximum masking ratio.
    """
    N, L = batch['mask'].shape
    device = batch['mask'].device

    ####################### mask all regions ########################
    if not random_mask or sele_ratio == 0:
        batch['mask_gen'] = batch['mask'].clone()
        batch['fragment_type'] = torch.ones(
            N, L, requires_grad = False, device = device
        ) * batch['mask_gen']
        return batch

    ################## randomly mask some regions ###################

    protein_size = batch['mask'].sum(dim=-1)  # (N,)
    sele_mask = torch.rand(N).to(device) < sele_ratio # (N,), True for selected samples
    mask_ratio = torch.rand(N).to(device) * (max_mask_ratio - min_mask_ratio) + min_mask_ratio
    mask_len = (mask_ratio * protein_size).int() # masked length for each sample; (N,)
    mask_gen = batch['mask'] * (~sele_mask.unsqueeze(-1)) # all False for selected samples
    for i, m_len in enumerate(mask_len):
        if not sele_mask[i]:
            continue
        start_idx = np.random.choice(range(int(protein_size[i] - m_len)))
        end_idx = start_idx + int(m_len)
        mask_gen[i][start_idx:end_idx] = True

    batch['mask_gen'] = mask_gen
    batch['fragment_type'] = torch.ones(
        N, L, requires_grad = False, device = device
    ) * batch['mask'] * 2
    batch['fragment_type'] -= mask_gen.int()

    return batch


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
    energy_guide = None, 
    fitness_guide = None,
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
    ### add type features
    batch = add_random_masking(
        batch, random_mask = args.random_mask, sele_ratio = args.sele_ratio, 
        min_mask_ratio = args.min_mask_ratio, max_mask_ratio = args.max_mask_ratio
    )

    if args.with_contact and args.contact_fix:
        contact_path_list_all = [
            os.path.join(args.contact_path, name + '.contact') for name in batch['id'] 
        ]
        name_idx = torch.arange(len(contact_path_list_all)).to(args.device)
    else:
        contact_path_list_all = None
        name_idx = None

    ###### centralize the data ######
    if args.centralize:
        mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
        mean = mean / batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1)  # (N, 3)
        batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 15, 3)
        batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

    ############################################################################
    # Forward and loss cal
    #     if args.debug: torch.set_anomaly_enabled(True) 
    ############################################################################

    loss_dict = model(
        batch = batch,
        micro = args.micro, posi_loss_version = args.posi_loss_version,
        with_dist_loss = args.with_dist_loss,
        dist_clamp = args.dist_clamp,
        loss_version = args.loss_version,
        with_clash = args.with_clash, threshold_clash = args.threshold_clash,
        with_gap = args.with_gap, threshold_gap = args.threshold_gap,
        with_consist_loss = args.with_consist_loss,
        cross_loss = args.cross_loss,
        consist_target = args.consist_target,
        with_CEP_loss = args.with_CEP_joint,
        with_energy_loss = args.with_energy_guide,
        with_fitness_loss = args.with_fitness_guide,
        energy_guide = energy_guide,
        energy_guide_type = args.energy_guide_type,
        struc_scale = args.struc_scale,
        temperature = args.temperature,
        energy_aggre = args.energy_aggre,
        RepulsionOnly = args.RepulsionOnly, 
        with_resi = args.with_resi, 
        multithread=args.multithread,
        with_contact=args.with_contact,
        contact_fix = args.contact_fix,
        contact_path_list_all = contact_path_list_all,
        name_idx = name_idx,
        atom_list = ['CA'],
        contact_thre = args.contact_thre,
        fitness_guide = fitness_guide,
        fitness_guide_type = args.fitness_guide_type,
        seq_scale=args.seq_scale,
        seq_sample=args.seq_sample,
        t_max=args.t_max,
        force_vs_diff = args.force_vs_diff
    )
    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
    loss_dict['overall'] = loss
    time_forward_end = current_milli_time()

    ############################################################################
    # Backpropogate 
    ############################################################################

    loss.backward()
    orig_grad_norm = clip_grad_norm_(
        model.parameters(), config.train.max_grad_norm
    )
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
        }, os.path.join(args.logdir, 'checkpoint_nan_%d.pt' % it))
        raise KeyboardInterrupt()

    torch.cuda.empty_cache()


def validate(
    model, scheduler, val_loader,
    logger, writer, args, config, it,
    energy_guide = None, fitness_guide = None,
):

    loss_tape = ValidationLossTape()
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
            ####### Prepare data ######
            batch = recursive_to(batch, args.device)
            ### add type features
            batch = add_random_masking(
                batch, random_mask = args.random_mask, sele_ratio = args.sele_ratio, 
                min_mask_ratio = args.min_mask_ratio, max_mask_ratio = args.max_mask_ratio
            )

            if args.with_contact and args.contact_fix:
                contact_path_list_all = [
                    os.path.join(args.contact_path, name + '.contact') for name in batch['id']
                ]
                name_idx = torch.arange(len(contact_path_list_all)).to(args.device)
            else:
                contact_path_list_all = None
                name_idx = None

            ### centralize the data
            if args.centralize:
                mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
                mean = mean / batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1)  # (N, 3)
                batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 15, 3)
                batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

            ###### Forward ######
            loss_dict = model(batch,
                micro = args.micro, posi_loss_version = args.posi_loss_version,
                with_dist_loss = args.with_dist_loss,
                dist_clamp = args.dist_clamp,
                loss_version = args.loss_version,
                with_clash = args.with_clash, threshold_clash = args.threshold_clash,
                with_gap = args.with_gap, threshold_gap = args.threshold_gap,
                with_consist_loss = args.with_consist_loss,
                cross_loss = args.cross_loss,
                consist_target = args.consist_target,
                with_CEP_loss = args.with_CEP_joint,
                with_energy_loss = args.with_energy_guide,
                with_fitness_loss = args.with_fitness_guide,
                energy_guide = energy_guide,
                energy_guide_type = args.energy_guide_type,
                struc_scale = args.struc_scale,
                temperature = args.temperature,
                energy_aggre = args.energy_aggre,
                RepulsionOnly = args.RepulsionOnly,
                with_resi = args.with_resi,
                multithread=args.multithread,
                with_contact=args.with_contact,
                contact_fix = args.contact_fix,
                contact_path_list_all = contact_path_list_all,
                name_idx = name_idx,
                atom_list = ['CA'],
                contact_thre = args.contact_thre,
                fitness_guide = fitness_guide,
                fitness_guide_type = args.fitness_guide_type,
                seq_scale=args.seq_scale,
                seq_sample=args.seq_sample,
                t_max=args.t_max,
                force_vs_diff = args.force_vs_diff
            )
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

    ######### ProteinMPNN (for consist loss) #########
    if args.with_consist_loss:
        config.model.proteinMPNN_path = args.proteinMPNN_path

        print('Loading ProteinMPNN...')
        mpnn_ckpt = torch.load(args.proteinMPNN_path)
        proteinMPNN_model = ProteinMPNN(
            num_letters=21, node_features=128, edge_features=128,
            hidden_dim = 128, num_encoder_layers=3, num_decoder_layers=3,
            vocab=21, k_neighbors=mpnn_ckpt['num_edges'], augment_eps=0.05,
            dropout=0.1, ca_only=True
        )
        proteinMPNN_model.load_state_dict(mpnn_ckpt['model_state_dict'])
        ###### freeze the parameters ######
        for param in proteinMPNN_model.parameters():
            param.requires_grad = False

    else:
        config.model.proteinMPNN_path = None
        proteinMPNN_model = None

    config.model.proteinMPNN_model = proteinMPNN_model

    ################## settings #####################
    ### CET
    config.model.with_CEP_joint = bool(args.with_CEP_joint)
    ### random masking
    config.model.random_mask = bool(args.random_mask)

    ################## Model ########################
    logger.info('Building model...')
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
        config.train.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        logger.info('%d GPUs detected. Applying parallel training and a batch size of %d.' % 
            (torch.cuda.device_count(), config.train.batch_size)
        )

    elif args.device == 'cuda':
        logger.info('Applying single GPU training with a batch size of %d.' % 
            (config.train.batch_size)
        )

    else:
        logger.info('Applying CPU training with a batch size of %d.' %
            (config.train.batch_size)
        )

    #######################################################################
    # Guidance Module
    #######################################################################
    if args.with_energy_guide or args.with_fitness_guide:
        oracle = Guidance_cal(
            with_energy_guide=args.with_energy_guide, openmm_version=args.openmm_version,
            with_fitness_guide=args.with_fitness_guide, esm_version=args.esm_version,
            input_voc_set=ressymb_order, device=args.device
        )

        if args.with_energy_guide and args.with_fitness_guide:
            logger.info('Applying energy (%s) and fitness guidance (%s)...' %
                (args.energy_guide_type, args.fitness_guide_type)
            )
            energy_guide = oracle.energy_guide
            if args.fitness_guide_type in {'cosine', 'mse'}:
                fitness_guide = oracle.fitness_guide
            else:
                fitness_guide = None

        elif args.with_fitness_guide:
            logger.info('Applying fitness guidance (%s) with no energy guidance...' % args.fitness_guide_type)
            energy_guide = None
            if args.fitness_guide_type in {'cosine', 'mse'}:
                fitness_guide = oracle.fitness_guide
            else:
                fitness_guide = None
        else:
            logger.info('Applying energy guidance (%s) with no fitness guidance...' % args.energy_guide_type)
            energy_guide = oracle.energy_guide
            fitness_guide = None
            args.fitness_guide_type = 'cosine'

        if args.openmm_version == 'CA':
            args.atom_list = ['CA']
        else:
            pass

    else:
        logger.info('Applying no guidance...')
        energy_guide = None
        fitness_guide = None

    #######################################################################
    # Data Loading
    #######################################################################
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    train_iterator = inf_iterator(DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=True,
        num_workers=args.num_workers
    ))
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers
    )
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    #######################################################################
    # Training
    #######################################################################
    try:
        for it in range(it_first, config.train.max_iters + 1):
            # train
            train(
                model, optimizer, scheduler, train_iterator, 
                logger, writer, args, config, it,
                energy_guide, fitness_guide,
            )

            # validation
            if (it % config.train.val_freq == 0 or it == it_first) and (not args.debug):
                try:
                    avg_val_loss = validate(
                        model, scheduler, val_loader, logger, writer, args, config, it,
                        energy_guide, fitness_guide
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
                        type=str, default='../configs/jointdiff_dim-128-64-4_step100_lr1.e-4_wd0._posiscale50.0.yml')
    parser.add_argument('--logdir', type=str,
                        default='../Logs/')
    parser.add_argument('--tag', type=str, default=None,
                        help='tag of the saved files')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--proteinMPNN_path', type=str, 
        default='../oracles/mpnn_ca_model/v_48_002.pt',
        help='pretrained proteinMPNN model'
    )

    ########################### setting #######################################
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seq_sample', type=str, default='multinomial')
    parser.add_argument('--centralize', type=int, default=0)
    parser.add_argument('--random_mask', type=int, default=0)
    parser.add_argument('--sele_ratio', type=float, default=0.7)
    parser.add_argument('--min_mask_ratio', type=float, default=0.3)
    parser.add_argument('--max_mask_ratio', type=float, default=1.)

    ########################### for losses ###################################

    ### basic loss
    parser.add_argument('--micro', type=int, default=1)
    parser.add_argument('--unnorm_first', type=int, default=0)
    parser.add_argument('--posi_loss_version', type=str, default='mse')
    ### distance loss
    parser.add_argument('--with_dist_loss', type=int, default=0)
    parser.add_argument('--dist_clamp', type=float, default=20.)
    parser.add_argument('--loss_version', type=str, default='mse')
    ### clash loss
    parser.add_argument('--with_clash', type=int, default=0)
    parser.add_argument('--threshold_clash', type=float, default=3.6)
    ### gap loss
    parser.add_argument('--with_gap', type=int, default=0)
    parser.add_argument('--threshold_gap', type=float, default=3.9)
    ### consist loss
    parser.add_argument('--with_consist_loss', type=int, default=0)
    parser.add_argument('--cross_loss', type=int, default=0)
    parser.add_argument('--consist_target', type=str, default='distribution')
    ### CEP loss
    parser.add_argument('--with_CEP_joint', type=int, default=0)
    ### engergy guidence
    parser.add_argument('--with_energy_guide', type=int, default=0)
    parser.add_argument('--openmm_version', type=str, default='CA')
    parser.add_argument('--energy_guide_type', type=str, default='cosine')
    parser.add_argument('--struc_scale', type=str, default='Boltzmann',
                        help='"none", "Boltzmann" or "negative-Boltzmann"')
    parser.add_argument('--temperature', type=float, default=300.,
                        help='default value is 300K')
    parser.add_argument('--energy_aggre', type=str, default='LJ 12 Repulsion Energy',
                        help='"all" or "LJ 12 Repulsion Energy" or other energy terms')
    parser.add_argument('--RepulsionOnly', type=int, default=1)
    parser.add_argument('--with_resi', type=int, default=0)
    parser.add_argument('--force_vs_diff', type=int, default=0)
    parser.add_argument('--multithread', type=int, default=0)
    ### contact
    parser.add_argument('--with_contact', type=int, default=0),
    parser.add_argument('--contact_fix', type=int, default=1)
    parser.add_argument('--contact_path', type=str, default='../data/ContactMap_CA/')
    parser.add_argument('--contact_thre', type=float, default=12.)

    ### fitness guidance
    parser.add_argument('--with_fitness_guide', type=int, default=0)
    parser.add_argument('--fitness_guide_type', type=str, default='cosine')
    parser.add_argument('--esm_version', type=str, default='ESM-1b')
    parser.add_argument('--seq_scale', type=str,
        default='none', help='"none", "length", "negative" or float'
    )

    ### guide step
    parser.add_argument('--t_max', type=str, default=None)

    ########################### constraints ##################################

    parser.add_argument('--with_mpnn_force', type=int, default=0)

    args = parser.parse_args()

    ###### lables ######
    if args.tag is None or args.tag.upper() == 'NONE':
        args.tag = ''
    if args.resume is not None and args.resume.upper() == 'NONE':
        args.resume = None
    if args.finetune is not None and args.finetune.upper() == 'NONE':
        args.finetune = None

    if args.t_max is None or args.t_max.upper() == 'NONE':
        args.t_max = None
    else:
        args.t_max = int(args.t_max)

    ###### arguments setting ######
    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    args.centralize = bool(args.centralize)
    args.random_mask = bool(args.random_mask)
    args.micro = bool(args.micro)
    args.unnorm_first = bool(args.unnorm_first)
    args.with_dist_loss = bool(args.with_dist_loss)
    args.with_clash = bool(args.with_clash)
    args.with_gap = bool(args.with_gap)
    args.with_consist_loss = bool(args.with_consist_loss)
    args.cross_loss = bool(args.cross_loss)
    args.with_energy_guide = bool(args.with_energy_guide)
    args.RepulsionOnly = bool(args.RepulsionOnly)
    if args.RepulsionOnly:
        args.energy_aggre = 'LJ 12 Repulsion Energy' 
    args.with_resi = bool(args.with_resi)
    args.multithread = bool(args.multithread)
    args.with_contact = bool(args.with_contact)
    args.contact_fix = bool(args.contact_fix)
    args.force_vs_diff = bool(args.force_vs_diff)

    args.with_fitness_guide = bool(args.with_fitness_guide)
    if not args.seq_scale in {'none', 'length', 'negative'}:
        args.seq_scale = float(args.seq_scale)

    ###### running ######
    main(args)
