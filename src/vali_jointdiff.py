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
import pickle
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn  # SZ
import torch.nn.functional as F  # SZ
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.networks_proteinMPNN import ProteinMPNN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')  # SZ
torch.autograd.set_detect_anomaly(True)

######################################################################################
# Utility Functions                                                                  #
######################################################################################

def dict_save(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0

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
# Loss cal ###########                                                               #
######################################################################################

def validate(
    model, val_loader, args, config,
    energy_guide = None, fitness_guide = None,
):

    with torch.no_grad():
        model.eval()
        loss_dict_all = {'summary':{}, 'sepa':{}}

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
            for key in loss_dict:
                loss_dict[key] = float(loss_dict[key].mean().cpu())
            torch.cuda.empty_cache()

            loss_dict_all['sepa'][i] = loss_dict

    ###### statistics ######
    for i in loss_dict_all['sepa']:
        for key in loss_dict_all['sepa'][i]:
            if key not in loss_dict_all['summary']:
                loss_dict_all['summary'][key] = []
            loss_dict_all['summary'][key].append(loss_dict_all['sepa'][i][key])

    for key in loss_dict_all['summary']:
        loss_dict_all['summary'][key] = np.mean(loss_dict_all['summary'][key])

    return loss_dict_all

######################################################################################
# Main Function                                                                      #
######################################################################################


def main(args):

    #######################################################################
    # Model 
    #######################################################################

    ckpt = torch.load(args.model_path, map_location=args.device)
    config = ckpt['config'] 
    seed_all(config.train.seed)

    ################## Main Model ###################
    model = get_model(config.model).to(args.device)
    print('Number of parameters: %d' % count_parameters(model))

    ckpt['model'] = {''.join(key.split('module.')[:]) : ckpt['model'][key]
        for key in ckpt['model']
    }
    model.load_state_dict(ckpt['model'])

    ################## Parallel  ##################
    if args.multi_gpu:
        config.train.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        print('%d GPUs detected. Applying parallel training and a batch size of %d.' % 
            (torch.cuda.device_count(), config.train.batch_size)
        )

    elif args.device == 'cuda':
        print('Applying single GPU training with a batch size of %d.' % 
            (config.train.batch_size)
        )

    else:
        print('Applying CPU training with a batch size of %d.' %
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
        print('Applying no guidance...')
        energy_guide = None
        fitness_guide = None

    #######################################################################
    # Data Loading
    #######################################################################
    print('Loading dataset...')
    val_dataset = get_dataset(config.dataset.val)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers
    )

    #######################################################################
    # Loss cal
    #######################################################################
    try:
        avg_val_loss = validate(
            model, val_loader, args, config,
            energy_guide, fitness_guide,
        )
        _ = dict_save(avg_val_loss, args.save_path)
        print('Loss saved.')
    except Exception as e:
        avg_val_loss = torch.nan
        print('Validation error: %s' % e)

######################################################################################
# Running the Script                                                                 #
######################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##################### paths and name ######################################
    parser.add_argument('--model_path', type=str,
        default='../../Logs/logs_originDiff/codesign_diffab_complete_gen_share-true_dim-256-128-2_step100_lr1.e-4_wd0._posiscale50.0_sc_center_2024_10_20__23_57_14_loss-1-mse-1-1/checkpoints/500000.pt'
    )
    # parser.add_argument('--proteinMPNN_path', type=str, 
    #     default='../../Pretrained/ca_model_weights/v_48_002.pt',
    #     help='pretrained proteinMPNN model'
    # )
    parser.add_argument('--save_path', type=str,
        default='../../Logs/logs_originDiff/codesign_diffab_complete_gen_share-true_dim-256-128-2_step100_lr1.e-4_wd0._posiscale50.0_sc_center_2024_10_20__23_57_14_loss-1-mse-1-1/val_losses/loss_500000.pkl'
    )

    ########################### setting #######################################
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seq_sample', type=str, default='multinomial')
    parser.add_argument('--sele_ratio', type=float, default=0.7)
    parser.add_argument('--min_mask_ratio', type=float, default=0.3)
    parser.add_argument('--max_mask_ratio', type=float, default=1.)

    ########################### for losses ###################################

    ### distance loss
    parser.add_argument('--dist_clamp', type=float, default=20.)
    ### clash loss
    parser.add_argument('--threshold_clash', type=float, default=3.6)
    ### gap loss
    parser.add_argument('--threshold_gap', type=float, default=3.9)
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
    parser.add_argument('--contact_path', type=str, default='../../Data/Processed/CATH_forDiffAb/ContactMap_CA/')
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

    ##################### settings ###########################################

    model_info = [token for token in args.model_path.split('/') if token != '']
    name = model_info[-3]

    args.centralize = True if ('_center_' in name) else False
    args.random_mask = True if ('_rm_' in name or name.endswith('_rm')) else False
    args.micro = False if ('_macro_' in name) else True
    args.unnorm_first = True if ('_uf_' in name or name.endswith('_uf')) else False
    args.posi_loss_version = 'rmsd' if ('_rmsd' in name) else 'mse'

    ### distance related losses
    loss_terms = name.split('_loss-')[-1].split('_')[0].split('-')

    args.with_dist_loss = bool(int(loss_terms[0]))
    args.loss_version = loss_terms[1]
    args.with_clash = bool(int(loss_terms[2]))
    args.with_gap = bool(int(loss_terms[3]))

    ### consistency losses 
    args.with_consist_loss = True if ('_consist-' in name) else False
    if args.with_consist_loss:
         args.consist_target = name.split('_consist-')[-1].split('_')[0].split('-')[0]
         args.cross_loss = True if ('_cross' in name) else False
    else:
         args.consist_target = 'distribution'
         args.cross_loss = False

    ### energy related losses ###
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
