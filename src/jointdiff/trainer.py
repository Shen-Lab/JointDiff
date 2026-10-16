import os
import shutil
import argparse
from tqdm.auto import tqdm
import yaml
from easydict import EasyDict
import pickle
import random
import numpy as np

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn  
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.multiprocessing

from jointdiff.modules.utils.misc import (
    BlackHole, seed_all, current_milli_time
)
from jointdiff.modules.utils.train import (
    recursive_to, sum_weighted_losses, log_losses, ValidationLossTape
)
from jointdiff.modules.data import get_transform
from jointdiff.dataset import SingleChainDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

###############################################################################
# IO / loader
###############################################################################

def dict_load(path):
    with open(path, 'rb') as handle:
        result = pickle.load(handle)
    return result


def dict_save(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


###############################################################################
# data
###############################################################################

def get_dataset(cfg):
    transform = get_transform(cfg.transform) if 'transform' in cfg else None
    return SingleChainDataset(
        transform = transform,
        **cfg
    )


###############################################################################
# Train & Validation
###############################################################################

def get_triple_contacted_mask(dist_mat, pair_mask, distance_threshold=6.0, sequence_gap=10):
    """
    dist_mat: (N, L, L) tensor of Cβ–Cβ pairwise distances
    pair_mask: (N, L, L) boolean tensor indicating valid entries in dist_mat
    Returns:
        A binary mask of shape (N, L) with 1s at selected residues and 0s elsewhere
    """
    assert dist_mat.shape == pair_mask.shape, "Distance matrix and mask must have the same shape"
    N, L, _ = dist_mat.shape
    result_mask = torch.zeros((N, L), dtype=torch.bool, device=dist_mat.device)

    for n in range(N):
        valid_triples = []

        for i in range(L):
            for j in range(i + sequence_gap + 1, L):
                for k in range(j + sequence_gap + 1, L):
                    # Check sequence separation
                    if j - i <= sequence_gap or k - j <= sequence_gap or k - i <= sequence_gap:
                        continue

                    # Skip if any pair is invalid
                    if not (pair_mask[n, i, j] and pair_mask[n, i, k] and pair_mask[n, j, k]):
                        continue

                    # Get distances
                    dij = dist_mat[n, i, j].item()
                    dik = dist_mat[n, i, k].item()
                    djk = dist_mat[n, j, k].item()

                    if dij < distance_threshold and dik < distance_threshold and djk < distance_threshold:
                        valid_triples.append((i, j, k))

        if not valid_triples:
            continue

        triple = random.choice(valid_triples)
        selected = list(triple)

        for idx in triple:
            if random.random() < 0.5:
                if 0 < idx < L - 1:
                    selected.append(random.choice([idx - 1, idx + 1]))
                elif idx == 0 and L > 1:
                    selected.append(1)
                elif idx == L - 1 and L > 1:
                    selected.append(L - 2)

        result_mask[n, selected] = 1

    return result_mask


def random_masking(
    mask, 
    protein_size, 
    sele_ratio = 0.7,
    min_mask_ratio = 0.2,
    max_mask_ratio = 1.,
    consecutive_prob = 0.5,
):

    N = mask.shape[0]
    device = mask.device

    ### masked sample selection
    sele_mask = torch.rand(N).to(device) < sele_ratio # (N,), True for selected samples

    ### mask size
    mask_ratio = torch.rand(N).to(device) * (
        max_mask_ratio - min_mask_ratio
    ) + min_mask_ratio
    mask_len = (mask_ratio * protein_size).int() # masked length for each sample; (N,)

    ### masking
    mask_gen = mask * (~sele_mask.unsqueeze(-1)) # all False for selected samples

    for i, m_len in enumerate(mask_len):
        if not sele_mask[i]:
            continue

        ### whether consecutive
        consecutive = (random.random() < consecutive_prob)

        if consecutive:
            ### consecutive masking
            start_idx = np.random.choice(range(int(protein_size[i] - m_len)))
            end_idx = start_idx + int(m_len)
            mask_gen[i][start_idx:end_idx] = True
        else:
            ### random masking
            p_len = protein_size[i] # protein size
            idx_sele = random.sample(range(p_len), m_len)
            mask_gen[i][idx_sele] = True

    return mask_gen 


def add_random_masking(
    batch, 
    random_mask = False,
    sele_ratio = 0.7, 
    min_mask_ratio = 0.2, 
    max_mask_ratio = 1., 
    consecutive_prob = 0.5,
    seq_thre = 10,
    dist_thre = 6.,
    flanking_prob = 0.5,
    separate_mask = False,
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
    batch['mask_gen'] = random_masking(
        batch['mask'], protein_size = protein_size,
        sele_ratio = sele_ratio, 
        min_mask_ratio = min_mask_ratio, max_mask_ratio = max_mask_ratio,
        consecutive_prob = consecutive_prob,
    )
    ### fragment type
    batch['fragment_type'] = torch.ones(
        N, L, requires_grad = False, device = device
    ) * batch['mask'] * (2 + int(separate_mask))
    batch['fragment_type'] -= batch['mask_gen'].int()

    ###### separate masking for sequence and structure ######
    if separate_mask:
        #print('separate mask')
        batch['mask_gen_seq'] = random_masking(
            batch['mask'], protein_size = protein_size,
            sele_ratio = sele_ratio, 
            min_mask_ratio = min_mask_ratio, max_mask_ratio = max_mask_ratio,
            consecutive_prob = consecutive_prob,
        )
        batch['fragment_type'] -= batch['mask_gen_seq'].int() * 2
    else:
        batch['mask_gen_seq'] = batch['mask_gen']

    return batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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
    it,
):

    time_start = current_milli_time()
    model.train()
    train_args = args.train

    ############################################################################
    # Prepare data 
    ############################################################################

    batch = recursive_to(next(train_iterator), train_args.device)
    ### add type features
    batch = add_random_masking(
        batch, 
        random_mask = train_args.random_mask, 
        sele_ratio = train_args.sele_ratio, 
        min_mask_ratio = train_args.min_mask_ratio, 
        max_mask_ratio = train_args.max_mask_ratio,
        consecutive_prob = train_args.consecutive_prob,
        separate_mask = train_args.separate_mask,
    )

    ###### centralize the data ######
    if train_args.centralize:

        ### motif-scaffolding
        if train_args.random_mask and (batch['mask_gen'] != batch['mask']).any():
            posi_sele = batch['pos_heavyatom'] * (~batch['mask_gen']).unsqueeze(-1).unsqueeze(-1)
            mean = posi_sele.sum(dim = (1, 2))  # (N, 3)
            denorm = batch['mask_heavyatom'] * (~batch['mask_gen']).unsqueeze(-1)
            denorm = denorm.sum(dim = (1, 2)).unsqueeze(1) + 1e-6
        ### monomer
        else:
            mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
            denorm =  batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1) + 1e-6 # (N, 3)

        mean = mean / denorm  # (N, 3)
        batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 15, 3)
        batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

    ############################################################################
    # Forward and loss cal
    #     if args.debug: torch.set_anomaly_enabled(True) 
    ############################################################################

    loss_dict = model(
        batch,
        micro = train_args.micro, 
        posi_loss_version = train_args.posi_loss_version,
        unnorm_first = train_args.unnorm_first,
        with_dist_loss = train_args.with_dist_loss,
        dist_loss_version = train_args.dist_loss_version,
        threshold_dist = train_args.threshold_dist,
        dist_clamp = train_args.dist_clamp,
        with_clash = train_args.with_clash, 
        threshold_clash = train_args.threshold_clash,
        with_gap = train_args.with_gap, 
        threshold_gap = train_args.threshold_gap,
        motif_factor = train_args.motif_factor,
    )
    #print(loss_dict.keys())
    loss = sum_weighted_losses(loss_dict, train_args.loss_weights)
    loss_dict['overall'] = loss
    time_forward_end = current_milli_time()

    ############################################################################
    # Backpropogate 
    ############################################################################

    loss.backward()
    orig_grad_norm = clip_grad_norm_(
        model.parameters(), train_args.max_grad_norm
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
            'config': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': it,
            'batch': recursive_to(batch, 'cpu'),
        }, os.path.join(args.path.logdir, 'checkpoint_nan_%d.pt' % it))
        raise KeyboardInterrupt()

    torch.cuda.empty_cache()


def validate(
    model, 
    scheduler, 
    val_loader,
    logger, 
    writer, 
    args, 
    it,
):
    train_args = args.train
    loss_tape = ValidationLossTape()
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
            ####### Prepare data ######
            batch = recursive_to(batch, train_args.device)
            ### add type features
            batch = add_random_masking(
                batch, 
                random_mask = train_args.random_mask, 
                sele_ratio = train_args.sele_ratio, 
                min_mask_ratio = train_args.min_mask_ratio, 
                max_mask_ratio = train_args.max_mask_ratio,
                consecutive_prob = train_args.consecutive_prob,
                separate_mask = train_args.separate_mask,
            )

            ### centralize the data
            if train_args.centralize:
        
                if train_args.random_mask and (batch['mask_gen'] != batch['mask']).any():
                    posi_sele = batch['pos_heavyatom'] * (~batch['mask_gen']).unsqueeze(-1).unsqueeze(-1)
                    mean = posi_sele.sum(dim = (1, 2))  # (N, 3)
                    denorm = batch['mask_heavyatom'] * (~batch['mask_gen']).unsqueeze(-1)
                    denorm = denorm.sum(dim = (1, 2)).unsqueeze(1) + 1e-6
                else:
                    mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
                    denorm =  batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1) + 1e-6 # (N, 3)

                mean = mean / denorm  # (N, 3)
                batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 15, 3)
                batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

            ###### Forward ######
            loss_dict = model(
                batch,
                micro = train_args.micro, 
                posi_loss_version = train_args.posi_loss_version,
                unnorm_first = train_args.unnorm_first,
                with_dist_loss = train_args.with_dist_loss,
                dist_loss_version = train_args.dist_loss_version,
                threshold_dist = train_args.threshold_dist,
                dist_clamp = train_args.dist_clamp,
                with_clash = train_args.with_clash, 
                threshold_clash = train_args.threshold_clash,
                with_gap = train_args.with_gap, 
                threshold_gap = train_args.threshold_gap,
                motif_factor = train_args.motif_factor,
            )
            loss = sum_weighted_losses(loss_dict, train_args.loss_weights)
            loss_dict['overall'] = loss
            loss_tape.update(loss_dict, 1)
            torch.cuda.empty_cache()

    avg_loss = loss_tape.log(it, logger, writer, 'val')
    # Trigger scheduler
    if train_args.scheduler.type == 'plateau':
        scheduler.step(avg_loss)
    else:
        scheduler.step()

    return avg_loss


######################################################################################
# Inference                                                                          #
######################################################################################

RESIDUE_dict = {'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'Q':'GLN', 'E':'GLU',
                'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS', 'M':'MET', 'F':'PHE',
                'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP', 'Y':'TYR', 'V':'VAL', 'B':'ASX',
                'Z':'GLX', 'X':'UNK'}

RESIDUE_reverse_dict = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
                        'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F',
                        'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V', 'ASX':'B',
                        'GLX':'Z', 'UNK':'X'}

ELEMENT_dict = {'N':'N', 'CA':'C', 'C':'C', 'O':'O'}


def pdb_line_write(chain, aa, resi_idx, atom, a_idx, vec, wf):
    """Write a single line in the PDB file."""

    line = 'ATOM  '
    line += '{:>5} '.format(a_idx)
    line += '{:<4} '.format(atom)
    line += aa + ' '
    line += chain
    line += '{:>4}    '.format(resi_idx) # residue index

    ### coordinates
    for coor_val in vec:
        coor_val = float(coor_val)

        # to avoid the coordinate value is longer than 8
        if len('%.3f' % coor_val) <= 8:
            ### -999.9995 < coor_val < 1000.9996
            coor_val = '%.3f' % coor_val

        elif coor_val >= 99999999.5:
            coor_val = '%.2e' % coor_val

        elif coor_val <= -9999999.5:
            coor_val = '%.1e' % coor_val

        else:
            # length of the interger part
            inte_digit = 1 + int(np.log10(abs(coor_val))) + int(coor_val < 0)
            deci_digit = max(7 - inte_digit, 0)
            coor_val = '%f' % round(coor_val, deci_digit)
            coor_val = coor_val[:8]

        line += '{:>8}'.format(coor_val)

    line += '{:>6}'.format('1.00') # occupancy
    line += '{:>6}'.format('0.00') # temperature
    line += ' ' * 10
    element = ELEMENT_dict[atom]
    line += '{:>2}'.format(element)  # element

    wf.write(line + '\n')


def pdb_write(coor_dict, path):
    """
    Transform the given info into the pdb format.
    """
    with open(path, 'w') as wf:
        a_idx = 0

        for chain in coor_dict.keys():
            c_coor_dict = coor_dict[chain]['coor']
            if 'seq' in coor_dict[chain].keys():
                seq = coor_dict[chain]['seq']
                if len(c_coor_dict.keys()) != len(seq):
                    print(
                        'Error! The size of the strutctue and the sequence do not match for chain %s! (%d and %d)' % (
                        len(c_coor_dict.keys()), len(seq), chain
                    ))
                    continue
            else:
                seq = None

            for i,resi_idx in enumerate(coor_dict[chain]['ordered_idx']):
                if seq is not None:
                    aa = RESIDUE_dict[seq[i]]
                else:
                    aa = 'GLY'

                for atom in c_coor_dict[resi_idx].keys():
                    vec = c_coor_dict[resi_idx][atom]
                    a_idx += 1

                    pdb_line_write(chain, aa, resi_idx, atom, a_idx, vec, wf)


def inference_pdb_write(coor, path, seq = None, chain = 'A',
              atom_list = ['N', 'CA', 'C', 'O']):
    """
    Args:
        coor: (L, atom_num, 3)
        seq: str of length L
    """
    if seq is not None and coor.shape[0] != len(seq):
        print('Error! The size of the strutctue and the sequence do not match! (%d and %d)'%(coor.shape[0],
                                                                                             len(seq)))
    elif coor.shape[1] != len(atom_list):
        print('Error! The size of the resi-wise coor and the atom_num do not match! (%d and %d)'%(coor.shape[1],
                                                                                             len(atom_list)))
    else:
        with open(path, 'w') as wf:
            a_idx = 0

            for i, resi in enumerate(seq):
                ### residue-wise info
                r_idx = i + 1
                aa = RESIDUE_dict[resi]

                for j,vec in enumerate(coor[i]):
                    ### atom-wise info
                    atom = atom_list[j]
                    a_idx += 1
                    pdb_line_write(chain, aa, r_idx, atom, a_idx, vec, wf)
