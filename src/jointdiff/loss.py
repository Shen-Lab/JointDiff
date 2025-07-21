import typing as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import ml_collections
from einops import einsum
import functools
from tqdm.auto import tqdm
from typing import Dict, Optional, Tuple

from jointdiff.modules.data.constants import make_bb_coordinate_tensors_AF2, BBHeavyAtom
# make_bb_coordinate_tensors_AF2: Tensor; (21, 4, 3)

try:
    from openfold.config import model_config
    from openfold.utils.rigid_utils import Rotation, Rigid
    from openfold.utils.loss import compute_renamed_ground_truth  # AlphaFoldLoss, lddt_ca
except Exception as e:
    print(e)

#############################################################
# for sequences
#############################################################

def loss_smoothed(
    S, log_probs, mask, 
    vocab_size = 21, weight=0.1
):
    """Negative log probabilities."""
    S_onehot = torch.nn.functional.one_hot(S, vocab_size).float() # (N, L, 21)

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True) # (N, L, 21)

    loss = -(S_onehot * log_probs).sum(-1) # (N, L)
    loss_av = torch.sum(loss * mask) / 2000.0  # fixed
    return loss, loss_av

###### overall ######
def sequence_loss(
    s_noisy, s_0, c_denoised, t, mask_res, micro = True, 
    version = 'jointdiff', trans_seq = None, mask_factor = None 
):
    if micro:
        n_tokens = mask_res.sum().float()  # scalar
    else:
        n_tokens = mask_res.sum(dim = -1).float()  # (N,)

    ###### jointdiff ######
    if version == 'jointdiff':
        ### the posterior q(s_(t-1) | s_t, s_0) derived from the forward process
        post_true = trans_seq.posterior(s_noisy, s_0, t)
        ### predicted posterior: p(s_(t-1) | R_t)
        post_pred = trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8
        log_post_pred = torch.log(post_pred)
        kldiv = F.kl_div(
            input = log_post_pred, target = post_true,
            reduction = 'none', log_target = False
        ).sum(dim=-1)    # (N, L)
        if micro:
            loss_seq = (kldiv * mask_res).sum() / (n_tokens + 1e-8)
        else:
            loss_seq = (kldiv * mask_res).sum(dim=1) / (n_tokens + 1e-8)
            loss_seq = loss_seq.mean()

    ###### jointdiff-x ######
    else:
        loss_seq = F.cross_entropy(
            c_denoised.transpose(1, 2), s_0 * mask_res, reduction = 'none'
        )  * mask_res # (N, L)
           
        if mask_factor is not None:
            loss_seq = loss_seq * mask_factor  

        if micro:
            loss_seq = loss_seq.sum() / (n_tokens + 1e-8)            
        else:
            loss_seq = (loss_seq * mask_res).sum(dim=1) / (n_tokens + 1e-8)
            loss_seq = loss_seq.mean()

    return loss_seq


#############################################################
# for position
#############################################################

###################### position loss ########################

###### alignment ######
def weighted_rigid_align(
    true_coords,
    pred_coords,
    mask,
    weights = None,
    detach = True,
):
    """Compute weighted alignment.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates
    pred_coords: torch.Tensor
        The predicted atom coordinates
    weights: torch.Tensor
        The weights for alignment
    mask: torch.Tensor
        The atoms mask

    Returns
    -------
    torch.Tensor
        Aligned coordinates
    torch.Tensor
        Rotation matrix; (B, 3, 3)

    """

    batch_size, num_points, dim = true_coords.shape
    if weights is None:
        weights = mask
    else:
        weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    if detach:
        aligned_coords.detach_()

    return aligned_coords, rot_matrix.transpose(-1, -2)

############################## FAPE ###########################################

def rigid_transform(points, R, t):
    """
    Applies a rigid transformation (rotation + translation) to points using frames.

    Parameters:
      points: Shape (B, *, 3), atomic positions.
      R: Shape (B, L, 3, 4), rotation (3x3)
      t: Shape (B, L, 3), translation.

    Returns:
      torch.Tensor: Transformed coordinates, shape (B, L, *, 3).
    """

    out_shape = list(points.shape)
    B, L, _ = t.shape
    if len(out_shape) > 3:
        points = points.reshape(B, -1, 3)  # (B, N, 3)
    out_shape.insert(1, L)

    ### reverse translation
    p_align = points.unsqueeze(1) - t.unsqueeze(-2) # (B, L, N, 3)
    ### reverse rotation
    p_align = torch.einsum('blqk, blik->bliq', R.transpose(-1, -2), p_align)
    ### reshaping
    p_align = p_align.reshape(tuple(out_shape))

    return p_align


def fape(
    coord_pred, R_pred, t_pred,
    coord_true, R_true, t_true,
    mask = None,
    Z = 10.0,
    clamp_distance=10.0, 
    eps=1e-4,
    mask_factor = None,
    micro = True
):
    """
    Computes batch-wise Frame Aligned Point Error (FAPE) loss.

    Returns:
      torch.Tensor: Scalar FAPE loss value (averaged across batch).
    """
    
    B = coord_pred.shape[0]

    ### Align predicted and true coordinates using the true frames
    pred_aligned = rigid_transform(coord_pred, R_pred, t_pred)  # (B, L, *, 3)
    true_aligned = rigid_transform(coord_true, R_true, t_true)  # (B, L, *, 3)

    ### Compute Euclidean distances per atom
    errors = torch.norm(pred_aligned - true_aligned, dim=-1)  # (B, L, *)

    ### Apply clamping to stabilize large errors
    errors = torch.clamp(errors, max=clamp_distance) / Z # (B, L, *)

    ### Compute mean FAPE per structure, then take the batch mean
    if mask_factor is None and (mask is not None):
        mask_factor = mask.float()

    if mask_factor is not None:
        ### (B, L) to (B, L, *)
        if mask is None:
            mask = (mask_factor > 0).float()

        if len(coord_pred.shape) == 4:
            mask_factor = mask_factor.unsqueeze(-1).expand(coord_pred.shape[:-1])  # (B, L, N)
            mask_factor = torch.einsum('bl, bij->blij', mask, mask_factor)
        else:
            mask_factor = torch.einsum('bl, bi->bli', mask, mask_factor)

        ### errors cal
        errors = errors * mask_factor
        if micro:
            fape_value = torch.sum(errors) / torch.sum(mask)
        else:
            errors = errors.reshape(B, -1)
            mask_factor = mask_factor.reshape(B, -1)
            fape_value = torch.sum(errors, dim=-1)  # (B,)
            fape_value = fape_value / torch.sum(mask, dim = -1)  # (B,)
            fape_value = torch.mean(fape_value)

    else:
        ### no mask, then micro == macro
        fape_value = torch.mean(errors)

    return fape_value


###### orientation loss ######
def rotation_matrix_cosine_loss(R_pred, R_true):
    """Rotation loss from diffab.

    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).

    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )

    return loss


###### overall  ######
def structure_loss(
    p_pred, p_ref, R_pred, R_ref, mask_res, 
    version = 'mse', micro = True, mask_factor = None
):
    """Calculate the structure loss between the predicted structures and
    groundtruths.

    Args:
        p_pred: predicted coordinates; (N, L, 3) or (N, L, 4, 3)
        p_ref: groundtruth coordinates; (N, L, 3) or (N, L, 4, 3)
        R_pred: predicted rotation matrix; (N, L, 3, 3)
        R_ref: groundtruth rotation matrix; (N, L, 3, 3)
        mask_res: 1 for valid positions; (N, L)
        version: versions of the loss functions; 'mse', 'fape', 'rmsd'
        micro: whether to use micro or macro averaging
        mask_factor: weight mask for the loss function; (N, L)
        
    """

    #################### denormolization ######################
    if micro:
        n_tokens = mask_res.sum().float()  # scalar
    else:
        n_tokens = mask_res.sum(dim = -1).float()  # (N,)

    ##################### alignment ############################
    ### alignment with detach
    if version.endswith('align-detach'):
        ### pos alignment 
        p_ref, rot_matrix = weighted_rigid_align(
            p_ref, p_pred, mask = mask_res.unsqueeze(-1), detach = True
        )  # p_ref = bmm(p_ref, rot_matrix) + pos_center  
        ### rotation alignment 
        if R_ref is not None:
            R_ref = einsum(
                rot_matrix.transpose(-1, -2), R_ref, "b n i, b k i j -> b k n j"
            )  # (N, 3, 3)
    ### alignment without detach
    elif version.endswith('align'):
        ### pos alignment 
        p_pred, rot_matrix = weighted_rigid_align(
            p_pred, p_pred, mask = mask_res.unsqueeze(-1), detach = True
        ) 
        ### rotation alignment 
        if R_pred is not None:
            R_pred = einsum(
                rot_matrix.transpose(-1, -2), R_pred, "b n i, b k i j -> b k n j"
            )  # (N, 3, 3)

    ### pos loss version
    version = version.split('-')[0] 

    ################# rotation loss ############################################
    loss_rot = None
    if (R_ref is not None) and (R_pred is not None):
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_ref) * mask_res # (N, L)
        if mask_factor is not None:
            loss_rot = loss_rot * mask_factor

        if micro:
            loss_rot = loss_rot.sum() / (n_tokens + 1e-8)
        else:
            loss_rot = loss_rot.sum(dim=1) / (n_tokens + 1e-8)
            loss_rot = loss_rot.mean()

    ####################### position loss #####################################

    ### FAPE loss
    if version == 'fape':
        t_pred = p_pred[:, :, BBHeavyAtom.CA] if len(p_pred.shape) == 4 else p_pred
        t_ref = p_ref[:, :, BBHeavyAtom.CA] if len(p_ref.shape) == 4 else p_ref

        loss_pos = fape(
            coord_pred = p_pred, R_pred = R_pred, t_pred = t_pred,
            coord_true = p_ref, R_true = R_ref, t_true = t_ref,
            mask = mask_res, mask_factor = mask_factor,
            micro = micro
        )

    ### MSE loss
    elif version == 'mse':
        loss_pos = F.mse_loss(p_pred, p_ref, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = loss_pos * mask_res 

        if mask_factor is not None:
            loss_pos = loss_pos * mask_factor

        if micro:
            loss_pos = loss_pos.sum() / (n_tokens  + 1e-8)
        else:
            loss_pos = loss_pos.sum(dim=1) / (n_tokens + 1e-8)
            loss_pos = loss_pos.mean()

    ### RMSD loss
    elif verion == 'rmsd':
        loss_pos = ((p_pred - p_ref) ** 2).sum(-1) # (N, L)
        if micro:
            loss_pos = (loss_pos * mask_res).sum() / (n_tokens  + 1e-8)
            loss_pos = loss_pos.sqrt()
        else:
            loss_pos = (loss_pos * mask_res).sum(dim=1) / (n_tokens + 1e-8) # (N,)
            loss_pos = loss_pos.sqrt()
            loss_pos = loss_pos.mean()

    ### others
    else:
        raise Exception('Error! No position loss named %s!' % version)

    return loss_rot, loss_pos


############################################################################### 
# Distogram prediction loss 
###############################################################################

# modified based on the openfold implementation: 
# https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    """
    Args:
        logits: distogram prediction log-likelihood, (B, L_max, L_max, # of bins=64) 
        pseudo_beta: coordinates of the beta carbon (alpha carbon for glycine), (B, L_max, 3)
        pseudo_beta_mask: mask indicating if the beta carbon (alpha carbon for glycine) atom has coordinates, 
                          (B, L_max)
        *** default values in config *** 
        min_bin: 2.3125
        max_bin: 21.6875
        no_bins: 64
        eps: 1e-8,
    """

    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2  # (no_bins - 1 = 63,)

    # distance matrix
    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )  # (B, L_max, L_max, 1)

    true_bins = torch.sum(dists > boundaries, dim=-1)  # (B, L_max, L_max, 63)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = pseudo_beta_mask[..., None] * \
        pseudo_beta_mask[..., None, :]  # (B, L_max, L_max)

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


############################################################################### 
# Distance loss 
###############################################################################

def smooth_lddt_loss(
    pred_dists,
    true_dists,
    mask,
):
    """Compute the smooth lddt loss.

    """
    B, N, _ = true_dists.shape
    ### Compute distances between all pairs of atoms
    dist_diff = torch.abs(true_dists - pred_dists)  # (B, L, L)
    ### Compute epsilon values
    eps = (((
        F.sigmoid(0.5 - dist_diff)
        + F.sigmoid(1.0 - dist_diff)
        + F.sigmoid(2.0 - dist_diff)
        + F.sigmoid(4.0 - dist_diff)
    ) / 4.0 ).mean(dim=0))  # (B, L, L)

    ### Calculate masked averaging
    num = (eps * mask).sum(dim=(-1, -2))
    den = mask.sum(dim=(-1, -2)).clamp(min=1)
    lddt = num / den

    return 1.0 - lddt.mean()


def distance_loss(
    coor_pred, dist_ref = None, coor_ref = None, mask_res = None,
    loss_version = 'mse', threshold_dist = 15.0, dist_clamp = None,
    with_clash = False, threshold_clash = 3.6,
    with_gap = False, threshold_gap = 3.9,
):
    """Calculate the distance similarity between the predicted structures and 
    groundtruths.

    Args:
        coor_pred: predicted coordinates; (N, L, 3)
        dist_ref: groundtruth distance mat; (N, L, L)
        coor_ref: groundtruth coordinates or distance mat; (N, L, 3)
        mask_res: 1 for valid positions; (N, L)
        with_dist: whether ref is the distance mat.
        dist_clamp: maximum clamp of the distance loss.
        loss_version: versions of the loss functions; 'mse', 'l2' or 'l1'
    """
    N, L, _ = coor_pred.shape
    device = coor_pred.device
    if mask_res is None:
        mask_res = torch.ones(N, L).to(device) 

    ### predicted distance mat
    dist_pred = torch.cdist(coor_pred, coor_pred)  # (N, L, L)
    if dist_ref is None:
        dist_ref = torch.cdist(coor_ref, coor_ref)  # (N, L, L)

    ### dist mask
    pair_mask = torch.bmm(
        mask_res[..., None].float(), mask_res.unsqueeze(1).float()
    ) * (dist_ref <= threshold_dist) * (1 - torch.eye(L).to(device))  # (N, L, L)
    denorm = pair_mask.sum(dim=(1,2)).float() + 1e-8  # (N,)

    dist_pred_sele = dist_pred * pair_mask
    dist_ref = dist_ref * pair_mask

    ############################ distance loss ################################

    if loss_version == 'smooth-lddt':
        ### smooth lddt loss
        loss_dist = smooth_lddt_loss(
            dist_pred_sele, dist_ref, pair_mask 
        )
    else:
        ### MSE loss or L2 loss
        if loss_version == 'mse' or loss_version == 'l2':
            loss_dist = F.mse_loss(
                dist_pred_sele, dist_ref, reduction='none'
            )  # (N, L, L)
        ### Abs (l1-norm) loss
        else:
            loss_dist = F.l1_loss(
                dist_pred_sele, dist_ref, reduction='none'
            ) # (N, L, L)
        ### clamp 
        if dist_clamp is not None:
            loss_dist = torch.clamp(loss_dist, max = dist_clamp)
        ### average
        loss_dist = loss_dist.sum(dim=(1,2)) # (N,) 
        if loss_version == 'l2':
            loss_dist = loss_dist.sqrt()
        loss_dist = loss_dist / denorm
        loss_dist = loss_dist.mean()

    ############################ clash loss ###################################

    if with_clash:
        clash_flag = (dist_pred < threshold_clash) * pair_mask # (N, L, L)
        loss_clash = torch.clamp(
            threshold_clash - dist_pred, min = 0
        ) * clash_flag # (N, L, L)
        loss_clash = loss_clash.sum(dim=(1,2)) / (clash_flag.sum(dim=(1,2)) + 1e-8)  # (N,)
        loss_clash = loss_clash.mean()
    else:
        loss_clash = None

    ############################ clash loss ###################################

    if with_gap:
        idx_arange_1 = torch.arange((L - 1) * N)  # (N * (L - 1),)
        idx_arange_2 = torch.arange(1, L).repeat(N)  # (N * (L - 1),)
        dist_sele = dist_pred[:,:-1,:].reshape(-1, L)[idx_arange_1, idx_arange_2]  # (N, L-1)
        dist_sele = dist_sele.reshape(N, -1)
        gap_flag = (dist_sele > threshold_gap) * mask_res[:,1:] # (N, L - 1)
        loss_gap = torch.clamp(
            dist_sele - threshold_gap, min = 0
        ) * gap_flag # (N, L-1)
        loss_gap = loss_gap.sum(dim=1) / (gap_flag.sum(dim=1) + 1e-8) # (N,)
        loss_gap = loss_gap.mean()
    else:
        loss_gap = None

    return loss_dist, loss_clash, loss_gap


#############################################################
# oracle feedback
#############################################################

################### consistency loss ########################

def consistency_loss( 
    coor_pred, c_denoised, s_gt, mask_res, proteinMPNN_model, 
    coor_gt = None, micro = True, 
    consist_target = 'distribution', cross_loss = False,
):
    """Consistency loss with proteinMPNN."""

    if proteinMPNN_model is None:
        print('Warning! ProteinMPNN model is not provided!')
        return None
    N, L = s_gt.shape

    ###### for MPNN input ######
    residue_idx = torch.arange(1, L+1).repeat(N,1).to(s_gt.device)  # (N, L)
    residue_idx[~mask_res.bool()] = 0
    randn = torch.randn(N, L).to(s_gt.device)
    ### residue mapping
    # diffab: 'ACDEFGHIKLMNPQRSTVWYX'
    # MPNN:   'ACDEFGHIKLMNPQRSTVWYX'
    s_gt[~mask_res.bool()] = 0

    ###### proteinMPNN prediction ######
    log_probs_mpnn = proteinMPNN_model(
        X = coor_pred,
        S = s_gt,
        mask = mask_res.int(),
        chain_M = mask_res.float(),
        residue_idx = residue_idx,
        chain_encoding_all = mask_res.int(),
        randn = randn
    )  # (N, L, 21)

    ###### loss calculation ######
    ### cross-entropy with groundtruth sequence
    if consist_target == 'gt':
        _, loss_cyclic = loss_smoothed(
            S = s_gt,
            log_probs = log_probs_mpnn,
            mask = mask_res,
            vocab_size = log_probs_mpnn.shape[-1]
        )

        if cross_loss:
            log_probs_mpnn_2 = proteinMPNN_model(
                X = coor_gt,
                S = s_gt * mask_res,
                mask = mask_res.int(),
                chain_M = mask_res.float(),
                residue_idx = residue_idx,
                chain_encoding_all = mask_res.int(),
                randn = randn
            )
            loss_cyclic_2 = F.kl_div(
                input = log_probs_mpnn_2[:,:,:20], 
                target = c_denoised,
                reduction='none',
                log_target=False
            ).sum(dim=-1)  # (N, L)
            loss_cyclic_2 = loss_cyclic_2 * mask_res.float()
            if micro:
                loss_cyclic_2 = loss_cyclic_2.sum() / (mask_res.sum() + 1e-8)
            else:
                loss_cyclic_2 = loss_cyclic_2.sum(dim=-1) / (mask_res.sum(dim=-1) + 1e-8)
                loss_cyclic_2 = loss_cyclic_2.mean()

            loss_cyclic = (loss_cyclic + loss_cyclic_2) / 2
        
    ### KLD with the predicted distribution 
    else:
        ### loss calculation
        loss_cyclic = F.kl_div(
            input = log_probs_mpnn[:,:,:20],
            target = c_denoised,
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # (N, L)
        loss_cyclic = loss_cyclic * mask_res.float()
        if micro:
            loss_cyclic = loss_cyclic.sum() / (mask_res.sum() + 1e-8)
        else:
            loss_cyclic = loss_cyclic.sum(dim=-1) / (mask_res.sum(dim=-1) + 1e-8)
            loss_cyclic = loss_cyclic.mean()

    return loss_cyclic


