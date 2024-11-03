import typing as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import ml_collections
from typing import Dict, Optional, Tuple

try:
    from openfold.config import model_config
    from openfold.utils.rigid_utils import Rotation, Rigid
    from openfold.utils.loss import compute_renamed_ground_truth  # AlphaFoldLoss, lddt_ca
except Exception as e:
    print(e)

#############################################################
# embedding (masked MSE loss)
#############################################################

def emb_loss(tensor_1, tensor_2, mask=None, mode='train'):
    """Masked MSE loss."""
    if mask is None and mode == 'train':
        ### train without mask
        loss = F.mse_loss(tensor_1, tensor_2)  # (1,)
    elif mask is None:
        ### eval with mask
        loss = F.mse_loss(
            tensor_1, 
            tensor_2,
            reduction = 'none'
        )  # (B, *)
        B = loss.shape[0]
        loss = loss.reshape(B, -1).mean(dim = -1)  # (B,)
    else: 
        loss = F.mse_loss(
            tensor_1, 
            tensor_2,
            reduction = 'none'
        )  # (B, *)
        B = loss.shape[0]

        if mode == 'train':
            ### train with mask
            loss = loss[mask == 1].mean()
        else:
            ### eval with mask
            loss_out = torch.zeros(B,).to(loss.device)
            for i in range(B):
                loss_out[i] = loss[i][mask[i] == 1].mean()
            loss = loss_out

    return loss


def contrastive_loss(emb_1, emb_2, mask = None, temperature = 1.):
    """
    Contrastive loss.
    Positive pairs: emb_1[i, :] and emb_2[i,:]
    Negative pairs: emb_1[i, :] and emb_2[j,:], i != j.

    Args:
        emb_1: (N, *)
        emb_2: (N, *)
        mask: (N, *)
    """
    N = emb_1.shape[0]
    if N <= 1:
        return 0

    if mask is not None:
        emb_1[mask == 0] = 0
        emb_2[mask == 0] = 0
    emb_1 = F.normalize(emb_1.reshape(N, -1), dim=1)  # (N, l)
    emb_2 = F.normalize(emb_2.reshape(N, -1), dim=1)  # (N, l)
    emb_all = torch.cat([emb_1, emb_2], dim = 0)  # (2N, l)
    similarity_matrix = torch.mm(emb_all, emb_all.t())  # (2N, 2N)

    labels = torch.cat([torch.arange(N) for _ in range(2)], dim=0).to(emb_1.device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Mask to remove the diagonal from the similarity matrix
    diag_mask = torch.eye(labels.shape[0], dtype=torch.bool).to(emb_1.device)
    labels = labels[~diag_mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~diag_mask].view(similarity_matrix.shape[0], -1)
    
    # Compute contrastive loss
    loss = -torch.sum(labels * F.log_softmax(similarity_matrix, dim=1), dim=1).mean()

    return loss

#############################################################
# for VAE (KLD loss)
#############################################################

def kld_cal(mu, sig, mask, habits_lambda = 0.2):
    """KL-divergence between N(mu, sig) and N(0,1)."""
    if mask is not None:
        mu[mask == 0] = 0
        sig[mask == 0] = 0
        denorm = mask.sum()
    else:
        denorm = torch.prod(torch.tensor(mu.shape).float())
        
    out = -0.5 * torch.sum(
        sig - torch.pow(mu, 2) - torch.exp(sig) + 1, 1
    ).sum() / denorm 
    if habits_lambda is not None:
        out = torch.clamp(out, min=habits_lambda)

    return out


def kld_loss(out_dict, mask, pair_mask, habits_lambda = 0.2):
    """KLD loss for VAE."""

    KLD = 0.

    for key in [
        'seq_mu_feat', 
        'seq_mu_feat_pair', 
        'struc_mu_feat', 
        'struc_mu_feat_pair', 
    ]:
        if key in out_dict and out_dict[key] is not None:
            mask_sele = pair_mask if key.endswith('pair') else mask

            KLD += kld_cal(
                out_dict[key], out_dict['sigma'.join(key.split('mu'))],
                mask_sele, habits_lambda
            )

    return KLD

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


#############################################################
# for structure
#############################################################

# modified based on the openfold implementation: 
# https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py

########### Distogram prediction loss #######################

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


def distance_loss(
    coor_pred, ref, mask_res = None,
    with_dist = False, dist_clamp = None, loss_version = 'mse',
    with_clash = False, threshold_clash = 3.6,
    with_gap = False, threshold_gap = 3.9,
):
    """Calculate the distance similarity between the predicted structures and 
    groundtruths.

    Args:
        coor_pred: predicted coordinates; (N, L, 3)
        ref: groundtruth coordinates or distance mat; (N, L, 3) or (N, L, L)
        mask_res: 1 for valid positions; (N, L)
        with_dist: whether ref is the distance mat.
        dist_clamp: maximum clamp of the distance loss.
        loss_version: versions of the loss functions; 'mse' of 'l1'
    """
    if not with_dist:
        ref = torch.cdist(ref, ref)  # (N, L, L)
    N, L, _ = ref.shape
    if mask_res is None:
        mask_res = torch.ones(N, L).to(ref.device) 

    ### predicted distance mat
    dist_pred = torch.cdist(coor_pred, coor_pred)  # (N, L, L)

    ############################ distance loss ################################

    ### loss functions
    if loss_version == 'mse':
        loss_dist = F.mse_loss(dist_pred, ref, reduction='none') # (N, L, L)
    else:
        loss_dist = F.l1_loss(dist_pred, ref, reduction='none') # (N, L, L)

    ### clamp 
    if dist_clamp is not None:
        loss_dist = torch.clamp(loss_dist, max = dist_clamp)
    
    ### average
    mask_pair = torch.einsum(
        'bp,bq->bpq', mask_res, mask_res
    ) * (1 - torch.eye(L).to(mask_res.device))
    loss_dist = (loss_dist * mask_pair).sum(dim=(1,2)) 
    loss_dist = loss_dist / (mask_pair.sum(dim=(1,2)).float() + 1e-8) # (N,)
    loss_dist = loss_dist.mean()

    ############################ clash loss ###################################

    if with_clash:
        clash_flag = (dist_pred < threshold_clash) * mask_pair # (N, L, L)
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


######################### FAPE ################################################

try:
    def compute_fape(
        pred_frames: Rigid,
        target_frames: Rigid,
        frames_mask: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        positions_mask: torch.Tensor,
        length_scale: float,
        l1_clamp_distance: Optional[float] = None,
        eps=1e-8,
    ) -> torch.Tensor:
        """Computes FAPE loss.
    
        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    
        """
        # [*, N_frames, N_pts, 3]
        local_pred_pos = pred_frames.invert()[..., None].apply(
            pred_positions[..., None, :, :],
        )
        local_target_pos = target_frames.invert()[..., None].apply(
            target_positions[..., None, :, :],
        )
    
        #print(local_pred_pos.shape, local_target_pos.shape)
        error_dist = torch.sqrt(
            torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
        )
    
        if l1_clamp_distance is not None:
            error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)
    
        normed_error = error_dist / length_scale
        normed_error = normed_error * frames_mask[..., None]
        normed_error = normed_error * positions_mask[..., None, :]
    
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = (
            normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        )
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    
        return normed_error
    
    
    def backbone_loss(
        backbone_rigid_tensor: torch.Tensor,
        backbone_rigid_mask: torch.Tensor,
        traj: torch.Tensor,
        # use_clamped_fape: Optional[torch.Tensor] = None,
        use_clamped_fape: float = None,  # 0.9 for ESMFold
        clamp_distance: float = 10.0,
        loss_unit_distance: float = 10.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            backbone_rigid_tensor: (B, L_max, 4, 4)
            backbone_rigid_mask: (B, L_max)
            traj: structure["frames"], (8, B, L_max, 7)
            use_clamped_fape: 0.9 for ESMFold
            *** default value in config ***
            clamp_distance: 10.0
            loss_unit_distance: 10.0
            eps: 0.0001
        """
    
        pred_aff = Rigid.from_tensor_7(traj)
        pred_aff = Rigid(
            Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
            pred_aff.get_trans(),
        )
    
        # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
        # backbone tensor, normalizes it, and then turns it back to a rotation
        # matrix. To avoid a potentially numerically unstable rotation matrix
        # to quaternion conversion, we just use the original rotation matrix
        # outright. This one hasn't been composed a bunch of times, though, so
        # it might be fine.
        gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)  # (*)
    
        # clamped loss
        fape_loss = compute_fape(
            # predicted frames: Rigid object of predicted frames, [*, N_frames]
            pred_aff,
            # groundtruth frames: Rigid object of ground truth frames, [*, N_frames]
            gt_aff[None],
            # frames_mask: binary mask for the frames, [*, N_frames]
            backbone_rigid_mask[None],
            # predites positions: predicted atom positions, [*, L_max, 3]
            pred_aff.get_trans(),
            # target positions: ground truth positions, [*, L_max, 3]
            gt_aff[None].get_trans(),
            # positions_mask: positions mask, [*, N_pts]
            backbone_rigid_mask[None],
            # Cutoff above which distance errors are disregarded, float
            l1_clamp_distance=clamp_distance,
            # Length scale by which the loss is divided, float
            length_scale=loss_unit_distance,
            eps=eps,  # Small value used to regularize denominators. float
        )
    
        # unclamped loss
        if use_clamped_fape is not None:
            unclamped_fape_loss = compute_fape(
                pred_aff,
                gt_aff[None],
                backbone_rigid_mask[None],
                pred_aff.get_trans(),
                gt_aff[None].get_trans(),
                backbone_rigid_mask[None],
                l1_clamp_distance=None,
                length_scale=loss_unit_distance,
                eps=eps,
            )
    
            fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
                1 - use_clamped_fape
            )
    
        # Average over the batch dimension
        fape_loss = torch.mean(fape_loss)
    
        return fape_loss
    
    
    def sidechain_loss(
        sidechain_frames: torch.Tensor,
        sidechain_atom_pos: torch.Tensor,
        rigidgroups_gt_frames: torch.Tensor,
        rigidgroups_alt_gt_frames: torch.Tensor,
        rigidgroups_gt_exists: torch.Tensor,
        renamed_atom14_gt_positions: torch.Tensor,
        renamed_atom14_gt_exists: torch.Tensor,
        alt_naming_is_better: torch.Tensor,
        use_clamped_fape: float = None,  # 0.9 for ESMFold, added by SZ
        clamp_distance: float = 10.0,
        length_scale: float = 10.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            sidechain_frames: [8*B, 1, L_max, 8, 4, 4]
            sidechain_atom_pos: [8*B, 1, L_max, 14, 3]
            rigidgroups_gt_frames: 
            rigidgroups_alt_gt_frames: 
            rigidgroups_gt_exists: 
            renamed_atom14_gt_positions: 
            renamed_atom14_gt_exists: 
            alt_naming_is_better: 
            *** default value in config ***
            clamp_distance: 10.0
            loss_unit_distance: 10.0
            eps: 0.0001
        """
    
        renamed_gt_frames = (
            1.0 - alt_naming_is_better[..., None, None, None]
        ) * rigidgroups_gt_frames + alt_naming_is_better[
            ..., None, None, None
        ] * rigidgroups_alt_gt_frames
    
        # Steamroll the inputs
        #print(sidechain_frames.shape, sidechain_atom_pos.shape)
        #print(rigidgroups_gt_frames.shape, renamed_gt_frames.shape, rigidgroups_gt_exists.shape)
        #print(renamed_atom14_gt_positions.shape, renamed_atom14_gt_exists.shape)
        # quit()
    
        length = sidechain_frames.shape[-4]
        # side-chain frames
        sidechain_frames = sidechain_frames.reshape(8, -1, length, 8, 4, 4)
        sidechain_frames = sidechain_frames[-1]  # (B, L_max, 8, 4, 4)
        batch_dims = sidechain_frames.shape[:-4]
        sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
        sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
        # gt frames
        renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
        renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
        rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
        # side-chain position
        sidechain_atom_pos = sidechain_atom_pos.reshape(8, -1, length, 14, 3)
        sidechain_atom_pos = sidechain_atom_pos[-1]
        sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
        renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
            *batch_dims, -1, 3
        )
        # renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)  # by SZ
        renamed_atom14_gt_exists = renamed_atom14_gt_exists.reshape(
            *batch_dims, -1)
    
        fape_loss = compute_fape(
            # predicted frames: Rigid object of predicted frames, [*, N_frames = L_max * 8]
            sidechain_frames,
            # groundtruth frames: Rigid object of ground truth frames, [*, N_frames]
            renamed_gt_frames,
            # frames_mask: binary mask for the frames, [*, N_frames]
            rigidgroups_gt_exists,
            # predites positions: predicted atom positions, [*, N_pts = L_max * 14, 3]
            sidechain_atom_pos,
            # target positions: ground truth positions, [*, N_pts, 3]
            renamed_atom14_gt_positions,
            renamed_atom14_gt_exists,  # positions_mask: positions mask, [*, N_pts]
            # Cutoff above which distance errors are disregarded, float
            l1_clamp_distance=clamp_distance,
            length_scale=length_scale,  # Length scale by which the loss is divided, float
            eps=eps,  # Small value used to regularize denominators. float
        )
        # unclamped loss (added by SZ)
        if use_clamped_fape is not None:
            unclamped_fape_loss = compute_fape(
                # predicted frames: Rigid object of predicted frames, [*, N_frames]
                sidechain_frames,
                # groundtruth frames: Rigid object of ground truth frames, [*, N_frames]
                renamed_gt_frames,
                # frames_mask: binary mask for the frames, [*, N_frames]
                rigidgroups_gt_exists,
                # predites positions: predicted atom positions, [*, N_pts, 3]
                sidechain_atom_pos,
                # target positions: ground truth positions, [*, N_pts, 3]
                renamed_atom14_gt_positions,
                # positions_mask: positions mask, [*, N_pts]
                renamed_atom14_gt_exists,
                l1_clamp_distance=None,  # Cutoff above which distance errors are disregarded, float
                length_scale=length_scale,  # Length scale by which the loss is divided, float
                eps=eps,  # Small value used to regularize denominators. float
            )
    
            fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
                1 - use_clamped_fape
            )
    
        return fape_loss
    
    
    def fape_loss(
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        config: ml_collections.ConfigDict,
    ) -> torch.Tensor:
        """
        Args:
            out:
                frames: float, (8, B, L_max, 7)
                sidechain_frames: float, (8, B, L_max, 8, 4, 4)
                positions: float, (8, B, L_max, atom_num=14, 3)
            batch:
                ### for backbone
                backbone_rigid_tensor: float, (B, L_max, 4, 4)
                backbone_rigid_mask: binary, (B, L_max)
                use_clamped_fape: 0.9
                ### for sidechain
                rigidgroups_gt_frames: float, (B, L_max, 8, 4, 4)
                rigidgroups_alt_gt_frames: float, (B, L_max, 8, 4, 4)
                rigidgroups_gt_exists: binary, (B, L_max, 8)
                renamed_atom14_gt_positions
                renamed_atom14_gt_exists 
                alt_naming_is_better
            config:
                backbone:
                    clamp_distance: 10.0
                    loss_unit_distance: 10.0
                    weight: 0.5
                sidechain:
                  clamp_distance: 10.0
                  length_scale: 10.0
                  weight: 0.5
                eps: 0.0001
                weight: 1.0
        """
        # backbone loss
        bb_loss = backbone_loss(
            traj=out['frames'],  # (8, B, L_max, 7)
            **{**batch, **config.backbone},
        )
    
        # side-chain loss
        sc_loss = sidechain_loss(
            out['sidechain_frames'],  # (8, B, L_max, 8, 4, 4)
            out['positions'],  # (8, B, L_max, atom_num=14, 3)
            **{**batch, **config.sidechain},
        )
    
        loss = config.backbone.weight * bb_loss + config.sidechain.weight * sc_loss
    
        # Average over the batch dimension
        loss = torch.mean(loss)
    
        return loss
    
    # esmfold structure loss: 
    #     developed based on the SI of ESMFold;
    #     contain Frame Aligned Point Error (FAPE) and distogram losses introduced in AlphaFold2;
    #     for FAPE, both clamped and unclamped losses and take the sum, with weights of 0.9 and 0.1 respectively
    
    
    def strutcure_loss(
            out: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            config: ml_collections.ConfigDict,
            **kwargs,):
        """
        Args:
            out:
                ### for distogram
                distogram_logits, float, (B, L_max, L_max, 64)
                ### for fape
                frames: float, (8, B, L_max, 7)
                sidechain_frames: float, (8, B, L_max, 8, 4, 4)
                positions: float, (8, B, L_max, atom_num=14, 3)
            batch:
                ### for distogram
                pseudo_beta: float, (B, L_max, 3)
                pseudo_beta_mask: binary, (B, L_max)
                ### for backbone fape
                backbone_rigid_tensor: float, (B, L_max, 4, 4)
                backbone_rigid_mask: binary, (B, L_max)
                use_clamped_fape: 0.9
                ### for sidechain fape
                rigidgroups_gt_frames: float, (B, L_max, 8, 4, 4)
                rigidgroups_alt_gt_frames: float, (B, L_max, 8, 4, 4)
                rigidgroups_gt_exists: binary, (B, L_max, 8)
                ### for ground truth atom14
                atom14_gt_positions: float, (B, L_max, 14, 3)
                atom14_alt_gt_positions: float, (B, L_max, 14, 3)
                atom14_gt_exists: binary, (B, L_max, 14)
                atom14_atom_is_ambiguous: binary, (B, L_max, 14)
                atom14_alt_gt_exists: binary, (B, L_max, 14)
            config:
                distogram:
                    eps: 1.0e-08
                    max_bin: 21.6875
                    min_bin: 2.3125
                    no_bins: 64
                    weight: 0.3
                fape:
                    backbone:
                        clamp_distance: 10.0
                        loss_unit_distance: 10.0
                        weight: 0.5
                    sidechain:
                      clamp_distance: 10.0
                      length_scale: 10.0
                      weight: 0.5
                    eps: 0.0001
                    weight: 1.0
        """
        # feature update
        batch.update(
            compute_renamed_ground_truth(
                batch,
                out['positions'][-1],
            )
        )
        # Add:
        #     renamed_atom14_gt_positions
        #     renamed_atom14_gt_exists
        #     alt_naming_is_better
    
        # distogram losses
        dist_loss = distogram_loss(
            logits=out['distogram_logits'],
            **{**batch, **config.distogram},
        )
    
        # fape loss
        fape = fape_loss(
            out=out,
            batch=batch,
            config=config.fape,
        )
    
        # final loss
        loss = config.fape.weight * fape + config.distogram.weight * dist_loss
    
        return loss

except Exception as e:
    print(e)

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
        # loss_cyclic = F.cross_entropy(
        #     input=log_probs_mpnn.transpose(1,2).to(s_gt.device), 
        #     target=s_gt
        # ) # (N, L)
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


################### energy-guided loss ######################

def energy_guided_loss(
    p_noisy, eps_p_pred, s_pred, mask_res,
    with_energy_loss = True, with_fitness_loss = True,
    energy_guide = None,
    energy_guide_type = 'cosine',
    struc_scale = 'Boltzmann',
    Boltzmann_constant = 8.314462618e-3,
    temperature = 300,
    energy_aggre = 'all',
    RepulsionOnly = False,
    with_resi = False,
    multithread = False,
    with_contact = True,
    contact_path_list = None,
    atom_list = ['CA'],
    contact_thre = 12,
    fitness_guide = None,
    fitness_guide_type = 'cosine',
    seq_scale = 'none',
    t_max = None,
    force_vs_diff = False,
):
    ########################################################
    # Position Guidence loss (sequence and CA-structure are both required) 
    ########################################################

    if with_energy_loss:

        if RepulsionOnly:
            energy_aggre = 'LJ 12 Repulsion Energy'

        ### force calculation 
        #* (force / negative gradient); energy is the lower the better
        force_dict, _ = energy_guide(
            p_noisy,
            s_pred,
            mask = mask_res,
            atom_list = atom_list,
            with_contact = with_contact,
            contact_path_list = contact_path_list,
            contact_thre = contact_thre,
            get_force = True,
            get_energy = False,
            multithread = multithread,
            sum_result = (energy_aggre == 'all'),
            RepulsionOnly = RepulsionOnly,
            with_resi = with_resi
        )
        #* force_dict: {key: (N, L, 3)}
        #*     if sum_result is True, key = 'all';
        #*     otherwise key in ['Harmonic Bond Force', 'Harmonic Angle', 'Periodic Torsion', 'LJ 12-10 Contact', 'LJ 12 Repulsion'].

        ### energy guidance loss: || score_pred - force ||^2
        #* q(x_t | x_(t-1)) = N(x | sqrt(1 - beta_t) * x_(t-1), beta_t * I)
        if struc_scale == 'Boltzmann':
            struc_scale = 1 / (Boltzmann_constant * temperature)
        elif struc_scale == 'negative-Boltzmann':
            struc_scale = -1 / (Boltzmann_constant * temperature)
        elif struc_scale == 'none':
            struc_scale = 1.

        ### MSE loss
        if energy_aggre != 'scheduled' and energy_aggre in energy_aggre:
            force_mat = force_dict[energy_aggre]  # (N, L, 3)

        elif energy_aggre == 'scheduled':
            pass

        else:
            print('No energy aggregating method named %s!'%energy_aggre)
            quit()

        ### prepare the term to compare with force
        if force_vs_diff:
            ### rather than make force close to G, make force close to G
            p_next = self.trans_pos.denoise(p_noisy, eps_p_pred, t)
            eps_p_pred = p_noisy - p_next
        else:
            eps_p_pred = self._unnormalize_position(eps_p_pred, protein_size = protein_size)
            #* eps_p_pred = G(R_t); (N, L, 3)
            #* mu_(t-1) = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * G(R_t))
            #*          = (1/sqrt(alpha_t)) * x_t - (1/sqrt(alpha_t)) * (beta_t / sqrt(1 - alpha_bar_t)) * G(R_t)
            #* force is the negative gradient

        if energy_guide_type == 'cosine':
            ### minus cosine similaity: minus G should be on a close direction of force
            loss_energy = - F.cosine_similarity(- eps_p_pred, force_mat, dim = -1)  # (N, L) 

        elif energy_guide_type == 'mse':
            ### mse: minus G should be close to the force
            loss_energy = F.mse_loss(- eps_p_pred,
                force_mat * struc_scale, reduction='none').sum(dim=-1)  # (N, L) 

        if t_max is None:
            loss_energy = (loss_energy * mask_res).sum() / (mask_res.sum().float() + 1e-8)
        else:
            t_mask = (t <= t_max)  # (N,)
            if t_mask.any():
                t_mask = mask_res * (t_mask.reshape(-1, 1))
                loss_energy = (loss_energy * t_mask).sum() / (t_mask.sum().float() + 1e-8)
            else:
                loss_energy = torch.zeros(1)[0].to(t_mask.device)

    else:
        loss_energy = None

    ########################################################
    # Fitness Guidence loss (sequence is required) 
    ########################################################

    if with_fitness_loss:
        ###### fitness guidance loss ######
        #* || (pred_multinomial - noised_multinomial) * seq_scale - fitness_grad ||^2
        #* q(s_t | s_(t-1)) = Multonomial((1 - beta_t) * onehot(s_(t-1)) + beta_t * (1/20) * one-vec)

        ### fitness calculation
        if fitness_guide_type not in {'direction', 'gt_fitness'}:
            ### scale weight of the fitness graduate 
            if seq_scale == 'length':
                seq_scale = mask_res.sum(dim = -1).reshape(-1,1,1)  # (N,1,1)
            elif seq_scale == 'none':
                seq_scale = 1.
            elif seq_scale == 'negative':
                seq_scale = - 1.

            ### gradient of fitness score; fitness score is the higher the better
            _, fitness_grad = fitness_guide(s_noisy, mask = mask_res,
                                  with_grad = True, seq_transform = True, with_padding = False)
            fitness_grad = fitness_grad.to(mask_res.device)
            #* fitness_grad: gradient of fitness score, (N, L, 20)

        ###  claculation
        if fitness_guide_type == 'direction':
            ### direction of (post_pred - s_noise_prob) should be close to that of (s0_onehot - s_noise_prob)
            s0_onehot = F.one_hot(s_0, num_classes = self.token_size + 1)  # (N, L, token_size + 1)
            s0_onehot = s0_onehot[:, :, : s_noise_prob.shape[-1]]  # (N, L, K)
            loss_fitness = - F.cosine_similarity(post_pred - s_noise_prob,
                s0_onehot - s_noise_prob, dim = -1)  # (N, L)

        elif fitness_guide_type == 'gt_fitness':
            ### fitness on post_pred should be large than that on s_noise_prob
            s0_onehot = F.one_hot(s_0, num_classes = self.token_size + 1)  # (N, L, token_size + 1)
            s0_onehot = s0_onehot[:, :, : s_noise_prob.shape[-1]]  # (N, L, K)
            fitness_noise = (s0_onehot * s_noise_prob).max(-1)[0]
            fitness_pred = (s0_onehot * post_pred).max(-1)[0]
            loss_fitness = fitness_noise - fitness_pred  # maximize (fitness_pred - fitness_noise)

        elif fitness_guide_type == 'cosine':
            ### minus cosine similaity: direction of (post_pred - s_noise_prob) should be close to that of fitness_grad
            loss_fitness = - F.cosine_similarity(post_pred - s_noise_prob,
                fitness_grad * seq_scale, dim = -1)  # (N, L)

        elif fitness_guide_type == 'mse':
            ### MSE loss: (post_pred - s_noise_prob) should be close to fitness_grad
            #loss_fitness = F.mse_loss((log_post_pred - torch.log(s_noise_prob + 1e-8)), 
            loss_fitness = F.mse_loss(post_pred - s_noise_prob,
                fitness_grad * seq_scale, reduction='none').sum(dim=-1)  # (N, L)

        if t_max is None:
            loss_fitness = (loss_fitness * mask_res).sum() / (mask_res.sum().float() + 1e-8)
        else:
            t_mask = (t <= t_max)  # (N,)
            if t_mask.any():
                t_mask = mask_res * t_mask.reshape(-1, 1)
                loss_fitness = (loss_fitness * t_mask).sum() / (t_mask.sum().float() + 1e-8)
            else:
                loss_fitness = torch.zeros(1)[0].to(t_mask.device)
    else:
        loss_fitness = None

    return loss_energy, loss_fitness
