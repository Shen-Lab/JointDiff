"""Modules to add noise on the original sample (forward diffusion process).

By SZ; 01/19/24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from diffab.modules.common.geometry import reconstruct_backbone, construct_3d_basis
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition
from diffab.utils.protein.constants import BBHeavyAtom

from diffab.modules.diffusion.dpm_full import seq_recover 
# diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'


class ForwardDPM(nn.Module):

    def __init__(
        self, 
        num_steps,
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        #Boltzmann_constant = 8.314462618e-3,
    ):
        super().__init__()

        self.num_steps = num_steps
        #self.Boltzmann_constant = Boltzmann_constant

        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    ###########################################################################
    # forward diffusion
    ###########################################################################

    @torch.no_grad()
    def forward_diffusion(
        self, 
        v_0, 
        p_0, 
        s_0,
        t = None,
        seq_sample = 'multinomial',
        posi_scale = None 
    ):
        """
        Add noise to the original samples for predefined t.

        Args:
            v_0: orientation vector, (N, L, 3).
            p_0: CA coordinates, (N, L, 3).
            s_0: aa sequence, (N, L).
            t: None (than will do the random sampling) or (N, ).
            seq_sample: how to sample the sequences based on the multinomial distribution.
            posi_scale: scaling weight of the coordinates; if None, use the default scaling weight of the model.
        """
        if t is None:  # t: None or (N,)
            N, L = v_0.shape[:2]
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)

        ###### for structure ######
 
        ### Add noise to rotation
        v_noisy, _ = self.trans_rot.add_noise(v_0, t)
        #* v_noisy: noised orientation vector, (N, L, 3) 
 
        ### position normalization
        if posi_scale is not None:
            p_0 = (p_0 - self.position_mean) / posi_scale
        else:
            p_0 = self._normalize_position(p_0)
        ### Add noise to positions
        p_noisy, eps_p = self.trans_pos.add_noise(p_0, t)
        ### denormalization 
        if posi_scale is not None:
            p_noisy = p_noisy * posi_scale + self.position_mean
        else:
            p_noisy = self._unnormalize_position(p_noisy)
        #* p_noisy: noised position, (N, L, 3) 
        #* eps_p: Gaussian noise, (N, L, 3)

        ###### for sequence #######
        ### Add noise to sequence
        s_noise_prob, s_noisy = self.trans_seq.add_noise(s_0, t, method = seq_sample)
        #* s_noise_prob: noised sequence distribution; (N, L, K = 20)
        #*               s_noise_prob = alpha_bar * onehot(s_0) + ((1 - alpha_bar) / K); 
        #* s_noisy: noised sequence, s_noisy = sample(s_noise_prob); (N, L)

        return v_noisy, p_noisy, s_noise_prob, s_noisy

    def forward_trajectory(
        self,
        v_0,
        p_0,
        s_0,
        seq_sample = 'multinomial',
        posi_scale = None
    ):
        """
        Apply the forward trajectory on the original samples.

        Args:
            v_0: orientation vector, (N, L, 3).
            p_0: CA coordinates, (N, L, 3).
            s_0: aa sequence, (N, L).
            seq_sample: how to sample the sequences based on the multinomial distribution.
            posi_scale: scaling weight of the coordinates.
        """
        traj = {}

        for t in range(1, self.num_steps + 1):

            t_tensor = torch.ones(v_0.shape[0], device=self._dummy.device).long() * t
            v_noisy, p_noisy, s_noise_prob, s_noisy = self.forward_diffusion(
                                                               v_0 = v_0,
                                                               p_0 = p_0,
                                                               s_0 = s_0,
                                                               t = t_tensor,
                                                               seq_sample = seq_sample,
                                                               posi_scale = posi_scale
                                                           )
            traj[t] = (v_noisy.cpu(), p_noisy.cpu(), s_noisy.cpu(), s_noise_prob.cpu())

        return traj


    def backbone_trajectory(
        self,
        batch,
        seq_sample = 'multinomial',
        posi_scale = None
    ):
        """
        Apply the forward trajectory on the original samples and get the noised backbones.

        Args:
            batch: dictionary containing the backbone coordinates, sequences and masks.
                pos_heavyatom: (N, L, 4, 3)
                aa: (N, L)
                mask: (N, L)
            posi_scale: scaling weight of the coordinates.
        """
        ### data process
        R_0 = construct_3d_basis(
           batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
           batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
           batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        v_0 = rotation_to_so3vec(R_0)  # transform SO(3) to 3d vectors (N, L, 3)
        p_0 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        s_0 = batch['aa']  # amino acid index; int, 0~19, 21 for padding;  (N, L)
        mask_res = batch['mask'].cpu()

        ###### forward trajectory ######
        traj = self.forward_trajectory(
                        v_0,
                        p_0,
                        s_0,
                        seq_sample = seq_sample,
                        posi_scale = posi_scale
                    )

        ###### get the backbones and sequences ######
        N, L = mask_res.shape
        res_nb = torch.zeros(N,L).int().to(mask_res.device)
        res_nb[:] = torch.arange(1, L+1).to(mask_res.device)
        res_nb = (res_nb * mask_res).int() # residue idx to determine consecutive residues; int, ordinal, e.g. 1,2,...,L, 0, ..., 0; (N, L_max)
        chain_nb = mask_res.int()
        lengths = mask_res.sum(-1)

        out_dict = {}
        for t in traj.keys():
            out_dict[t] = []
            R = so3vec_to_rotation(traj[t][0])
            bb_coor_batch = reconstruct_backbone(R, traj[t][1], traj[t][2], chain_nb, res_nb, mask_res)  # (N, L_max, 4, 3)
            for i, bb_coor in enumerate(bb_coor_batch):
                seq = seq_recover(traj[t][2][i], length = lengths[i]) 
                out_dict[t].append({'coor': bb_coor[:lengths[i]], 'seq': seq})

        return out_dict, traj
