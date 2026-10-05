import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from diffab.modules.encoders.ga import GAEncoder
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
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


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={},
        LLM_encoder = None, trans_mat = None, LLM_dim = 1280, cross_attn = False
    ):
        super().__init__()

        self.LLM_encoder = LLM_encoder
        if self.LLM_encoder is None:
            self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        else:
            self.trans_mat = trans_mat
            self.aa_dim_map = nn.Linear(LLM_dim, res_feat_dim)

        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, cross_attn = cross_attn, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1) 
        )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, chain_nb = None):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_res:       (N, L).
            chain_nb:       (N, L). 

        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        if self.LLM_encoder is None:
            res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)) # [Important] Incorporate sequence at the current step.
        else:
            #print('LLM encoding')
            #print(self.trans_mat.shape)
            #print(self.trans_mat)
            #print(s_t)
            padding_mask = (s_t >= 21)
            s_t[padding_mask] = 20
            aa_feat = self.trans_mat[s_t]  # (N, L) 
            aa_feat[padding_mask] = 1
            aa_feat = self.LLM_encoder(aa_feat, repr_layers=[33])['representations'][33]  # (N, L, 1280)
            aa_feat = self.aa_dim_map(aa_feat)  # (N, L, feat)
            res_feat = self.res_feat_mixer(torch.cat([res_feat, aa_feat], dim=-1))
            #print('Done')

        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res, chain_nb = chain_nb)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)

        return v_next, R_next, eps_pos, c_denoised


class FullDPM(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        token_size = 21,
        reweighting_term = 0.001,
        ps_adapt_scale = 1.0,
        modality='joint',
        train_version = 'noise',
        proteinMPNN_model = None,
        LLM_encoder = None, 
        trans_mat = None, 
        LLM_dim = 1280,
        with_dist = False,
        dist_clamp = None,
        with_rmsd = False,
        with_center_loss = False,
        with_clash_loss = False,
        clash_thre = 3.6,
        cross_attn = False
    ):
        super().__init__()

        ########################### settings ##################################
        self.num_steps = num_steps
        self.token_size = token_size
        self.train_version = train_version
        self.ps_adapt_scale = ps_adapt_scale

        self.with_rmsd = with_rmsd
        self.dist_clamp = dist_clamp
        self.clash_thre = clash_thre

        if train_version == 'gt':
            self.with_dist = with_dist
            self.with_center_loss = with_center_loss,
            self.with_clash_loss = with_clash_loss,
        else:
            if with_dist:
                print('Distance loss only works for self-conditioning version!')
            if with_center_loss:
                print('Center loss only works for self-conditioning version!')
            if with_clash_loss:
                print('Clash loss only works for self-conditioning version!')

            self.with_dist = False
            self.with_center_loss = False
            self.with_clash_loss = False

        ### for ablation study
        self.modality = modality
        if self.modality not in {'joint', 'sequence', 'structure'}:
            raise Exception('No modality version named %s!' % self.modality)
 
        ########################### status encoder ############################

        self.eps_net = EpsilonNet(
            res_feat_dim, pair_feat_dim, 
            LLM_encoder = LLM_encoder, trans_mat = trans_mat, LLM_dim = LLM_dim,
            cross_attn = cross_attn,
            **eps_net_opt
        )

        ########################### modules ###################################

        ###### rotation diffusion ######
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)

        ###### position diffusion ######
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)

        ###### sequence diffsuion  ######
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        ################################# buffer ##############################
        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        if isinstance(position_scale, str):
            self.position_scale =  position_scale
        else:
            self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))  # (1, 1, 1)
        self.register_buffer('_dummy', torch.empty([0, ]))

    ###########################################################################
    # Position scale and unscale, and other transformations
    ###########################################################################

    def _normalize_position(self, p, protein_size = None):
        """Normalize the coodinates.

        Args:
            p: coordinates matrix; (N, L, 3) 
            protein_size: protein size; (N, )
        """
        if self.position_scale == 'adapt':
            posi_scale = (protein_size.float() * 0.01999327 + 5.91968673) # (N,)
            posi_scale *= self.ps_adapt_scale
            posi_scale = posi_scale.view(-1, 1, 1) # (N, 1, 1)

        elif self.position_scale == 'adapt_all':
            posi_scale = torch.FloatTensor([
                [0.02006428, 5.73314863],  # x
                [0.02043748, 5.69885825],  # y
                [0.02168806, 5.63076041],  # z
            ]).to(p.device)
            posi_scale *= self.ps_adapt_scale
            posi_scale = torch.matmul(
                protein_size.float().reshape(-1, 1), posi_scale[:, 0].reshape(1, -1)
            ) + posi_scale[:, 1]   # (N, 3)
            posi_scale = posi_scale.unsqueeze(dim = 1)  # (N, 1, 3)

        else:
            posi_scale = self.position_scale

        p_norm = (p - self.position_mean) / posi_scale
        return p_norm


    def _unnormalize_position(self, p_norm, protein_size = None):
        """Unnormalize the coodinates.

        Args:
            p: coordinates matrix; (N, L, 3) 
            protein_size: protein size; (N, )
        """
        if self.position_scale == 'adapt':
            posi_scale = (protein_size.float() * 0.01999327 + 5.91968673) # (N,)
            posi_scale *= self.ps_adapt_scale
            posi_scale = posi_scale.view(-1, 1, 1) # (N, 1, 1)

        elif self.position_scale == 'adapt_all':
            posi_scale = torch.FloatTensor([
                [0.02006428, 5.73314863],  # x
                [0.02043748, 5.69885825],  # y
                [0.02168806, 5.63076041],  # z
            ]).to(p.device)
            posi_scale = torch.matmul(
                protein_size.float().reshape(-1, 1), posi_scale[:, 0].reshape(1, -1)
            ) + posi_scale[:, 1]   # (N, 3)
            posi_scale *= self.ps_adapt_scale
            posi_scale = posi_scale.unsqueeze(dim = 1)  # (N, 1, 3)

        else:
            posi_scale = self.position_scale

        p = p_norm * posi_scale + self.position_mean
        return p


    def gt_noise_transfer(self, feat, eps_pred, t):
        alpha_bar = self.trans_pos.var_sched.alpha_bars[t]  # (N,) 
        c0 = 1 / (1 - alpha_bar + 1e-8).view(-1, 1, 1)
        c1 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        eps_pred = c0 * (feat - c1 * eps_pred)
        return eps_pred

    ###########################################################################
    # forward function (get the loss)
    ###########################################################################

    def forward(self, 
        v_0, 
        p_0, 
        s_0, 
        res_feat, 
        pair_feat, 
        mask_generate, 
        mask_res, 
        denoise_structure, 
        denoise_sequence, 
        t=None,
        protein_size=None,
        chain_nb = None,
    ):
        """
        Args:
            ### basic inputs
            v_0: orientation vector, (N, L, 3)
            p_0: CA coordinates, (N, L, 3)
            s_0: aa sequence, (N, L) 
            res_feat: residue feature, (N, L, res_feat_dim) or None
            pair_feat: pair-wise edge feature, (N, L, L, pair_feat_dim) or None
            mask_res: True for valid tokens other than paddings; (N, L)
            denoise_structure: whether do the structure diffusion; bool
            denoise_sequence: whether do the sequence diffusion; bool
            t: None (than will do the random sampling) or (N, )
            protein_size: size of the samples; (N,)
        """

        #############################################
        # data preprocess 
        #############################################

        N, L = res_feat.shape[:2]
        denoise_structure = denoise_structure and (self.modality in {'joint', 'structure'})
        denoise_sequence = denoise_sequence and (self.modality in {'joint', 'sequence'})

        if self.modality == 'sequence':
            ### only sequence is needed
            v_0 = torch.zeros(v_0.shape, device = v_0.device)
            p_0 = torch.zeros(p_0.shape, device = p_0.device)

        elif self.modality == 'structure':
            ### only structure is needed 
            s_0 = torch.zeros(s_0.shape, device = s_0.device).long()

        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        #############################################
        # forward (add noise, 0 to t) 
        #############################################

        if denoise_structure:
            # Add noise to rotation
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        #############################################
        # reverse (denoise, t to t-1)
        #############################################

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res, chain_nb
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)

        #############################################
        # Loss Calculation
        #############################################

        loss_dict = {}

        ######################### Rotation loss ################################

        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        ######################### Position loss ################################
        
        if self.train_version == 'gt':
            p_ref = p_0
        else:
            p_ref = eps_p

        if self.with_rmsd:
            loss_pos = ((eps_p_pred - p_ref) ** 2).sum(-1) # (N, L)
            loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
            loss_pos = loss_pos.sqrt()
        else:
            loss_pos = F.mse_loss(eps_p_pred, p_ref, reduction='none').sum(dim=-1)  # (N, L)
            loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)

        loss_dict['pos'] = loss_pos

        ################### Sequence categorical loss ##########################

        if self.train_version == 'gt':
            mask_seq_design = mask_generate * (s_0 < 20)
            loss_seq = F.cross_entropy(
                 c_denoised[mask_seq_design == 1], s_0[mask_seq_design == 1]
            )
        else: 
            post_true = self.trans_seq.posterior(s_noisy, s_0, t)
            log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
            kldiv = F.kl_div(
                input=log_post_pred, 
                target=post_true, 
                reduction='none',
                log_target=False
            ).sum(dim=-1)    # (N, L)
            loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)

        loss_dict['seq'] = loss_seq

        ################# other constraint losses ##############################

        ###### distance loss ######
        if self.with_dist:
            dist_true = torch.cdist(p_ref, p_ref)  # (N, L, L)
            dist_pred = torch.cdist(eps_p_pred, eps_p_pred)  # (N, L, L)
            loss_dist = F.mse_loss(dist_pred, dist_true, reduction='none')  # (N, L, L)
            if self.dist_clamp is not None:
                loss_dist = torch.clamp(loss_dist, max = self.dist_clamp)
            mask_pair = torch.einsum('bp,bq->bpq', mask_res, mask_res) * (1 - torch.eye(L).to(mask_res.device))
            loss_dist = (loss_dist * mask_pair).sum() / (mask_pair.sum().float() * 2 + 1e-8)
            loss_dict['dist'] = loss_dist

        ###### chain position loss ######
        if self.with_center_loss:
            pred_center = eps_p_pred[mask_generate].mean(dim=1) # (N, 3)
            true_center = p_0[mask_generate].mean(dim=1) # (N, 3) 
            loss_dict['center'] = ((pred_center - true_center) ** 2).sum(-1).sqrt().mean()

        ###### clash loss ######
        if self.with_clash_loss:
            pos_all = torch.where(mask_generate[:, :, None].expand_as(p_0), eps_p_pred, p_0)
            ### map the position back to the original space
            #print(protein_size)
            pos_all = self._unnormalize_position(pos_all, protein_size = protein_size)

            self_dist = torch.triu(torch.cdist(pos_all, pos_all)) # (N, L, L)

            # clash_mask = (self_dist < self.clash_thre) * torch.einsum('bp,bq->bpq', mask_generate, mask_generate) # (N, L, L), for intra-only
            clash_flag = (self_dist < self.clash_thre) # (N, L, L)

            loss_clash = torch.clamp(self.clash_thre - self_dist, min = 0) # (N, L, L)
            loss_clash = loss_clash.sum(dim=(1,2)) / (clash_flag.sum(dim=(1,2)) + 1e-8)

            loss_dict['clash'] = loss_clash 


        return loss_dict

    ###########################################################################
    # sample with context info
    ###########################################################################

    @torch.no_grad()
    def sample(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res,
        chain_nb = None,
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]

        ##############################################################
        # preprocess
        ##############################################################

        ###### structure inititialization ######

        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        ###### structure inititialization ######
        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        ###### trajectory inititialization ######
        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        ##############################################################
        # generation
        ##############################################################

        for t in pbar(range(self.num_steps, 0, -1)):
            #print(t)
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            #print('Prediction')

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, chain_nb = chain_nb
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            #print('Transformation')

            ###### rotation ######
            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)

            ###### CA-position ######
            if self.train_version == 'gt':
                ## eps_p is the predicted ground truth, get x_(t-1)
                p_next = torch.where(mask_generate[:, :, None].expand_as(p_t), eps_p, p_t)

                if t > 1:
                    p_next, _ = self.trans_pos.add_noise(p, mask_generate, t_tensor - 1)
                    p_next = torch.where(mask_generate[:, :, None].expand_as(p_t), p_next, p_t)

            else:
                ## eps_p is the predicted noise
                p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)

            ### sequence
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            ###### sequence only (fix the structure) ###### 
            if not sample_structure:
                v_next, p_next = v_t, p_t
            ###### structure only (fix the sequence) ###### 
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj


    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
