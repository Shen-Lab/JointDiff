import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
from typing import Dict

### difhusion in the original space based on diffab
from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix, reconstruct_backbone
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from diffab.modules.encoders.ga import GAEncoder
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition

from diffab.modules.encoders.residue import ResidueEmbedding 
from diffab.modules.encoders.pair import PairEmbedding  
from diffab.utils.protein.constants import ressymb_order 
# diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'

### discrete diffusion and autoregressive diffusion based on evodiff
from evodiff.model import ByteNetLMTime
from evodiff.utils import TokenizerSingleSeq 
from evodiff.collaters import OAMaskCollaterSingleSeq, D3PMCollaterSingleSeq
from evodiff.losses import OAMaskedCrossEntropyLoss, D3PMCELoss, D3PMLVBLoss

### for advanced structure decoder
# try:
#     from utils_modules import TriangularSelfAttentionNetwork
#     from esm.esmfold.v1.trunk import (
#         StructureModuleConfig,
#         FoldingTrunkConfig,
#         FoldingTrunk
#     )
# except Exception as e:
#     print('Failed to load ESMFold decoder! (%s)' % e)

### for losses 
from losses import distance_loss, consistency_loss, energy_guided_loss


##########################################################################################
# Auxiliary Functions
##########################################################################################

def seq_recover(aa:torch.Tensor, length:int = None) -> str:
    """Recover sequence from the tensor.

    Args:
        aa: embedded sequence tensor; (L,).
        length: length of the sequence; if None consider the paddings.

    Return:
        seq: recovered sequence string. 
    """

    length = aa.shape[0] if length is None else min(length, aa.shape[0])
    seq = ''
    for i in range(length):
        idx = int(aa[i])
        if idx > 20:
            print('Error! Index %d is larger than 20.'%idx)
            break
        seq += ressymb_order[idx]
    return seq


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


def aa_sampling(c, seq_sample_method = 'multinomial'):
    """
    Args:
        c:    probalility sample; (N, L, K).
    Returns:
        x:    (N, L).
    """
    if seq_sample_method == 'multinomial':
        N, L, K = c.size()
        c = c.view(N*L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
    else:
        x = torch.max(c, dim = -1).indices
    return x


##########################################################################################
# Epsilon Network
##########################################################################################

class EpsilonNet(nn.Module):
    def __init__(self, 
        res_feat_dim: int, 
        pair_feat_dim: int, 
        num_layers: int, 
        encoder_opt: Dict = {},
        num_atoms: int = 4, 
        gen_version: str = 'complete_gen', 
        residue_embed: torch.Tensor = None, 
        pair_embed: torch.Tensor = None,
        decoder_version = 'mlp',
        folding_trunk_opt: Dict = {}
    ):
        super().__init__()
        """Encoder of the sample states.
        
        Revised by SZ for single-chain generation: use the same resi encoder 
        and the pair encoder from the encoder.

        Args:
            res_feat_dim: dimension of residue-wise embedding features. 
            pair_feat_dim: dimension of pair-wise embedding features. 
            num_layers: number of layers.
            encoder_opt: other hyper-parameters of GAEncoder; if empty use the 
                default values.

            ### the arguments below were added by SZ.
            num_atoms: atoms considered for embedding.
            gen_version: version of the encoder (for implementation comparison).
                diffab_gen: original diffab version; the resi_feat and the 
                    pair_feat are from the encoder.
                complete_gen: updated version; the resi_feat and the 
                    pair_feat are calculated based on the current stat
            residue_embed: sequence encoder.
            pair_embed: edge feature encoder.
        """

        self.gen_version = gen_version

        #######################################################################
        # feature embedding module 
        #######################################################################

        ###### original diffab implementation ######
        if self.gen_version == 'diffab_gen':
            ### embedding layer
            self.current_sequence_embedding = nn.Embedding(
                    25, 
                    res_feat_dim
            )  # 22 is padding

            ### residue-wise feature mixer
            self.res_feat_mixer = nn.Sequential(
                nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, res_feat_dim),
            )

        ###### updated diffab for resi_feat and pair feat ######
        elif self.gen_version == 'complete_gen':

            ### residue (node) info 
            if residue_embed is not None:  # utilize the sequence encoder from the encoder
                self.residue_embed = residue_embed
            else:
                self.residue_embed = None
                self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
                self.res_feat_encode = nn.Sequential(
                     nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
                     nn.Linear(res_feat_dim, res_feat_dim),
                 )

            ### struture (edge) info
            if pair_embed is not None:
                self.pair_embed = pair_embed
                self.num_atoms = self.pair_embed.max_num_atoms
            else:
                self.pair_embed = PairEmbedding(pair_feat_dim, num_atoms)
                self.num_atoms = num_atoms

        ###### others: raise error ######
        else:
            raise Exception('Error! No generation version called %s!'%self.gen_version)

        #######################################################################
        # massage passing module 
        #######################################################################

        ###### embed the current features rather than the groundtruth ######
        self.encoder = GAEncoder(
            res_feat_dim, pair_feat_dim, num_layers, **encoder_opt
        )

        #######################################################################
        # prediction projector
        #######################################################################

        self.decoder_version = decoder_version

        ###### for energy prediction ######
        if self.decoder_version == 'CEP':
            self.energy_net = self.eps_crd_net = nn.Sequential(
                nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, 1)
            )

        ###### MLP decoders (the orgiginal diffab implementation) ######
        elif self.decoder_version == 'mlp':

            ### position decoder
            self.eps_crd_net = nn.Sequential(
                nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, 3)
            )

            ### rotation decoder
            self.eps_rot_net = nn.Sequential(
                nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, 3)
            )

        ###### TriangularSelfAttentionNetwork decoder ######
        elif self.decoder_version == 'tab':

            ### position decoder
            self.eps_crd_net = TriangularSelfAttentionNetwork(
                in_resi_features = res_feat_dim+3,
                in_pair_features = pair_feat_dim,
                out_resi_features = 3,
                out_pair_features = None,
                num_heads = 1,
                num_blocks = 2
            )

            ### rotation decoder
            self.eps_rot_net = TriangularSelfAttentionNetwork(
                in_resi_features = res_feat_dim+3,
                in_pair_features = pair_feat_dim,
                out_resi_features = 3,
                out_pair_features = None,
                num_heads = 1,
                num_blocks = 2
            )

        elif self.decoder_version == 'esmfold_decoder':
            ### structure decoder
            self.eps_struc_net = FoldingTrunk(**folding_trunk_opt) 

        else:
            raise Exception('No decoder version named %s!' % self.decoder_version)

        ###### sequence decoder ######
        if self.decoder_version != 'CEP':
            self.eps_seq_net = nn.Sequential(
                nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
                nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1)
            )

    #######################################################################
    # Forward Function
    #######################################################################

    def forward(self, 
        v_t, 
        p_t, 
        s_t, 
        beta, 
        mask_res, 
        mask_gen, 
        res_feat = None, 
        pair_feat = None, 
        batch = None,
        seq_sample_method = 'multinomial',
        no_recycles = 1,
    ):
        """Embedding of state[t].

        Args:
            v_t: orienation vectors at t; (N, L, 3).
            p_t: position vectors at t; (N, L, 3).
            s_t: sequence at time t; (N, L).
            beta: Beta_t; (N,).
            mask_res: mask with True for valid tokens (N, L).
            mask_gen: mask with True for target tokens (N, L).
            res_feat: None or (N, L, res_dim).
            pair_feat: None or (N, L, L, pair_dim).

        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        #######################################################################
        # Antibody design (from diffab)
        #######################################################################

        if self.gen_version == 'diffab_gen':
            # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
            res_feat = self.res_feat_mixer(
                    torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)
            ) # [Important] Incorporate sequence at the current step.

        #######################################################################
        # single-chain design
        #     The previous res_feat and pair_feat contain the original struc&seq 
        #     info and would have info leakage. The updated version only use the 
        #     sequence of the current step.
        #######################################################################

        elif self.gen_version == 'complete_gen':

            #################### backbone atom construction ###################
            ##  get the coor of the backbone atoms; (N, L, 4, 3)
            p_t_backbone = reconstruct_backbone(
                R = R,  
                t = p_t,  
                aa = s_t, 
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask = mask_res
            )

            ######################## Residue info #############################
            if self.residue_embed is not None:
                ### The residue embedding module is the same as the context 
                ### embedding module from diffab which requires both the 
                ### sequence and the structure.

                res_feat = self.residue_embed(
                    aa = s_t,
                    res_nb = batch['res_nb'], 
                    chain_nb = batch['chain_nb'], 
                    pos_atoms = p_t_backbone, 
                    mask_atoms = batch['mask_heavyatom'][:, :, :self.num_atoms],  # (N, L, 4)
                )

            else:
                res_feat = self.res_feat_encode(self.current_sequence_embedding(s_t))

            ################################ pair info ########################
            pair_feat = self.pair_embed(
                aa = s_t,
                res_nb = batch['res_nb'],
                chain_nb = batch['chain_nb'],
                pos_atoms = p_t_backbone,
                mask_atoms = batch['mask_heavyatom'][:, :, :self.num_atoms],
            )

        ########################### message passing ###########################
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        ########################### concatenation #############################
        t_embed = torch.stack(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )[:, None, :].expand(N, L, 3)  # (N, L, 3)
        in_feat = torch.cat([res_feat, t_embed], dim=-1)  # (N, L, res_dim+3)

        ########################### decoding ##################################

        ###### for energy prediction ######
        if self.decoder_version == 'CEP':
            if mask_res is not None:
                in_feat[~mask_res] = 0
            in_feat = torch.sum(in_feat, dim=1)  # (N, res_dim+3)
            in_feat = in_feat / torch.sum(mask_res, dim = -1).reshape(-1,1) 
            energy = self.energy_net(in_feat)    # (N, 1)

            return energy

        ###### direct predict the structure ######
        elif self.decoder_version == 'esmfold_decoder':
            ### sequence distribution
            c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)
            ### sequence sampling
            seq = aa_sampling(c_denoised, seq_sample_method) # (N, L) 
             
            ### structure decoding
            structure = self.eps_struc_net(
                seq_feats = in_feat, 
                pair_feats = pair_feat, 
                true_aa = seq, 
                residx = batch['res_nb'], 
                mask = mask_res, 
                no_recycles = no_recycles
            )  # (N, L, 4, 3)

            return structure, seq, c_denoised

        ###### seperate decoder (following diffab) ######
        else:
            ### Position changes
            if self.decoder_version == 'mlp':
                eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
            elif self.decoder_version == 'tab':
                eps_crd, _ = self.eps_crd_net(in_feat, pair_feat, mask_res.int()) # (N, L, 3)
            eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
            eps_pos = torch.where(mask_gen[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

            ### New orientation
            if self.decoder_version == 'mlp':
                eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
            elif self.decoder_version == 'tab':
                eps_rot, _ = self.eps_rot_net(in_feat, pair_feat, mask_res.int()) # (N, L, 3)
            U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
            R_next = R @ U
            v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
            v_next = torch.where(mask_gen[:, :, None].expand_as(v_next), v_next, v_t)

            ### New sequence categorical distributions
            c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)

            return v_next, R_next, eps_pos, c_denoised


##########################################################################################
# Diffusion Module
##########################################################################################

class FullDPM(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps,
        num_atoms = 4, 
        gen_version = 'complete_gen',
        residue_embed = None,
        pair_embed = None,   
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        Boltzmann_constant = 8.314462618e-3,
        seq_diff_version = 'multinomial',
        remember_padding = False,
        token_size = 21,
        reweighting_term = 0.001,
        ps_adapt_scale = 1.0,
        path_to_blosum = "../../Data/Origin/blosum62-special-MSA.mat",
        seq_model_opt={},
        modality='joint',
        train_version = 'noise',
        proteinMPNN_model = None,
        with_CEP_joint = False
    ):
        super().__init__()

        ########################### settings ##################################
        self.num_steps = num_steps
        self.token_size = token_size
        self.train_version = train_version

        ### for ablation study
        self.modality = modality
        if self.modality not in {'joint', 'sequence', 'structure'}:
            raise Exception('No modality version named %s!' % self.modality)

        ### for energy guidance
        self.Boltzmann_constant = Boltzmann_constant

        ### for sequence diffusion 
        self.seq_diff_version = seq_diff_version
        self.seq_diff_name = self.seq_diff_version.split('-')[0]
        self.remember_padding = remember_padding
        self.reweighting_term = reweighting_term # lambda reweighting term from Austin D3PM
        self.ps_adapt_scale = ps_adapt_scale

        ### for contrastive guidance
        self.with_CEP_joint = with_CEP_joint

        ########################### status encoder ############################

        self.eps_net = EpsilonNet(
             res_feat_dim, pair_feat_dim, num_atoms = num_atoms,
             gen_version = gen_version, residue_embed = residue_embed, pair_embed = pair_embed,
             **eps_net_opt
        )

        ########################### modules ###################################

        ###### rotation diffusion ######
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)

        ###### position diffusion ######
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)

        ###### sequence diffsuion  ######
        if self.seq_diff_version == 'multinomial':  # multinomial diffusion
            self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        elif self.seq_diff_name == 'autoregressive':  # autoregressive diffusion
            self.trans_seq = OAMaskCollaterSingleSeq(mask_id = token_size + 1, num_steps = num_steps)
            self.seq_loss_func = OAMaskedCrossEntropyLoss(reweight=True)

        elif self.seq_diff_name == 'discrete':  # discrete diffusion
            tokenizer = TokenizerSingleSeq(path_to_blosum = path_to_blosum)

            if self.seq_diff_version == 'discrete-random':  # uniform distribution
                Q_prod, Q_t = tokenizer.q_random_schedule(timesteps = num_steps)
            elif self.seq_diff_version == 'discrete-blosum':  # with blosum matrix
                Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps = num_steps)
            else:
                raise Exception('No sequence diffusion named %s!' % self.seq_diff_version)

            self.trans_seq = D3PMCollaterSingleSeq(token_size = token_size, Q=Q_t, Q_bar=Q_prod)
            self.seq_loss_func1 = D3PMLVBLoss(tmax = num_steps, tokenizer=tokenizer)
            self.seq_loss_func2 = D3PMCELoss(tokenizer=tokenizer)
            self.Q, self.Q_bar = Q_prod, Q_t

        else:
            raise Exception('No sequence diffusion named %s!' % self.seq_diff_version)

        ### sequence decoding module (for autoregressive and discrete diffusion)
        if self.seq_diff_version != 'multinomial':
            self.seq_model = ByteNetLMTime(
                n_tokens = token_size + 2, timesteps = num_steps, padding_idx = token_size + 1,
                 **seq_model_opt
            )

        ############################# consistency oracle #####################

        self.proteinMPNN_model = proteinMPNN_model
        if self.proteinMPNN_model is not None:
            for param in self.proteinMPNN_model.parameters():
                param.requires_grad = False

        ################################# for CEP #####################
        if self.with_CEP_joint:
            self.CEP_joint_pred = EpsilonNet(
                 res_feat_dim, pair_feat_dim, num_atoms = num_atoms,
                 gen_version = 'complete_gen', residue_embed = residue_embed, pair_embed = pair_embed,
                 decoder_version = 'CEP', num_layers = eps_net_opt['num_layers']
            )
        else:
            self.CEP_joint_pred = None

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
            posi_scale = torch.matmul(
                protein_size.float().reshape(-1, 1), posi_scale[:, 0].reshape(1, -1)
            ) + posi_scale[:, 1]   # (N, 3)
            posi_scale *= self.ps_adapt_scale
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
    # Losses
    ###########################################################################

    def basic_losses(self,
        R_pred, R_0, p_pred, p_ref, s_noisy, s_0, c_denoised, t, mask_res,
        posi_loss_version = 'mse', micro = True, 
        denoise_structure = True, denoise_sequence = True,
    ):
        """Basic losses for joint diffusion."""

        if micro:
            n_tokens = mask_res.sum().float()  # scalar
        else:
            n_tokens = mask_res.sum(dim = -1).float()  # (N,)

        if denoise_structure:
            #################### Rotation loss ####################################
            loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
            if micro:
                loss_rot = (loss_rot * mask_res).sum() / (n_tokens + 1e-8)
            else:
                loss_rot = (loss_rot * mask_res).sum(dim=1) / (n_tokens + 1e-8)
                loss_rot = loss_rot.mean()

            ######################### Position loss ###############################
            if posi_loss_version == 'mse':
                ### mse loss
                loss_pos = F.mse_loss(p_pred, p_ref, reduction='none').sum(dim=-1)  # (N, L)
                if micro:
                    loss_pos = (loss_pos * mask_res).sum() / (n_tokens  + 1e-8)
                else:
                    loss_pos = (loss_pos * mask_res).sum(dim=1) / (n_tokens + 1e-8)
                    loss_pos = loss_pos.mean()
            else:
                ### rmsd loss
                loss_pos = F.mse_loss(p_pred, p_ref, reduction='none').sum(dim=-1)  # (N, L)
                loss_pos = ((p_pred - p_ref) ** 2).sum(-1) # (N, L)
                if micro:
                    loss_pos = (loss_pos * mask_res).sum() / (n_tokens  + 1e-8)
                    loss_pos = loss_pos.sqrt()
                else:
                    loss_pos = (loss_pos * mask_res).sum(dim=1) / (n_tokens + 1e-8) # (N,)
                    loss_pos = loss_pos.sqrt()
                    loss_pos = loss_pos.mean()
 
        else:
            loss_rot = None
            loss_pos = None

        ####################### Sequence categorical loss #####################

        if denoise_sequence:   
            ###### multinomial diffusion & DDPM ######
            if self.seq_diff_version == 'multinomial' and self.train_version == 'noise':
                ### the posterior q(s_(t-1) | s_t, s_0) derived from the forward process
                post_true = self.trans_seq.posterior(s_noisy, s_0, t) 
                ### predicted posterior: p(s_(t-1) | R_t)
                post_pred = self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8
                log_post_pred = torch.log(post_pred)
                kldiv = F.kl_div(
                    input=log_post_pred,
                    target=post_true,
                    reduction='none',
                    log_target=False
                ).sum(dim=-1)    # (N, L)
                if micro:
                    loss_seq = (kldiv * mask_res).sum() / (n_tokens + 1e-8)
                else:
                    loss_seq = (kldiv * mask_res).sum(dim=1) / (n_tokens + 1e-8)
                    loss_seq = loss_seq.mean()

            ###### multinomial diffusion & self-conditioning ######
            elif self.seq_diff_version == 'multinomial' and self.train_version == 'gt':
                if micro:
                    loss_seq = F.cross_entropy(
                        c_denoised[mask_res == 1], s_0[mask_res == 1]
                    )
                else:
                    loss_seq = F.cross_entropy(
                        c_denoised.transpose(1, 2), s_0 * mask_res, reduction = 'none'
                    )
                    loss_seq = (loss_seq * mask_res).sum(dim=1) / (n_tokens + 1e-8)
                    loss_seq = loss_seq.mean()

            ###### autoregressive diffusion ######
            elif self.seq_diff_name == 'autoregressive':  # autoregressive diffusion
                loss_Seq, nll_loss = self.seq_loss_func(
                    seq_pred, s_0, forward_masks, num_mask, mask_res.float()
                )  # sum(loss per token)

            ###### discrete diffusion ######
            else: 
                lvb_loss = self.seq_loss_func1(
                    src_onehot[:,:,:self.token_size], q_x, seq_pred, s_0, 
                    tgt_onehot[:,:,:self.token_size], mask_res.float(), t, 
                    self.Q.to(s_0.device), self.Q_bar.to(s_0.device)
                ).to(torch.float32)
                ce_loss = self.seq_loss_func2(seq_pred, s_0, mask_res).to(torch.float32)
                loss_seq = (lvb_loss + (self.reweighting_term * ce_loss)) * n_tokens

        else:
            loss_seq = None

        return loss_rot, loss_pos, loss_seq


    def unnormalize_for_loss(self, coor_1, coor_2, protein_size, mask_res):
        """Utility unnormalization function for loss call. """
        if protein_size is None and mask_res is not None:
            protein_size = mask_res.sum(dim = -1)
        coor_out_1 = self._unnormalize_position(coor_1, protein_size = protein_size)
        coor_out_2 = self._unnormalize_position(coor_2, protein_size = protein_size)
        return coor_out_1, coor_out_2, protein_size


    def loss_cal(self,
        R_pred, R_ref, p_pred, p_ref, s_noisy, c_denoised, s_ref, 
        t, mask_res, mask_gen, p_noisy = None, protein_size = None, 
        posi_loss_version = 'mse', micro = True, unnorm_first = False,
        denoise_structure = True, denoise_sequence = True,
        with_dist_loss = False, dist_clamp = 20., loss_version = 'mse',
        with_clash = False, threshold_clash = 3.6,
        with_gap = False, threshold_gap = 3.9, 
        with_consist_loss = False,
        consist_target = 'distribution',
        cross_loss = False,
        with_CEP_loss = False,
        with_energy_loss = True, with_fitness_loss = True,
        energy_guide = None,
        energy_guide_type = 'cosine',
        struc_scale = 'Boltzmann',
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
        """Loss calculation."""

        loss_dict = {}

        ########### basic losses ####################

        if self.train_version == 'gt' and unnorm_first:
            p_pred, p_ref, protein_size = self.unnormalize_for_loss(
                p_pred, p_ref, protein_size, mask_res
            )
            coor_pred, coor_ref = p_pred, p_ref
        else:
            coor_pred, coor_ref = None, None
         
        if self.train_version != 'gt':
            posi_loss_version = 'mse'

        loss_rot, loss_pos, loss_seq = self.basic_losses(
            R_pred = R_pred, R_0 = R_ref, 
            p_pred = p_pred, p_ref = p_ref, 
            s_noisy = s_noisy, s_0 = s_ref, c_denoised = c_denoised, 
            t = t, mask_res = mask_gen, 
            micro = micro, posi_loss_version = posi_loss_version,
            denoise_structure = denoise_structure, 
            denoise_sequence = denoise_sequence
        )
        if denoise_structure:
            loss_dict['rot'] = loss_rot
            loss_dict['pos'] = loss_pos
        if denoise_sequence:
            loss_dict['seq'] = loss_seq

        ########## distance related losses ##########

        if self.train_version == 'gt' and denoise_structure \
        and (with_dist_loss or with_clash or with_gap):
            if coor_pred is None:
                coor_pred, coor_ref, protein_size = self.unnormalize_for_loss(
                    p_pred, p_ref, protein_size, mask_res
                )
            loss_dist, loss_clash, loss_gap = distance_loss(
                coor_pred = coor_pred, ref = coor_ref, mask_res = mask_gen,
                with_dist = False, dist_clamp = dist_clamp, loss_version = loss_version,
                with_clash = with_clash, threshold_clash = threshold_clash,
                with_gap = with_gap, threshold_gap = threshold_gap,
            )
            if with_dist_loss:
                loss_dict['dist(%s)' % loss_version] = loss_dist
            if with_clash:
                loss_dict['clash'] = loss_clash
            if with_gap:
                loss_dict['gap'] = loss_gap
 
        ########## oracle losses ####################

        if self.train_version == 'gt' and with_consist_loss \
        and denoise_structure and denoise_sequence and self.proteinMPNN_model is not None:
            if coor_pred is None:
                coor_pred, coor_ref, protein_size = self.unnormalize_for_loss(
                    p_pred, p_ref, protein_size, mask_res
                )
            loss_dict['consist'] = consistency_loss(
                coor_pred = coor_pred, c_denoised = c_denoised, 
                coor_gt = coor_ref, s_gt = s_ref,
                mask_res = mask_gen, proteinMPNN_model = self.proteinMPNN_model,        
                micro = micro, consist_target = consist_target,
                cross_loss = cross_loss,
            )

        ############# CEP lossses ###################

        if with_CEP_loss and self.CEP_joint_pred is not None \
        and denoise_structure and denoise_sequence:
            energy_scores = self.CEP_joint_pred(
                v_noisy, p_noisy, s_noisy, beta, mask_gen,
                res_feat = res_feat, pair_feat = pair_feat, batch = batch
            ).reshape(-1)  # (N, ) 
            energy_scores = torch.exp(-energy_scores) # (N, )
            denorm = torch.exp(-energy_scores).sum()
            loss_CEP = - (1 / N) * torch.log(energy_scores / denorm).sum()
            loss_dict['CEP'] = loss_CEP

        elif with_CEP_loss:
            print('Warning! CEP network is not defined!')

        ########## energy lossses ###################

        with_energy_loss = with_energy_loss and denoise_structure and self.train_version != 'gt'
        with_fitness_loss = with_fitness_loss and denoise_sequence and self.train_version != 'gt'

        if with_energy_loss or with_fitness_loss:
            loss_energy, loss_fitness = energy_guided_loss(
                p_noisy, p_pred, s_0, mask_gen,
                with_energy_loss = with_energy_loss, 
                with_fitness_loss = with_fitness_loss,
                energy_guide = energy_guide,
                energy_guide_type = energy_guide_type,
                struc_scale = struc_scale,
                Boltzmann_constant = self.Boltzmann_constant,
                temperature = temperature,
                energy_aggre = energy_aggre,
                RepulsionOnly = RepulsionOnly,
                with_resi = with_resi,
                multithread = multithread,
                with_contact = with_contact,
                contact_path_list = contact_path_list,
                atom_list = atom_list,
                contact_thre = contact_thre,
                fitness_guide = fitness_guide,
                fitness_guide_type = fitness_guide_type,
                seq_scale = seq_scale,
                t_max = t_max, 
                force_vs_diff = force_vs_diff,
            )
            if with_energy_loss:
                loss_dict['energy'] = loss_energy
            if with_fitness_loss:
                loss_dict['fitness'] = loss_fitness

        return loss_dict


    ###########################################################################
    # forward function (get the loss)
    ###########################################################################

    def forward(self, 
        v_0, 
        p_0, 
        s_0, 
        res_feat, 
        pair_feat, 
        mask_res,
        mask_gen,
        protein_size = None, 
        denoise_structure = True, 
        denoise_sequence = True, 
        t = None, 
        batch = None,
        ###### for losses ######
        micro = True, posi_loss_version = 'mse', unnorm_first = False,
        ### distance loss
        with_dist_loss = False, dist_clamp = 20., loss_version = 'mse',
        ### clash loss
        with_clash = False, threshold_clash = 3.6,
        ### gap loss
        with_gap = False, threshold_gap = 3.9,
        ### consist loss
        with_consist_loss=False,
        consist_target = 'distribution', cross_loss = False,
        ### CEP loss
        with_CEP_loss = False,
        ### energy loss
        with_energy_loss = False,
        with_fitness_loss = False,
        energy_guide = None, 
        energy_guide_type = 'cosine',
        struc_scale = 'Boltzmann', 
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
        seq_sample = 'multinomial',
        t_max = None,
        force_vs_diff = False,
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
            mask_gen: True for target tokens; (N, L)
            protein_size: True for valid tokens other than paddings; (N, L)
            denoise_structure: whether do the structure diffusion; bool
            denoise_sequence: whether do the sequence diffusion; bool
            t: None (than will do the random sampling) or (N, )

            ### energy guidance
            energy_guite: energy gradient (force) predictor
            energy_guide_type: loss type for the energy guidance; 'cosine' or 'mse'
            struc_scale: "Boltzmann", "none" or float; 
                         "Boltzmann" = 1 / (K_B, T)
                         "none" = 1
            temperature: temperature for scaling
            energy_aggre: how to deal with the energy guidance; 'all' or 'schedule' or energy_type
            multithread: whether do the calculation with multithread
            # for contact
            with_contact: list of str or bool; list of str for predefined contact path, bool for whether include the contact energy

            ### fitness guidance 
            fitness_guide: fitness score gradient predictor
            fitness_guide_type: loss type for the fitness guidance; 'cosine' or 'mse'
            seq_scale: "length", "none" or float; "length" = length; for other options the same as 'struc_scale'
            seq_sample: how to sample the sequence
        """

        N, L = v_0.shape[:2]
        denoise_structure = denoise_structure and (self.modality in {'joint', 'structure'})
        denoise_sequence = denoise_sequence and (self.modality in {'joint', 'sequence'})
        if protein_size is None:
            protein_size = mask_res.sum(dim=1)  # (N,) 

        #############################################
        # data preprocess 
        #############################################

        ### step 
        if t is None:  # t: None or (N,)
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)

        if self.modality == 'sequence':
            ### only sequence is needed
            v_0 = torch.zeros(v_0.shape, device = v_0.device)
            p_0 = torch.zeros(p_0.shape, device = p_0.device)
        elif self.modality == 'structure':
            ### only structure is needed 
            s_0 = torch.zeros(s_0.shape, device = s_0.device).long()

        ### position normalization
        p_0 = self._normalize_position(p_0, protein_size = protein_size)
        ### transform orientation vector to SO(3)
        R_0 = so3vec_to_rotation(v_0)  # (N, L, 3, 3)

        #############################################
        # forward (add noise, 0 to t) 
        #############################################

        ############## for structure ################

        ###### add noise to structure ######
        if denoise_structure:
            ### Add noise to rotation 
            # v_noisy: noised orientation vector; (N, L, 3)
            v_noisy, _ = self.trans_rot.add_noise(v_0, t, mask_generate = mask_gen)  
            
            ### Add noise to positions 
            # p_noisy: noised position, (N, L, 3); ~ N(sqrt(alpha_bar[t]) * p_0, (1 - alpha_bar[t]))
            # eps_p: Gaussian noise, (N, L, 3)
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, t, mask_generate = mask_gen)

        ###### fix the structure ######
        else:
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        ################# for sequence ##############

        ###### multinomial diffusion ######
        if denoise_sequence and self.seq_diff_version == 'multinomial':
            ### Add noise to sequence (multinomial diffusion)
            # s_noise_prob: noised sequence distribution; (N, L, K = 20)
            #               s_noise_prob = alpha_bar * onehot(s_0) + ((1 - alpha_bar) / K); 
            # s_noisy: noised sequence, s_noisy = sample(s_noise_prob); (N, L)
            s_noise_prob, s_noisy = self.trans_seq.add_noise(
                s_0, t, method = seq_sample, mask_generate = mask_gen
            )

        ###### autoregressive diffusion ######
        elif denoise_sequence and self.seq_diff_name == 'autoregressive':
            s_noisy, num_mask, forward_masks = self.trans_seq(s_0, t, mask_res.int())
            
        ###### discrete diffusion ######
        elif denoise_sequence:
            s_noisy, src_onehot, tgt_onehot, q_x = self.trans_seq(s_0, t, mask_res.int())

        ###### fix the sequence ######
        else:
            s_noisy = s_0.clone()

        #############################################
        # reverse (denoise, t to t-1)
        #############################################

        ################# sequence diffusion ###################################
        if denoise_sequence and self.seq_diff_version != 'multinomial':
            # input_mask: True for valid and False for padding, (N, L, 1)
            seq_pred = self.seq_model(s_noisy, t, input_mask=mask_res.unsqueeze(-1)) # (N, L, token_size + 2~padding&mask)

            ###### autoregressive diffusion ######
            # the original noised sequence contain the unkown masked token
            if self.seq_diff_version == 'autoregressive':
                s_noisy = torch.multinomial(
                    torch.nn.functional.softmax(seq_pred.reshape(-1, seq_pred.shape[-1])[:, :self.token_size - 1], dim = -1),
                    num_samples=1
                ).reshape(N, L)
                s_noisy[~mask_res] = s_0[~mask_res]

            elif self.seq_diff_version == 'autoregressive-maskonly':
                s_noisy = torch.multinomial(
                    torch.nn.functional.softmax(seq_pred.reshape(-1, seq_pred.shape[-1])[:, :self.token_size - 1], dim = -1),
                    num_samples=1
                ).reshape(N, L)
                s_noisy[~forward_masks] = s_0[~forward_masks]

            elif self.seq_diff_version == 'autoregressive-random': 
                ### rather than sample from the predicted dist, sample from random
                s_noisy = torch.randint(0, self.token_size - 1, (N, L)).to(s_noisy.device)
                s_noisy[~forward_masks] = s_0[~forward_masks]

        elif denoise_sequence and self.remember_padding:
            s_noisy[~mask_res] = s_0[~mask_res]

        ############################## joint diffusion #########################
        beta = self.trans_pos.var_sched.betas[t]  # (N,)
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, beta, mask_res, mask_gen, 
            res_feat = res_feat, pair_feat = pair_feat, batch = batch
        )   
        # v_pred: dire_(t-1); (N, L, 3)
        # R_pred: O_(t-1); (N, L, 3, 3)
        # eps_p_pred: G(R_t); (N, L, 3);
        #     mu_(t-1) = (1/sqrt(alpha_t)) * 
        #         (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * G(R_t))
        # c_denoised: F(R_t), predicted multinomial distribution; (N, L, K); s_(t-1) ~ F(R_t)

        #############################################
        # Loss Calculation
        #############################################

        if self.train_version == 'gt':
            p_ref = p_0
        else:
            p_ref = eps_p
 
        loss_dict = self.loss_cal(
            R_pred = R_pred, R_ref = R_0, 
            p_pred = eps_p_pred, p_ref = p_ref, 
            s_noisy = s_noisy, c_denoised = c_denoised, s_ref = s_0,
            t = t, mask_res = mask_res, mask_gen = mask_gen, micro = micro, 
            posi_loss_version = posi_loss_version, unnorm_first = unnorm_first,
            denoise_structure = denoise_structure, 
            denoise_sequence = denoise_sequence,
            with_dist_loss = with_dist_loss, 
            dist_clamp = dist_clamp, 
            loss_version = loss_version,
            with_clash = with_clash, threshold_clash = threshold_clash,
            with_gap = with_gap, threshold_gap = threshold_gap,
            with_consist_loss = with_consist_loss,
            consist_target = consist_target,
            with_CEP_loss = with_CEP_loss,
            with_energy_loss = with_energy_loss,
            with_fitness_loss = with_fitness_loss,
            energy_guide = energy_guide,
            energy_guide_type = energy_guide_type,
            struc_scale = struc_scale,
            temperature = temperature,
            energy_aggre = energy_aggre,
            RepulsionOnly = RepulsionOnly,
            with_resi = with_resi,
            multithread = multithread,
            with_contact = with_contact,
            contact_path_list = contact_path_list,
            atom_list = atom_list,
            contact_thre = contact_thre,
            fitness_guide = fitness_guide,
            fitness_guide_type = fitness_guide_type,
            seq_scale = seq_scale,
            t_max = t_max,
            force_vs_diff = force_vs_diff,
        )

        return loss_dict


    ###########################################################################
    # sample with context info (from diffab)
    ###########################################################################

    @torch.no_grad()
    def sample(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_res,
        protein_size = None,
        sample_structure=True, sample_sequence=True,
        pbar=False,
        batch = None 
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        if protein_size is None:
            protein_size = mask_res.sum(dim=1)  # (N,)
        p = self._normalize_position(p, protein_size = protein_size)

        ########################### initialization ####################################

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_init = random_uniform_so3([N, L], device=self._dummy.device)
            p_init = torch.randn_like(p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            #s_init = torch.randint_like(s, low=0, high=19)
            s_init = torch.randint_like(s, low=0, high=20) # fixed a bug from diffab; 01/31/24
            if self.remember_padding:
                s_noisy[~mask_res] = s[~mask_res]
        else:
            s_init = s

        ########################### denoising steps ####################################

        ### container
        traj = {self.num_steps: (
            v_init, self._unnormalize_position(p_init, protein_size = protein_size), s_init)
        }
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        for t in pbar(range(self.num_steps, 0, -1)): # from T to 1
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t, protein_size = protein_size)
           
            ###### beta coefficient calculation ######
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])  # (N,)
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            ###### noise sampling ######
            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, beta, mask_res,
                res_feat = res_feat, pair_feat = pair_feat, batch = batch
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            ###### denoising ######
            v_next = self.trans_rot.denoise(v_t, v_next, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, t_tensor)

            if self.remember_padding:
                s_next[~mask_res] = s[~mask_res]

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next, protein_size = protein_size), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    ###########################################################################
    # single-chain sampling from scratch
    ###########################################################################

    @torch.no_grad()
    def sample_complete(
        self,
        mask_res, res_nb, chain_nb, mask_heavyatom,
        protein_size = None,
        res_feat = None, pair_feat = None,
        pbar=False, 
        with_wiener = True, 
        self_condition = True,
        save_pred = False,
        seq_sample = 'multinomial',
        t_bias = -1
    ):
        """
        Sampling from scratch. (by SZ)

        Args:
            res_feat:  (N, L_max); True for valid tokens and False for the others.
            res_nb: (N, L_max); 1, 2, ..., L for valid tokens and 0 for paddings 
            chain_nb: (N, L_max); 1 for valid tokens and 0 for paddings
            mask_heavyatom: (N, L_max, atom_num (>=4))
        """
        N, L = mask_res.shape
        if protein_size is None:
            protein_size = mask_res.sum(dim=1)  # (N,)

        #######################################################################
        # Initialization 
        #######################################################################

        v_init = random_uniform_so3([N, L], device=mask_res.device)  
        p_init = torch.randn(N, L, 3).to(mask_res.device)  # (N, L, 3), random CA coordinates
        #s_init = torch.randint(size = (N,L), low=0, high=19).to(mask_res.device) # (N, L)
        s_init = torch.randint(size = (N,L), low=0, high=20).to(mask_res.device) # (N, L), fix the bug
        if self.remember_padding:
            s_init[~mask_res] = self.token_size

        batch = {'res_nb': res_nb,
                 'chain_nb': chain_nb,
                 'mask_heavyatom': mask_heavyatom}

        ### container
        traj = {self.num_steps: (
            v_init, self._unnormalize_position(p_init, protein_size = protein_size), s_init
        )}

        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        ######################################################################## 
        # denoising steps 
        ########################################################################

        for t in pbar(range(self.num_steps, 0, -1)): # from T to 0
            if save_pred and t < self.num_steps:
                v_t, p_t, s_t, _ = traj[t]
            else:
                v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t, protein_size = protein_size)

            ###### beta coefficient calculation ######
            beta = self.trans_pos.var_sched.betas[t + t_bias].expand([N, ])
            t_tensor = torch.full(
                [N, ], fill_value=t + t_bias, 
                dtype=torch.long, device=mask_res.device
            )

            ###### noise sampling ######
            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, beta, mask_res, mask_gen = mask_res,
                res_feat = res_feat, pair_feat = pair_feat, batch = batch
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            ###### denoising ######

            ### orientation
            if with_wiener:
                v_next = self.trans_rot.denoise(v_t, v_next, t_tensor)

            ### CA position
            if self.train_version == 'gt' and t > 1:
                ## eps_p is the predicted ground truth, get x_(t-1)

                if self_condition: 
                    p_next, _ = self.trans_pos.add_noise(eps_p, t_tensor - 1) 
                else:
                    eps_p = self.gt_noise_transfer(p_t, eps_p, t_tensor)
                    p_next = self.trans_pos.denoise(
                        p_t, eps_p, t_tensor, with_wiener = with_wiener
                    )

            elif self.train_version == 'gt':
                ## eps_p is already x_0
                p_next = eps_p
            else:
                ## eps_p is the predicted noise
                p_next = self.trans_pos.denoise(p_t, eps_p, t_tensor, with_wiener = with_wiener)

            ### sequence
            if self.train_version == 'gt':
                s_next = aa_sampling(c_denoised, seq_sample_method = seq_sample)
                if t > 1:
                    _, s_next = self.trans_seq.add_noise(s_next, t_tensor - 1, method = seq_sample)

            else:
                _, s_next = self.trans_seq.denoise(s_t, c_denoised, t_tensor, seq_sample)

            if self.remember_padding:
                s_next[~mask_res] = self.token_size

            ###### record current step ######

            if save_pred:
                traj[t-1] = (
                    v_next, self._unnormalize_position(p_next, protein_size = protein_size), s_next, eps_p
                )
            else:
                traj[t-1] = (
                    v_next, self._unnormalize_position(p_next, protein_size = protein_size), s_next
                )
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.\

        # more the last states (0) to cpu memory
        traj[t-1] = tuple(x.cpu() for x in traj[t-1])

        return traj


    def backbone_gen(self, 
        mask_res, protein_size = None, res_feat = None, pair_feat = None, 
        pbar=False, with_wiener=True, self_condition = True, t_bias = -1,
    ):
        """
        Generate the backbone structure given the size. (by SZ)

        Args:
            res_feat:  (N, L_max); True for valid tokens and False for the others.
        Return:
            dict: t: {coor: (L, 4, 3); seq: str} x N
        """
        N, L = mask_res.shape
        if protein_size is None:
            protein_size = mask_res.sum(dim=1)  # (N,)
        res_nb = torch.zeros(N,L).int().to(mask_res.device)
        res_nb[:] = torch.arange(1, L+1).to(mask_res.device)
        res_nb = (res_nb * mask_res).int() # residue idx to determine consecutive residues; int, ordinal, e.g. 1,2,...,L, 0, ..., 0; (N, L_max)
        chain_nb = mask_res.int() # chain idx; int; (N, L); 1 for valid token and 0 for Padding
        mask_heavyatom = mask_res[:,:,None].repeat(1,1,4) # (N, L_max, atom_num (>=4))
        lengths = mask_res.sum(-1).cpu()  # (N,)

        traj = self.sample_complete(
           mask_res = mask_res, res_nb = res_nb, 
           chain_nb = chain_nb, mask_heavyatom = mask_heavyatom,
           protein_size = protein_size,
           res_feat = res_feat, pair_feat = pair_feat,
           pbar = pbar, with_wiener = with_wiener, 
           self_condition = self_condition, t_bias = t_bias
        )

        out_dict = {}
        for t in traj.keys():
            out_dict[t] = []
            R = so3vec_to_rotation(traj[t][0])
            bb_coor_batch = reconstruct_backbone(R, traj[t][1], traj[t][2], chain_nb.cpu(), res_nb.cpu(), mask_res.cpu())  # (N, L_max, 4, 3)
            for i, bb_coor in enumerate(bb_coor_batch):
                seq = seq_recover(traj[t][2][i], length = lengths[i]) 
                out_dict[t].append({'coor': bb_coor[:lengths[i]], 'seq': seq})
        return out_dict, traj

    ###########################################################################
    # likelihood esitimation (WIP)
    ###########################################################################

    @torch.no_grad()
    def neg_loglikelihood(self, v_0, p_0, s_0, mask_res, protein_size = None):
        """
        Calculate the negative loglikelihood given the samples. (by SZ)
        Args:
            v_0: orientation vector, (N, L, 3)
            p_0: CA coordinates, (N, L, 3)
            s_0: aa sequence, (N, L) 
            mask_res: True for valid tokens other than paddings; (N, L)
        """
        N, L = mask_res.shape
        if protein_size is None:
            protein_size = mask_res.sum(dim=1)  # (N,)

        ########################### coordinates ######################################


        ########################### initialization ####################################

        v_init = random_uniform_so3([N, L], device=mask_res.device)
        p_init = torch.randn(N, L, 3).to(mask_res.device)
        s_init = torch.randint(size = (N,L), low=0, high=19).to(mask_res.device)

        batch = {'res_nb': res_nb,
                 'chain_nb': chain_nb,
                 'mask_heavyatom': mask_heavyatom}

        ########################### denoising steps ####################################

        ### container
        traj = {self.num_steps: (
            v_init, self._unnormalize_position(p_init, protein_size = protein_size), s_init
        )}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        for t in pbar(range(self.num_steps, 0, -1)): # from T to 0
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t, protein_size = protein_size)

            ###### beta coefficient calculation ######
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=mask_res.device)

            ###### noise sampling ######
            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, beta, mask_res,
                res_feat = res_feat, pair_feat = pair_feat, batch = batch
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            ###### denoising ######
            v_next = self.trans_rot.denoise(v_t, v_next, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, t_tensor)

            traj[t-1] = (
                v_next, self._unnormalize_position(p_next, protein_size = protein_size), s_next
            )
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    ###########################################################################
    # status optimization (from diffab)
    ###########################################################################

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        res_feat, pair_feat, 
        mask_res,
        protein_size = None,
        sample_structure=True, sample_sequence=True,
        pbar=False,
        batch = None
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        if protein_size is None:
            protein_size = mask_res.sum(dim=1)  # (N,)
        p = self._normalize_position(p, protein_size = protein_size)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_init, _ = self.trans_rot.add_noise(v, t)  # v_init: (N, L, 3)
            # Add noise to positions
            p_init, _ = self.trans_pos.add_noise(p, t)  # p_init: (N, L, 3)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, t)
        else:
            s_init = s

        traj = {opt_step: (
            v_init, self._unnormalize_position(p_init, protein_size = protein_size), s_init
        )}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t, protein_size = protein_size)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, beta, mask_res,
                res_feat = res_feat, pair_feat = pair_feat, batch = batch
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (
                v_next, self._unnormalize_position(p_next, protein_size = protein_size), s_next
            )
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
