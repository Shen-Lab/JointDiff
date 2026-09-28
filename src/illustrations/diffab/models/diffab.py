import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.diffusion.dpm_full import FullDPM
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model

###### for LLMs (by SZ) ######
import esm
from diffab.utils.protein.constants import ressymb_order
from torch.nn.parameter import Parameter


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


@register_model('diffab')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        ################# by SZ #########################
        if not cfg.__contains__('chain_feat_version'):
            cfg.chain_feat_version = 'same'

        ###### training version ######
        if not 'train_version' in cfg.keys():
            cfg.train_version = 'noise'

        ###### LLM ######
        if cfg.__contains__('with_LLM') and cfg.with_LLM:
            LLM_encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D() 
            ### freeze the parameters
            for param in LLM_encoder.parameters():
                param.requires_grad = False

            trans_mat = []
            for char in ressymb_order:
                trans_mat.append(alphabet.get_idx(char))
            trans_mat = Parameter(data = torch.tensor(trans_mat), requires_grad = False)

            print('LLM ESM2-650M loaded.')

        else:
            LLM_encoder = None
            trans_mat = None

        if not cfg.__contains__('with_rmsd'):
            cfg.with_rmsd = False

        if not cfg.__contains__('with_dist'):
            cfg.with_dist = False

        if not cfg.__contains__('dist_clamp'):
            cfg.dist_clamp = None

        if not cfg.__contains__('with_center_loss'):
            cfg.with_center_loss = False

        if not cfg.__contains__('with_clash_loss'):
            cfg.with_clash_loss = None

        if not cfg.__contains__('with_af3_relpos'):
            cfg.with_af3_relpos = False

        if not cfg.__contains__('cross_attn'):
            cfg.cross_attn = False

        #################################################

        #num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        num_atoms = 4 # only for backbone atoms
        self.residue_embed = ResidueEmbedding(
            cfg.res_feat_dim, num_atoms,
            LLM_encoder = LLM_encoder, trans_mat = trans_mat, LLM_dim = 1280
        )
        self.pair_embed = PairEmbedding(
            cfg.pair_feat_dim, num_atoms, 
            chain_feat_version = cfg.chain_feat_version,
            with_af3_relpos = cfg.with_af3_relpos,
        )

        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            LLM_encoder = LLM_encoder, 
            trans_mat = trans_mat, 
            LLM_dim = 1280,
            with_dist = cfg.with_dist,
            with_rmsd = cfg.with_rmsd,
            dist_clamp = cfg.dist_clamp,
            with_center_loss = cfg.with_center_loss,
            with_clash_loss = cfg.with_clash_loss,
            train_version = cfg.train_version,
            cross_attn = cfg.cross_attn,
            **cfg.diffusion,
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generate_flag']     # Context means ``not generated''
        )  ## True for framework, False for CDR

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['resi_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )  # "chain_nb" is used for dihedral angles

        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['resi_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
            fragment_type = batch['fragment_type'], # by SZ
        )  # "chain_nb" is used for features

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p
    
    def forward(self, batch):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        #print('Diffusion', v_0.shape, p_0.shape, s_0.shape, mask_generate.sum(), mask_res.sum())
        loss_dict = self.diffusion(
            v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
            chain_nb = batch['chain_nb'],
        )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.sample(v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, **sample_opt)
        return traj

    @torch.no_grad()
    def optimize(
        self, 
        batch, 
        opt_step, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = optimize_opt.get('sample_structure', True),
            remove_sequence = optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_generate, mask_res, **optimize_opt)
        return traj
