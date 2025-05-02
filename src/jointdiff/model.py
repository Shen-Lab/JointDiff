import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
from typing import Dict

from jointdiff.modules.common.geometry import construct_3d_basis, reconstruct_backbone
from jointdiff.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec
from jointdiff.modules.encoders.ga import GAEncoder
from jointdiff.modules.encoders.residue import ResidueEmbedding
from jointdiff.modules.encoders.pair import PairEmbedding
from jointdiff.modules.diffusion.dpm_full import FullDPM, seq_recover
from jointdiff.modules.data.constants import ressymb_order, max_num_heavyatoms, BBHeavyAtom
# diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'

resolution_to_num_atoms = {
    'backbone+CB': 5,  # N, CA, C, O, CB
    'backbone': 4,     # for single chain; N, CA, C, O
    'full': max_num_heavyatoms   # 15; N, CA, C, O, CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2, OXT
}


################################################################################
# Diffusion Model
################################################################################

default_hyperparameters = {
    'train_version': 'jointdiff',   # 'jointdiff' or 'jointdiff-x'
    'modality': 'joint', # 'joint', 'structure', 'sequence', 'stru_pred', 'seq_pred'
    'embed_first': False,
    'with_type_emb': False,
    'max_relpos': 32,
    'with_distogram': False,
    'encode_share': True,
    'seq_diff_version': 'multinomial',
    'all_bb_atom': False,
}

available_versions = { 
    'train_version': {'jointdiff', 'jointdiff-x'},
    'modality': {'joint', 'structure', 'sequence', 'stru_pred', 'seq_pred'},
    'seq_diff_version': {'multinomial', 'ddpm'}
}

class DiffusionSingleChainDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        ########################################################################
        # Model Settings
        ########################################################################

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'backbone')]  
        # 15 (if cfg contains "resolution" and it is "full")
        # 5 (if cfg contains "resolution" and it is "backbone+CB")
        # 4 (other wise; will be "backbone")

        for key in default_hyperparameters:
            if key not in cfg.keys():
                cfg[key] = default_hyperparameters[key]
                
        for key in available_versions:
            val = cfg[key] 
            if val not in available_versions[key]:
                raise NameError('No %s version named %s!' % (key, val))

        self.cfg = cfg
        self.train_version = cfg.train_version
        print('Model: %s' % self.train_version)
        self.modality = cfg.modality
        self.embed_first = cfg.embed_first
        self.seq_diff_version = cfg.seq_diff_version
        self.with_distogram = cfg.with_distogram
        self.encode_share = cfg.encode_share
        self.max_relpos = cfg.max_relpos
        self.all_bb_atom = cfg.all_bb_atom

        self.with_sequence = False if self.modality == 'structure' else True
        self.with_structure = False if self.modality == 'sequence' else True
       
        model_name = self.train_version
        print('Model: %s' % self.train_version)
        print('Modality: %s' % self.modality)
        print('%s for sequence diffusion.' % self.seq_diff_version)
        if self.with_distogram:
            print('With distogram prediction.')

        ########################################################################
        # model architecture
        ########################################################################

        ############################## feature embedding #######################
        if self.embed_first:
            ### only random mask version require the context encoder
            self.residue_embed = ResidueEmbedding(
                cfg.res_feat_dim, num_atoms, with_type_emb = cfg.with_type_emb,
                with_sequence = self.with_sequence, with_structure = self.with_structure,
                with_seq_ddpm = (self.seq_diff_version == 'ddpm'),
            )
            self.pair_embed = PairEmbedding(
                cfg.pair_feat_dim, num_atoms, 
                with_sequence = self.with_sequence, with_structure = self.with_structure,
                with_seq_ddpm = (self.seq_diff_version == 'ddpm'), seq_ddpm_dim = cfg.res_feat_dim,
                max_relpos = self.max_relpos
            )
        else:
            self.residue_embed = None
            self.pair_embed = None

        ############################## main model ##############################
        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            num_atoms = num_atoms, 
            residue_embed = self.residue_embed if self.encode_share else 'same',
            pair_embed = self.pair_embed if self.encode_share else None,
            train_version = self.train_version,
            max_relpos = self.max_relpos,
            with_distogram = self.with_distogram,
            with_type_emb = cfg.with_type_emb,
            emb_first = cfg.embed_first,
            all_bb_atom = cfg.all_bb_atom,
            **cfg.diffusion,
        )

    ############################################################################
    # context encoding 
    ############################################################################

    def encode(self, batch):
        """
        Returns:
            res_feat: node-wise feature, (N, L, res_feat_dim) or None.
            pair_feat: pair-wise feature, (N, L, L, pair_feat_dim) or None.
            R: rotation matrix, (N, L, 3, 3) 
            p: CA position, (N, L, 3)
        """
        #################### context embedding #################################
        if self.embed_first:

            structure_mask = ((~batch['mask_gen']) * batch['mask']).bool()
            sequence_mask = ((~batch['mask_gen']) * batch['mask']).bool()

            ### residue features, (N, L, feat_dim)
            res_feat = self.residue_embed(
                aa = batch['aa'],  # amino acid index; int, 0~19, 21 for padding; (N, L)
                res_nb = batch['res_nb'],  # residue idx to determine consecutive residues; (N, L)
                chain_nb = batch['chain_nb'],  # chain idx; (N, L)
                pos_atoms = batch['pos_heavyatom'],  # heavy atom coordinates; (N, L, atom_num, 3)
                mask_atoms = batch['mask_heavyatom'],  # heavy atom mask; (N, L, atom_num )
                fragment_type = batch['fragment_type'],  # fragment indicating the type (motif or scaffold)
                structure_mask = structure_mask,
                sequence_mask = sequence_mask,
            )  

            ### edge features, (N, L, L, feat_dim)
            pair_feat = self.pair_embed(
                aa = batch['aa'],
                res_nb = batch['res_nb'],
                chain_nb = batch['chain_nb'],
                pos_atoms = batch['pos_heavyatom'],
                mask_atoms = batch['mask_heavyatom'],
                structure_mask = structure_mask,
                sequence_mask = sequence_mask,
            ) 

        ################## with the single-chain pipeline #####################
        # No res_feat or pair feat of the context are needed.
        else:
            res_feat = None
            pair_feat = None

        ############################ ground truth prepare ##################### 
        if self.all_bb_atom:
            p = batch['pos_heavyatom'][:, :, :4]  # (N, L, 4, 3)
            v = None
        else:
            p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA] # (N, L, 3)
            R = construct_3d_basis(
                batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
                batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
                batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
            )  # (N, L, 3, 3)
            v = rotation_to_so3vec(R)  # transform SO(3) to 3d vectors (N, L, 3)

        return res_feat, pair_feat, v, p
   
    ############################################################################
    # forward function 
    ############################################################################

    def forward(self, 
        batch,
        ###### for losses ######
        micro = True, posi_loss_version = 'mse', unnorm_first = False,
        ### distance loss 
        with_dist_loss = False, dist_loss_version = 'mse', threshold_dist = 15.0, dist_clamp = 20., 
        ### clash loss
        with_clash = False, threshold_clash = 3.6,
        ### gap loss
        with_gap = False, threshold_gap = 3.9,
        ### for motif scaffolding
        motif_factor = 0.0
    ):
        ###### masks ######
        mask_res = batch['mask'] # True for valid tokens other than paddings; (N, L)
        mask_gen = batch['mask_gen'] # True for target positions; (N, L)
 
        ###### context feature embedding ######
        s_0 = batch['aa']  # amino acid index; int, 0~19, 21 for padding;  (N, L) 
        res_feat, pair_feat, v_0, p_0 = self.encode(batch)
        # res_feat, (N, L, *); None for "complete_gen" version
        # pair_feat, (N, L, L, *); None for "complete_gen" version"
        # v_0: orientation, (N, L, 3)
        # p_0: position of the CA atoms (N, L, 3) or backbone atoms (N, L, 4, 3)

        ###### diffusion loss calculation ######
        loss_dict = self.diffusion(
            v_0 = v_0, p_0 = p_0, s_0 = s_0, 
            mask_res = mask_res, mask_gen = mask_gen,
            res_feat = res_feat, pair_feat = pair_feat,
            denoise_structure = self.with_structure,
            denoise_sequence  = self.with_sequence,
            t = None, batch = batch,
            micro = micro, posi_loss_version = posi_loss_version,
            unnorm_first = unnorm_first,
            with_dist_loss = with_dist_loss, 
            dist_loss_version = dist_loss_version,
            threshold_dist = threshold_dist, dist_clamp = dist_clamp,
            with_clash = with_clash, threshold_clash = threshold_clash,
            with_gap = with_gap, threshold_gap = threshold_gap,
            motif_factor = motif_factor,
        )
        return loss_dict


    ############################################################################
    # sampling 
    ############################################################################

    @torch.no_grad()
    def sample(self, 
        length_list = None,
        mask_res = None, mask_generate = None,
        batch = None,
        t_bias = -1,
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        """
        Sample generation from scratch (by SZ):
            For the diffab version, require a batch input as the training 
            process did (i.e. need some groundtruth structures); otherwise just 
            need a list of the lengths (int)
        """

        ####################### preprocess #####################################
        device = next(self.parameters()).device
        if mask_res is not None:
            length_list = mask_res.sum(-1)  # (N,)

        N = len(length_list)
        L_max = int(max(length_list))
        if mask_res is None:
            mask_res = torch.zeros(N, L_max).bool().to(device) 
            for i in range(N):
                mask_res[i,:int(length_list[i])] = True

        ### mask for generating
        if mask_generate is None:
            mask_generate = mask_res  # generate the complete monomer

        ### batch data
        if batch is None:  # sample from scratch
            batch = {
                'res_nb': torch.arange(1, L_max+1).repeat(N, 1).to(device) * mask_res,
                'mask': mask_res,
                'mask_gen': mask_generate,
                'chain_nb': mask_res.int(),
                'fragment_type': mask_res.float(),
            }

        ################################ sampling ##############################

        ###### encoding ######
        motif_flag = ('aa' in batch) and ('pos_heavyatom' in batch)

        ### with encoder or motif
        if motif_flag:
            if 'aa' not in batch:
                batch['aa'] = (torch.ones(N, L_max).to(device) * 21 * (~mask_res)).long()
            if 'pos_heavyatom' not in batch:
                batch['pos_heavyatom'] = torch.zeros(N, L_max, 15, 3).to(device)
                batch['mask_heavyatom'] = F.pad(
                    mask_res.unsqueeze(2).repeat(1, 1, 4), (0, 11), "constant", False
                )

            s_0 = batch['aa']
            res_feat, pair_feat, v_0, p_0 = self.encode(batch)
        
        ### monomer design without encoder
        else:
            res_feat, pair_feat = None, None
            v_0, p_0, s_0 = None, None, None

        ###### sampling ######
        out_dict, traj = self.diffusion.backbone_gen(
            mask_res = mask_res, mask_gen = mask_generate,
            protein_size = length_list,
            res_feat = res_feat, pair_feat = pair_feat,
            v = v_0, p = p_0, s = s_0,
            batch = batch,
            t_bias = t_bias,
            **sample_opt
        )

        return out_dict, traj


################################################################################
# Confidence Net
################################################################################

class ConfidenceNet(nn.Module):
    def __init__(self, 
        res_feat_dim: int, 
        pair_feat_dim: int, 
        num_layers: int, 
        num_atoms: int = 4,
        max_relpos: int = 30,
        binary: bool = True 
    ):
        super().__init__()
        """Encoder of the sample states.
        
        Revised by SZ for single-chain generation: use the same resi encoder 
        and the pair encoder from the encoder.

        Args:
            res_feat_dim: dimension of residue-wise embedding features. 
            pair_feat_dim: dimension of pair-wise embedding features. 
            num_layers: number of layers.
            num_atoms: atoms considered for embedding.
        """

        self.num_atoms = num_atoms
        self.binary = binary

        #######################################################################
        # feature embedding module 
        #######################################################################

        self.residue_embed = ResidueEmbedding(
            res_feat_dim, num_atoms, with_type_emb = False,
            with_sequence = True, with_structure = True,
            with_seq_ddpm = False,
        )
        self.pair_embed = PairEmbedding(
            pair_feat_dim, num_atoms,
            with_sequence = True, with_structure = True,
            with_seq_ddpm = False, seq_ddpm_dim = res_feat_dim,
            max_relpos = max_relpos
        )

        #######################################################################
        # massage passing module 
        #######################################################################

        self.encoder = GAEncoder(
            res_feat_dim, pair_feat_dim, num_layers, 
        )

        #######################################################################
        # prediction projector
        #######################################################################

        self.predictor = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 4)
        )

        ### binary classification
        if self.binary:
            self.sigmoid = nn.Sigmoid()

    #######################################################################
    # Forward Function
    #######################################################################

    def forward(self, batch):
        """Embedding of state[t].

        Args (batch):
            aa: sequence at time t; (N, L).
            coor: 
            mask_res: mask with True for valid tokens (N, L).

        Returns:
            logit: (N, 4); confidence prediction 
        """
        #######################################################################
        # preprocessing
        #######################################################################

        N, L = batch['mask'].size()
        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )

        #######################################################################
        # feature embedding
        #######################################################################

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = batch['mask'],
            sequence_mask = batch['mask'],
        )

        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = batch['mask'],
            sequence_mask = batch['mask'],
        )

        #######################################################################
        # message passing
        #######################################################################

        res_feat = self.encoder(
            R, batch['pos_heavyatom'][:, :, BBHeavyAtom.CA], 
            res_feat, pair_feat, batch['mask']
        )

        #######################################################################
        # prediction
        #######################################################################

        output = self.predictor(res_feat)  # (N, L, 4)

        ###### aggregation ######
        output = (output * batch['mask'].unsqueeze(-1)).sum(dim=1)  # (N, 4)
        output = output / batch['mask'].sum(dim=1).unsqueeze(-1)    # (N, 4)
        if self.binary:
            output = self.sigmoid(output)

        return output 


