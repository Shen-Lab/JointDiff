import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab.modules.common.geometry import construct_3d_basis, reconstruct_backbone
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.diffusion.dpm_full import FullDPM, seq_recover
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model

from networks_proteinMPNN import ProteinMPNN  # added by SZ

resolution_to_num_atoms = {
    'backbone+CB': 5,  # N, CA, C, O, CB
    'backbone': 4, # by SZ; for single chain; N, CA, C, O
    'full': max_num_heavyatoms   # 15; N, CA, C, O, CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2, OXT
}


@register_model('diffab')
class DiffusionSingleChainDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        ###########################################################################################
        # Model Settings
        ###########################################################################################

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'backbone')]  
        # 15 (if cfg contains "resolution" and it is "full")
        # 5 (if cfg contains "resolution" and it is "backbone+CB")
        # 4 (other wise; will be "backbone")

        if not 'gen_version' in cfg.keys():
            print('Warning. "gen_version" not found in the config. Use the default value "complete_gen".')
            cfg.gen_version = 'complete_gen'

        elif not cfg.gen_version in ['complete_gen', 'diffab_gen']:
            print('Error! No generation version called %s!' % self.gen_version)
            quit()

        ###### training version ######
        if not 'train_version' in cfg.keys():
            cfg.train_version = 'noise'

        ###### random masking ######
        if not 'random_mask' in cfg.keys():
            cfg.random_mask = False

        if cfg.random_mask and cfg.gen_version != 'diffab_gen':
            print("Random masking requires context information. Use 'diffab_gen' instead!")
            cfg.gen_version = 'diffab_gen'

        ############################# whether share the encoder ##################################
        if cfg.gen_version == 'complete_gen' and (not 'encode_share' in cfg.keys()):
            print('Warning. "encode_share" not found in the config. Set it to be True.')
            cfg.encode_share = True
        self.cfg = cfg

        ############################# generation version (by SZ) ##################################
        # How to incorporate the res_feat and pair_feat:
        #     diffab_gen: based on the groundtruth (based on the original code, masking all region)
        #     complete_gen: based on the denoised sample 
        ###########################################################################################

        ############################## feature embedding ##########################################
        if cfg.gen_version == 'diffab_gen' or cfg.encode_share:
            self.residue_embed = ResidueEmbedding(
                cfg.res_feat_dim, num_atoms, random_mask = cfg.random_mask
            )
            self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        ################################ consistency oracle #######################################
       
        if 'proteinMPNN_model' not in cfg.keys():
            cfg.proteinMPNN_model = None

        ################################ CEP ######################################################

        if 'with_CEP_joint' not in cfg.keys():
            with_CEP_joint = False
        else:
            with_CEP_joint = cfg.with_CEP_joint

        if with_CEP_joint:
            print('Appling CEP...')

        ############################## main model #################################################
        if cfg.gen_version == 'complete_gen' and cfg.encode_share: # add by SZ
            ### share the sequence (both resi-wise and pair-wise) embedd
            self.diffusion = FullDPM(
                cfg.res_feat_dim,
                cfg.pair_feat_dim,
                num_atoms = num_atoms, 
                gen_version = cfg.gen_version,
                residue_embed = self.residue_embed,
                pair_embed = self.pair_embed,
                train_version = cfg.train_version,
                proteinMPNN_model = cfg.proteinMPNN_model,
                with_CEP_joint = with_CEP_joint,
                **cfg.diffusion,
            )
        else:
            self.diffusion = FullDPM(
                cfg.res_feat_dim,
                cfg.pair_feat_dim,
                num_atoms = num_atoms,
                gen_version = cfg.gen_version,
                with_CEP_joint = with_CEP_joint,
                **cfg.diffusion,
            )

    ############################################################################
    # context and groundtruth 
    ############################################################################

    def encode(self, batch):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim) or None.
            pair_feat:  (N, L, L, pair_feat_dim) or None.
        """
        ########## with the diffab pipeline (for single-chain) ################
        if self.cfg.gen_version == 'diffab_gen':
            # structure_mask = torch.zeros(batch['mask'].shape, dtype=torch.bool).to(batch['mask'].device) 
            # sequence_mask = torch.zeros(batch['mask'].shape, dtype=torch.bool).to(batch['mask'].device)
            structure_mask = ((~batch['mask_gen']) * batch['mask']).bool()
            sequence_mask = ((~batch['mask_gen']) * batch['mask']).bool()

            ### residue features, (N, L, feat)
            res_feat = self.residue_embed(
                aa = batch['aa'], 
                res_nb = batch['res_nb'],
                chain_nb = batch['chain_nb'],
                pos_atoms = batch['pos_heavyatom'],
                mask_atoms = batch['mask_heavyatom'],
                fragment_type = batch['fragment_type'],
                structure_mask = structure_mask,
                sequence_mask = sequence_mask,
            )  
            ## aa: amino acid index; int, 0~19, 21 for padding;  (N, L).
            ## res_nb: residue idx to determine consecutive residues; 
            ##         int, ordinal, 0 after valid tokens for padding; e.g. 1,2,...,L; (N, L).
            ## chain_nb: chain idx; int; (N, L); original 2 for Heavy, 1 for Ag and 0 for 
            ##           Light or Padding; for single-chain protein, 1 for valid token and 
            ##           0 for Padding.
            ## pos_atoms: heavy atom coordinates; float; (N, L, atom_num = 15; 3).
            ## mask_atoms: heavy atom mask; bool, True for containing the corresponding atom; 
            ##             (N, L, atom_num = 15).

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
        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )  # (N, L, 3, 3)
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA] # (N, L, 3)

        return res_feat, pair_feat, R, p
   
 
    ############################################################################
    # forward function 
    ############################################################################

    def forward(self, 
        batch,
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
        contact_fix = True,
        contact_path_list_all = None,
        name_idx = None,
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
            batch:
                mask
                aa
            energy_guite: energy gradient (force) predictor
            fitness_guide: fitness score gradient predictor
            struc_scale: "Boltzmann", "none" or float; 
                         "Boltzmann" = 1 / (K_B, T)
                         "none" = 1
            seq_scale: "length", "none" or float; "length" = length; for other options the same as 'struc_scale'
        """
        ###### masks ######
        mask_res = batch['mask'] # True for valid tokens other than paddings; (N, L)
        mask_gen = batch['mask_gen'] # True for target positions; (N, L)
 
        ###### feature embedding (of ground truth) ######
        res_feat, pair_feat, R_0, p_0 = self.encode(batch)
        # res_feat, (N, L, res_feat_dim); None for "complete_gen" version
        # pair_feat, (N, L, L, pair_feat_dim); None for "complete_gen" version"
        # R_0: orientation, (N, L, 3, 3)
        # p: position of the CA atoms, (N, L, 3)

        v_0 = rotation_to_so3vec(R_0)  # transform SO(3) to 3d vectors (N, L, 3)
        s_0 = batch['aa']  # amino acid index; int, 0~19, 21 for padding;  (N, L) 

        ###### prepare contact maps ######
        if (energy_guide is not None) and with_contact and contact_fix: 
            contact_path_list = [
                contact_path_list_all[int(idx)] for idx in name_idx
            ]
        else:
            contact_path_list = None

        ###### diffusion loss calculation ######
        loss_dict = self.diffusion(
            v_0, p_0, s_0, res_feat, pair_feat, mask_res, mask_gen,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
            t = None, batch = batch,
            micro = micro, posi_loss_version = posi_loss_version,
            unnorm_first = unnorm_first,
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
            seq_sample = seq_sample,
            t_max = t_max,
            force_vs_diff = force_vs_diff,
        )
        return loss_dict

    ############################################################################
    # sampling with context info (from diffab) 
    ############################################################################

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.sample(v_0, p_0, s_0, res_feat, pair_feat, mask_res, batch = batch, **sample_opt)
        return traj

    ############################################################################
    # sampling from scratch 
    ############################################################################

    @torch.no_grad()
    def sample_from_scratch(self, 
        length_list = None, with_wiener=True, self_condition=True, t_bias = -1
    ):
        """
        Sample generation from scratch (by SZ):
        For the diffab version, require a batch input as the training process did (i.e. need some groundtruth structures);
        otherwise just need a list of the lengths (int)
        """
        N = len(length_list)
        L_max = int(max(length_list))
        device = next(self.parameters()).device
        mask_res = torch.zeros(N, L_max).bool().to(device) 
        for i in range(N):
            mask_res[i,:int(length_list[i])] = True               

        batch = {
            'aa': (torch.ones(N, L_max).to(device) * 21 * (~mask_res)).long(),
            'res_nb': torch.arange(1, L_max+1).repeat(N, 1).to(device) * mask_res,
            'mask': mask_res,
            'mask_gen': mask_res,
            'chain_nb': mask_res.int(),
            'pos_heavyatom': torch.zeros(N, L_max, 15, 3).to(device),
            'mask_heavyatom': F.pad(mask_res.unsqueeze(2).repeat(1, 1, 4), (0, 11), "constant", False),
            'fragment_type': mask_res.float(),
        }
 
        if self.cfg.gen_version == 'diffab_gen':
            res_feat, pair_feat, R_0, p_0 = self.encode(batch)
            out_dict, traj = self.diffusion.backbone_gen(
                mask_res, res_feat = res_feat, pair_feat = pair_feat, 
                with_wiener=with_wiener, self_condition = self_condition,
                t_bias = t_bias
            )

        else:
            out_dict, traj = self.diffusion.backbone_gen(
                mask_res, 
                with_wiener=with_wiener, self_condition = self_condition,
                t_bias = t_bias
            )

        return out_dict, traj

    ############################################################################
    # sampling for single modality 
    ############################################################################

    @torch.no_grad()
    def sample_SingleModal(
        self,
        batch,
        modality, 
        with_the_other_modality = True
    ):

        if modality == 'structure':
            sample_structure = True
            sample_sequence = False
        elif modality == 'sequence':
            sample_structure = False
            sample_sequence = True
        else:
            raise NameError('No modality named %s!' % modality) 

        mask_res = batch['mask']
        N, L = mask_res.shape
        ### for info extraction
        res_nb = torch.zeros(N,L).int().to(mask_res.device)
        res_nb[:] = torch.arange(1, L+1).to(mask_res.device)
        res_nb = (res_nb * mask_res).int().cpu() # residue idx to determine consecutive residues; int, ordinal, e.g. 1,2,...,L, 0, ..., 0; (N, L_max)
        chain_nb = mask_res.int().cpu() # chain idx; int; (N, L); 1 for valid token and 0 for Padding
        lengths = mask_res.sum(-1).cpu()  # (N,)

        ###### data preprocess ######
        ## for "complete_gen" res_feat and pair_feat are None
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
        )

        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        
        if not with_the_other_modality:
            v_0 = torch.zeros(v_0.shape, device = v_0.device)
            p_0 = torch.zeros(p_0.shape, device = p_0.device)
            s_0 = torch.zeros(s_0.shape, device = s_0.device).long()

        ###### inference ######
        traj = self.diffusion.sample(
                   v_0, p_0, s_0, 
                   res_feat, pair_feat, mask_res, batch = batch, 
                   sample_structure = sample_structure, sample_sequence = sample_sequence,
               )
        ### for t = 0 
        traj[0] = tuple(x.cpu() for x in traj[0])

        ###### extract the info ######
        out_dict = {}
        for t in traj.keys():
            out_dict[t] = []
            R = so3vec_to_rotation(traj[t][0])
            bb_coor_batch = reconstruct_backbone(
                 R, traj[t][1], traj[t][2], 
                 chain_nb, res_nb, mask_res.cpu()
            )  # (N, L_max, 4, 3)

            for i, bb_coor in enumerate(bb_coor_batch):
                seq = seq_recover(traj[t][2][i], length = lengths[i])
                out_dict[t].append({'coor': bb_coor[:lengths[i]], 'seq': seq})

        return out_dict, traj


    @torch.no_grad()
    def sample_SingleModal_from_scratch(self, 
        modality = 'structure', 
        length_list = None, 
    ):
        """Single-modality generation from scratch (set the other modality to be 0)."""

        ###### shape ######
        N = len(length_list)
        L_max = int(max(length_list))

        ###### inputs ######
        batch = {}

        ### mask: True for valid token and False for paddings; (N, L_max)
        batch['mask'] = torch.zeros(N, L_max).bool().to(next(self.parameters()).device)
        for i in range(N):
            batch['mask'][i,:int(length_list[i])] = True
        ### aa: sequence; zero-vectors in this function; (N, L_max)
        batch['aa'] = torch.zeros(N, L_max).long().to(batch['mask'].device)
        ### res_nb: residue index vector; (N, L_max) 
        batch['res_nb'] = torch.zeros(N, L_max).int().to(batch['mask'].device)
        batch['res_nb'][:] = torch.arange(1, L_max + 1).to(batch['mask'].device)
        ### chain_nb: 1 for valid token and 0 for paddings; (N, L_max)
        batch['chain_nb'] = batch['mask'].int() 
        ### position tensor: zero-vectors in this function; (N, L_max, 4, 3)
        batch['pos_heavyatom'] = torch.zeros(N, L_max, 4, 3).to(batch['mask'].device) 
        ### atom mask: (N, L_max, 4)
        batch['mask_heavyatom'] = batch['mask'][:,:,None].repeat(1,1,4)

        ###### inference ######
        out_dict, traj = self.sample_SingleModal(
            batch,
            modality = modality, 
            with_the_other_modality = False
        )

        return out_dict, traj


    ############################################################################
    # otimiztion given a current states (from diffab) 
    ############################################################################

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
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_res, batch = batch, **optimize_opt) 
        return traj
