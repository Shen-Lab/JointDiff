import torch
import torch.nn as nn

from jointdiff.modules.common.geometry import (
    construct_3d_basis, 
    global_to_local, 
    get_backbone_dihedral_angles
)
from jointdiff.modules.common.layers import AngularEncoding
from jointdiff.modules.data.constants import BBHeavyAtom, AA


class ResidueEmbedding(nn.Module):

    def __init__(self, 
        feat_dim, max_num_atoms, max_aa_types=22, 
        with_type_emb=False, with_sequence = True, with_structure = True,
        with_seq_ddpm = False
    ):
        super().__init__()

        ######################## version ######################################
        self.with_type_emb = with_type_emb
        self.with_sequence = with_sequence
        self.with_structure = with_structure
        self.with_seq_ddpm = with_seq_ddpm

        if not (with_sequence or with_structure):
            raise Exception('At least one modality is needed!')

        ######################## settings #####################################
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types

        ######################### architecture ################################
        infeat_dim = 0

        ### aa embedding layer 
        if self.with_sequence:
            if not self.with_seq_ddpm:
                self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
            infeat_dim += feat_dim
 
        ### dihedral embedding layer
        if self.with_structure:
            self.dihed_embed = AngularEncoding()
            infeat_dim += self.dihed_embed.get_out_dim(3)
            if self.with_sequence:
                infeat_dim += self.max_aa_types * self.max_num_atoms * 3
            else:
                infeat_dim += self.max_num_atoms * 3

        ### type features
        if self.with_type_emb:
             ## need to consider different position types 
             self.type_embed = nn.Embedding(3, feat_dim, padding_idx=0)  # 1: target, 2: scaffold
             infeat_dim += feat_dim

        ### MLP output layers
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )


    def forward(self, 
        aa, res_nb, chain_nb, pos_atoms, mask_atoms, fragment_type = None, 
        structure_mask=None, sequence_mask=None
    ):
        """
        Args:
            aa: amino acid sequence; (N, L).
            res_nb: residue indexes; (N, L), 
            chain_nb: chain indexes; (N, L); for single chain protein only contain 0 and 1 while 0 for padding
            pos_atoms: atom-wise coordinates; (N, L, A, 3).
            mask_atoms: atom-wise masks; (N, L, A).
            ## fragment_type:  (N, L).
            structure_mask: (N, L), mask out unknown structures to generate.
                            SZ: for complete gen, for groundtruth it will be a all False mat (mask all) 
            sequence_mask:  (N, L), mask out unknown amino acids to generate.
                            SZ: for complete gen, for groundtruth it will be a all False mat (mask all)
        """
        N = aa.shape[0]
        L = aa.shape[1]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # mask the invalid residues or those whose CA is unknown (N, L)
        out_feat = []

        ###### Amino acid identity features ######
        if self.with_sequence:
            if self.with_seq_ddpm:
                aa_feat = aa
            else:
                if sequence_mask is not None:
                    # Avoid data leakage at training time
                    aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
                aa_feat = self.aatype_embed(aa) # (N, L, feat)
            out_feat.append(aa_feat)

        ###### structure features ######

        ###### Remove other atoms ######
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]  # coordinates, (N, L, atom_sele_num, 3)
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms] # atom masks, mask if the corresponding atom missing, (N, L, atom_sele_num)

        ###### Coordinate features ######
        if self.with_structure: 
            R = construct_3d_basis(
                pos_atoms[:, :, BBHeavyAtom.CA], 
                pos_atoms[:, :, BBHeavyAtom.C], 
                pos_atoms[:, :, BBHeavyAtom.N]
            ) # A batch of orthogonal basis matrix, (N, L, 3, 3)
            t = pos_atoms[:, :, BBHeavyAtom.CA] # (N, L, 3)
            crd = global_to_local(R, t, pos_atoms)    # (N, L, atom_sele_num, 3)
            crd_mask = mask_atoms[:, :, :, None].expand_as(crd)  # (N, L, atom_sele_num, 3)
            crd = torch.where(crd_mask, crd, torch.zeros_like(crd))  # set the masked cdr feature to be 0, (N, L, atom_sele_num, 3)

            ### incorporate sequence info: (N, L, aa_num * atom_sele_num * 3)
            if self.with_sequence:
                aa_expand = aa[:, :, None, None, None].expand(
                    N, L, self.max_aa_types, self.max_num_atoms, 3
                )  # (N, L, aa_num=22, atom_sele_num, 3)
                rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(
                    N, L, self.max_aa_types, self.max_num_atoms, 3
                ).to(aa_expand)  # (N, L, aa_num=22, atom_sele_num, 3)
                place_mask = (aa_expand == rng_expand)  # retain the corresponding residue positions and mask the rest
                crd_expand = crd[:, :, None, :, :].expand(
                    N, L, self.max_aa_types, self.max_num_atoms, 3
                )  # (N, L, aa_num=22, atom_sele_num, 3)
                crd_expand = torch.where(
                    place_mask, crd_expand, torch.zeros_like(crd_expand)
                )  # (N, L, aa_num=22, atom_sele_num, 3)
                crd_feat = crd_expand.reshape(
                    N, L, self.max_aa_types * self.max_num_atoms * 3
                )  # (N, L, aa_num * atom_sele_num * 3)

            ### no sequence info: (N, L, atom_sele_num * 3)
            else:
                crd_feat = crd.reshape(N, L, -1) 
                
            ### mask: avoid data leakage at training time
            if structure_mask is not None:
                crd_feat = crd_feat * structure_mask[:, :, None]
            out_feat.append(crd_feat)

        ###### Backbone dihedral features ######
        if self.with_structure:
            bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(
                pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue
            )
            # bb_dihedral: Omega, Phi, and Psi angles in radian; (N, L, 3) 
            # mask_bb_dihed: Masks of dihedral angles; (N, L, 3) 
            dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # (N, L, 3, dihed/3)
            dihed_feat = dihed_feat.reshape(N, L, -1)  # (N, L, dihed)
            if structure_mask is not None:
                # Avoid data leakage at training time
                dihed_mask = torch.logical_and(
                    structure_mask,
                    torch.logical_and(
                        torch.roll(structure_mask, shifts=+1, dims=1), 
                        torch.roll(structure_mask, shifts=-1, dims=1)
                    ),
                )   # Avoid slight data leakage via dihedral angles of anchor residues
                dihed_feat = dihed_feat * dihed_mask[:, :, None]
            out_feat.append(dihed_feat)

        ###### Type feature ######
        if self.with_type_emb:
            type_feat = self.type_embed(fragment_type.long()) # (N, L, feat)
            out_feat.append(type_feat)

        out_feat = self.mlp(torch.cat(out_feat, dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]

        return out_feat

