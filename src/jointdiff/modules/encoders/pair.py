import torch
import torch.nn as nn
import torch.nn.functional as F

from jointdiff.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals
from jointdiff.modules.common.layers import AngularEncoding
from jointdiff.modules.data.constants import BBHeavyAtom, AA

def concat_pairs(X):
    """Concatenates all possible pairs of vectors in the input tensor.

    Args:
        X: A tensor of shape (N, L, D).

    Returns:
        A tensor of shape (N, L, L, 2 * D), where Y[k][i][j] is the 
        concatenation of X[k][i] and X[k][j].
    """
    N, L, D = X.shape
    X_i = X.unsqueeze(2).expand(-1, -1, L, -1)  # Shape: NxLxLxD
    X_j = X.unsqueeze(1).expand(-1, L, -1, -1)  # Shape: NxLxLxD
    Y = torch.cat((X_i, X_j), dim=-1)  # Concatenate along the last dimension to get NxLxLx2D

    return Y


class PairEmbedding(nn.Module):

    def __init__(self, 
        feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32,
        with_sequence = True, with_structure = True, 
        with_seq_ddpm = False, seq_ddpm_dim = None
    ):
        super().__init__()

        ######################## version ######################################
        self.with_sequence = with_sequence
        self.with_structure = with_structure
        self.with_seq_ddpm = with_seq_ddpm

        if not (with_sequence or with_structure):
            raise Exception('At least one modality is needed!')

        ######################## settings #####################################
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos

        ######################### architecture ################################
        infeat_dim = feat_dim

        ### relative position embedding
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)

        ### aa pair embedding
        if self.with_sequence:
            if self.with_seq_ddpm:
                self.aa_pair_embed = nn.Linear(
                    seq_ddpm_dim * 2, feat_dim
                )
            else:
                self.aa_pair_embed = nn.Embedding(
                    self.max_aa_types * self.max_aa_types, feat_dim
                )
            infeat_dim += feat_dim
 
        if self.with_structure: 
            ### distance embedding
            self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
            nn.init.zeros_(self.aapair_to_distcoef.weight)
            self.distance_embed = nn.Sequential(
                nn.Linear(max_num_atoms * max_num_atoms, feat_dim), nn.ReLU(),
                nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            )
            infeat_dim += feat_dim

            ### dihedral embedding
            self.dihedral_embed = AngularEncoding()
            feat_dihed_dim = self.dihedral_embed.get_out_dim(2) # Phi and Psi
            infeat_dim += feat_dihed_dim

        ### output layer
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )


    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            structure_mask: (N, L)
                            SZ: for complete gen, for groundtruth it will be an all False mat (mask all)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.
                            SZ: for complete gen, for groundtruth it will be an all False mat (mask all)

        Returns:
            (N, L, L, feat_dim)
        """
        N = aa.shape[0]
        L = aa.shape[1]
        feat_all = []

        ###### Pair identities ######

        ### discrete sequence feature
        if self.with_sequence and (not self.with_seq_ddpm):
            if sequence_mask is not None:
                # Avoid data leakage at training time
                aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
            aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]    # (N, L, L)
            feat_aapair = self.aa_pair_embed(aa_pair)
            feat_all.append(feat_aapair)
 
        ### continuous DDPM for sequence; aa: (N, L, feat_dim)
        elif self.with_sequence:
            if sequence_mask is not None:
                aa[~sequence_mask.bool()] = 0.
            aa_pair = concat_pairs(aa)  # (N, L, L, resi_dim * 2)
            feat_aapair = self.aa_pair_embed(aa_pair)
            feat_all.append(feat_aapair)

        ### no sequence feature
        else: 
            aa_pair = torch.zeros(N, L, L).int().to(aa.device)
    
        ### Relative sequential positions
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = torch.clamp(
            res_nb[:,:,None] - res_nb[:,None,:], 
            min=-self.max_relpos, max=self.max_relpos,
        )   # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]
        feat_all.append(feat_relpos)

        if self.with_structure:
            ### Remove other atoms
            pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
            mask_atoms = mask_atoms[:, :, :self.max_num_atoms]
            pair_structure_mask = structure_mask[:, :, None] * structure_mask[:, None, :] if structure_mask is not None else None
 
            ### Distances
            d = angstrom_to_nm(torch.linalg.norm(
                pos_atoms[:,:,None,:,None] - pos_atoms[:,None,:,None,:],
                dim = -1, ord = 2,
            )).reshape(N, L, L, -1) # (N, L, L, A*A)
            c = F.softplus(self.aapair_to_distcoef(aa_pair))    # (N, L, L, A*A)
            d_gauss = torch.exp(-1 * c * d**2)
            mask_atom_pair = (mask_atoms[:,:,None,:,None] * mask_atoms[:,None,:,None,:]).reshape(N, L, L, -1)
            feat_dist = self.distance_embed(d_gauss * mask_atom_pair)
            if pair_structure_mask is not None:
                # Avoid data leakage at training time
                feat_dist = feat_dist * pair_structure_mask[:, :, :, None]
            feat_all.append(feat_dist)

            ### Orientations
            dihed = pairwise_dihedrals(pos_atoms)   # (N, L, L, 2)
            feat_dihed = self.dihedral_embed(dihed)
            if pair_structure_mask is not None:
                # Avoid data leakage at training time
                feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]
            feat_all.append(feat_dihed)

        ### All
        feat_all = torch.cat(feat_all, dim=-1)
        feat_all = self.out_mlp(feat_all)   # (N, L, L, F)

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all

