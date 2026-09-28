import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals
from diffab.modules.common.layers import AngularEncoding
from diffab.utils.protein.constants import BBHeavyAtom, AA

################################################################################
# AF3 relative position encoding: https://github.com/lucidrains/alphafold3-pytorch/blob/main/alphafold3_pytorch/alphafold3.py
################################################################################

from functools import partial, wraps
from einops import rearrange, repeat, reduce, einsum, pack, unpack
try:
    import einx
except Exception as e:
    print(e)

LinearNoBias = partial(nn.Linear, bias = False)

class RelativePositionEncoding(nn.Module):
    """ Algorithm 3 """
    
    def __init__(
        self,
        *,
        r_max = 32,
        s_max = 2,
        dim_out = 128
    ):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        
        dim_input = (2*r_max+2) + (2*r_max+2) + 1 + (2*s_max+2)
        self.out_embedder = LinearNoBias(dim_input, dim_out)

    # @typecheck
    # def forward(
    #     self,
    #     *,
    #     additional_molecule_feats: Int[f'b n {ADDITIONAL_MOLECULE_FEATS}']
    # ) -> Float['b n n dp']:

    #     dtype = self.out_embedder.weight.dtype
    #     device = additional_molecule_feats.device

    #     res_idx, token_idx, asym_id, entity_id, sym_id = additional_molecule_feats.unbind(dim = -1)
    
    def forward(
        self, res_idx, token_idx, asym_id, entity_id, sym_id,
    ):
        """
        Args:
            res_idx: Residue number in the token’s original input chain; (N, L)
            token_idx: Token number. Increases monotonically; does not restart at 1 for new chains; (N, L)
            asym_id: Unique integer for each distinct chain; (N, L)
            entity_id: Unique integer for each distinct sequence; (N, L)
            sym_id: Unique integer within chains of this sequence; (N, L)
        """

        dtype = self.out_embedder.weight.dtype
        device = res_idx.device

        diff_res_idx = einx.subtract('b i, b j -> b i j', res_idx, res_idx)
        diff_token_idx = einx.subtract('b i, b j -> b i j', token_idx, token_idx)
        diff_sym_id = einx.subtract('b i, b j -> b i j', sym_id, sym_id)

        mask_same_chain = einx.subtract('b i, b j -> b i j', asym_id, asym_id) == 0
        mask_same_res = diff_res_idx == 0
        mask_same_entity = einx.subtract('b i, b j -> b i j 1', entity_id, entity_id) == 0
        
        d_res = torch.where(
            mask_same_chain,
            torch.clip(diff_res_idx + self.r_max, 0, 2*self.r_max),
            2*self.r_max + 1
        )

        d_token = torch.where(
            mask_same_chain * mask_same_res,
            torch.clip(diff_token_idx + self.r_max, 0, 2*self.r_max),
            2*self.r_max + 1
        )

        d_chain = torch.where(
            ~mask_same_chain,
            torch.clip(diff_sym_id + self.s_max, 0, 2*self.s_max),
            2*self.s_max + 1
        )
        
        def onehot(x, bins):
            dist_from_bins = einx.subtract('... i, j -> ... i j', x, bins)
            indices = dist_from_bins.abs().min(dim = -1, keepdim = True).indices
            one_hots = F.one_hot(indices.long(), num_classes = len(bins))
            return one_hots.type(dtype)

        r_arange = torch.arange(2*self.r_max + 2, device = device)
        s_arange = torch.arange(2*self.s_max + 2, device = device)

        a_rel_pos = onehot(d_res, r_arange)
        a_rel_token = onehot(d_token, r_arange)
        a_rel_chain = onehot(d_chain, s_arange)

        out, _ = pack((
            a_rel_pos,
            a_rel_token,
            mask_same_entity,
            a_rel_chain
        ), 'b i j *')

        return self.out_embedder(out)

################################################################################

class PairEmbedding(nn.Module):

    def __init__(self, 
        feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32, 
        chain_feat_version='same', with_af3_relpos = False 
    ):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        # self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)  # origin

        ###########################################################
        # how to incorporate the chain information (by SZ)
        ###########################################################
        self.chain_feat_version = chain_feat_version
        self.with_af3_relpos = with_af3_relpos
       
        if self.chain_feat_version != 'same':
            self.chain_embed = nn.Embedding(2, feat_dim)
            self.chain_feat_map = nn.Linear(feat_dim * 2, feat_dim)

        if self.with_af3_relpos:
            self.relpos_embed = RelativePositionEncoding(r_max = max_relpos, dim_out = feat_dim) 
        else:
            self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)  # origin

        ###########################################################

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2) # Phi and Psi

        infeat_dim = feat_dim+feat_dim+feat_dim+feat_dihed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None, fragment_type = None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        device = aa.device  # by SZ

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
        pair_structure_mask = structure_mask[:, :, None] * structure_mask[:, None, :] if structure_mask is not None else None

        # Pair identities
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]    # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)
    
        # Relative sequential positions
        
        if self.with_af3_relpos:
            feat_relpos = self.relpos_embed(
                res_idx = res_nb, 
                token_idx = torch.arange(L).repeat(N, 1).to(device) * mask_residue,
                asym_id = chain_nb, 
                entity_id = chain_nb, 
                sym_id = fragment_type,
            )

        else:  # original
            same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
            relpos = torch.clamp(
                res_nb[:,:,None] - res_nb[:,None,:], 
                min=-self.max_relpos, max=self.max_relpos,
            )   # (N, L, L)
            feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]
     
        ### incorporate interchain info: by SZ
        if self.chain_feat_version != 'same':
            feat_chain = self.chain_embed(same_chain.int()) # (N, L, L, dim)
            feat_relpos = torch.cat([feat_relpos, feat_chain], dim=-1)
            feat_relpos = self.chain_feat_map(feat_relpos)

        # Distances
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

        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)   # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]

        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)   # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all

