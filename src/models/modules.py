import typing as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import ml_collections
from typing import Dict, Optional, Tuple
import math

from einops import rearrange
from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock

from models.utils_modules import (
    MultiLayerPerceptron, 
    GraphAttentionLayer, 
    ConvLayers, 
    TriangularSelfAttentionNetwork,
    TransformerEncoder,
    TransformerDecoder,
    VarianceSchedule,
)
from ldm.modules.diffusionmodules.openaimodel import UNetModel

###############################################################################
# Auxiliary
###############################################################################

def mat_outer_prod(mat_1, mat_2):
    # mat_1, mat_2: (N, L)
    if mat_1 is None or mat_2 is None:
        return None
    else:
        return torch.bmm(mat_1.unsqueeze(2), mat_2.unsqueeze(1))


def emb_mask_prepare(ori_mask, kernel_size, with_pair = False, device = 'cpu'):
    if ori_mask is None:
        return None, None

    B, L = ori_mask.shape
    L = math.ceil((L - 1) / (kernel_size - 1))
    length_list = ori_mask.sum(1)
    length_list = torch.ceil((length_list - 1) / (kernel_size - 1))

    node_mask = torch.zeros(B, L)
    for i, l in enumerate(length_list):
        node_mask[i, :int(l)] = 1
    node_mask = node_mask.to(device)

    if with_pair:
        pair_mask = mat_outer_prod(
            node_mask.float(), node_mask.float()
        )
        pair_mask = (pair_mask == 1)  # (B, L_max, L_max)
    else:
        pair_mask = None

    return node_mask, pair_mask


def sequence_transform(
    mpnn_restypes='ACDEFGHIKLMNPQRSTVWYX',
    esm_restypes='ARNDCQEGHILKMFPSTWYVX'
):
    # mpnn_restypes: 'ACDEFGHIKLMNPQRSTVWYX'
    # esm_restypes: 'ARNDCQEGHILKMFPSTWYVX'

    length = len(mpnn_restypes)

    if set(mpnn_restypes) != set(esm_restypes):
        raise Exception('Error! The voxel sets do not match!')
    elif len(esm_restypes) != length:
        raise Exception('Error! The voxel size does not match! (%d and %d)' % (
            length, len(esm_restypes)))
    else:
        mpnn_to_esm_mat = torch.zeros(length + 1, length + 1)
        mpnn_to_esm_mat[-1][-1] = 1
        for mpnn_idx, token in enumerate(mpnn_restypes):
            esm_idx = esm_restypes.index(token)
            mpnn_to_esm_mat[mpnn_idx][esm_idx] = 1

    return mpnn_to_esm_mat


###############################################################################
# FeatureProjector
###############################################################################

class FeatureProjector(nn.Module):
    def __init__(self,
        in_resi_features:int,
        in_pair_features:int,
        hidden_features:int,
        out_features:int,
        version:str='mlp',
        num_heads:int=4,
        num_layers:int=3,
        dropout:float=0.0,
        max_length:int=202,
        with_edge_feature:bool=True,
        with_posi_embedding:bool=True
    ):
        super(FeatureProjector, self).__init__()

        self.version = version
        self.with_edge_feature = with_edge_feature
        if with_edge_feature and self.version == 'transformer':
            raise NameError(
                'Error! %s projector does not support the edge features!' % self.version
            ) 
        elif not with_edge_feature and self.version == 'tab':
            raise NameError(
                'Error! %s projector needs the edge features!' % self.version
            ) 

        ################## MLP Projector ######################################
        if self.version == 'mlp':
            def mlp_projector(in_feat, out_feat):
                return MultiLayerPerceptron(
                    input_dim = in_feat, 
                    hidden_dims = [hidden_features] * (num_layers - 1),
                    output_dim = out_feat
                )

            ###### node encoder
            self.node_encoder = mlp_projector(in_resi_features, out_features)
            self.node_decoder = mlp_projector(out_features, in_resi_features)

            ###### edge encoder
            if with_edge_feature:
                self.edge_encoder = mlp_projector(in_pair_features, out_features)
                self.edge_decoder = mlp_projector(out_features, in_pair_features)

        ################## Transformer Projector ###############################
        elif self.version == 'transformer':

            self.node_dim_map = nn.Linear(
                in_resi_features, hidden_features
            )
            self.node_encoder = TransformerEncoder(
                d_model = hidden_features, 
                out_features = out_features,
                num_heads = num_heads,
                num_layers = num_layers,
                dropout = dropout,
                with_embedding = False,
                with_posi_embedding = with_posi_embedding,
                max_length = max_length,
            )
            self.node_decoder = TransformerEncoder(
                d_model = out_features, 
                out_features = in_resi_features,
                num_heads = num_heads,
                num_layers = num_layers,
                dropout = dropout,
                with_embedding = False,
                with_posi_embedding = with_posi_embedding,
                max_length = max_length,
            )

        ################## TriangularSelfAttentionBlock Projector ##############
        elif self.version == 'tab':

            self.node_dim_map = nn.Linear(
                in_resi_features, hidden_features
            )
            self.edge_dim_map = nn.Linear(
                in_pair_features, hidden_features
            )
 
            self.encoder = TriangularSelfAttentionNetwork(
                in_resi_features = hidden_features,
                in_pair_features = hidden_features,
                out_resi_features = out_features,
                out_pair_features = out_features,
                num_heads = num_heads,
                num_blocks = num_layers,
                dropout = dropout
            )

            self.decoder = TriangularSelfAttentionNetwork(
                in_resi_features = out_features,
                in_pair_features = out_features,
                out_resi_features = in_resi_features,
                out_pair_features = in_pair_features,
                num_heads = num_heads,
                num_blocks = num_layers,
                dropout = dropout
            )

        ######################## outliers #####################################
        else:
            raise NameError(
                'Error! No projector version named %s!' % self.version
            )


    def encode(self, node_feat, edge_feat = None, mask = None, adj = None):
        """Projector."""
        if self.version == 'mlp':
            node_feat = self.node_encoder(node_feat)
            if self.with_edge_feature:
                edge_feat = self.edge_encoder(edge_feat)

        elif self.version == 'transformer':
            node_feat = self.node_dim_map(node_feat)
            node_feat = self.node_encoder(
                src = node_feat, 
                padding_mask = ~mask.to(torch.bool)
            )

        elif self.version == 'tab':
            ### dim match
            node_feat = self.node_dim_map(node_feat)
            edge_feat = self.edge_dim_map(edge_feat)
            ### project
            node_feat, edge_feat = self.encoder(node_feat, edge_feat, mask)

        return node_feat, edge_feat
         

    def decode(self, node_feat, edge_feat = None, mask = None, adj = None):
        """Projector."""
        if self.version == 'mlp':
            node_feat = self.node_decoder(node_feat)
            if self.with_edge_feature:
                edge_feat = self.edge_decoder(edge_feat)

        elif self.version == 'transformer':
            node_feat = self.node_decoder(
                src = node_feat,
                padding_mask = ~mask.to(torch.bool)
            )

        elif self.version == 'tab':
            node_feat, edge_feat = self.decoder(node_feat, edge_feat, mask)

        return node_feat, edge_feat


class JointFeatureProjector(nn.Module):
    def __init__(self,
        in_seq_resi_features:int,
        in_seq_pair_features:int,
        in_struc_resi_features:int,
        in_struc_pair_features:int,
        hidden_features:int,
        out_features:int,
        version:str='joint-mlp',
        num_heads:int=4,
        num_layers:int=3,
        dropout:float=0.0,
        max_length:int=202,
        with_edge_feature:bool=True,
        with_posi_embedding:bool=True
    ):
        super(JointFeatureProjector, self).__init__()

        self.version = version
        self.with_edge_feature = with_edge_feature

        ################## MLP Projector ######################################
        if self.version == 'joint-mlp':
            def mlp_projector(in_feat, out_feat):
                return MultiLayerPerceptron(
                    input_dim = in_feat,
                    hidden_dims = [hidden_features] * (num_layers - 1),
                    output_dim = out_feat
                )

            ###### node encoder
            self.node_encoder = mlp_projector(
                in_seq_resi_features + in_struc_resi_features, out_features
            )
            self.seq_node_decoder = mlp_projector(
                out_features, in_seq_resi_features
            )
            self.struc_node_decoder = mlp_projector(
                out_features, in_struc_resi_features
            )

            ###### else encoder
            if with_edge_feature:
                self.edge_encoder = mlp_projector(
                    in_seq_pair_features + in_struc_pair_features, out_features
                )
                self.seq_edge_decoder = mlp_projector(
                    out_features, in_seq_pair_features
                )
                self.struc_edge_decoder = mlp_projector(
                    out_features, in_struc_pair_features
                )

        ################## Transformer Projector ###############################
        elif self.version == 'joint-transformer':
            if with_edge_feature:
                raise ValueError(
                    'Error! Cannot process edge features with transformer projector!'
                )

            self.node_dim_map = nn.Linear(
                in_seq_resi_features + in_struc_resi_features, hidden_features
            )
            self.encoder = TransformerEncoder(
                d_model = hidden_features,
                out_features = out_features,
                num_heads = num_heads,
                num_layers = num_layers,
                dropout = dropout,
                with_embedding = False,
                with_posi_embedding = with_posi_embedding,
                max_length = max_length,
            )
            self.seq_decoder = TransformerEncoder(
                d_model = out_features,
                out_features = in_seq_resi_features,
                num_heads = num_heads,
                num_layers = num_layers,
                dropout = dropout,
                with_embedding = False,
                with_posi_embedding = with_posi_embedding,
                max_length = max_length,
            )
            self.struc_decoder = TransformerEncoder(
                d_model = out_features,
                out_features = in_struc_resi_features,
                num_heads = num_heads,
                num_layers = num_layers,
                dropout = dropout,
                with_embedding = False,
                with_posi_embedding = with_posi_embedding,
                max_length = max_length,
            )

        ################## TriangularSelfAttentionBlock Projector #############
        elif self.version == 'joint-tab':

            if not with_edge_feature:
                raise ValueError(
                    'Error! Need edge features for tab projector!'
                )

            self.node_dim_map = nn.Linear(
                in_seq_resi_features + in_struc_resi_features, hidden_features
            )
            self.edge_dim_map = nn.Linear(
                in_seq_pair_features + in_struc_pair_features, hidden_features
            )

            self.encoder = TriangularSelfAttentionNetwork(
                in_resi_features = hidden_features,
                in_pair_features = hidden_features,
                out_resi_features = out_features,
                out_pair_features = out_features,
                num_heads = num_heads,
                num_blocks = num_layers,
                dropout = dropout
            )

            self.seq_decoder = TriangularSelfAttentionNetwork(
                in_resi_features = out_features,
                in_pair_features = out_features,
                out_resi_features = in_seq_resi_features,
                out_pair_features = in_seq_pair_features,
                num_heads = num_heads,
                num_blocks = num_layers,
                dropout = dropout
            )

            self.struc_decoder = TriangularSelfAttentionNetwork(
                in_resi_features = out_features,
                in_pair_features = out_features,
                out_resi_features = in_struc_resi_features,
                out_pair_features = in_struc_pair_features,
                num_heads = num_heads,
                num_blocks = num_layers,
                dropout = dropout
            )

        ######################## outliers #####################################
        else:
            raise NameError(
                'Error! No projector version named %s!' % self.version
            )


    def encode(self, 
        seq_node_feat, 
        seq_edge_feat = None, 
        struc_node_feat = None, 
        struc_edge_feat = None, 
        mask = None, 
        adj = None
    ):
        """Joint Projector.

        Args:
            seq_node_feat: (B, L_max, seq_resi_dim).
            seq_edge_feat: (B, L_max, L_max, seq_pair_dim).
            struc_node_feat: (B, L_max, struc_resi_dim).
            struc_edge_feat: (B, L_max, L_max, struc_pair_dim).
            mask: (B, L_max).
            adj: (B, L_max, L_max).

        Returns:
        
        """
        ###### feature cat ######
        node_feat = torch.cat(
            [seq_node_feat, struc_node_feat],
            dim = -1
        )
        if self.with_edge_feature:
            edge_feat = torch.cat(
                [seq_edge_feat, struc_edge_feat],
                dim = -1
            )
        else:
            edge_feat = None

        ###### project ######
        if self.version == 'joint-mlp':
            node_feat = self.node_encoder(node_feat)
            if self.with_edge_feature:
                edge_feat = self.edge_encoder(edge_feat)

        elif self.version == 'joint-transformer':
            node_feat = self.node_dim_map(node_feat)
            node_feat = self.encoder(
                src = node_feat,
                padding_mask = ~mask.to(torch.bool)
            )

        elif self.version == 'joint-tab':
            ### dim match
            node_feat = self.node_dim_map(node_feat) 
            edge_feat = self.edge_dim_map(edge_feat) 
            
            ### project
            node_feat, edge_feat = self.encoder(node_feat, edge_feat, mask)

        return node_feat, edge_feat


    def decode(self, node_feat, edge_feat = None, mask = None, adj = None):
        """Projector."""

        if self.version == 'joint-mlp':
            seq_node_feat = self.seq_node_decoder(node_feat)
            struc_node_feat = self.struc_node_decoder(node_feat)

            if self.with_edge_feature:
                seq_edge_feat = self.seq_edge_decoder(edge_feat)
                struc_edge_feat = self.struc_edge_decoder(edge_feat)
            else:
                seq_edge_feat, struc_edge_feat = None, None

        elif self.version == 'joint-transformer':
            seq_node_feat = self.seq_decoder(
                src = node_feat,
                padding_mask = ~mask.to(torch.bool)
            )
            struc_node_feat = self.struc_decoder(
                src = node_feat,
                padding_mask = ~mask.to(torch.bool)
            )
            
            seq_edge_feat, struc_edge_feat = None, None

        elif self.version == 'joint-tab':
            seq_node_feat, seq_edge_feat = self.seq_decoder(
                node_feat, edge_feat, mask
            )
            struc_node_feat, struc_edge_feat = self.struc_decoder(
                node_feat, edge_feat, mask
            )

        return seq_node_feat, struc_node_feat, seq_edge_feat, struc_edge_feat


###############################################################################
# Sampler
###############################################################################

class Sampler(nn.Module):
    """Upsampler and downsampler.

    downsampling: L_out = (L_in + 2*padding - kernel_size) / stride + 1
        e.g. for default setting: L_in = 200, kernel_size = 3, 
             stride = 2, padding = 1, then L_out = 100.
    upsampling: L_out = (L_in - 1)*stride - 2*padding + kernel_size
    """

    def __init__(self, 
        feature_dim, 
        version='share_kernel', 
        k_size=3, 
        layer_num=1, 
        act_fn=nn.ReLU(), 
        padding=1, 
        with_coor = False, 
        coor_aggregate='last', 
        with_edge_feature = True
    ):
        super(Sampler, self).__init__()

        ###################### for feature sampling ##########################
        self.version = version
        self.padding = padding
        self.with_edge_feature = with_edge_feature

        def cnn_construct(name, padding):
            return ConvLayers(
                name,
                feature_dim,
                feature_dim,
                kernel_size=k_size,
                stride=k_size - 1,
                padding=padding,
                layer_num=layer_num,
                act_fn=act_fn
            )

        if self.version == 'share_kernel':
            ###### for resi-wise peatures ######
            ### downsampler
            self.node_feat_downsampler = cnn_construct('down1d', padding)
            ### upsampler
            self.node_feat_upsampler = cnn_construct('up1d', 0)

            ###### for pair-wise peatures ######
            if with_edge_feature: 
                self.edge_feat_downsampler = cnn_construct('down2d', padding)
                self.edge_feat_upsampler = cnn_construct('up2d', 0)

        elif self.version == 'different_kernel':
            ###### for resi-wise peatures ######
            ### downsampler
            self.seq_node_feat_downsampler = cnn_construct('down1d', padding)
            self.struc_node_feat_downsampler = cnn_construct('down1d', padding)
            ### upsampler
            self.seq_node_feat_upsampler = cnn_construct('up1d', 0)
            self.struc_node_feat_upsampler = cnn_construct('up1d', 0)
            
            ###### for pair-wise peatures ######
            if with_edge_feature: 
                self.seq_edge_feat_downsampler = cnn_construct('down2d', padding)
                self.struc_edge_feat_downsampler = cnn_construct('down2d', padding)
                self.seq_edge_feat_upsampler = cnn_construct('up2d', 0)
                self.struc_edge_feat_upsampler = cnn_construct('up2d', 0)

        else:
            raise TypeError(
                'Error! No sampling version named %s!' % self.version
            )

        ################################## for coordinates sampling ############################
        self.with_coor = with_coor
        if self.with_coor:
            self.coor_sampler = ConvLayers('coor', 3, 3,
                kernel_size=k_size, stride=k_size - 1, padding=padding,
                layer_num=layer_num, coor_aggregate=coor_aggregate
            )

    def downsample(self, 
        seq_feat, 
        struc_feat, 
        seq_pair_feat = None, 
        struc_pair_feat = None, 
        alpha_coor = None
    ):
        # seq_feat, (B, L_max, *)
        # seq_pair_feat, (B, L_max, L_max, *)
        # struc_feat, (B, L_max, *)
        # struc_pair_feat, (B, L_max, L_max, *)
        # alpha_coor, ((B, L_max, 3)

        ###### feature downsampling ######
        if self.version == 'share_kernel':
            # node_feat: (B, L_hidden, *); input for cov1d: (B, *, L_max)
            seq_feat = self.node_feat_downsampler(
                seq_feat.transpose(1, 2)
            ).transpose(1, 2)
            struc_feat = self.node_feat_downsampler(
                struc_feat.transpose(1, 2)
            ).transpose(1, 2)

            # edge_feat: (B, L_hidden, L_hidden, *); input for cov2d: (B, *, L_max, L_max)
            if self.with_edge_feature:
                seq_pair_feat = rearrange(seq_pair_feat, 'b c h w -> b w h c')
                seq_pair_feat = self.edge_feat_downsampler(seq_pair_feat)
                seq_pair_feat = rearrange(seq_pair_feat, 'b w h c -> b c h w')
                struc_pair_feat = rearrange(struc_pair_feat, 'b c h w -> b w h c')
                struc_pair_feat = self.edge_feat_downsampler(struc_pair_feat)
                struc_pair_feat = rearrange(struc_pair_feat, 'b w h c -> b c h w')

        elif self.version == 'different_kernel':
            seq_feat = self.seq_node_feat_downsampler(
                seq_feat.transpose(1, 2)
            ).transpose(1, 2)
            struc_feat = self.struc_node_feat_downsampler(
                struc_feat.transpose(1, 2)
            ).transpose(1, 2)

            if self.with_edge_feature:
                seq_pair_feat = rearrange(seq_pair_feat, 'b c h w -> b w h c')
                seq_pair_feat = self.seq_edge_feat_downsampler(seq_pair_feat)
                seq_pair_feat = rearrange(seq_pair_feat, 'b w h c -> b c h w')
                struc_pair_feat = rearrange(struc_pair_feat, 'b c h w -> b w h c')
                struc_pair_feat = self.struc_edge_feat_downsampler(struc_pair_feat)
                struc_pair_feat = rearrange(struc_pair_feat, 'b w h c -> b c h w')

        else:
            raise TypeError(
                'Error! No sampling version named %s!' % self.version)

        ###### coordinates downsampling ######
        if self.with_coor:
            alpha_coor = self.coor_sampler(
                alpha_coor.transpose(1, 2)
            ).transpose(1, 2)
            return seq_feat, seq_pair_feat, struc_feat, struc_pair_feat, alpha_coor

        else:
            return seq_feat, struc_feat, seq_pair_feat, struc_pair_feat


    def upsample(self, 
        seq_feat, 
        struc_feat, 
        seq_pair_feat = None, 
        struc_pair_feat = None, 
        L_max = None
    ):
        # seq_feat, (B, m, *)
        # seq_pair_feat, (B, m, m, *)
        # struc_feat, (B, m, *)
        # struc_pair_feat, (B, m, m, *)

        if self.version == 'share_kernel':
            # node_feat: (B, L, *); input for cov1d: (B, *, L_hidden)
            # L is close to L_max, the difference is due to the padding
            seq_feat = self.node_feat_upsampler(
                seq_feat.transpose(1, 2)
            ).transpose(1, 2)
            struc_feat = self.node_feat_upsampler(
                struc_feat.transpose(1, 2)
            ).transpose(1, 2)

            # edge_feat: (B, L, L, *); input for cov2d: (B, *, L_hidden, L_hidden)
            if self.with_edge_feature:
                seq_pair_feat = rearrange(seq_pair_feat, 'b c h w -> b w h c')
                seq_pair_feat = self.edge_feat_upsampler(seq_pair_feat)
                seq_pair_feat = rearrange(seq_pair_feat, 'b w h c -> b c h w')
                struc_pair_feat = rearrange(struc_pair_feat, 'b c h w -> b w h c')
                struc_pair_feat = self.edge_feat_upsampler(struc_pair_feat)
                struc_pair_feat = rearrange(struc_pair_feat, 'b w h c -> b c h w')

        elif self.version == 'different_kernel':
            seq_feat = self.seq_node_feat_upsampler(
                seq_feat.transpose(1, 2)
            ).transpose(1, 2)
            struc_feat = self.struc_node_feat_upsampler(
                struc_feat.transpose(1, 2)
            ).transpose(1, 2)

            if self.with_edge_feature:
                seq_pair_feat = rearrange(seq_pair_feat, 'b c h w -> b w h c')
                seq_pair_feat = self.seq_edge_feat_upsampler(seq_pair_feat)
                seq_pair_feat = rearrange(seq_pair_feat, 'b w h c -> b c h w')
                struc_pair_feat = rearrange(struc_pair_feat, 'b c h w -> b w h c')
                struc_pair_feat = self.struc_edge_feat_upsampler(struc_pair_feat)
                struc_pair_feat = rearrange(struc_pair_feat, 'b w h c -> b c h w')

        else:
            raise TypeError(
                'Error! No sampling version named %s!' % self.version)

        # discard padding
        if self.padding != 0 and L_max is not None:
            L = seq_feat.shape[1]
            left_padding = max(0, math.ceil((L - L_max) / 2))
            right_padding = max(0, (L - L_max) // 2)
            right_padding = -L if right_padding == 0 else right_padding
            # print(L, left_padding, right_padding)

            seq_feat = seq_feat[
                :, left_padding:-right_padding, :
            ]  # (B, L_max, *)
            struc_feat = struc_feat[
                :, left_padding:-right_padding, :
            ]  # (B, L_max, *)

            if self.with_edge_feature:
                seq_pair_feat = seq_pair_feat[
                    :, left_padding:-right_padding, left_padding:-right_padding, :
                ]  # (B, L_max, L_max, *)
                struc_pair_feat = struc_pair_feat[
                    :, left_padding:-right_padding, left_padding:-right_padding, :
                ]  # (B, L_max, L_max, *)

        return seq_feat, struc_feat, seq_pair_feat, struc_pair_feat


    def singlemodal_downsample(self,
        node_feat,
        pair_feat = None,
    ):
        # node_feat, (B, L_max, *)
        # pair_feat, (B, L_max, L_max, *)

        if self.version != 'share_kernel':
            print('SingleModal downsampling only works for "share_kernel" version!')
            return None, None

        ###### feature downsampling ######
        node_feat = self.node_feat_downsampler(
            node_feat.transpose(1, 2)
        ).transpose(1, 2)

        # edge_feat: (B, L_hidden, L_hidden, *); input for cov2d: (B, *, L_max, L_max)
        if self.with_edge_feature:
            pair_feat = rearrange(pair_feat, 'b c h w -> b w h c')
            pair_feat = self.edge_feat_downsampler(pair_feat)
            pair_feat = rearrange(pair_feat, 'b w h c -> b c h w')

        return node_feat, pair_feat


    def singlemodal_upsample(self,
        node_feat,
        pair_feat = None,
        L_max = None
    ):
        # node_feat, (B, L_max, *)
        # pair_feat, (B, L_max, L_max, *)

        if self.version != 'share_kernel':
            print('SingleModal upsampling only works for "share_kernel" version!')
            return None, None

        ###### feature downsampling ######
        node_feat = self.node_feat_upsampler(
            node_feat.transpose(1, 2)
        ).transpose(1, 2)

        # edge_feat: (B, L_hidden, L_hidden, *); input for cov2d: (B, *, L_max, L_max)
        if self.with_edge_feature:
            pair_feat = rearrange(pair_feat, 'b c h w -> b w h c')
            pair_feat = self.edge_feat_upsampler(pair_feat)
            pair_feat = rearrange(pair_feat, 'b w h c -> b c h w')

        # discard padding
        if self.padding != 0 and L_max is not None:
            L = node_feat.shape[1]
            left_padding = max(0, math.ceil((L - L_max) / 2))
            right_padding = max(0, (L - L_max) // 2)
            right_padding = -L if right_padding == 0 else right_padding # avoid -0
              
            node_feat = node_feat[
                :, left_padding:-right_padding, :
            ]  # (B, L_max, *)
            if self.with_edge_feature:
                pair_feat = pair_feat[
                    :, left_padding:-right_padding, left_padding:-right_padding, :
                ]  # (B, L_max, L_max, *)

        return node_feat, pair_feat

###############################################################################
# Distance Mat Predictor
###############################################################################

class DistPredictor(nn.Module):
    """Predictor for the distance mat."""

    def __init__(self,
        in_resi_features:int,
        in_pair_features:int,
        num_heads:int=4,
        num_blocks:int=3,
        dropout:float=0.0
    ):
        super(DistPredictor, self).__init__()

        self.blocks = nn.ModuleList(
            [
                TriangularSelfAttentionBlock(
                    sequence_state_dim = in_resi_features,
                    pairwise_state_dim = in_pair_features,
                    sequence_head_width = in_resi_features // num_heads,
                    pairwise_head_width = in_pair_features // num_heads,
                    dropout = dropout,
                )
                for i in range(num_blocks)
            ]
        )
        self.out = nn.Linear(in_pair_features, 1)

    def forward(self, node_feat, edge_feat, mask = None):
        """predict the distance matrices and neighbors.

        Args:
            node_feat: node features; (B, L)
            edge_feat: 
            mask: 1 for valid residues and 0 for others; (B, L_max)

        Returns:
            dist_mat: distance matrix; (B, L, L)
        """
        for block in self.blocks:
            node_feat, edge_feat = block(node_feat, edge_feat, mask)
        edge_feat = F.relu(edge_feat)  # (B, L, L, mpnn_dim)
        dist_mat = self.out(edge_feat) # (B, L, L, 1)
        dist_mat = dist_mat.squeeze(-1)  # (B, L, L)
        dist_mat = (dist_mat + dist_mat.transpose(1,2)) / 2

        return dist_mat


###############################################################################
# for Diffusion Models
###############################################################################

##################### for noising and denoising ###############################

class DiffusionTransition(nn.Module):
    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps = num_steps, s = s)


    def add_noise_singleStep(self, node_feat, t, mask = None, pair_feat = None, pair_mask = None):
        """Add the noise from t-1 to t.

        q(x^t_j | x^(t-1)_j) = N(x^t_j | sqrt(1 - beta^t) * x^(t-1)_j, beta^t * I)

        Args:
            node_feat: node-wise feature, (N, L, dim).
            t: (N,).
            mask: 1 for valid position and 0 for padding, (N, L).
            pair_feat: pair-wise feature, (N, L, L, dim).
            pair_mask: True for valid position and False for padding, (N, L, L).
        """
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(1 - beta).view(-1, 1, 1)
        c1 = torch.sqrt(beta).view(-1, 1, 1)

        err_node = torch.randn_like(node_feat)  # added noise
        if mask is not None:
            err_node[mask == 0] = 0
        node_noisy = c0 * node_feat + c1 * err_node    # noised feature 

        if pair_feat is not None:
            c0 = c0.view(-1, 1, 1, 1)
            c1 = c1.view(-1, 1, 1, 1)
            err_pair = torch.randn_like(pair_feat)
            if pair_mask is not None:
                err_pair[pair_mask == 0] = 0
            pair_noisy = c0 * pair_feat + c1 * err_pair
        else:
            err_pair = None
            pair_noisy = None

        return node_noisy, err_node, pair_noisy, err_pair


    def add_noise(self, node_feat, t, mask = None, pair_feat = None, pair_mask = None):
        """Directly add noise from 0 to t.

        q(x^t_j | x^0_j) = N(x^t_j | sqrt(alpha_bar^t) * x^0_j, (1-alpha_bar^t) * I)

        Args:
            node_feat: node-wise feature, (N, L, dim).
            t: (N,).
            mask: 1 for valid position and 0 for padding, (N, L).
            pair_feat: pair-wise feature, (N, L, L, dim).
            pair_mask: True for valid position and False for padding, (N, L, L).
        """
        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        err_node = torch.randn_like(node_feat)  # added noise
        if mask is not None:
            err_node[mask == 0] = 0
        node_noisy = c0 * node_feat + c1 * err_node    # noised feature 

        if pair_feat is not None:
            c0 = c0.view(-1, 1, 1, 1)
            c1 = c1.view(-1, 1, 1, 1)
            err_pair = torch.randn_like(pair_feat)
            if pair_mask is not None:
                err_pair[pair_mask == 0] = 0
            pair_noisy = c0 * pair_feat + c1 * err_pair
        else:
            err_pair = None
            pair_noisy = None

        return node_noisy, err_node, pair_noisy, err_pair


    def denoise(self, 
        node_t, eps_node, t, 
        pair_t=None, eps_pair=None,
        with_Wiener = True    
    ):
        """Denoise from t to t-1.

        p(x^(t-1)_j | x^t_j) = N(x^(t-1)_j | mu(x^t_j, t), sigma^t * I)
        mu(x^t_j, t) = 1 / sqrt(alpha_t) * ( x_t - (beta_t / sqrt(1 - alpha_bar^t)) * epislon(x_t, t))

        Args:
            node_t: node feature at t; (N, L, dim).
            eps_node: predicted node-wise noise; (N, L, dim).
            pair_t: pair feature at t; (N, L, L, dim).
            eps_pair: predicted pair-wise noise; (N, L, L, dim).
            t: (N,).
            mask: 1 for valid position and 0 for padding, (N, L).
        """
        ################ scheduler ##########################
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t]

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(-1, 1, 1)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(-1, 1, 1)

        ################ denoise #############################
        node_next = c0 * (node_t - c1 * eps_node)
        if with_Wiener:
            node_z = torch.where(
                (t > 1)[:, None, None].expand_as(node_t),
                torch.randn_like(node_t),
                torch.zeros_like(node_t),
            )
            node_next += sigma.view(-1, 1, 1) * node_z

        if pair_t is not None:
            c0 = c0.view(-1, 1, 1, 1)
            c1 = c1.view(-1, 1, 1, 1)
            pair_next = c0 * (pair_t - c1 * eps_pair) 
            if with_Wiener:
                pair_z = torch.where(
                    (t > 1)[:, None, None, None].expand_as(pair_t),
                    torch.randn_like(pair_t),
                    torch.zeros_like(pair_t),
                )
                pair_next += sigma.view(-1, 1, 1, 1) * pair_z
        else:
            pair_next = None

        return node_next, pair_next


################################## epsilon network #############################

class EpsilonNet(nn.Module):
    def __init__(self,
        res_feat_dim: int,
        pair_feat_dim: int = None,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.,
        max_length: int = 200,
        architecture: str = 'transformer',
        with_posi_embedding: bool = True,
        ### for UNet
        model_channels: int = 320,
        attention_resolutions: tuple = (4, 2, 1),
        channel_mult: tuple = (1, 2, 4, 4)
    ):
        super(EpsilonNet, self).__init__()

        self.architecture = architecture
        self.node_dim_map = nn.Linear(
            res_feat_dim + 3, res_feat_dim
        )

        ############################# Transformer-encoder ######################
        if self.architecture == 'transformer':
            self.pred_net = TransformerEncoder(
                d_model = res_feat_dim,
                out_features = res_feat_dim,
                num_heads = num_heads,
                num_layers = num_layers,
                dropout = dropout,
                with_embedding = False,
                with_posi_embedding = with_posi_embedding,
                max_length = max_length
            )

        ############################# Transformer-decoder ######################
        elif self.architecture == 'transformer-decoder':
            self.pred_net = TransformerDecoder(
                d_model = res_feat_dim,
                out_features = res_feat_dim,
                num_heads = num_heads,
                num_layers = num_layers,
                dropout = dropout,
                with_embedding = False,
                with_posi_embedding = with_posi_embedding,
                max_length = max_length
            )

        ############################### UNet ###################################
        elif self.architecture == 'unet':
            self.pred_net = UNetModel(
                image_size = max_length, 
                in_channels = res_feat_dim, 
                model_channels = model_channels, 
                out_channels = res_feat_dim, 
                num_res_blocks = num_layers, 
                attention_resolutions = attention_resolutions, 
                num_heads = num_heads,
                channel_mult = channel_mult, 
                dims = 1
            )

        ############################### TAB ####################################
        elif self.architecture == 'tab':
            self.pair_dim_map = nn.Linear(
                pair_feat_dim + 3, pair_feat_dim
            )
            self.pred_net = TriangularSelfAttentionNetwork(
                in_resi_features = res_feat_dim,
                in_pair_features = pair_feat_dim,
                out_resi_features = res_feat_dim,
                out_pair_features = pair_feat_dim,
                num_heads = num_heads,
                num_blocks = num_layers,
                dropout = dropout
            )

        else:
            raise NameError(
                'Error! Architecture %s is not supported!' % self.architecture
            )

    def forward(self,
        node_feat,
        beta,
        memory:torch.Tensor=None,
        mask:torch.Tensor=None,
        pair_feat:torch.Tensor=None,
        timesteps:torch.Tensor=None,
    ):
        N, L, _ = node_feat.shape

        t_embed = torch.stack(
            [beta, torch.sin(beta), torch.cos(beta)],
            dim=-1
        )[:, None, :].expand(N, L, 3)
        node_feat = torch.cat([node_feat, t_embed], dim=-1)
        node_feat = self.node_dim_map(node_feat)
 
        if self.architecture == 'transformer':
            node_feat = self.pred_net(
                src = node_feat,
                padding_mask = ~mask.to(torch.bool)
            )

        elif self.architecture == 'transformer-decoder':
            node_feat = self.pred_net(
                tgt = node_feat,
                memory = memory,
                padding_mask = ~mask.to(torch.bool)
            )

        elif self.architecture == 'unet':
            if mask is not None:
                node_feat[mask == 0] = 0
            node_feat = self.pred_net(
                x = node_feat.transpose(1,2),
                timesteps = timesteps,
            ).transpose(1,2)

        elif self.architecture == 'tab':
            t_embed_pair = torch.stack(
                [beta, torch.sin(beta), torch.cos(beta)],
                dim=-1
            )[:, None, None, :].expand(N, L, L, 3)
            pair_feat = torch.cat([pair_feat, t_embed_pair], dim=-1)
            pair_feat = self.pair_dim_map(pair_feat)
            node_feat, pair_feat = self.pred_net(node_feat, pair_feat, mask)

        return node_feat, pair_feat

