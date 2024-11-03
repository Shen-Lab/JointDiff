import typing as T
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import ml_collections
from typing import Dict, Optional, Tuple

from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock

###############################################################################
# MLP
###############################################################################

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Define the input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # Forward pass through the input layer
        x = F.relu(self.input_layer(x))

        # Forward pass through the hidden layers
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        # Forward pass through the output layer
        x = self.output_layer(x)

        return x


###############################################################################
# GAT (wip)
###############################################################################

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_feature_dim, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight matrices for each attention head
        self.W = nn.Parameter(torch.Tensor(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(num_heads, 2*out_features + edge_feature_dim))

        # Initialize biases
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Initialize edge embedding weight
        self.edge_embedding_weight = nn.Parameter(torch.Tensor(out_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        nn.init.zeros_(self.bias.data)
        nn.init.xavier_uniform_(self.edge_embedding_weight.data)

    def forward(self, x, edge_index, edge_attr):
        # x: Node features (N x in_features)
        # edge_index: Edge indices (2 x E)
        # edge_attr: Edge features (E x edge_feature_dim)

        N, _ = x.size()
        E = edge_index.size(1)

        # Compute attention coefficients
        x_transformed = torch.matmul(x.unsqueeze(1), self.W)
        attention_weights = F.leaky_relu(
            torch.matmul(
                torch.cat(
                    [x_transformed[edge_index[0]], x_transformed[edge_index[1]], edge_attr],
                    dim=-1
                ),
                self.a
            ),
            negative_slope=0.2
        )
        attention_weights = F.softmax(attention_weights, dim=1)

        # Compute edge embeddings
        edge_embeddings = torch.matmul(edge_attr, self.edge_embedding_weight)

        # Aggregate neighborhood features
        aggregated_messages = torch.matmul(
            attention_weights.unsqueeze(-1),
            edge_embeddings.unsqueeze(1)
        ).squeeze(1)

        # Apply activation function and bias
        aggregated_messages = F.relu(aggregated_messages + self.bias)

        return aggregated_messages, edge_embeddings


class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_feature_dim, num_heads, num_layers):
        super(GraphAttentionNetwork, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()

        # Add multiple GraphAttentionLayers
        for _ in range(num_layers):
            self.gat_layers.append(GraphAttentionLayer(input_dim, hidden_dim, edge_feature_dim, num_heads))
            input_dim = hidden_dim  # Update input_dim for subsequent layers

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        for gat_layer in self.gat_layers:
            x, edge_attr = gat_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        x = self.fc(x)
        return x, edge_attr


###############################################################################
# CNN
###############################################################################

class ConvLayers(nn.Module):
    def __init__(self,
        version,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=1,
        layer_num=1,
        act_fn=nn.ReLU(),
        coor_aggregate='last'
    ):
        """CNN layer for k-mer sampling."""

        super(ConvLayers, self).__init__()

        conv_module = {'down1d': nn.Conv1d,
                       'down2d': nn.Conv2d,
                       'up1d': nn.ConvTranspose1d,
                       'up2d': nn.ConvTranspose2d,
                       'coor': nn.Conv1d
        }[version]

        bias = False if version == 'coor' else True
        self.version = version
        self.coor_aggregate = coor_aggregate
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        ###### for coordinates aggregating ######
        if version == 'coor':
            if padding >= kernel_size:
                raise Exception(
                    'For coordinates aggregating the padding must be smaller \
                     than the kernel size. Got %d and %d.' % (
                        padding,
                        kernel_size
                    )
                )

            weight = torch.zeros(out_channels, in_channels, kernel_size)
            if coor_aggregate == 'last':
                for i in range(out_channels):
                    weight[i, i, -1] = 1
            elif coor_aggregate == 'first':
                for i in range(out_channels):
                    weight[i, i, 0] = 1
            elif coor_aggregate == 'mean':
                for i in range(out_channels):
                    weight[i, i, :] = 1 / kernel_size
            else:
                raise NameError(
                    'Error! No coordinates aggregating method named %s!' % coor_aggregate
                )
            weight = nn.Parameter(weight, requires_grad=False)

        ###### layers ######
        self.conv_layer = []
        for i in range(layer_num):
            self.conv_layer.append(conv_module(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias)
            )
            if version == 'coor':
                self.conv_layer[-1].weight = weight
            elif i != layer_num - 1:  # add activation layer
                self.conv_layer.append(act_fn)

        self.conv_layer = nn.Sequential(*self.conv_layer)

    def forward(self, feat_input):
        ### feat_input: (N, feat_dim, L_max)
        L = feat_input.shape[-1]
        feat_input = self.conv_layer(feat_input)

        if self.version == 'coor' and self.padding != 0:
            # for left padding
            left_count = self.kernel_size - self.padding
            feat_input[:, :, 0] = feat_input[:, :, 0] * \
                (self.kernel_size / left_count)
            # for right padding
            L += 2 * self.padding
            L_covered = self.kernel_size + \
                ((L - self.kernel_size) // self.stride) * self.stride
            right_padding = self.padding - (L - L_covered)
            if right_padding:
                right_count = self.kernel_size - right_padding
                feat_input[:, :, -1] = feat_input[:, :, -1] * \
                    self.kernel_size / right_count

        return feat_input


###############################################################################
# Transformer Encoder
###############################################################################

class PositionEmbedding(nn.Module):
    def __init__(self, dim, max_length):
        super(PositionEmbedding, self).__init__()

        theta = np.array([
            [p / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for p in range(max_length)
        ])
        self.embedding = nn.Embedding(max_length, dim)
        self.embedding.weight.detach_()
        self.embedding.weight.requires_grad = False
        self.embedding.weight[:, 0::2] = torch.FloatTensor(np.sin(theta[:, 0::2]))
        self.embedding.weight[:, 1::2] = torch.FloatTensor(np.cos(theta[:, 1::2]))

    def forward(self, seq):
        """
        Args:
            seq: input sequence embedding; (N,L,dim)    
        """
        seq_length = seq.shape[1]
        position_ids = torch.arange(
            seq_length, 
            dtype=torch.long, 
            device=seq.device
        ) # (L,)
        position_ids = position_ids.unsqueeze(0).expand_as(seq[:,:,0])
        posi_emb = self.embedding(position_ids)  # (N,L,dim)

        return seq + posi_emb 


class TransformerEncoder(nn.Module):
    def __init__(self,
        d_model:int,
        dim_feedforward:int=2048,
        out_features:int=None,
        num_heads:int=4,
        num_layers:int=6,
        dropout:float=0.0,
        batch_first:bool=True,
        with_embedding=False,
        with_posi_embedding=False,
        vocab_size = 22,
        max_length = 500,
    ):
        super(TransformerEncoder, self).__init__()

        self.with_embedding = with_embedding
        self.with_posi_embedding = with_posi_embedding

        ###### embedding layers ######
        if self.with_embedding:
            self.embedding = nn.Embedding(vocab_size, d_model) 

        if self.with_posi_embedding:
            self.posi_embedding = PositionEmbedding(d_model, max_length) 

        ###### main encoder ######
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            dim_feedforward = dim_feedforward,
            nhead = num_heads,
            dropout = dropout,
            batch_first = batch_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers = num_layers,
        )
        ###### output layer ######
        if out_features is not None:
            self.out_layer = nn.Linear(d_model, out_features)
        else:
            self.out_layer = None

    def forward(self, src, src_mask=None, padding_mask=None):
        """
        Args:
            src: input sequence or representations; (N,L) or (N,L,dim)
            src_mask: the additive mask for the src sequence; 
                (L,L) or (N⋅num_heads,L,L).
            padding_mask: the Tensor mask for src keys per batch;
                (N,L)

        Return:
            out: (N,L,dim)
        """
        if self.with_embedding:
            # (N, L) to (N, L, dim)
            src = self.embedding(src) 

        if self.with_posi_embedding:
            src = self.posi_embedding(src)  # (N, L, dim)

        #print('Transformer in:', src.shape, padding_mask.shape, padding_mask[:,0], src_mask)
        out = self.encoder(
            src = src,
            mask = src_mask,
            src_key_padding_mask = padding_mask,
        ) 

        ### for the issue in https://discuss.pytorch.org/t/\
        ### transformerencoder-output-size-doesnt-match-input-size/181946
        if out.shape[1] != src.shape[1]:
            pad_length = src.shape[1] - out.shape[1]
            out = F.pad(out, (0,0,0,pad_length,0,0), 'constant', 0.)
        #print('Transformer out:', out.shape)

        if self.out_layer is not None:
            out = self.out_layer(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self,
        d_model:int,
        dim_feedforward:int=2048,
        out_features:int=None,
        num_heads:int=4,
        num_layers:int=6,
        dropout:float=0.0,
        batch_first:bool=True,
        with_embedding=False,
        with_posi_embedding=False,
        vocab_size = 22,
        max_length = 500,
    ):
        super(TransformerDecoder, self).__init__()

        self.with_embedding = with_embedding
        self.with_posi_embedding = with_posi_embedding

        ###### embedding layers ######
        if self.with_embedding:
            self.embedding = nn.Embedding(vocab_size, d_model)

        if self.with_posi_embedding:
            self.posi_embedding = PositionEmbedding(d_model, max_length)

        ###### main decoder ######
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            dim_feedforward = dim_feedforward,
            nhead = num_heads,
            dropout = dropout,
            batch_first = batch_first
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers = num_layers,
        )
        ###### output layer ######
        if out_features is not None:
            self.out_layer = nn.Linear(d_model, out_features)
        else:
            self.out_layer = None

    def forward(self, tgt, memory, tgt_mask=None, padding_mask=None):
        """
        Args:
            tgt: input sequence or representations; (N,L) or (N,L,dim)
            memory: contect information; (N,L_m,dim)
            tgt_mask: the additive mask for the tgt sequence; 
                (L,L) or (N⋅num_heads,L,L).
            padding_mask: the Tensor mask for tgt keys per batch;
                (N,L)

        Return:
            out: (N,L,dim)
        """
        if self.with_embedding:
            # (N, L) to (N, L, dim)
            tgt = self.embedding(tgt)

        if self.with_posi_embedding:
            tgt = self.posi_embedding(tgt)  # (N, L, dim)

        out = self.decoder(
            tgt = tgt,
            memory = memory,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = padding_mask,
        )

        if out.shape[1] != tgt.shape[1]:
            pad_length = tgt.shape[1] - out.shape[1]
            out = F.pad(out, (0,0,0,pad_length,0,0), 'constant', 0.)

        if self.out_layer is not None:
            out = self.out_layer(out)
        return out

###############################################################################
# TriangularSelfAttentionNetwork
###############################################################################

class TriangularSelfAttentionNetwork(nn.Module):
    def __init__(self,
        in_resi_features:int,
        in_pair_features:int,
        out_resi_features:int = None,
        out_pair_features:int = None,
        num_heads:int=4,
        num_blocks:int=3,
        dropout:float=0.0
    ):
        super(TriangularSelfAttentionNetwork, self).__init__()

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
        if out_resi_features is not None:
            self.seq_out = nn.Linear(in_resi_features, out_resi_features)
        else:
            self.seq_out = None

        if out_pair_features is not None:
            self.pair_out = nn.Linear(in_pair_features, out_pair_features)
        else:
            self.pair_out = None       

    def forward(self, node_feat, edge_feat, mask = None):
        for block in self.blocks:
            node_feat, edge_feat = block(node_feat, edge_feat, mask)

        if self.seq_out:
            node_feat = F.relu(node_feat)
            node_feat = self.seq_out(node_feat)
        
        if self.pair_out:
            edge_feat = F.relu(edge_feat)
            edge_feat = self.pair_out(edge_feat)

        return node_feat, edge_feat

###############################################################################
# for Diffusion Model
###############################################################################

class VarianceSchedule(nn.Module):
    """Scheduler from diffab."""
    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)  # (num_steps+1,); [0,...,num_steps+1]
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2  # (num_steps+1,)
        alpha_bars = f_t / f_t[0]  # (num_steps+1,)

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])  # (num_steps,)
        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # (num_steps+1,)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)

