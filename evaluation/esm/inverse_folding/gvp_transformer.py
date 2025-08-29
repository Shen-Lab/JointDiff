# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from scipy.spatial import transform

from esm.data import Alphabet

from .features import DihedralFeatures
from .gvp_encoder import GVPEncoder
from .gvp_utils import unflatten_graph
from .gvp_transformer_encoder import GVPTransformerEncoder
from .transformer_decoder import TransformerDecoder
from .util import rotate, CoordBatchConverter 


class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )
        decoder_embed_tokens = self.build_embedding(
            args, alphabet, args.decoder_embed_dim, 
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        decoder = self.build_decoder(args, alphabet, decoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        logits, extra = self.decoder(
            prev_output_tokens,
            enc = encoder_out["encoder_out"][0],
            padding_mask = encoder_out["encoder_padding_mask"],
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra
    
    def sample(self, coords, partial_seq=None, temperature=1.0, confidence=None, device=None):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )
      
        # Start with prepend token
        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int)
        sampled_tokens[0, 0] = self.decoder.dictionary.get_idx('<cath>')
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i+1] = self.decoder.dictionary.get_idx(c)
            
        # Save incremental states for faster sampling
        incremental_state = dict()
        
        # Run encoder only once
        encoder_out = self.encoder(batch_coords, padding_mask, confidence)
        
        # Make sure all tensors are on the same device if a GPU is present
        if device:
            sampled_tokens = sampled_tokens.to(device)
        
        # Decode one token at a time
        for i in range(1, L+1):
            logits, _ = self.decoder(
                sampled_tokens[:, :i], 
                enc = encoder_out["encoder_out"][0],
                padding_mask = encoder_out["encoder_padding_mask"][0],
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            if sampled_tokens[0, i] == mask_idx:
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        sampled_seq = sampled_tokens[0, 1:]
        
        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])

    ########################################################################
    # encoder and decoder for autoencoder and latent diffusion (by SZ)
    ########################################################################

    def struc_encoder(
        self,
        coords,
        padding_mask,
        confidence = None,
        protein_size: torch.Tensor = None,
        return_all_hiddens: bool = False,
    ):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: backbone coordinates; (B, L_max, 3, 3)
            padding_mask: 1 for valid token and 0 for padding; (B, L_max)
            protein_size: 1D tensor for protein size; (B,)

        Returns:
            encoder_out['encoder_out']: (L_max, B, dim)
            encoder_out['encoder_padding_mask']: (B, L_max)
        """
        ################## data preprocess ##########################

        ### protein size
        B = padding_mask.shape[0]
        if protein_size is None:
            protein_size = padding_mask.sum(dim = -1).int()  # (B,)

        ### add inf to the coordinates
        coords = F.pad(coords, (0,0,0,0,1,1), 'constant', 0)
        coords[:,0] = torch.inf
        coords[torch.arange(B), protein_size + 1] = torch.inf  

        ### padding and confidence
        padding_mask = F.pad(padding_mask, (1,1), 'constant', 0)
        if confidence is None:
            confidence = 1. * padding_mask
        padding_mask[:,0] = 1
        padding_mask[torch.arange(B), protein_size + 1] = 1
        padding_mask = ~padding_mask.to(dtype=torch.bool)  # False for valid token

        ################## encoding ##########################

        encoder_out = self.encoder(
            coords, 
            padding_mask, 
            confidence,
            return_all_hiddens=return_all_hiddens
        )
        return encoder_out['encoder_out'][0], encoder_out['encoder_padding_mask'][0]
   
 
    def seq_decoder(self, 
        encoder_emb, 
        prev_output_tokens = None,
        padding_mask = None,
        return_all_hiddens: bool = False,
        features_only: bool = False,
        partial_seq = None, 
        temperature = 1.0,
        mode = 'eval',
        return_string = False,
        protein_size: torch.Tensor = None, 
    ):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            encoder_emb: structure embedding; (L+2, B, esm_dim)
            prev_output_tokens: ground truth sequence; (B, L)
            padding_mask: 1 for valid and 0 for padding; (B, L+2) 
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
        """
        ### protein size
        L_pad, B, _ = encoder_emb.shape  # L_pad = L_max + 2
        if protein_size is None and padding_mask is not None:
            protein_size = padding_mask.sum(dim = -1).int()  # (B,)
        elif protein_size is None:
            protein_size = (torch.ones(B) * (L_pad - 2)).int()

        ### padding
        if padding_mask.shape[1] != L_pad:
            padding_mask = F.pad(padding_mask, (1,1), 'constant', 0)
            padding_mask[:,0] = 1
            padding_mask[torch.arange(B), protein_size + 1] = 1
        padding_mask = ~padding_mask.to(dtype=torch.bool)  # False for valid token

        ###### training with teacher-forcing ######

        if mode == 'train':
            logits, extra = self.decoder(
                prev_output_tokens,
                enc = encoder_emb,
                padding_mask = padding_mask,
                features_only=features_only,
                return_all_hiddens=return_all_hiddens,
            )
            return logits, extra

        ###### autoregressive inference
        else:
            L_max = int(max(protein_size))

            # Start with prepend token
            mask_idx = self.decoder.dictionary.get_idx('<mask>')
            sampled_tokens = torch.full((B, L_pad - 1), mask_idx, dtype=int)
            sampled_tokens[:, 0] = self.decoder.dictionary.get_idx('<cath>')
            sampled_tokens = sampled_tokens.to(encoder_emb.device)
                
            # Save incremental states for faster sampling
            incremental_state = dict()
            
            # Decode one token at a time
            for i in range(1, L_max+1):
                logits, _ = self.decoder(
                    sampled_tokens[:, :i], 
                    enc = encoder_emb,
                    padding_mask = padding_mask,
                    incremental_state=incremental_state,
                ) # (B, dim, 1)
                # logits = logits[0].transpose(0, 1)  
                logits = logits[:, :, -1] # (B, dim)
                logits /= temperature  # (B, dim)
                probs = F.softmax(logits, dim=-1)  # (B, dim)
                # if sampled_tokens[0, i] == mask_idx:
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)

            sampled_seq = sampled_tokens[:, 1:]
            if padding_mask is not None:
                sampled_seq[padding_mask[:,1:-1]] = mask_idx
            
            if return_string:
                # Convert back to string via lookup
                sampled_seq = [
                    ''.join([self.decoder.dictionary.get_tok(a) for a in s]) 
                    for s in sampled_seq
                ]

            return sampled_seq

