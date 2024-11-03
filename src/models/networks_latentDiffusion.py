import typing as T
import math
import random
import numpy as np
#from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
from typing import Dict, Optional, Tuple, Union

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

###### encoder and decoder ######
import esm
from esm.esmfold.v1.esmfold import ESMFold
from models.networks_proteinMPNN import ProteinMPNN, gather_edges
from openfold.config import model_config
from einops import rearrange

###### other modules ######
from models.utils_modules import (
    TriangularSelfAttentionNetwork,
    TransformerEncoder,
)
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from models.modules import (
    mat_outer_prod,
    sequence_transform,
    FeatureProjector,
    JointFeatureProjector,
    Sampler,
    DistPredictor,
    DiffusionTransition,
    EpsilonNet
)
from models.losses import (
    loss_smoothed, strutcure_loss, emb_loss, kld_loss, contrastive_loss
)

###############################################################################
# Utility Functions
###############################################################################

# ordered voxel set of ESMFold
ESM_ALPHABET    = 'ARNDCQEGHILKMFPSTWYVX'
# ordered voxel set of ESM-IF; from '*!**LAGVSERTIDPKQNFYMHWCXBUZO.-***'
ESM_IF_ALPHABET = 'LAGVSERTIDPKQNFYMHWCX'
# ordered voxel set of proteinMPNN
MPNN_ALPHABET   = 'ACDEFGHIKLMNPQRSTVWYX'

ESMIF2MPNN_MAT = sequence_transform(ESM_IF_ALPHABET, MPNN_ALPHABET)
MPNN2ESMIF_MAT = ESMIF2MPNN_MAT.T
ESM2MPNN_MAT = sequence_transform(ESM_ALPHABET, MPNN_ALPHABET)

def aa_transform(
    in_aa, version = 'prob', trans_mat = ESMIF2MPNN_MAT, voxel_size=21
):
    """Transform the sequence following MPNN alphabet.

    Args:
        in_aa: ordinal sequence or probability
        version: 'prob' for probability and 'seq' for ordinial sequence.
    """
    if version != 'seq' and version != 'prob':
        print(
            'Version %s not identified for sequence transformation!' % version
        )
        return None

    ######################### sequence transformation #####################
    if version == 'seq':
        in_aa = F.one_hot(in_aa, num_classes=voxel_size)  # (N,L,21)
        in_aa = torch.matmul(in_aa.float(), transmat.to(in_aa.device))
        in_aa = in_aa.max(dim=-1).indices  # (N, L)

    ######################### probability transformation ####################
    else:
        in_aa = torch.matmul(in_aa, transmat.to(in_aa.device)) # (N, L, 21)

    return in_aa


def esmif_mask_transform(mask):
    """Add 1's to the two ends of the mask."""
    N = mask.shape[0]
    protein_size = mask.sum(-1)  # (N,)
    padding_mask = F.pad(mask, (1,1), 'constant', 0)  # (N, L+2)
    padding_mask[:,0] = 1
    padding_mask[torch.arange(N), protein_size.long() + 1] = 1
    
    return padding_mask


def esmif_remove_ends(feat, protein_length):
    """remove the two ends of the feat.

    Args:
        feat: (N, L_max+2, d)
        protein_length: (N,)
    
    Output:
        feat_out: (N, L_max, d)
    """
    N = feat.shape[0]
    feat = feat[:,1:,:]  # (N, L_max+1, d)
    feat_out = []
    for i in range(N):
        length = int(protein_length[i])
        feat_out.append(torch.cat([
            feat[i][:length], 
            feat[i][length+1:]  # (L_max, d)
        ]))

    return torch.stack(feat_out)


###############################################################################
# DDPM
###############################################################################

class DenoiseDiffusionProbobilisticModel(nn.Module):
    def __init__(self, args):
        """Implementation for DDPM.

        Args:
            res_feat_dim: int,
            pair_feat_dim: int = None,
            num_layers: int = 3,
            num_heads: int = 4,
            dropout: float = 0.,
            max_length: int = 100,
            architecture = 'transformer',
            with_posi_embedding: bool = True,
            num_steps=100,
            s = 0.01
        """
        super(DenoiseDiffusionProbobilisticModel, self).__init__()

        ########################################################################
        # settings
        ########################################################################

        self.args = args
        self.device = args.device

        ######################### architecture #################################

        self.architecture = args.architecture
        self.res_feat_dim = args.res_feat_dim
        self.pair_feat_dim = args.pair_feat_dim
        self.num_steps = args.num_steps

        ############################# normalization ############################
        if not args.__contains__('normalization'):
            args.normalization = None
            args.node_bias = None
            args.node_var = None
            args.pair_bias = None
            args.pair_var = None

        if args.normalization is not None and args.normalization != 'all':
            self.node_bias = nn.parameter.Parameter(
                torch.from_numpy(args.node_bias), requires_grad = False
            ) if args.node_bias is not None else None
            self.node_var = nn.parameter.Parameter(
                torch.from_numpy(args.node_var), requires_grad = False
            ) if args.node_var is not None else None 
            self.pair_bias = nn.parameter.Parameter(
                torch.from_numpy(args.pair_bias), requires_grad = False
            ) if args.pair_bias is not None else None
            self.pair_var = nn.parameter.Parameter(
                torch.from_numpy(args.pair_var), requires_grad = False
            ) if args.pair_var is not None else None
        else:
            self.node_bias = args.node_bias
            self.node_var = args.node_var
            self.pair_bias = args.pair_bias
            self.pair_var = args.pair_var

        ###################### how to train the model ##########################
        if not args.__contains__('train_version'):
            args.train_version = 'noise'
        self.train_version = args.train_version
        # train_version: 
        #     "noise": train a score estimator
        #     "gt": train a groundtruth estimator
        
        ########################################################################
        # Main Network
        ########################################################################

        ###### length embedding ######
        if not args.__contains__('length_embedding'):
            self.length_embedding = None
        else:
            self.length_embedding = args.length_embedding

        ### multi-head attention
        if self.length_embedding == 'mha':
            self.length_emb = nn.Embedding(args.max_length_ori, args.res_feat_dim)
            self.mha_len = nn.MultiheadAttention(
                args.res_feat_dim, 
                args.num_heads,
                dropout = args.dropout,
                batch_first = True
            )
        ### concatenation + linear layer
        elif self.length_embedding == 'linear':
            self.length_emb = nn.Embedding(args.max_length_ori, args.res_feat_dim)
            self.linear_len = nn.Linear(args.res_feat_dim * 2, args.res_feat_dim)
        ### memory (only for Transformer decoder)
        elif self.length_embedding == 'memory':
            self.length_emb = nn.Embedding(args.max_length_ori, args.res_feat_dim)
            if args.architecture != 'transformer-decoder':
                raise ValueError(
                    '"Len-emb-memory" only works for transformer decoder!'
                )
        ### others
        elif self.length_embedding is not None:
            print('Warning! No length embedding module %s!' % self.length_embedding)
            self.length_embedding = None

        ###### reverse net ######
        if not args.__contains__('model_channels'):
            args.model_channels = 320

        self.reverse_net = EpsilonNet(
            res_feat_dim = args.res_feat_dim,
            pair_feat_dim = args.pair_feat_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            dropout = args.dropout,
            max_length = args.max_length,
            architecture = args.architecture,
            with_posi_embedding = args.with_posi_embedding,
            model_channels = args.model_channels,
        )
        self.transition = DiffusionTransition(
            num_steps=args.num_steps, s=args.s
        )

        ###### random masking ######
        if not args.__contains__('with_fragment_type'):
            self.with_fragment_type = False
        else:
            self.with_fragment_type = args.with_fragment_type

        if self.with_fragment_type:
            self.fragment_emb = nn.Embedding(3, args.res_feat_dim, padding_idx = 0)
            self.node_fead_merge = nn.Linear(args.res_feat_dim * 2, args.res_feat_dim)

        ###### loss ######
        if not args.__contains__('micro_loss'):
            self.micro_loss = False
        else:
            self.micro_loss = args.micro_loss

        if self.micro_loss:
            print('Calculate micro loss.')

    ########################################################
    # utility functions
    ########################################################

    def _normalize(self,
        node_feat,
        node_bias:float=None,
        node_var:float=None,
        pair_feat:torch.Tensor=None,
        pair_bias:float=None,
        pair_var:float=None,
        norm_version: str='default',
    ):
        if norm_version == 'default':
            node_bias = self.node_bias
            node_var = self.node_var
            pair_bias = self.pair_bias
            pair_var = self.pair_var

        if node_bias is not None:
            node_feat -= node_bias
        if node_var is not None:
            node_feat /= node_var

        if (pair_feat is not None) and (pair_bias is not None):
            pair_feat -= pair_bias 
        if (pair_feat is not None) and (pair_var is not None):
            pair_feat /= pair_var

        return node_feat, pair_feat


    def _unnormalize(self,
        node_feat:torch.Tensor,
        node_bias:float=None,
        node_var:float=None,
        pair_feat:torch.Tensor=None,
        pair_bias:float=None,
        pair_var:float=None,
        norm_version: str='default',
    ):
        if norm_version == 'default':
            node_bias = self.node_bias
            node_var = self.node_var
            pair_bias = self.pair_bias
            pair_var = self.pair_var

        if node_var is not None:
            node_feat *= node_var
        if node_bias is not None:
            node_feat += node_bias 

        if (pair_feat is not None) and (pair_var is not None):
            pair_feat *= pair_var
        if (pair_feat is not None) and (pair_bias is not None):
            pair_feat += pair_bias 

        return node_feat, pair_feat

    
    def infer_preprocess(self,
        length: Union[int, list] = None,
        batch_size: int=4,
        max_length: int=100,
        device: str='cuda',
    ):
        """prepare the masks for inference."""

        if type(length) == list:
            batch_size = len(length)
            if max_length is None:
                max_length = max(length)

            mask = torch.zeros(batch_size, max_length).int().to(device)
            for i, l in enumerate(length):
                mask[i,:l] = 1
            if self.architecture == 'tab':
                pair_mask = mat_outer_prod(mask.float(), mask.float())
            else:
                pair_mask = None

        else:
            if max_length is None:
                max_length = length
            mask = None
            pair_mask = None

        return mask, pair_mask, batch_size 
   

    def infer_init(self,
        length: Union[int, list] = None,
        batch_size: int=4,
        max_length: int=100,
        node_bias:float=None,
        node_var:float=None,
        pair_bias:float=None,
        pair_var:float=None,
        device: str='cuda',
        norm_version:str='default',
    ):
        """INitialization of the sampling process."""

        ###### repare the masks ######

        mask, pair_mask, batch_size = self.infer_preprocess(
            length, batch_size, max_length, device
        )

        ###### initialization and unnormalization ######
        node_feat = torch.randn(
            batch_size, max_length, self.res_feat_dim
        ).to(device)
        if self.architecture == 'tab':
            pair_feat = torch.randn(
                batch_size, max_length, max_length, self.pair_feat_dim
            ).to(device)
        else:
            pair_feat = None

        ### unnormalize
        node_feat, pair_feat = self._unnormalize(
            node_feat = node_feat,
            node_bias = node_bias,
            node_var = node_var,
            pair_feat = pair_feat,
            pair_bias = pair_bias,
            pair_var = pair_var,
            norm_version = norm_version
        )

        ### masking
        if mask is not None:
            node_feat[mask == 0] = 0
            node_feat = node_feat.detach().cpu()
        if pair_mask is not None:
            pair_feat[pair_mask == 0] = 0
            pair_feat = pair_feat.detach().cpu()

        traj = {self.num_steps: (node_feat, pair_feat)}
       
        return traj, mask, pair_mask

 
    def gt_noise_transfer(self,
        t,
        node_pred = None,
        node_feat = None,
        pair_pred = None,
        pair_feat = None,
    ):
        alpha_bar = self.transition.var_sched.alpha_bars[t]  # (N,) 
        c0 = 1 / (1 - alpha_bar + 1e-8).view(-1, 1, 1)
        c1 = torch.sqrt(alpha_bar).view(-1, 1, 1)

        if node_pred is not None and node_feat is not None:
            node_pred = c0 * (node_feat - c1 * node_pred)

        if pair_pred is not None and pair_feat is not None:
            c0 = c0.view(-1, 1, 1, 1)
            c1 = c1.view(-1, 1, 1, 1)
            pair_pred = c0 * (pair_feat - c1 * pair_pred)

        return node_pred, pair_pred


    ########################################################
    # for training
    ########################################################

    def forward(self,
        node_feat,
        mask:torch.Tensor=None,
        mask_gen:torch.Tensor=None,
        fragment_type:torch.Tensor=None,
        t:torch.Tensor=None,
        node_bias:float=None,
        node_var:float=None,
        pair_feat:torch.Tensor=None,
        pair_mask:torch.Tensor=None,
        pair_bias:float=None,
        pair_var:float=None,
        norm_version:str='default',
        return_pred:bool=False,
        protein_size:torch.Tensor=None,
    ):
        #############################################
        # preprocess 
        #############################################

        N, L_max, node_dim = node_feat.shape
        if pair_feat is not None:
            pair_dim = pair_feat.shape[-1]

        if mask_gen is None:
            mask_gen = mask

        ###### step ######
        if t is None:  # t: None or (N,)
            t = torch.randint(
                0, self.num_steps, (N,), 
                dtype=torch.long, device=node_feat.device
            )
            # t = torch.randint(
            #     1, self.num_steps+1, (N,), 
            #     dtype=torch.long, device=node_feat.device
            # )

        ###### normalize ######
        node_feat, pair_feat = self._normalize(
            node_feat = node_feat,
            node_bias = node_bias,
            node_var = node_var,
            pair_feat = pair_feat,
            pair_bias = pair_bias,
            pair_var = pair_var,
            norm_version = norm_version
        )

        #############################################
        # forward (add noise, 0 to t) 
        #############################################

        if self.with_fragment_type:
            fragment_feat = self.fragment_emb(fragment_type.long())  # (N, L, dim)
            node_feat_in = torch.cat([node_feat, fragment_feat], dim = -1)
            node_feat_in = self.node_fead_merge(node_feat_in)  # (N, L, dim)
        else:
            node_feat_in = node_feat

        (node_noisy,  # noised node feature
         err_node,    # Gaussian noise for the node feature
         pair_noisy,  # noised pair feature
         err_pair,    # Gaussian noise for the pair feature
        )= self.transition.add_noise(
            node_feat = node_feat_in, t = t, mask = mask,
            pair_feat = pair_feat, pair_mask = pair_mask
        )

        ### random masking
        node_noisy = torch.where(
            mask_gen.bool()[:,:,None].expand(node_noisy.shape), 
            node_noisy, node_feat_in
        )

        #############################################
        # reverse (denoise, t to t-1) 
        #############################################

        ###### scheduler ######
        beta = self.transition.var_sched.betas[t]  # (N,)

        ###### length embedding ######
        memory = None
        if self.length_embedding is not None:
            l_emb = self.length_emb(protein_size - 1) # (N, dim)
            l_emb = l_emb.unsqueeze(1) # (N, 1, dim)

            if self.length_embedding == 'mha':
                node_noisy,_ = self.mha_len(
                    query = node_noisy,
                    key = l_emb,
                    value = l_emb
                )
        
            elif self.length_embedding == 'linear':
                l_emb = l_emb.repeat(1, L_max, 1)  # (N, L, dim)
                node_noisy = torch.cat(
                    [node_noisy, l_emb], dim = -1
                )
                node_noisy = self.linear_len(node_noisy)

            elif self.length_embedding == 'memory':
                memory = l_emb

        ###### reverse net ######
        node_pred, pair_pred = self.reverse_net(
            node_feat = node_noisy,
            memory = memory,
            beta = beta,
            mask = mask,
            pair_feat = pair_noisy,
            timesteps = t
        )
        if self.train_version == 'gt':
            node_ref = node_feat 
            pair_ref = pair_feat
        else:
            node_ref = err_node
            pair_ref = err_pair
 
        node_pred = torch.where(
            mask.bool()[:,:,None].expand(node_pred.shape), 
            node_pred, node_ref
        ) 

        #############################################
        # Loss cal 
        #############################################

        out_dict = {'loss': 0.}
        if return_pred:
            out_dict['node_pred'] = node_pred
            out_dict['pair_pred'] = pair_pred

        ###### node-wise loss ######
        if node_pred is not None:
            out_dict['loss_node'] = F.mse_loss(
                node_pred, node_ref, reduction='none'
            ).mean(dim=-1) # (N, L)

            if mask_gen is not None:
                out_dict['loss_node'][mask_gen == 0] = 0

                if self.micro_loss:
                    out_dict['loss_node'] = (
                        out_dict['loss_node'].sum() / (mask_gen.sum() + 1e-8)
                    )  # (N,) 
                else:
                    out_dict['loss_node'] = (
                        out_dict['loss_node'].sum(dim=-1) / (mask_gen.sum(dim=-1) + 1e-8)
                    )  # (N,)
                    out_dict['loss_node'] = out_dict['loss_node'].mean()
   
            else: 
                out_dict['loss_node'] = out_dict['loss_node'].mean()

            out_dict['loss'] += out_dict['loss_node']

        ###### pair-wise loss ######
        if pair_pred is not None:
            out_dict['loss_pair'] = F.mse_loss(
                pair_pred, pair_ref, reduction='none'
            ).sum(dim=-1) / pair_dim  # (N, L, L)

            if pair_mask is not None:
                out_dict['loss_pair'][pair_mask == 0] = 0
                out_dict['loss_pair'] = (
                    out_dict['loss_pair'].sum(dim=(1,2)) / pair_mask.sum(dim=(1,2))
                )  # (N,)
            out_dict['loss_pair'] = out_dict['loss_pair'].mean()
            out_dict['loss'] += out_dict['loss_pair']

        return out_dict


    ########################################################
    # for inference
    ########################################################

    def sample(self,
        length: Union[int, list] = None, 
        batch_size: int=4,
        max_length: int=100,
        node_bias:float=None,
        node_var:float=None,
        pair_bias:float=None,
        pair_var:float=None,
        device: str='cuda',
        norm_version:str='default',
        self_condition:bool=False,
    ):
        """Sampling from scratch."""

        if self_condition and self.train_version != 'gt':
            raise ValueError('Self-conditioning only works for GT estimators!')

        #############################################
        # initialization 
        #############################################

        traj, mask, pair_mask = self.infer_init(
            length, batch_size, max_length
        )

        #############################################
        # reverse diffusion 
        #############################################

        #for t in range(self.num_steps, 0, -1):
        for t in reversed(range(0, self.num_steps)):

            ###### status from the last step ######
            #node_feat, pair_feat = traj[t]
            node_feat, pair_feat = traj[t+1]
            node_feat = node_feat.to(device)
            if pair_feat is not None:
                pair_feat = pair_feat.to(device)

            ###### normalization ######
            node_feat, pair_feat = self._normalize(
                node_feat=node_feat, node_bias=node_bias, node_var=node_var,
                pair_feat=pair_feat, pair_bias=pair_bias, pair_var=pair_var,
                norm_version = norm_version
            )

            ###### reverse step ######
            t_idx = torch.full(
                [node_feat.shape[0], ], fill_value=t, dtype=torch.long, device=device
            )
            beta = self.transition.var_sched.betas[t_idx]  # (N,)

            ###### length embedding ######
            if self.length_embedding is not None:
                length = torch.tensor(length).to(node_feat.device)
                l_emb = self.length_emb(length - 1) # (N, dim)
                l_emb = l_emb.unsqueeze(1) # (N, 1, dim)

                if self.length_embedding == 'mha':
                    node_feat, _ = self.mha_len(
                        query = node_feat,
                        key = l_emb,
                        value = l_emb
                    )

                elif self.length_embedding == 'linear':
                    l_emb = l_emb.repeat(1, node_feat.shape[1], 1)  # (N, L, dim)
                    node_feat = torch.cat(
                        [node_feat, l_emb], dim = -1
                    )
                    node_feat = self.linear_len(node_feat)

            ### noise or gt prediction
            node_pred, pair_pred = self.reverse_net(
                node_feat = node_feat, beta = beta,
                mask = mask, pair_feat = pair_feat,
                timesteps = t_idx,
            )
            # If self.train_version == 'gt', 
            #     node_pred and pair_pred are predicted ground truths;
            # Else,
            #     they are predicted noises.

            if self_condition:
                # based on the coarse prediction, add noise to t-1.
                #if t > 1:
                if t > 0:
                    node_feat, _, pair_feat, _ = self.transition.add_noise(
                        node_feat = node_pred, t = t_idx - 1, mask = mask,
                        pair_feat = pair_pred, pair_mask = pair_mask
                    )
                else:
                    node_feat, pair_feat = node_pred, pair_pred

            else:
                ### gt to noise
                if self.train_version == 'gt':
                    node_pred, pair_pred = self.gt_noise_transfer(
                        t_idx, node_pred, node_feat, pair_pred, pair_feat
                    )

                ### denoise
                node_feat, pair_feat = self.transition.denoise(
                    node_t = node_feat, eps_node = node_pred, t = t_idx, 
                    pair_t = pair_feat, eps_pair = pair_pred
                )

                ###### unnormalize #####
                node_feat, pair_feat = self._unnormalize(
                    node_feat=node_feat, node_bias=node_bias, node_var=node_var,
                    pair_feat=pair_feat, pair_bias=pair_bias, pair_var=pair_var,
                    norm_version = norm_version
                )

            ###### masking ######
            if mask is not None:
                node_feat[mask == 0] = 0
            if pair_mask is not None:
                pair_feat[pair_mask == 0] = 0

            node_feat = node_feat.detach().cpu()
            if pair_feat is not None:
                pair_feat = pair_feat.detach().cpu()

            #traj[t-1] = (node_feat, pair_feat)
            traj[t] = (node_feat, pair_feat)

        return traj 


    def forward_diffusion(self,
        node_feat:torch.Tensor,
        t:Union[int, torch.Tensor],
        mask:torch.Tensor=None,
        mask_gen:torch.Tensor=None,
        fragment_type:torch.Tensor=None,
        node_bias:float=None,
        node_var:float=None,
        pair_feat:torch.Tensor=None,
        pair_mask:torch.Tensor=None,
        pair_bias:float=None,
        pair_var:float=None,
        diffusion_mode='direct',
        norm_version:str='default'
    ): 

        N = node_feat.shape[0]
        device = node_feat.device
        if mask_gen is None:
            mask_gen = mask

        ###### normalize ######
        node_feat, pair_feat = self._normalize(
            node_feat = node_feat,
            node_bias = node_bias,
            node_var = node_var,
            pair_feat = pair_feat,
            pair_bias = pair_bias,
            pair_var = pair_var,
            norm_version = norm_version
        )

        if self.with_fragment_type:
            fragment_feat = self.fragment_emb(fragment_type.long())  # (N, L, dim)
            node_feat_in = torch.cat([node_feat, fragment_feat], dim = -1)
            node_feat_in = self.node_fead_merge(node_feat_in)  # (N, L, dim)
        else:
            node_feat_in = node_feat

        #############################################
        # add noise step by step 
        #############################################

        if diffusion_mode == 'step':
            node_noisy_batch = [node_feat_in] 
            err_node_batch = []
            if pair_feat is not None:
                pair_noisy_batch = [pair_feat]
                err_pair_batch = []
            else:
                pair_noisy_batch = None
                err_pair_batch = None

            for t_idx in range(1, t+1):
                t_idx = torch.ones(N).to(dtype=torch.long) * t_idx

                node_feat_in = node_noisy_batch[-1].to(device)
                if pair_feat is not None:
                    pair_feat = pair_noisy_batch[-1].to(device)

                (node_noisy,  # noised node feature
                 err_node,    # Gaussian noise for the node feature
                 pair_noisy,  # noised pair feature
                 err_pair,    # Gaussian noise for the pair feature
                )= self.transition.add_noise_singleStep(
                    node_feat = node_feat_in, t = t_idx, mask = mask,
                    pair_feat = pair_feat, pair_mask = pair_mask
                )

                ### random masking
                node_noisy = torch.where(
                    mask_gen.bool()[:,:,None].expand(node_noisy.shape),
                    node_noisy, node_feat_in
                )

                node_noisy_batch.append(node_noisy.detach().cpu())
                err_node_batch.append(err_node.detach().cpu())
                if pair_feat is not None:
                    pair_noisy_batch.append(pair_noisy.detach().cpu())
                    err_pair_batch.append(err_pair.detach().cpu())

            node_noisy_batch = node_noisy_batch[1:]
            if pair_noisy_batch is not None:
                pair_noisy_batch = pair_noisy_batch[1:]

            return node_noisy_batch, err_node_batch, pair_noisy_batch, err_pair_batch 

        #############################################
        # directly add the noise 
        #############################################

        else:
            ###### step ######
            if isinstance(t, int):
                t = torch.ones(N).to(dtype=torch.long) * t
            t = t.to(device = node_feat.device)

            (node_noisy,  # noised node feature
             err_node,    # Gaussian noise for the node feature
             pair_noisy,  # noised pair feature
             err_pair,    # Gaussian noise for the pair feature
            )= self.transition.add_noise(
                node_feat = node_feat_in, t = t, mask = mask,
                pair_feat = pair_feat, pair_mask = pair_mask
            )

            ### random masking
            node_noisy = torch.where(
                mask_gen.bool()[:,:,None].expand(node_noisy.shape),
                node_noisy, node_feat_in
            )

            return node_noisy, err_node, pair_noisy, err_pair


    def reverse_diffusion(self,
        node_noisy,
        t:Union[int, torch.Tensor],
        mask:torch.Tensor=None,
        node_bias:float=None,
        node_var:float=None,
        pair_noisy:torch.Tensor=None,
        pair_mask:torch.Tensor=None,
        pair_bias:float=None,
        pair_var:float=None,
        norm_version:str='default'
    ):

        ###### step ######
        if isinstance(t, int):
            N = node_noisy.shape[0]
            t = torch.ones(N).to(dtype=torch.long) * t
        t = t.to(device = node_noisy.device)

        #############################################
        # reverse (denoise, t to t-1) 
        #############################################

        beta = self.transition.var_sched.betas[t]  # (N,)
        node_pred, pair_pred = self.reverse_net(
            node_feat = node_noisy,
            beta = beta,
            mask = mask,
            pair_feat = pair_noisy,
        )

        ### denoise
        node_feat, pair_feat = self.transition.denoise(
            node_t = node_noisy,
            eps_node = node_pred,
            t = t,
            pair_t = pair_noisy,
            eps_pair = pair_pred
        )

        ###### unnormalize #####
        node_feat, pair_feat = self._unnormalize(
            node_feat = node_feat,
            node_bias = node_bias,
            node_var = node_var,
            pair_feat = pair_feat,
            pair_bias = pair_bias,
            pair_var = pair_var,
            norm_version = norm_version
        )

        ###### masking ######
        if mask is not None:
            node_feat[mask == 0] = 0
        if pair_mask is not None:
            pair_feat[pair_mask == 0] = 0

        return node_feat, node_pred, pair_feat, pair_pred


###############################################################################
# Oracle for the consistency
###############################################################################

def load_mpnn_oracle(args):
    oracle = ProteinMPNN(num_letters=21, node_features=128,
        edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False
    )
    oracle_ens = torch.load(args.proteinMPNN_path)
    oracle.load_state_dict(oracle_ens['model_state_dict'])
    for param in oracle.parameters():
        param.requires_grad = False
    print('Oracle proteinMPNN loaded!')

    return oracle


class ConsistencyPredictor(nn.Module):
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
        super(ConsistencyPredictor, self).__init__()

        #######################################################################
        # major predictor 
        #######################################################################

        self.architecture = architecture

        ###### Transformer ######
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

        ###### UNet ######
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

        ###### TAB ######
        elif self.architecture == 'tab':
            self.pred_net = TriangularSelfAttentionNetwork(
                in_resi_features = res_feat_dim,
                in_pair_features = pair_feat_dim,
                out_resi_features = res_feat_dim,
                out_pair_features = pair_feat_dim,
                num_heads = num_heads,
                num_blocks = num_layers,
                dropout = dropout
            )

        #######################################################################
        # out layer 
        #######################################################################

        self.out_layer = nn.Linear(res_feat_dim, 1)
        

    def forward(self,
        node_feat,
        mask:torch.Tensor=None,
        pair_feat:torch.Tensor=None,
        pair_mask:torch.Tensor=None,
    ):
        """Predict whether the input features are for positive or negative pairs.
        
        Args:
            node_feat: node-wise feature; (N, L, dim)
            mask: 1 for valid token and 0 for padding; (N, L)
            pair_feat: pair-wise feature, only for TAB; (N, L, L, dim)
            pair_mask: pair-wise mask; (N, L, L)

        Output:
            output: preobability; (N,)
        """

        ################################ Transformation ###########################

        if self.architecture == 'transformer':
            node_feat = self.pred_net(
                src = node_feat,
                padding_mask = ~mask.to(torch.bool)
            )

        elif self.architecture == 'unet':
            if mask is not None:
                node_feat[mask == 0] = 0
            node_feat = self.pred_net(
                x = node_feat.transpose(1,2),
                timesteps = torch.zeros(node_feat.shape[0]).to(node_feat.device),
            ).transpose(1,2)

        elif self.architecture == 'tab':
            node_feat, pair_feat = self.pred_net(node_feat, pair_feat, mask)

        ################################ global information #######################

        if mask is not None:
            node_feat[mask == 0] = 0
            node_feat = node_feat.sum(dim=1) / mask.sum(dim = 1).unsqueeze(dim=-1)
        else:
            node_feat = node_feat.mean(dim=1)

        ################################ outlayer #################################

        output = self.out_layer(node_feat)
        output = F.sigmoid(output)

        return output


###############################################################################
# Container of the pretrained Encoder/Decoder
###############################################################################

class PretrainedSOTA_Container(nn.Module):
    """Container of Pretrained SOTA methods.

    Encoder:
      * ESM2 (for sequence)
      * ProteinMPNN or ESM-IF encoder (for structure)
    
    Decoder:
      * edge predictor (not needed for FCG version) 
      * ProteinMPNN or ESM-IF decoder (for sequence)
      * ESMFold decoder (for structure)

    Inputs can be the sequence (ESM-2 tokenized) or the structure (4-atom of 3 
    -atom)
    """

    def __init__(self, args):
        super(PretrainedSOTA_Container, self).__init__()

        #######################################################################
        # Preprocess 
        #######################################################################

        self.args = args
        self.device = args.device
        if not args.__contains__('with_terminus_token'):
            args.with_terminus_token = True
        self.with_terminus_token = args.with_terminus_token

        ################## for sequence transfermation ########################
        # mpnn_restypes   = 'ACDEFGHIKLMNPQRSTVWYX'
        # esm-if_restypes = 'LAGVSERTIDPKQNFYMHWCX' 
        # (derived from '*!**LAGVSERTIDPKQNFYMHWCXBUZO.-***')
        # esm_restypes    = 'ARNDCQEGHILKMFPSTWYVX'

        self.voxel_size = len(args.mpnn_restypes) + 1
        self.mpnn_to_esm_mat = sequence_transform(
            args.mpnn_restypes, args.esm_restypes
        ).to(args.device)
        self.esm_to_mpnn_mat = self.mpnn_to_esm_mat.T

        self.mpnn_to_esm_mat = nn.Parameter(
            self.mpnn_to_esm_mat, requires_grad=False
        )
        self.esm_to_mpnn_mat = nn.Parameter(
            self.esm_to_mpnn_mat, requires_grad=False
        )

        ############## whether add paddings for esm node-wise feature #########
        # only when with ESM-IF and (mlp or joint projector)
        self.seq_design_module = args.seq_design_module
        if self.seq_design_module == 'ESM-IF':
            args.esm_add_padding = True
            args.with_edge_feature = False
        else:
            args.esm_add_padding = False
            args.with_edge_feature = True

        self.esm_add_padding = args.esm_add_padding
        self.with_edge_feature = args.with_edge_feature

        ############################### for loss ##############################
        self.config = model_config(
            args.config_preset, train=True,
            low_prec=(str(args.precision) == '16')
        )

        #######################################################################
        # ESMFold: sequence encoder and structure decoder
        #######################################################################

        if args.esmfold_path is not None:
            ### model information
            model_data = torch.load(args.esmfold_path)

            ### pair-wise features are needed to compare with ProteinMPNN
            model_data['cfg']['model']['use_esm_attn_map'] = self.with_edge_feature
            ### can select the suitable ESM such as "esm2_3B"
            model_data['cfg']['model']['esm_type'] = args.esm_type
            ### other hyperparameters
            model_data['cfg']['model']['trunk']['num_blocks'] = args.esm_num_blocks
            model_data['cfg']['model']['trunk']['sequence_state_dim'] = args.sequence_state_dim

            self.esmfold = ESMFold(esmfold_config=model_data['cfg']['model'])

            ###### load the model ######
            if args.load_pretrained_esmfold:
                self.esmfold.load_state_dict(model_data['model'], strict=False)
                print('ESMFold loaded from %s.' % args.esmfold_path)
            else:
                print('Structure decoder starts from the scratch.')

        else:
            ###### default esmfold ######
            self.esmfold = ESMFold()

        #######################################################################
        # ProteinMPNN or ESM-IF: structure encoder and sequence decoder
        #######################################################################

        if self.seq_design_module == 'ESM-IF':
            self.esm_if, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
            print('ESM-IF loaded.')

        elif self.seq_design_module == 'ProteinMPNN':
            self.proteinMPNN = ProteinMPNN(
                num_letters=args.num_letters,
                node_features=args.node_features,
                edge_features=args.edge_features,
                hidden_dim=args.proteinMPNN_hidden_dim,
                num_encoder_layers=args.proteinMPNN_encoder_layers,
                num_decoder_layers=args.proteinMPNN_decoder_layers,
                vocab=args.vocab,
                k_neighbors=args.k_neighbors,
                augment_eps=args.augment_eps,
                dropout=args.dropout,
                ca_only=args.ca_only
            )

            ##### load the model ######
            if args.load_pretrained_proteinMPNN and (args.proteinMPNN_path is not None):
                proteinMPNN_data = torch.load(args.proteinMPNN_path)
                self.proteinMPNN.load_state_dict(
                    proteinMPNN_data['model_state_dict']
                )
                print('ProteinMPNN loaded from %s.' % args.proteinMPNN_path)
            else:
                print('ProteinMPNN starts from the scratch.')

        else:
            raise NameError(
                'Error! No sequence design model named %s!' % args.proteinMPNN_path
            )

    ###########################################################################
    # Utility Fucntions
    ###########################################################################

    def aa_sample(self,
        seq_logit:torch.Tensor,
        topk:int=None,
        temperature=1.0
    ) -> torch.Tensor:
        """Sample the sequences based on the multinomial distribution.

        Args:
            seq_logit: (B, L_max, prob_dim).
            topk: k-value of top-k sampling; None for multinomial sampling.

        Returns:
            aa: the sampled sequences; (B, L_max).
        """
        if temperature != 1.0:
            seq_logit /= temperature

        ###### multinomial sampling ######
        if topk is None:
            B, L, dim = seq_logit.shape
            aa = torch.multinomial(
                seq_logit.reshape(-1, dim), 1
            ).reshape(B, L)

        ###### maximum sampling ######
        elif topk == 1:
            aa = seq_logit.max(dim=-1).indices  # (B, L_max)

        ###### top-k sampling ######
        else:
            top_v, top_k = seq_logit.data.topk(k) # (B, L_max, k), (B, L_max, k)
            B, L, _ = top_v.shape
            top_v_sele = torch.multinomial(
                top_v.reshape(-1, k), 1
            ).reshape(B, L, 1)  # (B, L_max, 1)
            aa = top_k.gather(dim = -1, index = top_v_sele).reshape(B, L)

        return aa

    def seq_transform(self, seq:torch.Tensor, to_esm:bool=True) -> torch.Tensor:
        """Transform sequence from ESM to other tokenizations or vice versa.

        mpnn_restypes   = 'ACDEFGHIKLMNPQRSTVWYX'
        esm-if_restypes = 'LAGVSERTIDPKQNFYMHWCX' 
            (derived from '*!**LAGVSERTIDPKQNFYMHWCXBUZO.-***')
        esm_restypes    = 'ARNDCQEGHILKMFPSTWYVX'

        Args:
            seq: original sequence; (B, L)
        
        Returns:
            seq: transformed sequence; (B, L)
        """
        ###### align the vocab set from ESM-IF to ESM ######
        if to_esm and self.seq_design_module == 'ESM-IF':
            seq -= 4  # for '*!**' in the alphabet
            seq[seq > 21] = 20  # assign outliers tokens with 'X'
        ###### one hot encoding ######
        seq = F.one_hot(seq, num_classes=self.voxel_size)  # (N, L_max, 22)
        ###### transformation ######
        trans_mat = self.mpnn_to_esm_mat if to_esm else self.esm_to_mpnn_mat
        seq = torch.matmul(seq.float(), trans_mat)  # (N, L_max, 22)
        ###### argmax ######
        seq = seq.max(dim=-1).indices  # (N, L_max)
        ###### align the vocab set from ESM to ESM-IF ######
        if (not to_esm) and self.seq_design_module == 'ESM-IF':
            ### align the vocab set from ESM-IF to ESM
            seq += 4  # for '*!**' in the alphabet

        return seq

    ###########################################################################
    # modules
    ###########################################################################

    ########################### encoder #######################################
    def encoding(self, 
        X: torch.Tensor = None,
        seq: torch.Tensor = None,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        chain_encoding_all: T.Optional[torch.Tensor] = None,
        protein_size: T.Optional[torch.Tensor] = None,
    ):
        """Sequence cncoding with ESMFold and Structure encoding with proteinMPNN.

        Args:
            X: coordinates info, (B, L_max, atom_num=4, 3)
            seq: sequence, (B, L_max)
            mask: 1 for valid residues and 0 for others, (B, L_max) 
            residx: from 1 to L for single-chain, (B, L_max)
            masking_pattern: Optional masking to pass to the input. Binary tensor 
                of the same size as `seq`. Positions with 1 will be masked. 
                ESMFold sometimes produces different samples when different masks 
                are provided.
            chain_encoding_all: perform like a chain mask (1 for valid token) for single-chain, (B, L_max)
            protein_size: tensor of the protein sizes, (B,)

        Returns:
            seq_feat_esm: residue-wise feature from ESMFold, (B, L_max, esm_dim) 
            seq_pair_feat_esm: pair-wise feature from ESMFold, (B, L_max, L_max, esm_dim)
            struc_feat_mpnn: residue-wise feature from ProteinMPNN or ESM-IF, (B, L_max, mpnn_dim) 
            struc_pair_feat_mpnn: pair-wise feature from ProteinMPNN, (B, L_max, K, mpnn_dim)
            dist_map: distance matrix, (B, L_max, L_max)
            E_idx: edge indexes from ProteinMPNN, (B, L_max, K)
        """

        #######################################################################
        # sequence encoding (with ESM-2) 
        #######################################################################

        if seq is not None:
            ###### esm encoding ######
            seq_feat_esm, seq_pair_feat_esm = self.esmfold.seq_encoder(
                aa = seq, mask = mask, residx = residx, masking_pattern = masking_pattern
            )
            # seq_feat_esm: node-wise seq emb (ESM-2), (B, L_max, esm_dim) 
            # seq_pair_feat_esm: pair-wise seq emb (ESM-2), (B, L_max, L_max, esm_dim)

            if not self.with_edge_feature:
                seq_pair_feat_esm = None
        else:
            seq_feat_esm = None
            seq_pair_feat_esm = None

        #######################################################################
        # structure encoding 
        #######################################################################

        ################### proteinMPNN encoding ##############################
        if X is not None and self.seq_design_module == 'ProteinMPNN':
            (struc_feat_mpnn,       # (B, L_max, mpnn_dim) 
             struc_pair_feat_mpnn,  # (B, L_max, K, mpnn_dim)
             dist_map,              # (B, L_max, L_max)
             E_idx,                 # (B, L_max, K)
            ) = self.proteinMPNN.struc_encoder(
                X = X,
                mask = mask,
                residue_idx = residx,
                chain_encoding_all = chain_encoding_all
            )

        #################### ESM-IF encoding ##################################
        elif X is not None and self.seq_design_module == 'ESM-IF':
            struc_feat_mpnn, _ = self.esm_if.struc_encoder(
                coords = X[:,:,:3,:],  # only need N, CA, C; (B, L, 3, 3)
                padding_mask = mask,
                protein_size = protein_size,
            )  # (L_max+2, B, esm_dim)
            struc_feat_mpnn = struc_feat_mpnn.transpose(0, 1)  # (B, L_max+2, esm_dim)
            if not self.with_terminus_token:
                struc_feat_mpnn = esmif_remove_ends(
                    feat = struc_feat_mpnn,  protein_length = protein_size
                )  # (B, L_max, esm_dim)

            struc_pair_feat_mpnn = None
            dist_map = None
            E_idx = None

        else:
            struc_feat_mpnn = None
            struc_pair_feat_mpnn = None
            dist_map = None
            E_idx = None

        return (
            seq_feat_esm, seq_pair_feat_esm,  
            struc_feat_mpnn, struc_pair_feat_mpnn,
            dist_map, E_idx,
        )


    ######################### decoder for training ############################
    def decoding_train(self, 
        seq_feat: torch.Tensor = None,
        seq_pair_feat: torch.Tensor = None,
        struc_feat: torch.Tensor = None,
        struc_pair_feat: torch.Tensor = None,
        seq_true: T.Optional[torch.Tensor] = None,
        mask: T.Optional[torch.Tensor] = None,
        mask_esmif: T.Optional[torch.Tensor] = None,
        with_aa_sampling: bool = False,
        chain_M: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        E_idx: T.Optional[torch.Tensor] = None,
        temperature: T.Optional[float] = 1.0,
        topk: T.Optional[int] = 1,
        with_true_seq: bool = False,
        num_recycles: T.Optional[int] = 3,
    ):
        """Sequence decoding with proteinMPNN and Structure decoding with ESMFold.

        * L = L_max (without downsampling) or m (with downsampling).

        Args:
            seq_feat: residue-wise features for sequence, (B, L, node_dim)
            seq_pair_feat: edge-wise features for sequence, (B, L, L, edge_dim)
            struc_feat: residue-wise features for structure, (B, L, node_dim)
            struc_pair_feat: edge-wise features for structure, (B, L, L, edge_dim)
            seq_true: true sequence, only for training, (B, L_max), following 
                ESM tokenization.
            mask: 1 for valid residues and 0 for others, (B, L_max)
            ****************** for sequence sampling **************************
            mask_esmif: 1 for valid residues and 0 for others, (B, L_max+2)
            chain_M: chain mask, 1.0 for the bits that need to be predicted, 0.0 
                for the bits that are given; (B, L_max)
            residx: from 1 to L for single-chain, (B, L_max); also needed for 
                sturcture sampling.
            temperature: temperature for sampling.
            topk: k value of the topk sampling, default = 1 (maximum sampling)
            ****************** for sturcture prediction ***********************
            with_true_seq: whether use the true sequence for ESMFold. 
            num_recycles (int): How many recycle iterations to perform. If None, 
                defaults to training max recycles, which is 3.

        Returns:
            seq_logit: sequence probability matrix; (N, L, dim), after softmax
            aa_gen: predicted sequence
            struc: predicted structure
        """

        ###### whether decode sequence ######
        seq_design_flag = struc_feat is not None
        if self.seq_design_module == 'ProteinMPNN':
            ### ProteinMPNN also requires pair-wise features
            seq_design_flag = seq_design_flag and struc_pair_feat is not None
        if not seq_design_flag:
            with_aa_sampling = False
            with_true_seq = True

        ###### whether decode structure ######
        struc_design_flag = seq_feat is not None
        if self.with_edge_feature:
            struc_design_flag = struc_design_flag and seq_pair_feat is not None

        #######################################################################
        # sequence decoding 
        #######################################################################

        ################# logistic distribution ###############################

        ###### sequence transformation (to decoder seq) ######
        seq_true = self.seq_transform(seq_true, to_esm = False)  # (N, L_max) 

        ###### ProteinMPNN decoding ######
        if seq_design_flag and self.seq_design_module == 'ProteinMPNN':
            seq_logit = self.proteinMPNN.seq_decoder(
                h_V = struc_feat,
                h_E = struc_pair_feat,
                E_idx = E_idx,
                mask = mask,
                chain_M = chain_M,
                S = seq_true,
                train_mode = True,
            ) # (N, L_max, 22)

        ###### ESM-IF decoding ######
        elif seq_design_flag:
            # add initial token to the sequence
            seq_true = F.pad(
                seq_true, (1,0), 'constant',
                self.esm_if.decoder.dictionary.get_idx('<cath>')
            )  # (N, L_max+1)

            if mask_esmif is None and mask is not None and self.with_terminus_token:
                mask_esmif = esmif_mask_transform(mask)  # (L_max+2, N, dim)
            elif mask_esmif is None:
                mask_esmif = mask  # (L_max, N, dim)

            seq_logit, _ = self.esm_if.seq_decoder(
                prev_output_tokens = seq_true, # (N, L_max+1)
                encoder_emb = struc_feat.transpose(0,1),  # (L_max(+2), N, dim)
                padding_mask = mask_esmif,  # (N, L_max(+2))
                protein_size = None,
                mode = 'train',
                with_inf = self.with_terminus_token,
            )  # (N, L_max, 35)

        else:
            seq_logit = None

        ################# sequence sampling ###################################
        if with_aa_sampling:
            ###### sampling ######
            aa_gen = self.aa_sample(seq_logit, topk=topk)  # (N, L_max)
            ###### transform to ESM seq ######
            aa_gen = self.seq_transform(aa_gen, to_esm = True)  # (N, L_max)

        else:
            aa_gen = None

        #######################################################################
        # structure decoding 
        #######################################################################

        if struc_design_flag:
            ### use the true or predicted sequence for ESMFold decoder
            aa_ref = seq_true if with_true_seq else aa_gen

            ###### structure decoding ######
            struc = self.esmfold.struc_decoder(
                s_s_0 = seq_feat,
                s_z_0 = seq_pair_feat,
                aa = aa_ref,
                mask = mask,
                residx = residx,
                num_recycles = num_recycles
            )
        else:
            struc = None

        return seq_logit, aa_gen, struc


    ######################## decoder for inference #############################
    @torch.no_grad()
    def decoding_inference(self,
        decode_seq: bool = True,
        decode_struc: bool = True,
        struc_feat: T.Optional[torch.Tensor] = None,     
        struc_pair_feat: T.Optional[torch.Tensor] = None,
        seq_feat: T.Optional[torch.Tensor] = None,       
        seq_pair_feat: T.Optional[torch.Tensor] = None,  
        mask: T.Optional[torch.Tensor] = None,
        mask_esmif: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        E_idx: T.Optional[torch.Tensor] = None,
        randn: T.Optional[torch.Tensor] = None,
        chain_encoding_all: T.Optional[torch.Tensor] = None,
        protein_size: T.Optional[torch.Tensor] = None,
        temperature: T.Optional[float] = 1.0,
        omit_AAs_np = None,
        bias_AAs_np = None,
        omit_AA_mask = None,
        pssm_coef = None,
        pssm_bias = None,
        pssm_multi = None,
        pssm_log_odds_flag = None,
        pssm_log_odds_mask = None,
        pssm_bias_flag = None,
        bias_by_res = None,
        with_true_seq: bool = False,
        seq_true: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 3,
    ):
        """Sequence decoding with ProteinMPNN or ESM-IF, and
        structure decoding with ESMFold.

        Args:
            mask: 1 for valid residues and 0 for others, (B, L_max).
            residx: from 1 to L for single-chain, (B, L_max).
            ************************* for seq decoding *************************
            decode_seq: whether predict the sequence.
            struc_feat: node-wise struc feat; (B, L_max, mpnn_dim) for ProteinMPNN 
                or (B, L_max+2, esmif_dim) for ESM-IF.  
            struc_pair_feat: pair-wise struc feat; (B, L_max, K, mpnn_dim) for 
                ProteinMPNN or None for ESM-IF.
            temperature: temperature for sampling.
            ###### for ProteinMPNN ######
            E_idx: edge indexes from ProteinMPNN, (B, L_max, K)
            randn: for decoding order, (B, L_max)
            chain_encoding_all: perform like a chain mask (1 for valid token) for 
                single-chain, (B, L_max)
            protein_size: tensor of the protein sizes, (B,)
            Other parameters for ProteinMPNN: 
                omit_AAs_np, bias_AAs_np, omit_AA_mask,
                pssm_coef, pssm_bias, pssm_multi, pssm_log_odds_flag,
                pssm_log_odds_mask, pssm_bias_flag, bias_by_res
            ###### for ESM-IF ######
            mask_esmif: 1 for valid residues and 0 for others, (B, L_max+2)
            ********************** structure only ******************************
            decode_struc: whether predict the structure.
            seq_feat: node-wise seq feat; (B, L_max, esm_dim) 
            seq_pair_feat: pair-wise seq feat; (B, L_max, L_max, esm_dim) or None.
            with_true_seq: for structure prediction whether use the predicted 
                sequence.
            seq_true: true sequence, used when with_true_seq=True; (B, L_max). 
            num_recycles: How many recycle iterations to perform. If None, 
                defaults to training max recycles, which is 3.

        Returns:
            aa_gen: designed sequence; (N, L_max).
            struc: Dict; designed structure.
        """

        #############################################################
        # preprocess 
        #############################################################
 
        if struc_feat is None:
            decode_seq = False
        elif self.seq_design_module == 'ProteinMPNN' and struc_pair_feat is None:
            decode_seq = False

        if seq_feat is None:
            decode_struc = False

        if not decode_seq:
            with_true_seq = True

        if protein_size is None:
            protein_size = mask.sum(-1)  # (N,)

        #############################################################
        # sequence decoding 
        #############################################################

        aa_gen = None

        ################## with ProteinMPNN #########################
        if decode_seq and self.seq_design_module == 'ProteinMPNN':
            if bias_by_res is None:
                L_max = max(protein_size)
                bias_by_res = torch.zeros(
                    [struc_feat.shape[0], L_max, 21]
                ).float().to(struc_feat.device)

            aa_gen = self.proteinMPNN.seq_decoder(
                h_V=struc_feat,
                h_E=struc_pair_feat,
                E_idx=E_idx,
                mask=mask,
                chain_M=chain_encoding_all.float(),
                S=None,
                randn=randn,
                train_mode=False,
                seq_only = True,
                temperature = temperature,
                omit_AAs_np = omit_AAs_np,
                bias_AAs_np = bias_AAs_np,
                omit_AA_mask = omit_AA_mask,
                pssm_coef = pssm_coef,
                pssm_bias = pssm_bias,
                pssm_multi = pssm_multi,
                pssm_log_odds_flag = pssm_log_odds_flag,
                pssm_log_odds_mask = pssm_log_odds_mask,
                pssm_bias_flag = pssm_bias_flag,
                bias_by_res = bias_by_res
            )  # (B, L_max)

        ###### with ESM-IF ######
        elif decode_seq and self.seq_design_module == 'ESM-IF':
            if mask_esmif is None and mask is not None and self.with_terminus_token:
                mask_esmif = esmif_mask_transform(mask)
            elif mask_esmif is None:
                mask_esmif = mask  # (L_max, N, dim)

            aa_gen = self.esm_if.seq_decoder(
                encoder_emb = struc_feat.transpose(0,1),
                padding_mask = mask_esmif,
                temperature = temperature,
                protein_size = protein_size,
                mode = 'eval',
                return_string = False,
                with_inf = self.with_terminus_token,
            )  # (B, L_max)

        if aa_gen is not None:
            aa_gen = self.seq_transform(aa_gen, to_esm = True)

        #############################################################
        # structure decoding 
        #############################################################

        struc = None
        ###### whether decode structure ######
        struc_design_flag = seq_feat is not None

        if struc_design_flag and self.with_edge_feature and seq_pair_feat is None:
            print(
                'Warning! Pairwise sequence feat is None for the version with ProteinMPNN.'
            )

        if struc_design_flag:
            ### use the true or predicted sequence for ESMFold decoder
            aa_ref = seq_true if with_true_seq else aa_gen

            ###### structure decoding ######
            struc = self.esmfold.struc_decoder(
                s_s_0 = seq_feat,
                s_z_0 = seq_pair_feat,
                aa = aa_ref,
                mask = mask,
                residx = residx,
                num_recycles = num_recycles
            )

        return aa_gen, struc


    ########################################################################### 
    # Losses                                                                  # 
    ###########################################################################

    def loss(self,
        out,
        batch,
        with_seq_recover_loss = False,
        with_struc_recover_loss = False,
        summary = False,
        loss_weight = None
    ):
        """ End2end loss (modality recover loss). """

        loss_all = {}

        #######################################################################
        # sequence (cross entropy; for proteinMPNN or ESMFold) 
        #######################################################################
        if with_seq_recover_loss:
            seq_true = self.seq_transform(
                batch['aatype'].to(self.device), to_esm = False
            )  # (N, L_max)
            seq_mask = batch['seq_mask'].to(self.device)
 
            ### smoothed cross entropy loss
            _, loss_all['seq'] = loss_smoothed(
                S = seq_true,
                log_probs = out['log_prob'],
                mask = seq_mask,
                vocab_size = out['log_prob'].shape[-1]
            )

        #######################################################################
        # structure (AF2 loss; for ESMFold)
        #######################################################################
        if with_struc_recover_loss:
            loss_all['struc'] = strutcure_loss(
                out=out, batch=batch, config=self.config.loss
            )

        #######################################################################
        # structure (AF2 loss; for ESMFold)
        #######################################################################
        if summary:
            loss_overall = 0.
            for key in loss_all:
                if loss_weight is not None and key in loss_weight:
                    loss_overall += loss_all[key] * loss_weight[key]
                else:
                    loss_overall += loss_all[key]
            loss_all['loss'] = loss_overall

        return loss_all

###############################################################################
# Projector
###############################################################################

class RepresentationAutoencoder(nn.Module):
    """Container of latent-diffusion model or autoencoder.

    Encoder:
      * feature projectors 
      * downsampler
    
    Decoder:
      * upsampler
      * feature projectors
      * edge predictor (for proteinMPNN only) 
    """
    def __init__(self, args):
        super(RepresentationAutoencoder, self).__init__()

        self.args = args
        self.projector_version = args.projector_version
        self.downsampling = args.downsampling
        self.with_edge_feature = args.with_edge_feature
        self.edge_pred_version = args.edge_pred_version
        self.k_neighbors = args.k_neighbors
        self.device = args.device
 
        #######################################################################
        # Feature Projectors: (N, L, D) to (N, L, d), D > d
        #######################################################################

        ######################## joint projector ###############################
        if 'joint' in args.projector_version:
            self.projector = JointFeatureProjector(
                in_seq_resi_features = args.proj_seq_resi_dim,
                in_seq_pair_features = args.proj_seq_pair_dim,
                in_struc_resi_features = args.proj_struc_resi_dim,
                in_struc_pair_features = args.proj_struc_pair_dim,
                hidden_features = args.projector_dim,
                out_features = args.latent_dim,
                version = args.projector_version,
                num_heads = args.projector_heads,
                num_layers = args.projector_num_layers,
                dropout = args.dropout,
                max_length = args.max_length,
                with_edge_feature = args.with_edge_feature,
            )

        ######################## separate projector ############################
        else:
            ###### sequence projector ######
            self.seq_projector = FeatureProjector(
                in_resi_features = args.proj_seq_resi_dim,
                in_pair_features = args.proj_seq_pair_dim,
                hidden_features = args.projector_dim,
                out_features = args.latent_dim,
                version = args.projector_version,
                num_heads = args.projector_heads,
                num_layers = args.projector_num_layers,
                dropout = args.dropout,
                max_length = args.max_length,
                with_edge_feature = args.with_edge_feature,
            )
            ###### structure projector ######
            self.struc_projector = FeatureProjector(
                in_resi_features = args.proj_struc_resi_dim,
                in_pair_features = args.proj_struc_pair_dim,
                hidden_features = args.projector_dim,
                out_features = args.latent_dim,
                version = args.projector_version,
                num_heads = args.projector_heads,
                num_layers = args.projector_num_layers,
                dropout = args.dropout,
                max_length = args.max_length,
                with_edge_feature = args.with_edge_feature,
            )

        #######################################################################
        # Downsampler and Upsampler: (N, L, d) to (N, m, d), L > m
        #######################################################################

        if args.downsampling:
            self.sampler = Sampler(
                feature_dim = args.latent_dim,
                version = args.sampler_version,
                k_size = args.sampler_kernel_size,
                layer_num = args.sampler_layer_num,
                padding = args.sampler_padding,
                with_coor = False,
                with_edge_feature = args.with_edge_feature
            )

        #######################################################################
        # Edge Predictor
        #######################################################################

        if args.with_edge_pred and args.with_edge_feature:
            if args.edge_pred_version == 'latent':
                edge_predictor_resi_dim = args.latent_dim
                edge_predictor_pair_dim = args.latent_dim
            else:
                edge_predictor_resi_dim = args.proj_struc_resi_dim
                edge_predictor_pair_dim = args.proj_struc_pair_dim

            ### predict the edge features 
            self.dist_predictor = DistPredictor(
                in_resi_features = edge_predictor_resi_dim,
                in_pair_features = edge_predictor_pair_dim,
                num_heads = args.projector_heads,
                num_blocks = args.dist_layer_num,
                dropout = 0.0
            )
            print('Edge predictor added.')

        elif args.with_edge_feature:
            print('Utilize fully-connected graph for ProteinMPNN.')

        else:
            print('ESM-IF: no edge predictor required.')

    ###########################################################################
    # Utility Fucntions
    ###########################################################################

    def pair_feat_transform(self,
        struc_pair_feat: T.Optional[torch.Tensor] = None,
        E_idx: T.Optional[torch.Tensor] = None,
    ):
        """Transform the shape of struc_pair_feat.

        Args:
            struc_pair_feat: (B, L_max, K, *)
            E_idx: edge indexes; (B, L_max, K)

        Return:
            struc_pair_feat_new: (B, L_max, L_max, *)
        """
        B, L, K = E_idx.shape

        idx_flatten = [
            torch.arange(0, B).reshape(-1, 1).repeat(1, L * K).reshape(-1),
            torch.arange(0, L).reshape(-1, 1).repeat(1, K).reshape(-1).repeat(B),
            E_idx.reshape(-1)
        ]
        struc_pair_feat_new = torch.zeros(
            B, L, L, struc_pair_feat.shape[-1]
        ).to(struc_pair_feat.device)
        struc_pair_feat_new[idx_flatten] = struc_pair_feat.reshape(
            -1, struc_pair_feat.shape[-1]
        )  # (B, L_max, L_max, *)
        return struc_pair_feat_new

    ###########################################################################
    # Modules
    ###########################################################################

    ################################ for Encoding #############################

    def negative_encode(self,
        seq_feat: T.Optional[torch.Tensor] = None,
        seq_pair_feat: T.Optional[torch.Tensor] = None,
        struc_feat: T.Optional[torch.Tensor] = None,
        struc_pair_feat: T.Optional[torch.Tensor] = None,
        mask: T.Optional[torch.Tensor] = None
    ):
        B, L_max, _ = seq_feat.shape
        if B <= 1:
            print('Batch size need to be larger than 1 for contrastive loss.')
            return None, None

        ### node-wise feature 
        struc_feat = torch.cat([struc_feat[1:], struc_feat[:1]], dim = 0)
        ### pair-wise feature
        if struc_pair_feat is not None:
            struc_pair_feat = torch.cat(
                [struc_pair_feat[1:], struc_pair_feat[:1]], dim = 0
            )
        ### mask
        if mask is not None:
            mask_nega = torch.cat([mask[1:], mask[:1]], dim = 0)
            mask_nega = mask_nega * mask
        else:
            mask_nega = None

        ### encoding
        node_feat_nega, pair_feat_nega = self.projector.encode(
            seq_feat, seq_pair_feat, struc_feat, struc_pair_feat, mask=mask_nega
        )
 
        return node_feat_nega, pair_feat_nega


    def encoding(self,
        seq_feat: T.Optional[torch.Tensor] = None,
        seq_pair_feat: T.Optional[torch.Tensor] = None,
        struc_feat: T.Optional[torch.Tensor] = None,
        struc_pair_feat: T.Optional[torch.Tensor] = None,
        E_idx: T.Optional[torch.Tensor] = None,
        mask: T.Optional[torch.Tensor] = None,
        with_contrastive: bool = False,
    ):
        """Map the original embedding from ESM2 or ProteinMPNN to the hidden space.
        
        Args:
            * L = L_max for proteinMPNN or (L_max + 2) for ESM-IF
            seq_feat: residue-wise sequence embedding from ESM2; (B, L, esm_dim)
            seq_pair_feat: pair-wise structure embedding from ESM2; None or 
                (B, L_max, L_max, esm_dim)
            struc_feat: residue-wise structure embedding from ProteinMPNN; 
                (B, L, mpnn_dim)
            struc_pair_feat: pair-wise structure embedding from ProteinMPNN; None 
                or (B, L_max, K, mpnn_dim)
            E_idx: index for proteinMPNN edge features; None or (B, L_max, K)
            mask: 1 for valid residues and 0 for others; (B, L)
            with_contrastive: If True, get both positive pair and negative pairs;
                only valid for joint version.

        Returns:
            * L = m if with downsampling else L_max or (L_max + 2)
            for joint projector:
                node_feat: (dimension reduced) residue-wise feature; (B, L, dim) 
                pair_feat: (dimension reduced) pair-wise feature; (B, L, L, dim)
            for separate projector: 
                seq_feat: (dimension reduced) residue-wise sequence feature; (B, L, dim)
                seq_pair_feat: (dimension reduced) pair-wise sequence feature; (B, L, L, dim)
                struc_feat: (dimension reduced) residue-wise structure feature; (B, L, dim) 
                struc_pair_feat: (dimension reduced) pair-wise structure feature; (B, L, L, dim)
        """
        #######################################################################
        # Inputs preprocess 
        #######################################################################

        ######### feature transformation for structure feature ################
        # (Only for ProteinMPNN)
        if struc_pair_feat is not None and E_idx is not None:
            # Make the struc_pair_feat (B, L_max, K, *) match the shape of 
            # seq_pair_feat (B, L_max, L_max, *).
            struc_pair_feat = self.pair_feat_transform(struc_pair_feat, E_idx)

        #######################################################################
        # Projecting and Downsampling 
        #######################################################################

        ############################# joint projector #########################
        if 'joint' in self.projector_version:
            ###### feature projecting ######
            node_feat, pair_feat = self.projector.encode(
                seq_feat, seq_pair_feat, struc_feat, struc_pair_feat, mask=mask
            )
            
            ### prepare negative pairs and get the embeddings
            if with_contrastive:
                node_feat_nega, pair_feat_nega = self.negative_encode(
                   seq_feat, seq_pair_feat, struc_feat, struc_pair_feat, mask=mask
                )
            else:
                node_feat_nega, pair_feat_nega = None, None

            ###### downsampling ######
            # reduce the dimension along the sequence (L to m)
            if self.downsampling:
                node_feat, pair_feat = self.sampler.singlemodal_downsample(
                    node_feat = node_feat,
                    pair_feat = pair_feat,
                )  # node_feat: (B, m, *); pair_feat: (B, m, m, *)

                if with_contrastive:
                    node_feat_nega, pair_feat_nega = self.sampler.singlemodal_downsample(
                        node_feat = node_feat_nega,
                        pair_feat = pair_feat_nega,
                    )

            return node_feat, pair_feat, node_feat_nega, pair_feat_nega

        ########################### separate projector ########################
        else:
            ###### for sequence features ######
            if seq_feat is not None or seq_pair_feat is not None:
                ### projecting
                seq_feat, seq_pair_feat = self.seq_projector.encode(
                    seq_feat, seq_pair_feat, mask = mask
                )
                ### downsampling 
                if self.downsampling:
                    seq_feat, seq_pair_feat = self.sampler.singlemodal_downsample(
                        node_feat = seq_feat,
                        pair_feat = seq_pair_feat,
                    )  # seq_feat: (B, m, *); seq_pair_feat: (B, m, m, *)

            ###### for structure features ######
            if struc_feat is not None or struc_pair_feat is not None:
                ### projecting
                struc_feat, struc_pair_feat = self.struc_projector.encode(
                    struc_feat, struc_pair_feat, mask = mask
                )
                ### downsampling 
                if self.downsampling:
                    struc_feat, struc_pair_feat = self.sampler.singlemodal_downsample(
                        node_feat = struc_feat,
                        pair_feat = struc_pair_feat,
                    )  # struc_feat: (B, m, *); struc_pair_feat: (B, m, m, *)

            return seq_feat, seq_pair_feat, struc_feat, struc_pair_feat,

    ############################### for Decoding ##############################
    def decoding(self,
        struc_feat: T.Optional[torch.Tensor] = None,
        struc_pair_feat: T.Optional[torch.Tensor] = None,
        seq_feat: T.Optional[torch.Tensor] = None,
        seq_pair_feat: T.Optional[torch.Tensor] = None,
        mask: T.Optional[torch.Tensor] = None,
        with_true_adj: bool = False,
        E_idx: T.Optional[torch.Tensor] = None,
        L_max: T.Optional[int] = None,
    ):
        """Map the hidden embedding to the original space.
        
        For the joint-projector, the structure features will be provided.
        * L = L_max (without downsampling) or m (with downsampling).

        Args:
          struc_feat: residue-wise features for structure, (B, L, node_dim)
          struc_pair_feat: edge-wise features for structure, (B, L, L, edge_dim)
          seq_feat: residue-wise features for sequence, (B, L, node_dim)
          seq_pair_feat: edge-wise features for sequence, (B, L, L, edge_dim)
          mask: 1 for valid residues and 0 for others, (B, L_max)
          with_true_adj: whether apply the true ajacency matrix for proteinMPNN
              decoding.
          E_idx: true ajacency matrix, only needed when applying proteinMPNN
              decoder and with_true_adj=True; (B, L_max, K)
          L_max: maximum protein size in the batch.

        Returns:
          seq_feat: recovered residue-wise sequence feature, (B, L_max, esm_dim)
          seq_pair_feat: recovered pair-wise sequence feature, (B, L_max, L_max, esm_dim)
          struc_feat: recovered residue-wise structure feature, (B, L_max, mpnn_dim)
          struc_pair_feat: recovered pair-wise structure feature, (B, L_max, L_max, mpnn_dim)
          dist_pred: predicted distance map, (B, L_max, L_max)
          E_idx_pred: predicted edge index, each row reders to the index of 
              the neighbor nodes; (B, L_max, top-k for neighbours)
        """

        ######## for joint version, take the struc features ad inputs #########

        if 'joint' in self.projector_version:
            if struc_feat is None:
                struc_feat, struc_pair_feat = seq_feat, seq_pair_feat
            seq_feat, seq_pair_feat = None, None

        #######################################################################
        # Upsampling: Increase the dimension along the sequence (m to L) 
        #######################################################################

        if self.downsampling:
            if (seq_feat is not None) or (seq_pair_feat is not None):
                seq_feat, seq_pair_feat = self.sampler.singlemodal_upsample(
                    node_feat=seq_feat, pair_feat=seq_pair_feat, L_max=L_max
                )  # seq_feat: (B, L_max, dim); seq_pair_feat: (B, L_max, L_max, dim)

            if (struc_feat is not None) or (struc_pair_feat is not None):
                struc_feat, struc_pair_feat = self.sampler.singlemodal_upsample(
                    node_feat=struc_feat, pair_feat=struc_pair_feat, L_max=L_max
                )  # struc_feat: (B, L_max, dim); struc_pair_feat: (B, L_max, L_max, dim)

        #######################################################################
        # Edge Prediction (with latent feature)
        #######################################################################

        with_edge_feature = self.with_edge_feature
        if struc_feat is None or struc_pair_feat is None:
            with_edge_feature = False

        if not with_edge_feature:
            dist_pred, E_idx_pred = None, None

        elif self.edge_pred_version == 'latent':
            dist_pred, E_idx_pred = self.edge_pred(
                struc_feat, struc_pair_feat, mask
            )
            # E_idx: edge index, each row reders to the index of the neighbor 
            #     nodes; (B, L_max, top-k for neighbours)

        #######################################################################
        # Projecting 
        #######################################################################

        if 'joint' in self.projector_version:
            ### joint projector
            (seq_feat,       # (B, L_max, esm_dim)
             struc_feat,     # (B, L_max, mpnn_dim)
             seq_pair_feat,  # (B, L_max, L_max, esm_dim)
             struc_pair_feat,  # (B, L_max, L_max, mpnn_dim)
            ) = self.projector.decode(
                struc_feat, struc_pair_feat, mask = mask
            )

        else:
            ### for structure features
            if struc_feat is not None:
                struc_feat, struc_pair_feat = self.struc_projector.decode(
                    struc_feat, struc_pair_feat, mask = mask
                )  # struc_feat: (B, L_max, mpnn_dim); struc_pair_feat: (B, L_max, L_max, mpnn_dim)

            ### for sequence features 
            if seq_feat is not None:
                seq_feat, seq_pair_feat = self.seq_projector.decode(
                    seq_feat, seq_pair_feat, mask = mask
                )  # struc_feat: (B, L_max, esm_dim); struc_pair_feat: (B, L_max, L_max, esm_dim)

        #######################################################################
        # Edge Prediction (with proteinMPNN feature)
        #######################################################################

        if with_edge_feature and self.edge_pred_version != 'latent':
            dist_pred, E_idx_pred = self.edge_pred(
                struc_feat, struc_pair_feat, mask
            )
            # E_idx: edge index, each row reders to the index of the neighbor 
            #     nodes; (B, L_max, top-k for neighbours)

        #######################################################################
        # Edge feature match: (B, L_max, L_max, *) to (B, L_max, K, *) for 
        #     ProteinMPNN. 
        #######################################################################
        if struc_pair_feat is not None:
            if with_true_adj and E_idx is not None:
                E_idx_sele = E_idx
            else:
                E_idx_sele = E_idx_pred
            struc_pair_feat = gather_edges(struc_pair_feat, E_idx_sele)

        return (
            seq_feat, seq_pair_feat,
            struc_feat, struc_pair_feat,
            dist_pred, E_idx_pred
        )


    def edge_pred(self, node_feat, edge_feat, mask = None):
        """predict the distance matrices and neighbors.

        Args:
            node_feat: node features; (B, L)
            edge_feat: 
            mask: 1 for valid residues and 0 for others; (B, L_max)

        Returns:
            dist_mat: distance matrix; (B, L, L)
            E_idx: kNN indexes; (B, L, K)
        """

        dist_pred = self.dist_predictor(
            node_feat, edge_feat, mask
        )  # (N, L_max, L_max)

        if mask is not None:
            mask = mat_outer_prod(mask.float(), mask.float())
            dist_pred[mask == 0] = torch.inf

        _, e_idx = torch.topk(dist_pred,
            np.minimum(self.k_neighbors, dist_pred.shape[1]), 
            dim=-1, largest=False
        )
        return dist_pred, e_idx

    ###########################################################################
    # Loss
    ###########################################################################

    def loss(self,
        out,
        seq_mask = None,
        struc_mask = None,
        with_esm_emb_recover_loss = True,
        with_mpnn_emb_recover_loss = True,
        with_joint_emb_loss = True,
        with_dist_pred_loss = True,
        with_contrastive = False,
        contrast_dist = 'norm',
        with_size_regular = False,
        posi_weight = 1.0,
        posi_threshold = 1.0,
        nega_threshold = 2.0,
        mode = 'train',
        loss_weight = None,
        summary = True
    ):
        loss_name_dict = {}
        ### pair masks
        pair_mask = mat_outer_prod(
            seq_mask.float(), seq_mask.float()
        )
        pair_mask = (pair_mask == 1)  # (B, L_max, L_max)
        hidden_mask = None

        ###### embedding recover loss (for projector) ######
        if with_esm_emb_recover_loss:
            loss_name_dict['esm_feat_recover'] = (
                'seq_feat_recover', 'seq_feat_esm', seq_mask
            )
            loss_name_dict['esm_pair_recover'] = (
                'seq_pair_feat_recover', 'seq_pair_feat_esm', pair_mask
            )

        if with_mpnn_emb_recover_loss:
            loss_name_dict['mpnn_feat_recover'] = (
                'struc_feat_recover', 'struc_feat_mpnn', struc_mask
            )
            loss_name_dict['mpnn_pair_recover'] = (
                'struc_pair_feat_recover', 'struc_pair_feat_mpnn', struc_mask
            )

        ###### joint embedding loss ######
        if with_joint_emb_loss:
            loss_name_dict['resi-wise-simi'] = (
                'seq_feat_hidden', 'struc_feat_hidden', hidden_mask
            )
            loss_name_dict['pair-wise-simi'] = (
                'seq_pair_feat_hidden', 'struc_pair_feat_hidden', hidden_mask
            )

        ###### distance prediction loss ######
        if with_dist_pred_loss:
            loss_name_dict['dist_pred'] = ('dist_true', 'dist_pred', pair_mask)

        #################### loss calculation #################################
        def valid_key_check(dictionary, key):
            return (key in dictionary) and (dictionary[key] is not None)

        loss_all = {}
        for loss_key in loss_name_dict:
            if valid_key_check(out, loss_name_dict[loss_key][0]) \
            and valid_key_check(out, loss_name_dict[loss_key][1]):
                #print(loss_key, out[loss_name_dict[loss_key][0]].shape, out[loss_name_dict[loss_key][1]].shape)

                loss_all[loss_key] = emb_loss(
                    out[loss_name_dict[loss_key][0]],
                    out[loss_name_dict[loss_key][1]],
                    mask = loss_name_dict[loss_key][2],
                    mode = mode,
                )

        ##################### contrastive loss ################################
        ###### concatenation version ######
        if 'joint' in self.projector_version:
            ### consine similarity
            if with_contrastive and contrast_dist == 'cosine':
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)

                if valid_key_check(out, 'struc_feat_hidden') \
                and valid_key_check(out, 'node_feat_nega'):
                    B = out['struc_feat_hidden'].shape[0]
                    ### maximize the difference 
                    loss_all['contrastive_node'] = cos(
                        out['struc_feat_hidden'].reshape(B, -1), 
                        out['node_feat_nega'].reshape(B, -1),
                    ).mean()
           
                if valid_key_check(out, 'struc_pair_feat_hidden') \
                and valid_key_check(out, 'pair_feat_nega'):
                    B = out['struc_pair_feat_hidden'].shape[0]
                    ### maximize the difference 
                    loss_all['contrastive_pair'] = cos(
                        out['struc_pair_feat_hidden'].reshape(B, -1), 
                        out['pair_feat_nega'].reshape(B, -1),
                    ).mean()

            ### mse loss
            elif with_contrastive:
                if valid_key_check(out, 'struc_feat_hidden') \
                and valid_key_check(out, 'node_feat_nega'):
                    ### maximize the difference 
                    loss_all['contrastive_node'] = - emb_loss(
                        out['struc_feat_hidden'], out['node_feat_nega'], hidden_mask,
                    ) 
           
                if valid_key_check(out, 'struc_pair_feat_hidden') \
                and valid_key_check(out, 'pair_feat_nega'):
                    ### maximize the difference 
                    loss_all['contrastive_pair'] = - emb_loss(
                        out['struc_pair_feat_hidden'], out['pair_feat_nega'],
                    )

            if with_size_regular:
                ### minimize the range of the positive shape
                for key in ['struc_feat_hidden', 'struc_pair_feat_hidden']:
                    if valid_key_check(out, key):
                        loss_all['regu_%s' % key] = max(
                            torch.norm(out[key]) * posi_weight, posi_threshold
                        )
                ### maximize the range of the negative shape
                for key in ['node_feat_nega', 'pair_feat_nega']:
                    if valid_key_check(out, key):
                        loss_all['regu_%s' % key] = - min(
                            torch.norm(out[key]), nega_threshold
                        )

        ###### versatile version ######
        else:
            if with_contrastive:
                if valid_key_check(out, 'seq_feat_hidden') \
                and valid_key_check(out, 'struc_feat_hidden'):
                    loss_all['contrastive'] = contrastive_loss(
                        out['seq_feat_hidden'], out['struc_feat_hidden'], hidden_mask
                    )

                if valid_key_check(out, 'seq_pair_feat_hidden') \
                and valid_key_check(out, 'struc_pair_feat_hidden'):
                    loss_all['contrastive_pair'] = contrastive_loss(
                        out['seq_pair_feat_hidden'], 
                        out['struc_pair_feat_hidden'],
                    )

            if with_size_regular:
                ### minimize the range of the positive shape
                for key in [
                    'seq_feat_hidden', 'seq_pair_feat_hidden', 
                    'struc_feat_hidden', 'struc_pair_feat_hidden'
                ]:
                    if valid_key_check(out, key):
                        loss_all['regu_%s' % key] = max(
                            torch.norm(out[key]) * posi_weight, posi_threshold
                        )

        ##################### wrap up the losses ##############################
        if summary:
            loss_overall = 0.
            for key in loss_all:
                if loss_weight is not None and key in loss_weight:
                    loss_overall += loss_all[key] * loss_weight[key]
                else:
                    loss_overall += loss_all[key]
            loss_all['loss'] = loss_overall

        return loss_all

###############################################################################
# Joint Container
###############################################################################

class Codeisgn_Container(nn.Module):
    """Container of latent-diffusion model or autoencoder.

    Encoder:
      * ESM2 (for sequence)
      * ProteinMPNN encoder (for structure)
      * feature projectors 
      * downsampler
    
    Decoder:
      * upsampler
      * feature projectors
      * edge predictor (not needed for FCG version) 
      * ProteinMPNN decoder (for sequence)
      * ESMFold decoder (for structure)

    """

    def __init__(self, args):
        super(Codeisgn_Container, self).__init__()

        if not args.__contains__('seq_design_module'):
            args.seq_design_module = 'ProteinMPNN'
        if not args.__contains__('esmif_pad_version'):
            args.esmif_pad_version = 'zero'
        if not args.__contains__('with_terminus_token'):
            args.with_terminus_token = True

        self.args = args
        self.seq_design_module = args.seq_design_module
        self.esmif_pad_version = args.esmif_pad_version
        self.device = args.device
        self.with_terminus_token = args.with_terminus_token

        ############## whether add paddings for esm node-wise feature #########
        # only when with ESM-IF and (mlp or joint projector)
        if self.seq_design_module == 'ESM-IF':
            args.esm_add_padding = self.with_terminus_token
            args.with_edge_feature = False
        else:
            args.esm_add_padding = False
            args.with_edge_feature = True

        self.esm_add_padding = args.esm_add_padding
        self.with_edge_feature = args.with_edge_feature

        #######################################################################
        # External Autoencoder (based on SOTA methods)
        #######################################################################

        self.external_ae = PretrainedSOTA_Container(args)

        #######################################################################
        # Internal Autoencoder (projectors and up-down samplers)
        #######################################################################

        ######################## Dimension Setting ############################
        args.proj_seq_resi_dim = self.external_ae.esmfold.cfg.trunk.sequence_state_dim
        args.proj_seq_pair_dim = self.external_ae.esmfold.cfg.trunk.pairwise_state_dim

        if self.seq_design_module == 'ProteinMPNN':
            args.proj_struc_resi_dim = args.proteinMPNN_hidden_dim
            args.proj_struc_pair_dim = args.proteinMPNN_hidden_dim
            args.with_edge_feature = True

        elif self.seq_design_module == 'ESM-IF':
            args.proj_struc_resi_dim = 512  # ESMIF_hidden_dim
            args.proj_seq_pair_dim = None
            args.proj_struc_pair_dim = None
            args.with_edge_feature = False

        else:
            raise ValueError(
                'Invalid sequence design model named %s!' % args.seq_design_module
            )

        self.with_edge_feature = args.with_edge_feature

        ######################## internal autoencoder #########################
        if args.seq_design_module == 'ESM-IF' and self.with_terminus_token:
            args.max_length += 2

            if args.esmif_pad_version != 'zero':
                self.esm_pad = nn.parameter.Parameter(
                    torch.randn(args.proj_seq_resi_dim), requires_grad = True
                ) 

        self.internal_ae = RepresentationAutoencoder(args) 

        #######################################################################
        # Diffusion Modules
        #######################################################################

        if not args.__contains__('diffusion_version'):
            args.diffusion_version = None

        ###### latent diffusion ######
        if args.diffusion_version == 'latent':
            self.diffusion_model = DenoiseDiffusionProbobilisticModel(args)
            print('Latent diffusion model constructed.')

        ###### VAE ######
        elif args.diffusion_version == 'vae':
            (self.vae_map, 
             self.vae_mu_layer, 
             self.vae_sigma_layer
            ) = self.vae_construct()

            if self.with_edge_feature:
                (self.vae_map_pair,
                 self.vae_mu_layer_pair,
                 self.vae_sigma_layer_pair
                ) = self.vae_construct()

            print('VAE constructed.')

        ###### autoencoder ######
        elif args.diffusion_version is None:
            print('Autoencoder constructed.')

        ###### undefined ######
        else:
            raise NameError(
                'Error! No diffusion version named %s!' % args.diffusion_version
            )

        #######################################################################
        # oracle
        #######################################################################

        if args.__contains__('with_cyclic_loss') and args.with_cyclic_loss:
            self.oracle = load_mpnn_oracle(args)
        else:
            self.oracle = None 

    #############################################################################
    # Utility Functions                                                         #
    #############################################################################

    def esmif_feat_pad(self, seq_feat, mask):
        """Add paddings to match the sequence feature (N,L,*) and structure 
        feature (N, L+2, *). Only for ESM-IF.
        """

        seq_feat[mask == 0] = 0
        seq_feat = F.pad(
           seq_feat, (0,0,1,1), 'constant', 0
        ) # (B, L_max+2, dim)
        if self.esmif_pad_version != 'zero':
            N = mask.shape[0]
            protein_size = mask.sum(-1)  # (N,)
            seq_feat[:,0] = self.esm_pad 
            seq_feat[torch.arange(N), protein_size + 1] = self.esm_pad

        return seq_feat

    
    def vae_construct(self):
        """Construct a VAE-based module."""

        if not self.args.__contains__('vae_layer_num'):
            self.vae_layer_num = 1
        else:
            self.vae_layer_num = self.args.vae_layer_num

        if self.vae_layer_num == 1:
            vae_map = None
        elif self.vae_layer_num == 2:
            vae_map = nn.Linear(
                self.args.latent_dim, self.args.latent_dim
            )
        else:
            vae_map = MultiLayerPerceptron(
                self.args.latent_dim, 
                [self.args.latent_dim] * (self.vae_layer_num - 3), 
                self.args.latent_dim
            )

        vae_mu_layer = nn.Linear(self.args.latent_dim, self.args.latent_dim)
        vae_sigma_layer = nn.Linear(self.args.latent_dim, self.args.latent_dim)

        return vae_map, vae_mu_layer, vae_sigma_layer


    #############################################################################
    # Modules                                                                   #
    #############################################################################

    ############################# VAE map #######################################
    def VAE_projection(self,
        latent_feat, latent_feat_pair = None, sampling = True,
    ):
        ###### node-wise feature ######
        if self.vae_map is not None:
            latent_feat = self.vae_map(latent_feat)
        latent_feat = F.relu(latent_feat)
        mu_feat = self.vae_mu_layer(latent_feat)
        sigma_feat = self.vae_sigma_layer(latent_feat)

        if sampling:
            latent_feat = torch.randn(latent_feat.shape).to(sigma_feat.device)
            latent_feat = latent_feat * sigma_feat + mu_feat
        else:
            latent_feat = None

        ###### pair-wise feature ######
        if self.with_edge_feature and (latent_feat_pair is not None):
            if self.vae_map_pair is not None:
                latent_feat_pair = self.vae_map_pair(latent_feat_pair)
                latent_feat_pair = F.relu(latent_feat_pair)

            mu_feat_pair = self.vae_mu_layer_pair(latent_feat_pair)
            sigma_feat_pair = self.vae_sigma_layer_pair(latent_feat_pair)

            if sampling:
                latent_feat_pair = torch.randn(
                    latent_feat_pair.shape
                ) * sigma_feat_pair + mu_feat_pair
            else:
                latent_feat_pair = None
        else:
            mu_feat_pair, sigma_feat_pair = None, None

        return (
            mu_feat, sigma_feat, latent_feat, 
            mu_feat_pair, sigma_feat_pair, latent_feat_pair
        )

    ############################# diffusion #####################################

    def diffusion_infer(self):
        pass

    def diffusion(self, node_feat, edge_feat, E_idx):
        pass

    ################################ Encoder ####################################
    def internal_encoder(self,
        seq_feat: torch.Tensor=None, 
        seq_pair_feat: torch.Tensor=None,
        struc_feat: torch.Tensor=None, 
        struc_pair_feat: torch.Tensor=None,
        E_idx: torch.Tensor=None, 
        mask: torch.Tensor=None,
        with_contrastive: bool=False,
    ):
        """Encoding with the projectors and downsamplers"""

        ########## no pairwise feat needed for ESM-IF #########################
        if not self.with_edge_feature:
            seq_pair_feat = None
            struc_pair_feat = None

        ###### make the length of the two embeddings match for (ESM-IF) ######
        if self.esm_add_padding:
            if seq_feat is not None:
                seq_feat = self.esmif_feat_pad(seq_feat, mask) # (N,L_max+2,*)
            if mask is not None:
                mask = esmif_mask_transform(mask)  # (N, L_max+2)

        elif (seq_feat is not None) and (struc_feat is not None) \
        and seq_feat.shape[1] != struc_feat.shape[1]:
            struc_feat = esmif_remove_ends(
                struc_feat, mask.sum(-1)
            )

        ###### projecting ######
        embedding_all = self.internal_ae.encoding(
            seq_feat = seq_feat, seq_pair_feat = seq_pair_feat,
            struc_feat = struc_feat, struc_pair_feat = struc_pair_feat,
            E_idx = E_idx, mask = mask, with_contrastive = with_contrastive
        )
        return embedding_all


    def encoder(self,
        X: torch.Tensor,
        seq: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        chain_encoding_all: T.Optional[torch.Tensor] = None,
        protein_size: T.Optional[torch.Tensor] = None,
        with_sequence: bool = True,
        with_structure: bool = True,
        with_projector: bool = True,
        sampling: bool = True,
        with_contrastive: bool=False,
    ):
        """Sequence cncoding with ESMFold and Structure encoding with proteinMPNN.

        Args:
            X: coordinates info, (B, L_max, atom_num=4, 3)
            seq: sequence, (B, L_max)
            mask: 1 for valid residues and 0 for others, (B, L_max) 
            residx: from 1 to L for single-chain, (B, L_max)
            masking_pattern: Optional masking to pass to the input. Binary tensor 
                of the same size as `seq`. Positions with 1 will be masked. 
                ESMFold sometimes produces different samples when different masks 
                are provided.
            chain_encoding_all: perform like a chain mask (1 for valid token) for single-chain, (B, L_max)
            protein_size: tensor of the protein sizes, (B,)
            with_sequence: whether have esm2 embedding.
            with_structure: whether have ProteinMPNN or esm-if embedding.
            with_projector: whether apply the projector.
            sampling: for VAE version only; true for training and false for encoding

        Returns:
            * L = m if with downsampling else L_max
            seq_feat_esm: residue-wise feature from ESMFold, (B, L_max, esm_dim) 
            seq_pair_feat_esm: pair-wise feature from ESMFold, (B, L_max, L_max, esm_dim)
            struc_feat_mpnn: residue-wise feature from ProteinMPNN, (B, L_max, mpnn_dim) 
            struc_pair_feat_mpnn: pair-wise feature from ProteinMPNN, (B, L_max, K, mpnn_dim)
            dist_map: distance matrix, (B, L_max, L_max)
            E_idx: edge indexes from ProteinMPNN, (B, L_max, K)
            for joint projector:
                node_feat: (dimension reduced) residue-wise feature;
                    (B, L, dim) 
                pair_feat: (dimension reduced) pair-wise feature;
                    (B, L, L, dim)
            for separate projector: 
                seq_feat: (dimension reduced) residue-wise sequence feature;
                    (B, L, dim)
                seq_pair_feat: (dimension reduced) pair-wise sequence feature;
                    (B, L, L, dim)
                struc_feat: (dimension reduced) residue-wise structure feature;
                    (B, L, dim) 
                struc_pair_feat: (dimension reduced) pair-wise structure feature;
                    (B, L, L, dim)
        """

        if with_projector and 'joint' in self.args.projector_version:
            ### for the joint version both features are required
            with_sequence = True
            with_structure = True

        if not with_sequence:
            seq = None
        else:
            B, L_max = seq.shape

        if not with_structure:
            X = None
        else:
            B, L_max, _, _ = X.shape

        #######################################################################
        # External Encoding 
        #######################################################################

        (seq_feat_esm, seq_pair_feat_esm,
         struc_feat_mpnn, struc_pair_feat_mpnn,
         dist_map, E_idx,
        ) = self.external_ae.encoding(
             X = X,
             seq = seq,
             mask = mask,
             residx = residx,
             masking_pattern = masking_pattern,
             chain_encoding_all = chain_encoding_all,
             protein_size = protein_size
        )

        ########### early return is no projector needed #######################
        if not with_projector:
            return (
                seq_feat_esm, seq_pair_feat_esm,
                struc_feat_mpnn, struc_pair_feat_mpnn,
                dist_map, E_idx,
            )

        #######################################################################
        # Internal Encoding (Projectors and Downsamplers) 
        #######################################################################

        embedding_all = self.internal_encoder(
            seq_feat = seq_feat_esm, seq_pair_feat = seq_pair_feat_esm, 
            struc_feat = struc_feat_mpnn, struc_pair_feat = struc_pair_feat_mpnn,
            E_idx = E_idx, mask = mask, with_contrastive = with_contrastive
        )

        ########################## joint projector #############################
        if 'joint' in self.args.projector_version:
            node_feat, pair_feat, node_feat_nega, pair_feat_nega = embedding_all

            if self.args.diffusion_version != 'vae':
                return (
                    node_feat, pair_feat, node_feat_nega, pair_feat_nega,
                    seq_feat_esm, seq_pair_feat_esm,
                    struc_feat_mpnn, struc_pair_feat_mpnn,
                    dist_map, E_idx
                )

            (mu_feat, sigma_feat, latent_feat, 
             mu_feat_pair, sigma_feat_pair, latent_feat_pair
            ) = self.VAE_projection(
                node_feat, pair_feat, sampling = sampling,
            )
            if sampling:
                node_feat, pair_feat = latent_feat, latent_feat_pair
            else:
                node_feat, pair_feat = mu_feat, mu_feat_pair

            return (
                node_feat, pair_feat, node_feat_nega, pair_feat_nega,
                seq_feat_esm, seq_pair_feat_esm,
                struc_feat_mpnn, struc_pair_feat_mpnn,
                dist_map, E_idx,
                mu_feat, sigma_feat, mu_feat_pair, sigma_feat_pair
            )

        ####################### separate projector #############################
        else:
            seq_feat, seq_pair_feat, struc_feat, struc_pair_feat = embedding_all

            if self.args.diffusion_version != 'vae':
                return (
                    seq_feat_esm, seq_pair_feat_esm, seq_feat, seq_pair_feat, 
                    struc_feat_mpnn, struc_pair_feat_mpnn, struc_feat, struc_pair_feat, 
                    dist_map, E_idx
                )

            (seq_mu_feat, seq_sigma_feat, seq_latent_feat, 
             seq_mu_feat_pair, seq_sigma_feat_pair, seq_latent_feat_pair
            ) = self.VAE_projection(
                seq_feat, seq_pair_feat, sampling = sampling
            )
            (struc_mu_feat, struc_sigma_feat, struc_latent_feat, 
             struc_mu_feat_pair, struc_sigma_feat_pair, struc_latent_feat_pair
            ) = self.VAE_projection(
                struc_feat, struc_pair_feat, sampling = sampling
            )

            if sampling:
                seq_feat, seq_pair_feat = seq_latent_feat, seq_latent_feat_pair
                struc_feat, struc_pair_feat = struc_latent_feat, struc_latent_feat_pair
            else:
                seq_feat, seq_pair_feat = seq_mu_feat, seq_mu_feat_pair
                struc_feat, struc_pair_feat = struc_mu_feat, struc_mu_feat_pair

            return (
                seq_feat_esm, seq_pair_feat_esm, seq_feat, seq_pair_feat, 
                struc_feat_mpnn, struc_pair_feat_mpnn, struc_feat, struc_pair_feat, 
                dist_map, E_idx,
                seq_mu_feat, seq_sigma_feat, seq_mu_feat_pair, seq_sigma_feat_pair,
                struc_mu_feat, struc_sigma_feat, struc_mu_feat_pair, struc_sigma_feat_pair
            )


    ################################ Decoder ####################################
    def decoder(self,
        ###### for internal decoding ######
        seq_feat: torch.Tensor,
        seq_pair_feat: torch.Tensor,
        struc_feat: torch.Tensor = None,
        struc_pair_feat: torch.Tensor = None,
        with_projector: bool = True,
        mask: T.Optional[torch.Tensor] = None,
        with_true_adj: bool = False,
        E_idx: T.Optional[torch.Tensor] = None,
        L_max: T.Optional[int] = None,
        ###### for external decoding (final decoding) ######
        mode: T.Optional[str] = 'train',
        with_final_seq: bool = False,
        with_final_struc: bool = False,
        with_aa_sampling: bool = False,
        with_true_seq: bool = False,
        seq_true: T.Optional[torch.Tensor] = None,
        chain_M: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        temperature: T.Optional[float] = 1.0,
        topk: T.Optional[int] = 1,
        num_recycles: T.Optional[int] = 3,
        ### for inference only
        protein_size: T.Optional[int] = None,
        randn: T.Optional[torch.Tensor] = None,
        omit_AAs_np = None,
        bias_AAs_np = None,
        omit_AA_mask = None,
        pssm_coef = None,
        pssm_bias = None,
        pssm_multi = None,
        pssm_log_odds_flag = None,
        pssm_log_odds_mask = None,
        pssm_bias_flag = None,
        bias_by_res = None
    ):
        """Sequence decoding with proteinMPNN and Structure decoding with ESMFold.

        * L = L_max (without downsampling) or m (with downsampling).

        Args:
            mode: "train" or "eval".
            ********************* for internal decoder *************************
            seq_feat: residue-wise features for sequence, (B, L, node_dim)
            seq_pair_feat: edge-wise features for sequence, (B, L, L, edge_dim)
            struc_feat: residue-wise features for structure, (B, L, node_dim)
            struc_pair_feat: edge-wise features for structure, (B, L, L, edge_dim)
            mask: 1 for valid residues and 0 for others, (B, L_max)
            with_true_adj: whether apply the true ajacency matrix for structure
                features and ProteinMPNN.
            E_idx: true ajacency matrix; (B, L_max, K)
            L_max: maximum protein size in the batch.
            ****************** for sequence sampling ***************************
            with_final_seq: whether output the final sequence; if False,
                the arguments in this module are not needed.
            seq_true: true sequence, only for training, (B, L_max)
            chain_M: chain mask, 1.0 for the bits that need to be predicted, 0.0 
                for the bits that are given; (B, L_max)
            residx: from 1 to L for single-chain, (B, L_max); also needed for 
                sturcture sampling.
            temperature: temperature for sampling.
            >>>>>> for training only <<<<<<
            with_aa_sampling: whether sample the sequence from logit.
            topk: k value of the topk sampling, default = 1 (maximum sampling)
            >>>>>> for inference only <<<<<<
            randn: for decoding order, (B, L_max)
            protein_size: tensor of the protein sizes, (B,)
            Other parameters for ProteinMPNN: 
                omit_AAs_np, bias_AAs_np, omit_AA_mask,
                pssm_coef, pssm_bias, pssm_multi, pssm_log_odds_flag,
                pssm_log_odds_mask, pssm_bias_flag, bias_by_res
            ****************** for sturcture prediction ***********************
            with_final_struc: whether output the final structure; if False,
                the arguments in this module are not needed.
            with_true_seq: whether use the true sequence for ESMFold. 
            num_recycles (int): How many recycle iterations to perform. If None, 
                defaults to training max recycles, which is 3.

        Returns:
            seq_feat: recovered residue-wise sequence feature, 
                (B, L_max, esm_dim)
            seq_pair_feat: recovered pair-wise sequence feature, 
                (B, L_max, L_max, esm_dim)
            struc_feat: recovered residue-wise structure feature, 
                (B, L_max, mpnn_dim)
            struc_pair_feat: recovered pair-wise structure feature, 
                (B, L_max, L_max, mpnn_dim)
            dist_pred: predicted distance map, (B, L_max, L_max)
            E_idx_pred: predicted edge index, each row reders to the index of 
                the neighbor nodes; (B, L_max, top-k for neighbours)
            seq_logit: sequence probability matrix
            aa: predicted sequence
            struc: predicted structure
        """

        if not (with_projector or with_final_seq or with_final_struc):
            raise ValueError('Nothing to be decoded!')

        if L_max is None:
            if protein_size is None:
                protein_size = mask.sum(-1)  # (N,)
            L_max = max(protein_size)

        if self.esm_add_padding and mask is not None:
            mask_projector = esmif_mask_transform(mask)  # (N, L_max+2)
        else:
            mask_projector = mask

        #######################################################################
        # Internal Decoding (Upsamplers and Projectors)
        #######################################################################

        if with_projector:
 
            if with_final_seq and struc_feat is None:
                struc_feat, struc_pair_feat = seq_feat, seq_pair_feat
            if with_final_struc and seq_feat is None:
                seq_feat, seq_pair_feat = struc_feat, struc_pair_feat

            (seq_feat, seq_pair_feat,
             struc_feat, struc_pair_feat,
             dist_pred, E_idx_pred
            ) = self.internal_ae.decoding(
                struc_feat = struc_feat,
                struc_pair_feat = struc_pair_feat,
                seq_feat = seq_feat,
                seq_pair_feat = seq_pair_feat,
                mask = mask_projector,
                E_idx = E_idx,
                L_max = L_max,
            )
            # seq_feat: (N,L_max,*) when or (N,L_max+2,*) when ESM-IF
            # seq_pair_feat: None for ESM-IF and (N,L_max,L_max,*) for ProteinMPNN
            # struc_feat: (N,L_max+2,*) aor ESM-IF and (N,L_max,*) for ProteinMPNN
            # struc_pair_feat: None for ESM-IF and (N,L_max,K,*) for ProteinMPNN
            # dist_pred: None for ESM-IF and (N,L_max,L_max) for ProteinMPNN
            # E_idx_pred: None for ESM-IF and (N,L_max,K) for ProteinMPNN

            if self.esm_add_padding:
                seq_feat = seq_feat[:, 1:-1] # (N,L_max,*)
                if mask is not None:
                    seq_feat[mask == 0] = 0 

            ############## early return without final outputs #####################
            if not (with_final_seq or with_final_struc):
                # If just need to predict the original features, not need to 
                # generate the final outputs.
                return (
                    seq_feat, seq_pair_feat, 
                    struc_feat, struc_pair_feat, 
                    dist_pred, E_idx_pred,
                    None, None, None    # seq_logit, aa, struc are set to None 
                )

        #######################################################################
        # final decoding:
        #     based on the hidden features generate the sequences and modalities
        #######################################################################

        if with_final_seq:
            struc_feat_in, struc_pair_feat_in = struc_feat, struc_pair_feat
        else:
            struc_feat_in, struc_pair_feat_in = None, None

        if with_final_struc:
            seq_feat_in, seq_pair_feat_in = seq_feat, seq_pair_feat
        else:
            seq_feat_in, seq_pair_feat_in = None, None

        if with_true_adj:
            E_idx_sele = E_idx
        else:
            E_idx_sele = E_idx_pred

        ####################### for training ##################################
        if mode == 'train': 
            seq_logit, aa_gen, struc = self.external_ae.decoding_train(
                seq_feat = seq_feat_in,
                seq_pair_feat = seq_pair_feat_in,
                struc_feat = struc_feat_in,
                struc_pair_feat = struc_pair_feat_in,
                seq_true = seq_true,
                mask = mask,
                mask_esmif = mask_projector,
                with_aa_sampling = with_aa_sampling, 
                chain_M = chain_M, 
                residx = residx,
                E_idx = E_idx_sele,
                topk = topk,
                with_true_seq = with_true_seq,
                num_recycles = num_recycles,
            )

        ####################### for inference ##################################
        else:
            seq_logit = None
            aa_gen, struc = self.external_ae.decoding_inference(
                decode_seq = with_final_seq,
                decode_struc = with_final_struc,
                seq_feat = seq_feat_in,
                seq_pair_feat = seq_pair_feat_in,
                struc_feat = struc_feat_in,
                struc_pair_feat = struc_pair_feat_in,
                mask = mask,
                mask_esmif = mask_projector,
                residx = residx,
                E_idx = E_idx_sele,
                randn = randn,
                chain_encoding_all = chain_M, 
                with_true_seq = with_true_seq,
                seq_true = seq_true,
                protein_size = protein_size, 
                temperature = temperature,
                num_recycles = num_recycles,
                omit_AAs_np = omit_AAs_np,
                bias_AAs_np = bias_AAs_np,
                omit_AA_mask = omit_AA_mask,
                pssm_coef = pssm_coef,
                pssm_bias = pssm_bias,
                pssm_multi = pssm_multi,
                pssm_log_odds_flag = pssm_log_odds_flag,
                pssm_log_odds_mask = pssm_log_odds_mask,
                pssm_bias_flag = pssm_bias_flag,
                bias_by_res = bias_by_res
            )

        return (
            seq_feat, seq_pair_feat,
            struc_feat, struc_pair_feat,
            dist_pred, E_idx_pred,
            seq_logit, aa_gen, struc
        )


    #############################################################################
    # Feedforward Functions                                                     #
    #############################################################################

    def forward(self,
        ###### for encoder ######
        struc: torch.Tensor,
        seq: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        chain_encoding_all: T.Optional[torch.Tensor] = None,
        protein_size: T.Optional[torch.Tensor] = None,
        with_in_sequence: bool = True,
        with_in_structure: bool = True,
        with_projector: bool = True,
        with_contrastive: bool=False,
        ###### for decoder ###### 
        mode: T.Optional[str] = 'train',
        emb_exchange: T.Optional[bool] = False,
        with_true_adj: T.Optional[bool] = True,
        with_final_seq: bool = False,
        with_final_struc: bool = False,
        with_aa_sampling: bool = True,
        with_true_seq: bool = False,
        temperature: T.Optional[float] = 1.0,
        topk: T.Optional[int] = 1,
        num_recycles: T.Optional[int] = 3,
        ### for inference only
        input_mode: str = 'seq',
        randn: T.Optional[torch.Tensor] = None,
        omit_AAs_np = None,
        bias_AAs_np = None,
        omit_AA_mask = None,
        pssm_coef = None,
        pssm_bias = None,
        pssm_multi = None,
        pssm_log_odds_flag = None,
        pssm_log_odds_mask = None,
        pssm_bias_flag = None,
        bias_by_res = None
    ):
        """
        Args:
            ************************* for encoder *****************************
            struc: coordinates info, (B, L_max, atom_num=4, 3)
            seq: sequence, (B, L_max)
            mask: 1 for valid residues and 0 for others, (B, L_max)
            residx: from 1 to L for single-chain, (B, L_max)
            masking_pattern: Optional masking to pass to the input. Binary tensor 
                of the same size as `seq`. Positions with 1 will be masked. 
                ESMFold sometimes produces different samples when different masks 
                are provided.
            chain_encoding_all: perform like a chain mask (1 for valid token) for 
                single-chain, (B, L_max)
            protein_size: tensor of the protein sizes, (B,)
            ************************* hidden space ****************************
            emb_exchange: whether exchange the representations in the hidden space.
            ********************** for decoder ********************************
            with_true_adj: whether use the true adjacency matrix (E_idx)
            with_final_outputs: whether generate the final outputs
            num_recycles (int): How many recycle iterations to perform. If None, 
                defaults to training max recycles, which is 3.
            topk: k value of the topk sampling, default = 1 (maximum sampling)
            temperature: temperature for sampling.
        """
        ########################################################################
        # Encoding 
        ########################################################################

        encoder_out = self.encoder(
            X=struc, seq=seq, mask=mask, 
            residx=residx, masking_pattern=masking_pattern,
            chain_encoding_all = chain_encoding_all, protein_size = protein_size,
            with_sequence = with_in_sequence, with_structure = with_in_structure, 
            with_projector = with_projector, sampling = True,
            with_contrastive = with_contrastive
        )
        node_feat_nega, pair_feat_nega = None, None

        ###### no internal autoencoding ######
        if not with_projector:
            (seq_feat_esm, seq_pair_feat_esm,
             struc_feat_mpnn, struc_pair_feat_mpnn,
             dist_map, E_idx,
            ) = encoder_out
            
            seq_feat, seq_pair_feat = seq_feat_esm, seq_pair_feat_esm
            struc_feat, struc_pair_feat = struc_feat_mpnn, struc_pair_feat_mpnn

        ###### joint internal autoencoding ######
        elif 'joint' in self.args.projector_version and self.args.diffusion_version == 'vae':
            (seq_feat, seq_pair_feat, node_feat_nega, pair_feat_nega,
             seq_feat_esm, seq_pair_feat_esm,
             struc_feat_mpnn, struc_pair_feat_mpnn,
             dist_map, E_idx, 
             seq_mu_feat, seq_sigma_feat, seq_mu_feat_pair, seq_sigma_feat_pair
            ) = encoder_out
            struc_feat, struc_pair_feat = None, None
            struc_mu_feat, struc_sigma_feat = None, None
            struc_mu_feat_pair, struc_sigma_feat_pair = None, None

        elif 'joint' in self.args.projector_version:
            (seq_feat, seq_pair_feat, node_feat_nega, pair_feat_nega,
             seq_feat_esm, seq_pair_feat_esm,
             struc_feat_mpnn, struc_pair_feat_mpnn,
             dist_map, E_idx
            ) = encoder_out
            struc_feat, struc_pair_feat = None, None
            seq_mu_feat, seq_sigma_feat = None, None
            seq_mu_feat_pair, seq_sigma_feat_pair = None, None
            struc_mu_feat, struc_sigma_feat = None, None
            struc_mu_feat_pair, struc_sigma_feat_pair = None, None

        ###### separate internal autoencoding ######
        elif self.args.diffusion_version == 'vae':
            (seq_feat_esm, seq_pair_feat_esm, seq_feat, seq_pair_feat,
             struc_feat_mpnn, struc_pair_feat_mpnn, struc_feat, struc_pair_feat,
             dist_map, E_idx,
             seq_mu_feat, seq_sigma_feat, seq_mu_feat_pair, seq_sigma_feat_pair,
             struc_mu_feat, struc_sigma_feat, struc_mu_feat_pair, struc_sigma_feat_pair
            ) = encoder_out

        else:
            (seq_feat_esm, seq_pair_feat_esm, seq_feat, seq_pair_feat,
             struc_feat_mpnn, struc_pair_feat_mpnn, struc_feat, struc_pair_feat,
             dist_map, E_idx
            ) = encoder_out
            seq_mu_feat, seq_sigma_feat = None, None
            seq_mu_feat_pair, seq_sigma_feat_pair = None, None
            struc_mu_feat, struc_sigma_feat = None, None
            struc_mu_feat_pair, struc_sigma_feat_pair = None, None

        ########################################################################
        # Latent Space 
        ########################################################################

        ########################### cross embedding ###########################

        if (not with_projector) or 'joint' in self.args.projector_version:
            emb_exchange = False

        if emb_exchange:
            seq_feat, struc_feat = struc_feat, seq_feat
            seq_pair_feat, struc_pair_feat = struc_pair_feat, seq_pair_feat

        ########################### latent features ###########################

        ###### latent diffusion ######
        if self.args.diffusion_version == 'latent':
            seq_feat, seq_pair_feat = self.diffusion(
                seq_feat, seq_pair_feat, E_idx
            )
            struc_feat, struc_pair_feat = self.diffusion(
                struc_feat, struc_pair_feat, E_idx
            )

        ########################################################################
        # Decoding 
        ########################################################################

        L_max = seq.shape[-1]
        if self.seq_design_module == 'ESM-IF' and self.with_terminus_token:
            L_max += 2

        (seq_feat_recover, seq_pair_feat_recover,
         struc_feat_recover, struc_pair_feat_recover,
         dist_pred, E_idx_pred, seq_logit, aa, struc
        ) = self.decoder(
            with_projector = with_projector,
            seq_feat = seq_feat,
            seq_pair_feat = seq_pair_feat,
            struc_feat = struc_feat,
            struc_pair_feat = struc_pair_feat,
            mask = mask,
            with_true_adj = with_true_adj,
            E_idx = E_idx,
            L_max = L_max,
            mode = mode,
            with_final_seq = with_final_seq,
            with_final_struc = with_final_struc,
            with_aa_sampling = with_aa_sampling,
            with_true_seq = with_true_seq,
            seq_true = seq,
            chain_M = chain_encoding_all.float(), 
            residx = residx,
            topk = topk,
            temperature = temperature,
            num_recycles = num_recycles,
            protein_size = protein_size,
            randn = randn,
            omit_AAs_np = omit_AAs_np,
            bias_AAs_np = bias_AAs_np,
            omit_AA_mask = omit_AA_mask,
            pssm_coef = pssm_coef,
            pssm_bias = pssm_bias,
            pssm_multi = pssm_multi,
            pssm_log_odds_flag = pssm_log_odds_flag,
            pssm_log_odds_mask = pssm_log_odds_mask,
            pssm_bias_flag = pssm_bias_flag,
            bias_by_res = bias_by_res
        )

        ########################################################################
        # Final Outputs 
        ########################################################################

        if struc is None:
            struc = {}
        else:
            struc = {
                'positions': struc[0],
                'frames': struc[1], 
                'sidechain_frames': struc[2],
                'disto_logits': struc[3]
            }

        struc.update(
            {'log_prob': seq_logit,
             'seq': aa,
             'seq_feat_hidden': seq_feat,
             'seq_pair_feat_hidden': seq_pair_feat,
             'seq_feat_esm': seq_feat_esm,
             'seq_pair_feat_esm': seq_pair_feat_esm,
             'seq_feat_recover': seq_feat_recover,
             'seq_pair_feat_recover': seq_pair_feat_recover,
             'struc_feat_hidden': struc_feat,
             'struc_pair_feat_hidden': struc_pair_feat,
             'struc_feat_mpnn': struc_feat_mpnn,
             'struc_pair_feat_mpnn': struc_pair_feat_mpnn,
             'struc_feat_recover': struc_feat_recover,
             'struc_pair_feat_recover': struc_pair_feat_recover,
             'node_feat_nega': node_feat_nega,
             'pair_feat_nega': pair_feat_nega,
             'dist_true': dist_map,
             'dist_pred': dist_pred,
             'seq_mu_feat': seq_mu_feat, 
             'seq_sigma_feat': seq_sigma_feat,
             'seq_mu_feat_pair': seq_mu_feat_pair,
             'seq_sigma_feat_pair': seq_sigma_feat_pair,
             'struc_mu_feat': struc_mu_feat, 
             'struc_sigma_feat': struc_sigma_feat,
             'struc_mu_feat_pair': struc_mu_feat_pair, 
             'struc_sigma_feat_pair': struc_sigma_feat_pair,
            }
        )

        return struc


    def forward_simple(self,
        ###### for encoder ######
        seq_feat: torch.Tensor=None,
        seq_pair_feat: torch.Tensor=None,
        struc_feat: torch.Tensor=None,
        struc_pair_feat: torch.Tensor=None,
        E_idx: torch.Tensor=None,
        dist_map: torch.Tensor=None,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        chain_encoding_all: T.Optional[torch.Tensor] = None,
        protein_size: T.Optional[torch.Tensor] = None,
        with_in_sequence: bool = True,
        with_in_structure: bool = True,
        with_projector: bool = True,
        with_contrastive = False,
        ###### for decoder ###### 
        mode: T.Optional[str] = 'train',
        emb_exchange: T.Optional[bool] = False,
        seq_true: torch.Tensor=None,
        with_true_adj: T.Optional[bool] = True,
        with_final_seq: bool = False,
        with_final_struc: bool = False,
        with_aa_sampling: bool = True,
        with_true_seq: bool = False,
        temperature: T.Optional[float] = 1.0,
        topk: T.Optional[int] = 1,
        num_recycles: T.Optional[int] = 3,
        ### for inference only
        input_mode: str = 'seq',
        randn: T.Optional[torch.Tensor] = None,
        omit_AAs_np = None,
        bias_AAs_np = None,
        omit_AA_mask = None,
        pssm_coef = None,
        pssm_bias = None,
        pssm_multi = None,
        pssm_log_odds_flag = None,
        pssm_log_odds_mask = None,
        pssm_bias_flag = None,
        bias_by_res = None
    ):
        """Autoencoder on the representations.

        Args:
            ************************* for encoder *****************************
            seq_feat: node-wise sequence embedding from ESM2, (B, L_max, *).
            seq_pair_feat: pair-wise sequence embedding from ESM2, 
                (B, L_max, L_max, *) or None.
            struc_feat: node-wise sequence embedding from ProteinMPNN or ESM-IF, 
                (B, L_max, *) or (B, L_max+2, *).
            struc_pair_feat: pair-wise sequence embedding from ESM2 ProteinMPNN 
                or ESM-IF, (B, L_max, K, *) or None.
            mask: 1 for valid residues and 0 for others, (B, L_max)
            residx: from 1 to L for single-chain, (B, L_max)
            masking_pattern: Optional masking to pass to the input. Binary tensor 
                of the same size as `seq`. Positions with 1 will be masked. 
                ESMFold sometimes produces different samples when different masks 
                are provided.
            chain_encoding_all: perform like a chain mask (1 for valid token) for 
                single-chain, (B, L_max)
            protein_size: tensor of the protein sizes, (B,)
            ************************* hidden space ****************************
            emb_exchange: whether exchange the representations in the hidden space.
            seq_true: ground truth sequence, (B, L_max)
            ********************** for decoder ********************************
            with_true_adj: whether use the true adjacency matrix (E_idx)
            with_final_outputs: whether generate the final outputs
            num_recycles (int): How many recycle iterations to perform. If None, 
                defaults to training max recycles, which is 3.
            topk: k value of the topk sampling, default = 1 (maximum sampling)
            temperature: temperature for sampling.
        """
        if self.with_terminus_token:
            B, L_max, _ = struc_feat.shape
        else:
            B, L_max, _ = seq_feat.shape
        seq_feat_esm = seq_feat
        seq_pair_feat_esm = seq_pair_feat
        struc_feat_mpnn = struc_feat
        struc_pair_feat_mpnn = struc_pair_feat

        ########################################################################
        # Encoding (encoder of the projector) 
        ########################################################################

        if with_projector:
            embedding_all = self.internal_encoder(
                seq_feat = seq_feat, seq_pair_feat = seq_pair_feat,
                struc_feat = struc_feat, struc_pair_feat = struc_pair_feat, 
                E_idx = E_idx, mask = mask, with_contrastive = with_contrastive,
            )

            ########################### joint projector ########################
            node_feat_nega, pair_feat_neg = None, None

            if 'joint' in self.args.projector_version:
                struc_feat, struc_pair_feat, node_feat_nega, pair_feat_neg = embedding_all
                seq_feat, seq_pair_feat = None, None
                emb_exchange = False

                ###### VAE sampling ######
                if self.args.diffusion_version == 'vae':
                    (struc_mu_feat, struc_sigma_feat, struc_feat,
                     struc_mu_feat_pair, struc_sigma_feat_pair, struc_pair_feat
                    ) = self.VAE_projection(
                        struc_feat, struc_pair_feat, sampling = True,
                    )

                else:
                    struc_mu_feat, struc_sigma_feat = None, None
                    struc_mu_feat_pair, struc_sigma_feat_pair = None, None

                seq_mu_feat, seq_sigma_feat = None, None
                seq_mu_feat_pair, seq_sigma_feat_pair = None, None

            ########################### joint projector ########################
            else:
                seq_feat, seq_pair_feat, struc_feat, struc_pair_feat = embedding_all

                ###### VAE sampling ######
                if self.args.diffusion_version == 'vae':
                    (seq_mu_feat, seq_sigma_feat, seq_feat,
                     seq_mu_feat_pair, seq_sigma_feat_pair, seq_pair_feat
                    ) = self.VAE_projection(
                        seq_feat, seq_pair_feat, sampling = True,
                    )
                    (struc_mu_feat, struc_sigma_feat, struc_feat,
                     struc_mu_feat_pair, struc_sigma_feat_pair, struc_pair_feat
                    ) = self.VAE_projection(
                        struc_feat, struc_pair_feat, sampling = True,
                    )

                else:
                    seq_mu_feat, seq_sigma_feat = None, None
                    seq_mu_feat_pair, seq_sigma_feat_pair = None, None
                    struc_mu_feat, struc_sigma_feat = None, None
                    struc_mu_feat_pair, struc_sigma_feat_pair = None, None

            ########################### embedding exchange #####################
            if emb_exchange:
                seq_feat, struc_feat = struc_feat, seq_feat
                seq_pair_feat, struc_pair_feat = struc_pair_feat, seq_pair_feat

        ########################################################################
        # Decoding 
        ########################################################################

        (seq_feat_recover, seq_pair_feat_recover,
         struc_feat_recover, struc_pair_feat_recover,
         dist_pred, E_idx_pred, seq_logit, aa, struc
        ) = self.decoder(
            with_projector = with_projector,
            seq_feat = seq_feat,
            seq_pair_feat = seq_pair_feat,
            struc_feat = struc_feat,
            struc_pair_feat = struc_pair_feat,
            mask = mask,
            with_true_adj = with_true_adj,
            E_idx = E_idx,
            L_max = L_max,
            mode = mode,
            with_final_seq = with_final_seq,
            with_final_struc = with_final_struc,
            with_aa_sampling = with_aa_sampling,
            with_true_seq = with_true_seq,
            seq_true = seq_true,
            chain_M = chain_encoding_all.float(),
            residx = residx,
            topk = topk,
            temperature = temperature,
            num_recycles = num_recycles,
            protein_size = protein_size,
            randn = randn,
            omit_AAs_np = omit_AAs_np,
            bias_AAs_np = bias_AAs_np,
            omit_AA_mask = omit_AA_mask,
            pssm_coef = pssm_coef,
            pssm_bias = pssm_bias,
            pssm_multi = pssm_multi,
            pssm_log_odds_flag = pssm_log_odds_flag,
            pssm_log_odds_mask = pssm_log_odds_mask,
            pssm_bias_flag = pssm_bias_flag,
            bias_by_res = bias_by_res
        )

        ########################################################################
        # Final Outputs 
        ########################################################################

        if struc is None:
            struc = {}
        else:
            struc = {
                'positions': struc[0],
                'frames': struc[1], 
                'sidechain_frames': struc[2],
                'disto_logits': struc[3]
            }

        struc.update(
            {'log_prob': seq_logit,
             'seq': aa,
             'seq_feat_hidden': seq_feat,
             'seq_pair_feat_hidden': seq_pair_feat,
             'seq_feat_esm': seq_feat_esm,
             'seq_pair_feat_esm': seq_pair_feat_esm,
             'seq_feat_recover': seq_feat_recover,
             'seq_pair_feat_recover': seq_pair_feat_recover,
             'struc_feat_hidden': struc_feat,
             'struc_pair_feat_hidden': struc_pair_feat,
             'struc_feat_mpnn': struc_feat_mpnn,
             'struc_pair_feat_mpnn': struc_pair_feat_mpnn,
             'struc_feat_recover': struc_feat_recover,
             'struc_pair_feat_recover': struc_pair_feat_recover,
             'node_feat_nega': node_feat_nega,
             'pair_feat_nega': pair_feat_neg,
             'dist_true': dist_map,
             'dist_pred': dist_pred,
             'seq_mu_feat': seq_mu_feat,
             'seq_sigma_feat': seq_sigma_feat,
             'seq_mu_feat_pair': seq_mu_feat_pair,
             'seq_sigma_feat_pair': seq_sigma_feat_pair,
             'struc_mu_feat': struc_mu_feat,
             'struc_sigma_feat': struc_sigma_feat,
             'struc_mu_feat_pair': struc_mu_feat_pair,
             'struc_sigma_feat_pair': struc_sigma_feat_pair,
            }
        )

        return struc


    def forward_diffusion(self,
        ###### for diffusion ######
        node_feat,
        mask:torch.Tensor=None,
        mask_gen:torch.Tensor=None,
        fragment_type:torch.Tensor=None,
        t:torch.Tensor=None,
        node_bias:float=None,
        node_var:float=None,
        pair_feat:torch.Tensor=None,
        pair_mask:torch.Tensor=None,
        pair_bias:float=None,
        pair_var:float=None,
        norm_version:str='default',
        ###### for decoder ###### 
        mode: T.Optional[str] = 'train',
        mask_decoder:torch.Tensor=None,
        protein_size:torch.Tensor=None, 
        seq_true: torch.Tensor=None,
        E_idx: torch.Tensor=None,
        with_true_adj: T.Optional[bool] = True,
        with_final_seq: bool = False,
        with_final_struc: bool = False,
        with_aa_sampling: bool = True,
        with_true_seq: bool = False,
        residx: torch.Tensor=None,
        temperature: T.Optional[float] = 1.0,
        topk: T.Optional[int] = 1,
        num_recycles: T.Optional[int] = 3,
        ### for inference only
        input_mode: str = 'seq',
        randn: T.Optional[torch.Tensor] = None,
        omit_AAs_np = None,
        bias_AAs_np = None,
        omit_AA_mask = None,
        pssm_coef = None,
        pssm_bias = None,
        pssm_multi = None,
        pssm_log_odds_flag = None,
        pssm_log_odds_mask = None,
        pssm_bias_flag = None,
        bias_by_res = None
    ):
         
        ########################################################################
        # Diffusion
        ########################################################################

        out_dict = self.diffusion_model(
            node_feat = node_feat,
            mask = mask,
            mask_gen = mask_gen,
            fragment_type = fragment_type,
            t = t,
            node_bias = node_bias,
            node_var = node_var,
            pair_feat = pair_feat,
            pair_mask= pair_mask,
            pair_bias = pair_bias,
            pair_var = pair_var,
            norm_version = norm_version,
            return_pred = True,
            protein_size = protein_size,
        )
        # out_dict:
        #     node_pred: recovered node feature
        #     pair_pred: recovered pair feature
        #     loss_node, loss_pair, loss

        ########################################################################
        # decoding
        ########################################################################

        if self.with_terminus_token:
            L_max = mask_decoder.shape[1] + 2
        else:
            L_max = mask_decoder.shape[1]

        (seq_feat_recover, seq_pair_feat_recover,
         struc_feat_recover, struc_pair_feat_recover,
         dist_pred, E_idx_pred, seq_logit, aa, struc
        ) = self.decoder(
            with_projector = True,
            seq_feat = out_dict['node_pred'],
            seq_pair_feat = out_dict['pair_pred'],
            struc_feat = None,
            struc_pair_feat = None,
            mask = mask_decoder,
            with_true_adj = with_true_adj,
            E_idx = E_idx,
            L_max = L_max,
            mode = mode,
            with_final_seq = with_final_seq,
            with_final_struc = with_final_struc,
            with_aa_sampling = with_aa_sampling,
            with_true_seq = with_true_seq,
            seq_true = seq_true,
            chain_M = mask_decoder.float(),
            residx = residx,
            topk = topk,
            temperature = temperature,
            num_recycles = num_recycles,
            protein_size = protein_size,
            randn = randn,
            omit_AAs_np = omit_AAs_np,
            bias_AAs_np = bias_AAs_np,
            omit_AA_mask = omit_AA_mask,
            pssm_coef = pssm_coef,
            pssm_bias = pssm_bias,
            pssm_multi = pssm_multi,
            pssm_log_odds_flag = pssm_log_odds_flag,
            pssm_log_odds_mask = pssm_log_odds_mask,
            pssm_bias_flag = pssm_bias_flag,
            bias_by_res = bias_by_res
        )

        ########################################################################
        # Final Outputs 
        ########################################################################

        if struc is not None:
            out_dict.update(
                {
                    'positions': struc[0],
                    'frames': struc[1], 
                    'sidechain_frames': struc[2],
                    'disto_logits': struc[3]
                }
            )

        out_dict.update(
            {'log_prob': seq_logit,
             'seq': aa,
             'seq_feat_recover': seq_feat_recover,
             'seq_pair_feat_recover': seq_pair_feat_recover,
             'struc_feat_recover': struc_feat_recover,
             'struc_pair_feat_recover': struc_pair_feat_recover,
             'dist_pred': dist_pred,
            }
        )

        return out_dict


    ########################################################################### 
    # Losses                                                                  # 
    ###########################################################################

    def autoencoder_loss(self, 
        out, 
        batch, 
        loss_weight = None,
        with_emb_recover_loss = True,
        force_emb_recover_loss = True,
        with_joint_emb_loss = True,
        with_dist_pred_loss = True,
        with_seq_recover_loss = False,
        with_struc_recover_loss = False,
        with_contrastive = False,
        contrast_dist = 'norm',
        with_size_regular = False,
        posi_weight = 1.0,
        posi_threshold = 1.0,
        nega_threshold = 2.0,
        vae_loss_weight = 0.0,
        habits_lambda = 0.2,
        mode = 'train',
        summary = True
    ):
        """Caculate the losses for the autoencoder.

        Args: 
            out:
                ### for embedding recover loss
                seq_feat_esm: (B, L_max, esm_dim)
                seq_pair_feat_esm: (B, L_max, L_max, esm_dim)
                seq_feat_recover: (B, L_max, esm_dim)
                seq_pair_feat_recover: (B, L_max, L_max, esm_dim)
                struc_feat_mpnn: (B, L_max, mpnn_dim)
                struc_pair_feat_mpnn: (B, L_max, K, mpnn_dim)
                struc_feat_recover: (B, L_max, mpnn_dim)
                struc_pair_feat_recover: (B, L_max, K, mpnn_dim)
                ### for embedding similarity and embedding recovery
                (* L = L_max or m (reduced length))
                seq_feat_hidden: (B, L, hidden_dim)
                struc_feat_hidden: (B, L, hidden_dim)
                seq_pair_feat_hidden: (B, L, L, hidden_dim)
                struc_pair_feat_hidden: (B, L, L, hidden_dim)
                ### for distance prediction
                dist_true: [B, L_max, L_max]
                dist_pred: [B, L_max, L_max]
                ### for sequence
                log_prob: (B, L_max, 21)
                ### for distogram
                distogram_logits, (B, L_max, L_max, 64)
                ### for fape
                frames: (8, B, L_max, 7)
                sidechain_frames: (8, B, L_max, 8, 4, 4)
                positions: (8, B, L_max, atom_num=14, 3)

            batch:
                seq_mask: (B, L_max)
                ### for sequence
                aatype: (B, L_max)
                ### for distogram
                pseudo_beta: (B, L_max, 3)
                pseudo_beta_mask: (B, L_max)
                ### for backbone fape
                backbone_rigid_tensor: (B, L_max, 4, 4)
                backbone_rigid_mask: (B, L_max)
                use_clamped_fape: 0.9
                ### for sidechain fape
                rigidgroups_gt_frames: [B, L_max, 8, 4, 4] 
                rigidgroups_alt_gt_frames: [B, L_max, 8, 4, 4] 
                rigidgroups_gt_exists: [B, L_max, 8]
        """
        ####################### preprocess #####################################

        seq_mask = batch['seq_mask'].to(self.device)  # (B, L_max)

        ###### with terminus tokens ######
        if self.seq_design_module == 'ESM-IF' and self.with_terminus_token:
            struc_mask = esmif_mask_transform(seq_mask) # (B, L_max+2)
        ###### without terminus tokens ######
        else:
            struc_mask = seq_mask  # (B, L_max)
            if self.seq_design_module == 'ESM-IF' \
            and out['struc_feat_mpnn'].shape[1] != struc_mask.shape[1]:
                out['struc_feat_mpnn'] = esmif_remove_ends(
                    feat=out['struc_feat_mpnn'], protein_length=seq_mask.sum(-1)
                )

        ############# embedding recovery loss ##################################

        if force_emb_recover_loss:
            with_esm_emb_recover_loss = True
            with_mpnn_emb_recover_loss = True
        elif with_emb_recover_loss:
            with_esm_emb_recover_loss = not with_struc_recover_loss
            with_mpnn_emb_recover_loss = not with_seq_recover_loss
        else: 
            with_esm_emb_recover_loss = False
            with_mpnn_emb_recover_loss = False

        loss_all = self.internal_ae.loss(
            out = out,
            seq_mask = seq_mask,
            struc_mask = struc_mask,
            with_esm_emb_recover_loss = with_esm_emb_recover_loss,
            with_mpnn_emb_recover_loss = with_mpnn_emb_recover_loss,
            with_joint_emb_loss = with_joint_emb_loss,
            with_dist_pred_loss = with_dist_pred_loss,
            with_contrastive = with_contrastive,
            contrast_dist = contrast_dist,
            with_size_regular = with_size_regular,
            posi_weight = posi_weight,
            posi_threshold = posi_threshold,
            nega_threshold = nega_threshold,
            mode = mode,
            summary = False
        )

        ############# VAE KLD loss ############################################

        if self.args.diffusion_version == 'vae' and vae_loss_weight > 0: 
            # if self.with_edge_feature:
            #     pair_mask = mat_outer_prod(mask.float(), mask.float())
            # else:
            #     pair_mask = None
            
            loss_all['kld'] = kld_loss(
                #out,  mask, pair_mask, habits_lambda
                out,  None, None, habits_lambda
            ) * vae_loss_weight

        ############# end2end loss (modality recover loss) ####################

        loss_end2end = self.external_ae.loss(
            out = out, batch = batch,
            with_seq_recover_loss = with_seq_recover_loss,
            with_struc_recover_loss = with_struc_recover_loss,
            summary = False,
        )
        loss_all.update(loss_end2end)

        ##################### wrap up the losses ##############################
        if summary:
            loss_overall = 0.
            for key in loss_all:
                if loss_weight is not None and key in loss_weight:
                    loss_overall += loss_all[key] * loss_weight[key]
                else:
                    print(key, loss_all[key])
                    loss_overall += loss_all[key]
            loss_all['loss'] = loss_overall

        return loss_all


    def diffusion_loss(self,
        output, batch,
        loss_weight = None,
        with_seq_recover_loss = True,
        with_struc_recover_loss = True,
        with_cyclic_loss = True,
        summary = True
    ):
        ################# diffusion loss ######################################
        loss_all = {}
        for key in ['loss_node', 'loss_pair']:
            if key in output and output[key] is not None:
                loss_all[key] = output[key] 

        ############# end2end loss (modality recover loss) ####################

        loss_end2end = self.external_ae.loss(
            out = output, batch = batch,
            with_seq_recover_loss = with_seq_recover_loss,
            with_struc_recover_loss = with_struc_recover_loss,
            summary = False,
        )
        loss_all.update(loss_end2end)

        ######################### cyclic loss #################################

        if with_cyclic_loss:

            seq_mask = batch['seq_mask'].to(self.device)

            ###### ESM seq to MPNN seq ######
            seq_true = batch['aatype']  # ESM sequence
            seq_true = aa_transform(
                seq_true, version = 'seq', trans_mat = ESM2MPNN_MAT
            )  # MPNN sequence

            ###### MPNN infer ######
            randn = torch.argsort(
                (seq_mask + 0.0001) * (torch.abs(torch.randn(seq_mask.shape, device=device)))
            )

            mpnn_post_pred = self.oracle(
                X = output['positions'][-1, :, :, :4],  # N, CA, C, O; (N, L, 4, 3)
                S = seq_true, 
                mask = seq_mask, 
                chain_M = seq_mask, 
                residue_idx = batch['residx'], 
                chain_encoding_all = seq_mask, 
                randn = randn
            )
            mpnn_post_pred = np.exp(mpnn_post_pred)  # log-prob to prob

            ###### MPNN prob to ESM-IF prob ######
            if self.seq_design_module == 'ESM-IF':
                mpnn_post_pred = aa_transform(
                    mpnn_post_pred, version = 'prob', trans_mat = MPNN2ESMIF_MAT
                )  # ESM-IF prob format, (N, L, 21)
                mpnn_post_pred = F.pad(
                    mpnn_post_pred, (4, 10, 0, 0, 0, 0), 'constant', 0
                )  # (N, L, 35)

            ###### KLD ######
            n_tokens = seq_mask.sum().float()
            kldiv = F.kl_div(
                input=out_dict['log_prob'],
                target=mpnn_post_pred,
                reduction='none',
                log_target=False
            ).sum(dim=-1)    # (N, L)
            loss_all['cyclic'] = (kldiv * seq_mask).sum() / (n_tokens + 1e-8)

        ##################### wrap up the losses ##############################
        if summary:
            loss_overall = 0.
            for key in loss_all:
                if loss_weight is not None and key in loss_weight:
                    loss_overall += loss_all[key] * loss_weight[key]
                else:
                    loss_overall += loss_all[key]
            loss_all['loss'] = loss_overall

        return loss_all

