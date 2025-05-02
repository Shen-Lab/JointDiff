# diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'
import os
import shutil
import argparse
import pickle

from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.stats import spearmanr

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jointdiff.modules.utils.train import *
from jointdiff.modules.utils.misc import *
from jointdiff.modules.data.data_utils import *
from jointdiff.dataset import ConfidenceDataset
from jointdiff.model import ConfidenceNet
from jointdiff.trainer import dict_save

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')  # SZ
torch.autograd.set_detect_anomaly(True)

######################################################################################
# Evaluation Fucntions                                                               #
######################################################################################

def evaluation(model, dataloader, args):

    model.eval()
    thre_dict = {
        'consist-seq': args.consist_seq_thre, 
        'consist-stru': args.consist_stru_thre, 
        'foldability': args.foldability_thre, 
        'designability': args.designability_thre
    }

    ############################################################################
    # batch-wise inference 
    ############################################################################

    pred_all = []
    label_all = []

    for i, batch in tqdm(enumerate(dataloader)):

        if i >= 100: # for debugging
            break

        batch = recursive_to(batch, args.device)
        batch['chain_nb'] = batch['mask']
        batch['pos_heavyatom'] = batch['pos_heavyatom'][:, :, :args.num_atoms, :] 
        batch['mask_heavyatom'] = batch['mask_heavyatom'][:, :, :args.num_atoms] 

        ###### centralize the data ######
        if args.centralize:
            mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
            mean = mean / batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1)  # (N, 3)
            batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 4, 3)
            batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

        ############################################################################
        # Inference & Loss
        ############################################################################

        pred = model(batch)  # (N, 4)
        pred = pred.detach().cpu()
        label = batch['label'].detach().cpu()

        batch = recursive_to(batch, 'cpu')
        torch.cuda.empty_cache()

        pred_all.append(pred)
        label_all.append(label)

    ### predictions
    pred_all = torch.cat(pred_all, dim = 0)  # (N, 4)
    ### raw labels (unnormailized)
    label_all = torch.cat(label_all, dim = 0)  # (N, 4)

    ############################################################################
    # label and prediction process 
    ############################################################################
 
    ### binary label
    label_binary = torch.zeros(label_all.shape)
    for i, metric in enumerate(
        ['consist-seq', 'consist-stru', 'foldability', 'designability']
    ):
        if metric == 'consist-seq' or metric == 'foldability':
            label_binary[:, i] = (label_all[:, i] >= thre_dict[metric])
        else:
            label_binary[:, i] = (label_all[:, i] <= thre_dict[metric])
    label_binary = label_binary.float()

    ### label for loss
    if args.binary:
        label_for_loss = label_binary
    elif args.label_norm:
        label_for_loss = torch.zeros(label_all.shape)
        for i, metric in enumerate(
            ['consist-seq', 'consist-stru', 'foldability', 'designability']
        ):
            ### normalize 
            min_val, max_val = args.min_max_dict[metric]
            label_vec = label_all[:, i]
            label_for_loss[:, i] = 2 * (label_vec - max_val) / (max_val - min_val) + 1
    else:
        label_for_loss = label_all

    ### binary prediction
    if args.binary:
        pred_binary = (pred_all >= 0.5)
    else:
        pred_binary = torch.zeros(pred_all.shape)
        for i, metric in enumerate(
            ['consist-seq', 'consist-stru', 'foldability', 'designability']
        ):
            pred_vec = pred_all[:, i]
            ### denormalization
            if args.label_norm:
                min_val, max_val = args.min_max_dict[metric]
                pred_vec = (pred_vec - 1) * (max_val - min_val) / 2 + max_val

            if metric == 'consist-seq' or metric == 'foldability': 
                pred_binary[:, i] = (pred_vec >= thre_dict[metric])
            else: 
                pred_binary[:, i] = (pred_vec <= thre_dict[metric]) 

        pred_binary = pred_binary.int()

    ############################################################################
    # metric cal 
    ############################################################################
 
    result_dict = {}
 
    for i, metric in enumerate(['consist-seq', 'consist-stru', 'foldability', 'designability']):

        print(metric)
        print('*********************************************')

        result_dict[metric] = {}

        pred_vec = pred_all[:, i]
        pred_vec_binary = pred_binary[:, i]

        label_vec = label_all[:, i]
        label_vec_loss = label_for_loss[:, i]
        label_vec_binary = label_binary[:, i]

        result_dict[metric]['pred'] = pred_vec
        result_dict[metric]['pred_binary'] = pred_vec_binary
        result_dict[metric]['label'] = label_vec
        result_dict[metric]['label_loss'] = label_vec_loss
        result_dict[metric]['label_binary'] = label_vec_binary

        ####### loss ######
        ### binary classification
        if args.binary:
            result_dict[metric]['loss-BCE'] = F.binary_cross_entropy(
                pred_vec, label_vec_loss
            )
            print('loss-BCE:', result_dict[metric]['loss-BCE'])
        ### regression
        else:
            result_dict[metric]['loss-MSE'] = F.mse_loss(
                pred_vec, label_vec_loss
            )
            print('loss-MSE:', result_dict[metric]['loss-MSE'])

        ###### binary classification ######
        result_dict[metric]['acc-unbalance'] = (label_vec_binary == pred_vec_binary).sum() / len(label_vec_binary)
        result_dict[metric]['acc-balance'] = balanced_accuracy_score(
            label_vec_binary, pred_vec_binary
        )
        result_dict[metric]['f1-score'] = f1_score(
            label_vec_binary, pred_vec_binary
        )
        ###### binary correlation ######
        if args.binary and (metric == 'consist-stru' or metric == 'designability'):
            correlation, pvalue = spearmanr(1 - pred_vec, label_vec)
        else:
            correlation, pvalue = spearmanr(pred_vec, label_vec)
        result_dict[metric]['correlation'] = correlation
        result_dict[metric]['p_value'] = pvalue

        ###### visualization ######
        for kind in ['acc-unbalance', 'acc-balance', 'f1-score', 'correlation']:
            print('%s:' % kind, result_dict[metric][kind])
        print('#######################################################')

    return result_dict


######################################################################################
# Main Function                                                                      #
######################################################################################

def main(args):

    #######################################################################
    # load the Model
    #######################################################################

    print('Loading from checkpoint: %s' % args.ckpt_path)
    ckpt = torch.load(args.ckpt_path, map_location=args.device) #, weights_only = True)
    model_args = ckpt['args']
    model_args.consist_seq_thre = args.consist_seq_thre
    model_args.consist_stru_thre = args.consist_stru_thre
    model_args.foldability_thre = args.foldability_thre
    model_args.designability_thre = args.designability_thre

    model = ConfidenceNet(
        res_feat_dim = model_args.res_feat_dim,
        pair_feat_dim = model_args.pair_feat_dim,
        num_layers = model_args.num_layers,
        num_atoms = model_args.num_atoms,
        max_relpos = model_args.max_relpos,
        binary = model_args.binary
    ).to(args.device)
    print('Number of parameters: %d' % count_parameters(model))

    ckpt['model'] = {''.join(key.split('module.')[:]) : ckpt['model'][key]
        for key in ckpt['model']
    }
    model.load_state_dict(ckpt['model'])

    ################## Parallel  ##################
    if args.multi_gpu:
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        print('%d GPUs detected. Applying parallel training and a batch size of %d.' % 
            (torch.cuda.device_count(), args.batch_size)
        )

    elif args.device == 'cuda':
        print('Applying single GPU training with a batch size of %d.' % 
            (args.batch_size)
        )

    else:
        print('Applying CPU training with a batch size of %d.' %
            (args.batch_size)
        )

    #######################################################################
    # Data Loading
    #######################################################################

    args.binary = False
    args.label_norm = False
    args.balance = False
    test_dataset = ConfidenceDataset(args)
    test_dataloader = DataLoader(
        test_dataset, batch_size = args.batch_size,
        collate_fn = PaddingCollate(), shuffle=False, 
        num_workers = args.num_workers
    )
    print('%d test samples loaded.'  % (len(test_dataset)))

    #######################################################################
    # Evaluation
    #######################################################################
    
    model_args.min_max_dict = {
        'consist-seq': args.consist_seq_range,
        'consist-stru': args.consist_stru_range,
        'foldability': args.foldability_range,
        'designability': args.designability_range,
    }
    eval_dict = evaluation(model, test_dataloader, model_args)
    if not args.debug:
        _ = dict_save(eval_dict, args.result_path)


######################################################################################
# Running the Script                                                                 #
######################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##################### paths and name ######################################
    parser.add_argument('--summary_path',
        type=str, default='../../Documents/Data/Distillation/Confidence/confidence_summary_all.tsv'
    )
    parser.add_argument('--pdb_dir', type=str,
        default='../../Documents/Data/Distillation/Confidence/pdbs/'
    )
    parser.add_argument('--data_list_path', type=str,
        default='../../Documents/Data/Distillation/Confidence/test_data_list.pkl'
    )
    parser.add_argument('--processed_dir', type=str,
        default='../../Documents/Data/Distillation/Confidence/'
    )
    parser.add_argument('--ckpt_path', type=str,
        default='../../Documents/TrainedModels/JointDiff/logs_confidence_development/confidence_128-64-6_posiscale50.0000_norm-regression_/checkpoints/checkpoint_20.pt'
    )
    parser.add_argument('--result_path', type=str,
        default='../Results/confidence_development/confidence_128-64-6_posiscale50.0000_norm-regression_/result_ckpt_20.pkl'
    )

    ########################### setting #######################################
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    ###### data ######
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    ###### for label process ######
    ### label threshold
    parser.add_argument('--consist_seq_thre', type=float, default=0.3)
    parser.add_argument('--consist_stru_thre', type=float, default=2.0)
    parser.add_argument('--foldability_thre', type=float, default=0.3)
    parser.add_argument('--designability_thre', type=float, default=2.0)
    ### label normalization
    parser.add_argument(
        '--consist_seq_range', type=float, nargs='*', default=[0.004364, 0.800595]
    )
    parser.add_argument(
        '--consist_stru_range', type=float, nargs='*', default=[0.0, 6.67]
    )
    parser.add_argument(
        '--foldability_range', type=float, nargs='*', default=[0.010417, 0.807143]
    )
    parser.add_argument(
        '--designability_range', type=float, nargs='*', default=[0.20375, 5.76]
    )
    ###### setting ######
    parser.add_argument('--debug', type=int, default=0)

    args = parser.parse_args()

    ###### arguments setting ######
    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    args.debug = bool(args.debug)

    ###### running ######
    main(args)
