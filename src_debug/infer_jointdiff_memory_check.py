#####################################################
# sample and generate the pdb files 
#####################################################

import os
import shutil
import argparse
import math
import random

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn 
import torch.nn.functional as F_ 
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.multiprocessing 
torch.multiprocessing.set_sharing_strategy('file_system') 
import time

from jointdiff.model import DiffusionSingleChainDesign
from jointdiff.trainer import inference_pdb_write, count_parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

###############################################################################
# utility functions
###############################################################################

def sample_for_all_size(model, args, length_pool):
    """Generate args.num samples for each desired protein size.
    """

    start_time = time.time()
    sample_num = 0
    infer_func = model.module.sample if args.multi_gpu else model.sample

    for size in tqdm(length_pool):
        #################### for certain protein size #########################

        pdb_idx = 0
        num = args.num

        while num > 0:
            bz_temp = min(args.batch_size, num)
            num -= bz_temp

            len_list = [size] * bz_temp

            ####################### inference #################################

            ###### joint sampling ######
            len_list = torch.Tensor(len_list).cuda()
            print('Memory Usage (size=%d)' % size)
            try:
                out_dict, traj = infer_func(
                    length_list = len_list, 
                    t_bias = args.t_bias
                )

                print('  Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
                print('  Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')
                print('*******************************************')
                del out_dict
                del traj
            except Exception as e:
                print(e)
            torch.cuda.empty_cache()


def sample_for_random_size(model, args, length_pool):
    """Generate args.num samples for each desired protein size.
    """
    pass

####################################### main function ##########################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ###### paths ######
    parser.add_argument('--model_path', type=str, 
        default='../Debug/checkpoints/jointdiff-x_joint_multinomial_model6-128-64-step100_posi-scale-50.0_rm-10.0_allbbatom_micro-posi+mse-align-dist+mse-distogram-clash-gap_2025_05_01__23_47_24/checkpoints/1.pt'
    )
    parser.add_argument('--result_path', type=str, 
        default='../Debug/samples/'
    )
    ###### devices ######
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=0)
    ###### inference setting #####
    parser.add_argument('--sample_method', type=str, 
        default='all', help='"all" or "random"'
    )
    parser.add_argument('--size_range', nargs='*', type=int, default=[20,200])
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_type', type=str, default='sele', help='"sele", "all" or "last"')
    parser.add_argument('--save_steps', type=int, nargs='*', default=[0]) #, 1, 2])
    parser.add_argument('--modality', type=str, default='joint')
    parser.add_argument('--t_bias', type=int, default=-1)

    args = parser.parse_args()

    if args.modality != 'sequence' and (not os.path.exists(args.result_path)):
        os.mkdir(args.result_path)
    elif args.modality == 'sequence':
        if (not args.result_path.endswith('.fa')):
            while args.result_path and args.result_path.endswith('/'):
                args.result_path = args.result_path[:-1]
            args.result_path += '.fa'

        if os.path.exists(args.result_path):
            print('Warning! %s exists. Will cover the previous version...' % args.result_path)
            os.rename(args.result_path, args.result_path + '.previous')

    if args.device == 'cuda' and ( not torch.cuda.is_available() ):
        print('GPUs are not available! Use CPU instead.')
        args.device = 'cpu'
        args.multi_gpu = 0

    ###########################################################
    # Model Loading 
    ###########################################################

    checkpoint = torch.load(args.model_path, map_location = args.device)
    config = checkpoint['config']

    ###### define the model
    model = DiffusionSingleChainDesign(config.model).to(args.device)

    ###### parameters prepare ######
    parameter_dict = model.state_dict()
    parameter_set = set(parameter_dict.keys())
    ### map the parameters
    for key in checkpoint['model'].keys():
        if key.startswith('module'):
            key_new = key[7:]
        else:
            key_new = key

        if key_new in parameter_dict:
            parameter_dict[key_new] = checkpoint['model'][key]
            parameter_set.remove(key_new)
        else:
            print('parameter %s not needed.' % key_new)

    ### load the dictionary
    model.load_state_dict(parameter_dict)
    print('Model loaded from %s.' % args.model_path)
    print('Number of parameters: %d' % count_parameters(model))

    ### unloaded parameters
    for name in parameter_set:
        print('%s not loaded.' % name)
    print('**********************************************************')


    ### Parallel
    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if args.multi_gpu:
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        print("%d GPUs detected. Applying parallel computation."%(torch.cuda.device_count()))

    ###########################################################
    # Sampling
    ###########################################################

    ###### length pool (target length list) ######

    if len(args.size_range) > 2:
        length_interval = args.size_range[2]
    else:
        length_interval = 1

    length_pool = range(args.size_range[0], args.size_range[1] + 1, length_interval)
    length_pool = [300, 500, 1000, 2000, 5000]

    ###### sampling for the selected length ######

    sample_for_all_size(model, args, length_pool) 
