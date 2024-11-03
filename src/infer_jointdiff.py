#####################################################
# sample and generate the pdb files 
# by SZ; 5/15/2023
#####################################################

import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn 
import torch.nn.functional as F_ 
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *

import torch.multiprocessing 
torch.multiprocessing.set_sharing_strategy('file_system') 
import time

from utils_infer import inference_pdb_write

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ###### paths ######
    parser.add_argument('--model_path', type=str, 
        default='../checkpoints/JointDiff_model.pt'
    )
    parser.add_argument('--result_path', type=str, 
        default='../samples/'
    )
    ###### devices ######
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    ###### inference setting #####
    parser.add_argument('--size_range', nargs='*', type=int, default=[100,200,20])
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--save_type', type=str, default='sele', help='"sele", "all" or "last"')
    parser.add_argument('--save_steps', type=int, nargs='*', default=[0])
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

    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']
    seed_all(config.train.seed)
    ### no need of proteinMPNN for inference
    #config.model.proteinMPNN_path = None

    model = get_model(config.model).to(args.device)
    print('Number of parameters: %d' % count_parameters(model))

    checkpoint = torch.load(args.model_path)
    parameter_dict = {}
    for key in checkpoint['model'].keys():
        if key.startswith('module'):
            key_new = key[7:]
            parameter_dict[key_new] = checkpoint['model'][key]
        else:
            parameter_dict[key] = checkpoint['model'][key]
    model.load_state_dict(parameter_dict)

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

    start_time = time.time()
    sample_num = 0

    if len(args.size_range) > 2:
        length_interval = args.size_range[2]
    else:
        length_interval = 1

    for size in tqdm(range(args.size_range[0], args.size_range[1] + 1, length_interval)):
        pdb_idx = 0
        num = args.num

        while num > 0:
            bz_temp = min(args.batch_size, num)
            num -= bz_temp

            len_list = [size] * bz_temp

            ####################### inference #################################

            ###### joint sampling ######
            if args.modality == 'joint': 
                if args.multi_gpu:
                    len_list = torch.Tensor(len_list).cuda()
                    out_dict, traj = model.module.sample_from_scratch(length_list = len_list, t_bias = args.t_bias)
                else:
                    out_dict, traj = model.sample_from_scratch(length_list = len_list, t_bias = args.t_bias)

            ###### single-modality generation
            else:
                if args.multi_gpu:
                    len_list = torch.Tensor(len_list).cuda()
                    out_dict, traj = model.module.sample_SingleModal_from_scratch(modality = args.modality, length_list = len_list)
                else:
                    out_dict, traj = model.sample_SingleModal_from_scratch(modality = args.modality, length_list = len_list)

            ######################## Save the generated samples ###############

            if args.save_type == 'last':
                ### save the pdb file of the last reverse diffusion process (t = 0) 
                diff_step_list = [0]

            if args.save_type == 'sele':
                diff_step_list = args.save_steps
 
            elif args.save_type == 'all':
                ### save the complete trajectory
                diff_step_list = out_dict.keys()
 
            else:
                raise Exception('Error! No sampling type called %s!'%args.save_type)

            ###### save the samples ######
            for idx in range(bz_temp):
                pdb_idx += 1

                ### for different diffusion steps
                for diff_step in diff_step_list: 
                    if args.modality != 'sequence':
                        ### record pdb files
                        pdb_path = os.path.join(
                            args.result_path, 
                            'len%d_%d_%d.pdb'%(size, diff_step, pdb_idx)
                        )
            
                        inference_pdb_write(
                            coor = out_dict[diff_step][idx]['coor'], 
                            path = pdb_path, 
                            seq = out_dict[diff_step][idx]['seq']
                        )
                    else:
                        ### record as a FASTA file
                        with open(args.result_path, 'a') as wf:
                            wf.write('>len%d_%d_%d\n' % (size, diff_step, pdb_idx))
                            wf.write('%s\n' % out_dict[diff_step][idx]['seq'])

                    sample_num += 1
                 
    ###### summarizing ######
    print('%d samples genrated in %.4fs.'%(sample_num, time.time() - start_time))

