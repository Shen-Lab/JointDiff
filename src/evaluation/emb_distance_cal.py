import math
import numpy as np
from scipy.sparse import csc_matrix
import torch
import pickle
import argparse
from tqdm.auto import tqdm

from utils_eval import dict_load, dict_save

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tar_dict', type=str, 
        default='../../Results/latentDiff_DiT/000-DiT-S-withLen/emb-ddpm_dict.pkl'
    )
    parser.add_argument('--ref_dict', type=str, 
        default='../../Results/autoencoder/embedding/autoencoder-simple_with-ESM-IF_joint-mlp-4-512_pad-zero_dim16_vae-0.001_NoEnd_sele.pkl'
    )
    parser.add_argument('--save_dict', type=str, 
        default='../../Results/latentDiff_DiT/000-DiT-S-withLen/dist_emb2nature_ddpm_dict.pkl'
    )
    parser.add_argument('--kernel_size', type=str, default=3) 

    args = parser.parse_args()

    tar_dict = dict_load(args.tar_dict)
    ref_dict = dict_load(args.ref_dict)
    out_dict = dict()

    for size in tar_dict:
        if size not in ref_dict:
            continue

        emb_size = math.ceil((size - 1) / (args.kernel_size - 1))
        tar_num = len(tar_dict[size])
        ref_num = len(ref_dict[size])
        ref_emb = np.vstack([
            ref_dict[size][sample][:emb_size].reshape(-1) for sample in ref_dict[size]
        ])  # (ref_num, dim)
        out_dict[size] = np.zeros([tar_num, ref_num])
        
        for i in range(tar_num):
            tar_emb = tar_dict[size][i][0][0][:emb_size].reshape(-1)
            if torch.is_tensor(tar_emb):
                tar_emb = tar_emb.cpu().numpy()
            out_dict[size][i] = np.linalg.norm(ref_emb - tar_emb, axis = -1)

        print('Length=%d (%s): dist_ave=%.4f, dist_min_ave=%.4f' % (
            size, out_dict[size].shape, np.mean(out_dict[size]), np.mean(out_dict[size].min(axis=-1))
        ))

    _ = dict_save(out_dict, args.save_dict)

            





 
