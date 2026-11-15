import os
import argparse
import time
from tqdm.auto import tqdm
import numpy as np
import random
import torch

from utils_eval import dict_load, dict_save

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--go_emb', type=str, default='../../Data/Processed/mf_go_all_emb.pkl')
    parser.add_argument('--seq_emb', type=str, default='../../Transfer/Nature/Nature_protrek_emb_seq.pkl')
    parser.add_argument('--struc_emb', type=str, default='../../Transfer/Nature/Nature_protrek_emb_struc.pkl')
    parser.add_argument('--out_path', type=str, default='../../Transfer/Nature/Nature_protrek_similarity.pkl')
    parser.add_argument('--max_k', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.0187)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--random_pair', type=int, default=0)

    args = parser.parse_args()

    args.random_pair = bool(args.random_pair)

    ########################## load the GO terms #################################

    go_emb_dict = dict_load(args.go_emb)
    go_emb_all = []
    idx2go = {}

    for i, go in enumerate(go_emb_dict):
        go_emb_all.append(torch.from_numpy(go_emb_dict[go]['emb']))
        idx2go[i] = go

    go_emb_all = torch.cat(go_emb_all, dim = 0)
    print('GO embedding:', go_emb_all.shape)

    ####################  sequence&struc embedding #################################

    seq_emb_dict = dict_load(args.seq_emb)
    struc_emb_dict = dict_load(args.struc_emb)

    seq_emb_all = []
    struc_emb_all = []

    idx = 0
    idx2sample = {}

    if args.random_pair:
        seq_sample_list = list(seq_emb_dict.keys())

    for sample_struc in struc_emb_dict:
        if args.random_pair:
            sample = random.choice(seq_sample_list)
            idx2sample[idx] = sample_struc[:-1] if sample_struc.endswith('.') else sample_struc
        else:
            if sample_struc.endswith('.'):
                sample = sample_struc[:-1]
            else:
                sample = sample_struc 

            if sample not in seq_emb_dict:
                print(sample)
                continue
            idx2sample[idx] = sample

        idx += 1

        seq_emb_all.append(torch.from_numpy(seq_emb_dict[sample]))
        struc_emb_all.append(torch.from_numpy(struc_emb_dict[sample_struc]))

    seq_emb_all = torch.cat(seq_emb_all, dim = 0)  # (N, d)
    struc_emb_all = torch.cat(struc_emb_all, dim = 0) # (N, d)

    inner_prot = torch.matmul(seq_emb_all.unsqueeze(1), struc_emb_all.unsqueeze(2)).squeeze() # (N, )
    seq_norm = torch.norm(seq_emb_all, dim = -1)
    struc_norm = torch.norm(struc_emb_all, dim = -1)
    
    score = (inner_prot / args.temperature).numpy()  # (N,)
    cos_simi = (inner_prot / (seq_norm * struc_norm)).numpy() # (N,)

    seq_struc_score = (score, cos_simi)
    print('%d samples loaded.' % idx, seq_struc_score[0].mean(), seq_struc_score[1].mean())
 
    ######################  similarity ###########################################

    seq_scores = (seq_emb_all.to(args.device) @ go_emb_all.to(args.device).T / args.temperature) # (N, L)
    seq_argsort = seq_scores.argsort(dim = -1, descending=True).to('cpu')
    seq_scores = seq_scores.to('cpu')

    struc_scores = (struc_emb_all.to(args.device) @ go_emb_all.to(args.device).T / args.temperature) # (N, L)
    struc_argsort = struc_scores.argsort(dim = -1, descending=True).to('cpu')
    struc_scores = struc_scores.to('cpu')

    print('Scores calculated.')

    ########################## summarization #####################################

    out_dict = {}
    for idx in tqdm(idx2sample):
        sample = idx2sample[idx]
        out_dict[sample] = {}

        ### seq&stru
        out_dict[sample]['seq-struc'] = (
            seq_struc_score[0][idx], seq_struc_score[1][idx]
        )

        ### seq&GO
        seq_go_score = seq_scores[idx]  # (L,)
        order_vec = seq_argsort[idx][:args.max_k] 
        seq_go_simi = []
        seq_go_emb = []

        for j in order_vec:
            j = int(j)
            go = idx2go[j]
            score = float(seq_go_score[j])
            seq_go_simi.append((score, go))
            seq_go_emb.append(torch.from_numpy(go_emb_dict[go]['emb']))

        out_dict[sample]['seq-go'] = seq_go_simi
        seq_go_emb = torch.cat(seq_go_emb, dim = 0)

        ### struc&GO
        struc_go_score = struc_scores[idx]  # (L,)
        order_vec = struc_argsort[idx][:args.max_k] 
        struc_go_simi = []
        struc_go_emb = []

        for j in order_vec:
            j = int(j)
            go = idx2go[j]
            score = float(struc_go_score[j])
            struc_go_simi.append((score, go))
            struc_go_emb.append(torch.from_numpy(go_emb_dict[go]['emb']))

        out_dict[sample]['struc-go'] = struc_go_simi
        struc_go_emb = torch.cat(struc_go_emb, dim = 0)
       
        ### seq-go&struc-go
        simi_score = seq_go_emb @ struc_go_emb.T  # (N, N)
        simi_score = simi_score / torch.norm(seq_go_emb, dim=-1).unsqueeze(1)
        simi_score = simi_score / torch.norm(struc_go_emb, dim=-1)
        out_dict[sample]['go-go'] = simi_score.numpy()
 
    print('GO terms aligned.')

    _ = dict_save(out_dict, args.out_path)
    print('Results saved at %s.' % args.out_path)
