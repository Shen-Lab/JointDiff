######################################################################
# foldability: the sequence recovery between the ProteinMPNN-sequences
#     and the generated sequence; already provided by ProteinMPNN
######################################################################

import os
import numpy as np
import argparse
from tqdm.auto import tqdm

from utils_eval import dict_save, SeqRecovery, Identity

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_path', 
        type=str, default='../../Results/Nature/sample_sele/ProteinMPNN_design/seqs/'
    )
    parser.add_argument('--gt_path', 
        type=str, default='../../Results/Nature/sample_sele/sequences/'
    )
    parser.add_argument('--out_path', 
        type=str, default='../../Results/Nature/foldability_SR_dict.pkl'
    )
    parser.add_argument('--alignment', 
        type=int, default=1
    )
    parser.add_argument('--mpnn_format', 
        type=int, default=1
    )
    parser.add_argument('--token_match', 
        type=str, default=None, help="token_match=<token_gt>/<token_design>"
    )

    args = parser.parse_args()

    # whether do the alignment with NW algorithm
    args.alignment = bool(args.alignment)
    # whether the target sequences are from ProteinMPNN
    args.mpnn_format = bool(args.mpnn_format)

    if args.token_match is None or '/' not in args.token_match:
        args.token_match = None
    else:
        args.token_match = args.token_match.split('/')

    ############################################################################
    # reference sequence 
    ############################################################################

    gt_dict = {}

    ###### Directory ######
    if os.path.isdir(args.gt_path):
        seq_list = [f for f in os.listdir(args.gt_path) 
            if f.endswith('.fa') or f.endswith('.fasta')
        ]
        for seq_file in seq_list:
            name = '.'.join(seq_file.split('.')[:-1])
            gt_path = os.path.join(args.gt_path, seq_file)

            with open(gt_path, 'r') as rf:
                lines = rf.readlines()
                gt_seq = lines[1].strip('\n')
                if '{' in gt_seq:
                    gt_seq = gt_seq.split("'")[3]

            gt_dict[name] = gt_seq

    ###### FASTA file ######
    else:
        with open(args.gt_path, 'r') as rf:
            for line in rf:
                if line.startswith('>'):
                    name = line.strip('>\n')
                    name = name.split(';')[0]
                else:
                    gt_dict[name] = line.strip('\n')

    print('%d reference sequences.' % len(gt_dict))

    ############################################################################
    # target sequence 
    ############################################################################

    tar_dict = {}
    tar_num = 0
  
    seq_list = [f for f in os.listdir(args.seq_path)
        if f.endswith('.fa') or f.endswith('.fasta')
    ]
    for seq_file in seq_list:
        ###### target name ######
        name = '.'.join(seq_file.split('.')[:-1])
        if '_attempt' in name:
            name = name.split('_attempt')[0]
        ### target name match
        if args.token_match is not None:
            name = args.token_match[-1].join(
                name.split(args.token_match[0])
            )

        if not name in tar_dict:
            tar_dict[name] = [] 

        ###### load the sequences ######
        seq_path = os.path.join(args.seq_path, seq_file)
        with open(seq_path, 'r') as rf:
            lines = rf.readlines()
            if args.mpnn_format:
                lines = lines[2:]

            for line in lines:
                if not line.startswith('>'):
                    tar_dict[name].append(line.strip('\n'))
                    tar_num += 1

    print('%d target sequences.' % tar_num)

    ############################################################################
    # Identity Calculation 
    ############################################################################

    result_dict = {}
    sr_micro = []
    sr_macro = []

    for name in tqdm(gt_dict):

        if name not in tar_dict:
            print('Designs for %s not found!' % name)
            continue

        if name not in result_dict:
            result_dict[name] = []

        gt_seq = gt_dict[name]
        for seq_design in tar_dict[name]:
 
            ###### recovery calculation ######
            try:
                if args.alignment: 
                    sr = Identity(seq_design, gt_seq)
                else:
                    sr = SeqRecovery(seq_design, gt_seq)
                
                if sr is not None:
                    result_dict[name].append(sr)
                    sr_micro.append(sr)

            except Exception as e:
                print('Failed for %s! %s' % (name, e))

        if result_dict[name]:
            sr_macro.append(np.mean(result_dict[name]))

    ################################# statistics ##############################

    print('%d samples in all: micro-SR=%.4f; macro-SR=%.4f' % (len(sr_macro),
                                                               np.mean(sr_micro),
                                                               np.mean(sr_macro)))

    _ = dict_save(result_dict, args.out_path)

