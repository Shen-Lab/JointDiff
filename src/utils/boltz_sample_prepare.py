import os
from tqdm.auto import tqdm
import argparse
from utils_infer import dict_load

parser = argparse.ArgumentParser()

parser.add_argument('--in_dir', type=str,
    default='../results/RBX1Binder/dict/'
)
parser.add_argument('--out_dir', type=str,
    default='../results/RBX1Binder/yaml/'
)
parser.add_argument('--sele_num', type=int, default=5)
parser.add_argument('--chain_A_len', type=int, default=100)

args = parser.parse_args()

dict_list = os.listdir(args.in_dir)

for sample in tqdm(dict_list):

    name = '.'.join(sample.split('.')[:-1])
    in_path = os.path.join(args.in_dir, sample)
    in_dict = dict_load(in_path)

    idx_list = sorted(in_dict.keys())[:args.sele_num]

    for idx in idx_list:
        seq = in_dict[idx]['seq']
        seq_dict = { 
            'A': seq[:args.chain_A_len],
            'B': seq[args.chain_A_len:],
        }

        out_path = os.path.join(args.out_dir, "%s_att%d.yaml" % (name, idx))
                
        ### write 
        with open(out_path, 'w') as wf:
            wf.write("sequences:\n")
            
            for chain in seq_dict:
                
                wf.write("  - protein:\n")
                wf.write("      id: %s\n" % chain)
                wf.write("      sequence: %s\n" % seq_dict[chain])
                
                #msa_path = os.path.join(in_dir, sample, "%s.a3m" % chain)
                #if os.path.exists(msa_path):
                #    wf.write("      msa: %s\n" % msa_path)
                #else:
                #    print("%s was not found." % msa_path)
