######################################################
# extract the sequences from the pdb file
# by SZ; 8/12/2024
######################################################

import os
import argparse
from tqdm.auto import tqdm

from utils_eval import seq_extract

####################################### main function #######################################

parser = argparse.ArgumentParser()

parser.add_argument('--pdb_path', type=str, 
    default='../../Results/protein_generator/samples/'
)
parser.add_argument('--out_path', type=str, 
    default='../../Results/protein_generator/seq_gen.fa'
)

args = parser.parse_args()

pdb_list = [p for p in os.listdir(args.pdb_path) if p.endswith('.pdb')]
print('%d pdb samples in all.' % len(pdb_list))

with open(args.out_path, 'w') as wf:
    for p in tqdm(pdb_list):
        name = p[:-4]
        p_path = os.path.join(args.pdb_path, p)
        seq = seq_extract(p_path)[0]
    
        wf.write('>%s\n' % name)
        wf.write('%s\n' % seq)





















