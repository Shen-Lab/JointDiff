######################################################
# extract the sequences from the pdb file
# by SZ; 8/12/2024
######################################################

import os
import argparse
from tqdm.auto import tqdm

import Bio.PDB
from Bio.PDB import PDBParser, internal_coords, kdtrees

RESIDUE_dict = {'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'Q':'GLN', 'E':'GLU',
                'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS', 'M':'MET', 'F':'PHE',
                'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP', 'Y':'TYR', 'V':'VAL', 'B':'ASX',
                'Z':'GLX', 'X':'UNK'}

RESIDUE_reverse_dict = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
                        'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F',
                        'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V', 'ASX':'B',
                        'GLX':'Z', 'UNK':'X'}

ELEMENT_dict = {'N':'N', 'CA':'C', 'C':'C', 'O':'O'}


def seq_extract(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('target', file = pdb_path)

    seq_list = []
    for model in structure:
        for chain in model:
            poly = Bio.PDB.Polypeptide.Polypeptide(chain)
            seq_list.append(str(poly.get_sequence()))

    if len(seq_list) > 1:
        print('Warning! Multiple (%d) sequences detected!'%len(seq_list))
    return seq_list

def seq_extract_direct(pdb_path):
    seq_list = []

    chain_pre = None
    idx_pre = None
    seq = ''
    with open(pdb_path, 'r') as rf:
        for line in rf:
            chain = line[21]
            idx = chain + line[23:26]
            resi = RESIDUE_reverse_dict[line[17:20]]

            if chain_pre is not None and chain != chain_pre:
                seq_list.append(seq)
                seq = ''

            if idx != idx_pre:
                seq += resi
            idx_pre = idx
            chain_pre = chain

    seq_list.append(seq)

    if len(seq_list) > 1:
        print('Warning! Multiple (%d) sequences detected!'%len(seq_list))
    return seq_list


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

        try:
            seq = seq_extract(p_path)[0]
        except Exception as e:
            print(name, e)
            seq = seq_extract_direct(p_path)[0]
    
        wf.write('>%s\n' % name)
        wf.write('%s\n' % seq)





















