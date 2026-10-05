import numpy as np
import os
from numpy import linalg
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import Bio.PDB
from Bio.PDB import PDBParser, internal_coords, kdtrees
from Bio import pairwise2

try:
    from Bio.SubsMat import MatrixInfo as matlist
    blosum_matrix = matlist.blosum62
except:
    from Bio.Align import substitution_matrices  
    blosum_matrix = substitution_matrices.load('BLOSUM62') 

def dict_save(dictionary,path):
    with open(path,'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0

def dict_load(path):
    with open(path,'rb') as handle:
        result = pickle.load(handle)
    return result

#################################### for models #####################################

def model_size_check(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model parameters: {}'.format(param_size))
    print('buffer parameters: {}'.format(buffer_size))
    print('model size: {:.3f}MB'.format(size_all_mb))

#################################### PDB coor read and write #####################################

RESIDUE_dict = {'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'Q':'GLN', 'E':'GLU',
                'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS', 'M':'MET', 'F':'PHE',
                'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP', 'Y':'TYR', 'V':'VAL', 'B':'ASX',
                'Z':'GLX', 'X':'UNK'}

RESIDUE_reverse_dict = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
                        'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F',
                        'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V', 'ASX':'B',
                        'GLX':'Z', 'UNK':'X'}

ELEMENT_dict = {'N':'N', 'CA':'C', 'C':'C', 'O':'O'}

def read_pdb_coor(file_path):
    parser = PDBParser()
    structure = parser.get_structure("protein", file_path)
    model = structure[0]  # Assuming a single model in the PDB file

    # Access chains, residues, atoms, and their properties
    out_dict = {}

    for chain in model:
        chain_id = chain.full_id[2]
        out_dict[chain_id] = {'coor':{}, 'ordered_idx':[]}

        for residue in chain:
            resi_id = str(residue.full_id[3][1]) + residue.full_id[3][2]
            out_dict[chain_id]['coor'][resi_id] = {}
            out_dict[chain_id]['ordered_idx'].append(resi_id)

            for atom in residue:
                atom_name = atom.get_name()
                atom_coord = atom.get_coord()

                out_dict[chain_id]['coor'][resi_id][atom_name] = atom_coord
                # Process the atom as needed
                # print(f"Atom Name: {atom_name}, Atom Coordinates: {atom_coord}")
    return out_dict


def pdb_line_write(chain, aa, resi_idx, atom, a_idx, vec, wf):
    """Write a single line in the PDB file."""

    line = 'ATOM  '
    line += '{:>5} '.format(a_idx)
    line += '{:<4} '.format(atom)
    line += aa + ' '
    line += chain
    line += '{:>4}    '.format(resi_idx) # residue index

    ### coordinates
    for coor_val in vec:
        coor_val = float(coor_val)

        # to avoid the coordinate value is longer than 8
        if len('%.3f' % coor_val) <= 8:
            ### -999.9995 < coor_val < 1000.9996
            coor_val = '%.3f' % coor_val

        elif coor_val >= 99999999.5:
            coor_val = '%.2e' % coor_val

        elif coor_val <= -9999999.5:
            coor_val = '%.1e' % coor_val

        else:
            # length of the interger part
            inte_digit = 1 + int(np.log10(abs(coor_val))) + int(coor_val < 0)
            deci_digit = max(7 - inte_digit, 0)
            coor_val = '%f' % round(coor_val, deci_digit)
            coor_val = coor_val[:8]

        line += '{:>8}'.format(coor_val)

    line += '{:>6}'.format('1.00') # occupancy
    line += '{:>6}'.format('0.00') # temperature
    line += ' ' * 10
    element = ELEMENT_dict[atom]
    line += '{:>2}'.format(element)  # element

    wf.write(line + '\n')


def pdb_write(coor_dict, path):
    """
    Transform the given info into the pdb format.
    """
    with open(path, 'w') as wf:
        a_idx = 0

        for chain in coor_dict.keys():
            c_coor_dict = coor_dict[chain]['coor']
            if 'seq' in coor_dict[chain].keys():
                seq = coor_dict[chain]['seq']
                if len(c_coor_dict.keys()) != len(seq):
                    print('Error! The size of the strutctue and the sequence do not match for chain %s! (%d and %d)'%(len(c_coor_dict.keys()),
                                                                                                         len(seq), chain))
                    continue
            else:
                seq = None

            for i,resi_idx in enumerate(coor_dict[chain]['ordered_idx']):
                if seq is not None:
                    aa = RESIDUE_dict[seq[i]]
                else:
                    aa = 'GLY' 

                for atom in c_coor_dict[resi_idx].keys():
                    vec = c_coor_dict[resi_idx][atom] 
                    a_idx += 1

                    pdb_line_write(chain, aa, resi_idx, atom, a_idx, vec, wf)



def inference_pdb_write(coor, path, seq = None, chain = 'A',
              atom_list = ['N', 'CA', 'C', 'O']):
    """
    Args: 
        coor: (L, atom_num, 3)
        seq: str of length L
    """
    if seq is not None and coor.shape[0] != len(seq):
        print('Error! The size of the strutctue and the sequence do not match! (%d and %d)'%(coor.shape[0],
                                                                                             len(seq)))
    elif coor.shape[1] != len(atom_list):
        print('Error! The size of the resi-wise coor and the atom_num do not match! (%d and %d)'%(coor.shape[1],
                                                                                             len(atom_list)))
    else:
        with open(path, 'w') as wf:
            a_idx = 0

            for i, resi in enumerate(seq):
                ### residue-wise info
                r_idx = i + 1
                aa = RESIDUE_dict[resi]

                for j,vec in enumerate(coor[i]):
                    ### atom-wise info
                    atom = atom_list[j]
                    a_idx += 1 
                    pdb_line_write(chain, aa, r_idx, atom, a_idx, vec, wf)


def synthe_pdb_write(coor, path, seq = None, chain = 'A',
              atom_list = ['N', 'CA', 'C', 'O']):
    """
    Args: 
        coor: (L, atom_num, 3)
        seq: str of length L
    """
    if seq is not None and coor.shape[0] != len(seq):
        print('Error! The size of the strutctue and the sequence do not match! (%d and %d)'%(coor.shape[0],
                                                                                             len(seq)))
    elif coor.shape[1] != len(atom_list):
        print('Error! The size of the resi-wise coor and the atom_num do not match! (%d and %d)'%(coor.shape[1],
                                                                                             len(atom_list)))
    else:
        with open(path, 'w') as wf:
            a_idx = 1
            n = coor.shape[0]

            for i in range(n):
                if seq is not None:
                    resi = seq[i] 
                else:
                    resi = 'G'
                ### residue-wise info
                r_idx = i + 1
                aa = RESIDUE_dict[resi]

                for j,vec in enumerate(coor[i]):
                    ### atom-wise info
                    atom = atom_list[j]
                    element = ELEMENT_dict[atom]

                    line = 'ATOM  '
                    line += '{:>5} '.format(a_idx)
                    line += '{:<4} '.format(atom)
                    line += aa + ' '
                    line += chain
                    line += '{:>4}    '.format(r_idx) # residue index
                    # line += '{:>8}'.format('%.3f'%vec[0]) # x
                    # line += '{:>8}'.format('%.3f'%vec[1]) # y
                    # line += '{:>8}'.format('%.3f'%vec[2]) # z

                    ### to avoid too long coordinates
                    vec_str = []
                    for value in vec:
                        value = '%.3f'%value
                        while len(value) > 8:
                            value = value[:-1]
                        vec_str.append(value)
                    line += '{:>8}'.format(vec_str[0]) # x
                    line += '{:>8}'.format(vec_str[1]) # y
                    line += '{:>8}'.format(vec_str[2]) # z

                    line += '{:>6}'.format('1.00') # occupancy
                    line += '{:>6}'.format('0.00') # temperature
                    line += ' ' * 10
                    line += '{:>2}'.format(element)  # element

                    wf.write(line + '\n')
                    a_idx += 1

##################################### Sequence #########################################

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


##################################### Angles #########################################

def angles_cal_singlechain(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('target', file = pdb_path)
    out_dict = {'phi':{}, 'psi':{}, 'omega':{}, 'seq':[]}

    ### sequences 

    for model in structure:
        for chain in model:
            poly = Bio.PDB.Polypeptide.Polypeptide(chain)
            out_dict['seq'].append(str(poly.get_sequence()))

    if len(out_dict['seq']) > 1:
        print('Warning! Multiple (%d) sequences detected!'%(len(out_dict['seq'])))

    ### angles

    ic_chain_bound = internal_coords.IC_Chain(structure)
    ic_chain_bound.atom_to_internal_coordinates()

    dihedra_dict = ic_chain_bound.dihedraNdx
    angles = ic_chain_bound.dihedraICr

    for k in dihedra_dict.keys():
        name = [str(s) for s in k]
        comb = ''.join([s.split('_')[-1] for s in name])
        if comb == 'NCACN':
            kind = 'psi'
            resi_idx = int(name[0].split('_')[0])
        elif comb == 'CACNCA':
            kind = 'omega'
            resi_idx = int(name[0].split('_')[0])
        elif comb == 'CNCAC':
            kind = 'phi'
            resi_idx = int(name[-1].split('_')[0])
        else:
            # print('Warning! No angle for %s!'%name)
            continue

        out_dict[kind][resi_idx] = angles[dihedra_dict[k]]

    return out_dict

##################################### Clash #########################################

###### from EGCN ######
def remove(string,char):
    '''
    Remove the character in a string.
    '''
    #string_char = [i for i in string.split(char) if i != '']
    #return string_char[0]
    string_char = string.split(char)
    return ''.join(string_char)

def get_coor(path):
    coor = []
    with open(path, "r") as f:
       for lines in f:
           if len(lines)>4 and lines[0:4]=='ATOM' and lines[12]!='H' and lines[12:14]!=' H':
               x = float(remove(lines[30:38],' '))
               y = float(remove(lines[38:46],' '))
               z = float(remove(lines[46:54],' '))
               coor.append(np.array([x,y,z]))
    return coor

def detect_clash(x,y):
    if np.sum((x-y)**2)<=9:
        return True
    return False

def clash_cal_singlechain(pdb_path):
    clash_num = 0
    coors = get_coor(pdb_path)
    atom_num = len(coors)
    for i in range(atom_num):
        for j in range(i+1, atom_num):
            if detect_clash(coors[i], coors[j]):
                clash_num += 1
    return clash_num, atom_num

###### from https://www.blopig.com/blog/2023/05/checking-your-pdb-file-for-clashing-atoms/ ######

atom_radii = {
#    "H": 1.20,  # Who cares about hydrogen??
    "C": 1.70, 
    "N": 1.55, 
    "O": 1.52,
    "S": 1.80,
    "F": 1.47,
    "P": 1.80,
    "CL": 1.75,
    "MG": 1.73,
}

def count_clashes(pdb_file, clash_cutoff=0.63):
    # Create a PDB parser
    parser = PDBParser()

    # Parse the PDB file
    structure = parser.get_structure('protein', pdb_file)

    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j])) for i in atom_radii for j in atom_radii}
    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")
    # Build a KDTree (speedy!!!)
    kdt = kdtrees.KDTree(coords)
    # Initialize a list to hold clashes
    clashes = []
    # Iterate through all atoms
    atom_num = len(atoms)

    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))
        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]
        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue
            # Exclude clashes from peptide bonds
            elif (atom_2.name == "C" and atom_1.name == "N") or (atom_2.name == "N" and atom_1.name == "C"):
                continue
            # Exclude clashes from disulphide bridges
            elif (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue
            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))
    return len(clashes) // 2, atom_num

##################################### TM-score #########################################

def TM_score(pdb_1,pdb_2):
    command_1 = './TMalign ' + pdb_1 + ' ' + pdb_2 + ' -a'
    output_1 = os.popen(command_1)
    out_1 = output_1.read()
    if "(if normalized by length of Chain_1)" in out_1:
        k_1 = out_1.index("(if normalized by length of Chain_1)")
        tms_1 = out_1[k_1-8:k_1-1]
    else:
        return None

    command_2 = './TMalign ' + pdb_2 + ' ' + pdb_1 + ' -a'
    output_2 = os.popen(command_2)
    out_2 = output_2.read()
    if "(if normalized by length of Chain_1)" in out_2:
        k_2 = out_2.index("(if normalized by length of Chain_1)")
        tms_2 = out_2[k_2-8:k_2-1]
    else:
        return None
    return (float(tms_1) + float(tms_2))/2

##################################### Sequence Identity #########################################

def Identity(x,y,matrix = blosum_matrix):
    X = x.upper()
    Y = y.upper()
    alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
    max_iden = 0
    for i in alignments:
        same = 0
        for j in range(i[-1]):
            if i[0][j] == i[1][j] and i[0][j] != '-':
                same += 1
        iden = float(same)/float(i[-1])
        if iden > max_iden:
            max_iden = iden
    return max_iden


##################################### Dataset #########################################


def add_right_padding(tensor_ori, dim=[0], pad_length=[1], val=0):
    dim_num = len(tensor_ori.shape)
    pad = [0] * (dim_num * 2)
    for i, d in enumerate(dim):
        pad[2*(dim_num - d)-1] = pad_length[i]
    return F.pad(tensor_ori, tuple(pad), 'constant', val)


class AutoencoderDataset(Dataset):
    def __init__(self,
        args,
        key_list=[
            'name',
            'aatype',
            'seq_mask',
            'pseudo_beta',
            'pseudo_beta_mask',
            'backbone_rigid_tensor',
            'backbone_rigid_mask',
            'rigidgroups_gt_frames',
            'rigidgroups_alt_gt_frames',
            'rigidgroups_gt_exists',
            'atom14_gt_positions',
            'atom14_alt_gt_positions',
            'atom14_gt_exists',
            'atom14_atom_is_ambiguous',
            'atom14_alt_gt_exists',
        ],
        ignore_set = set(),
        padded_length = 200
    ):

        self.voxel_size = len(args.esm_restypes)

        data_info_all = dict_load(args.data_path)
        entry_list = [name 
            for name in dict_load(args.entry_list_path)
            if name not in ignore_set
        ]
        if args.debug_num is not None:
            entry_list = entry_list[:args.debug_num]

        self.data = []
        self.name_list = []
        self.padded_length = padded_length
        discard_num = 0
        sample_idx = 0

        for entry in entry_list:
            if entry in data_info_all.keys() and data_info_all[entry]['aatype'].shape[0] <= args.max_length:
                data_info = {key: data_info_all[entry][key]
                             for key in key_list if key in data_info_all[entry]}
                length = data_info['aatype'].shape[0]
                self.padded_length = max(self.padded_length, length)
                data_info['residx'] = torch.arange(length) + 1
                data_info['name'] = entry
                data_info['sample_idx'] = torch.tensor(sample_idx)
                sample_idx += 1

                self.data.append(data_info)
                self.name_list.append(entry)

            else:
                # print(entry in data_info_all.keys())
                discard_num += 1

        print('%d entries loaded. %d entries discarded.' %
              (self.__len__(), discard_num))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        aatype: (L,),
        seq_mask: (L,),
        residx: (L,),
        pseudo_beta: (L, 3),
        pseudo_beta_mask: (L,),
        backbone_rigid_tensor: (L, 4, 4),
        backbone_rigid_mask: (L,),
        rigidgroups_gt_frames: (L, 8, 4, 4),
        rigidgroups_alt_gt_frames: (L, 8, 4, 4),
        rigidgroups_gt_exists: (L, 8),
        atom14_gt_positions: (L, 14, 3),
        atom14_alt_gt_positions: (L, 14, 3),
        atom14_gt_exists: (L, 14),
        atom14_atom_is_ambiguous: (L, 14),
        atom14_alt_gt_exists: (L, 14),
        """

        data_info = self.data[idx]
        data_info['length'] = data_info['aatype'].shape[0]
        pad_length = self.padded_length - data_info['length']

        for key in data_info.keys():
            if key not in {'sample_idx', 'name', 'length'}:
                if key == 'aatype':
                    # pad_val = self.voxel_size ## would cause error for ESMFold as the index can only be from 0 to 20
                    pad_val = 0
                else:
                    pad_val = 0
                data_info[key] = add_right_padding(
                    data_info[key], dim=[0], pad_length=[pad_length], val=pad_val
                )

        return data_info
