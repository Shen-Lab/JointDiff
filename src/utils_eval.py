import numpy as np
import os
from numpy import linalg
import pickle

import Bio.PDB
from Bio.PDB import PDBParser, internal_coords, kdtrees
from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist
#blosum_matrix = matlist.blosum62

try: # updated on 10/04/23 by SZ for biopython v1.81
    from Bio.SubsMat import MatrixInfo as matlist
    blosum_matrix = matlist.blosum62
except:
    from Bio.Align import substitution_matrices
    blosum_matrix = substitution_matrices.load("BLOSUM62")

from scipy.special import kl_div

#################################### Auxiliary functions #####################################

def dict_save(dictionary,path):
    with open(path,'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0


def dict_load(path):
    with open(path,'rb') as handle:
        result = pickle.load(handle)
    return result


def stat_print(token, val_list):
    print('%s (%d samples): mean=%.4f, median=%.4f, min=%.4f, max=%.4f, std=%.4f'%(token,
                                                                      len(val_list),
                                                                      np.mean(val_list),
                                                                      np.median(val_list),
                                                                      min(val_list),
                                                                      max(val_list),
                                                                      np.std(val_list)))

#################################### PDB coor read and write #####################################

RESIDUE_dict = {'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'Q':'GLN', 'E':'GLU',
                'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS', 'M':'MET', 'F':'PHE',
                'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP', 'Y':'TYR', 'V':'VAL', 'B':'ASX',
                'Z':'GLX', 'X':'UNK'}

RESIDUE_dict_reverse = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
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
                    element = ELEMENT_dict[atom]
                    a_idx += 1

                    line = 'ATOM  '
                    line += '{:>5} '.format(a_idx)
                    line += '{:<4} '.format(atom)
                    line += aa + ' '
                    line += chain
                    line += '{:>4}    '.format(resi_idx) # residue index
                    line += '{:>8}'.format('%.3f'%vec[0]) # x
                    line += '{:>8}'.format('%.3f'%vec[1]) # y
                    line += '{:>8}'.format('%.3f'%vec[2]) # z
                    line += '{:>6}'.format('1.00') # occupancy
                    line += '{:>6}'.format('0.00') # temperature
                    line += ' ' * 10
                    line += '{:>2}'.format(element)  # element

                    wf.write(line + '\n')

##################################### Sequence #########################################

def seq_extract(pdb_path):
    try:
        parser = PDBParser()
        structure = parser.get_structure('target', file = pdb_path)

        seq_list = []
        for model in structure:
            for chain in model:
                poly = Bio.PDB.Polypeptide.Polypeptide(chain)
                seq_list.append(str(poly.get_sequence()))

    except:
        print('Directly extract the sequence from the PDB.')

        seq_dict = {}
        with open(pdb_path, 'r') as rf:
            for line in rf:
                if line.startswith('ATOM'):
                    chain = line[21]
                    index = line[22:27]
                    resi = ''.join(line[17:20].split(' '))
                    if resi in RESIDUE_dict_reverse:
                        token = RESIDUE_dict_reverse[resi] 
                    else:
                        token = 'X'
                    
                    if not chain in seq_dict:
                        seq_dict[chain] = {'seq': token, 'last_idx': index}
                    elif index != seq_dict[chain]['last_idx']:
                        seq_dict[chain]['seq'] += token
                        seq_dict[chain]['last_idx'] = index

        seq_list = [seq_dict[chain]['seq'] for chain in seq_dict]


    if len(seq_list) > 1:
        print('Warning! Multiple (%d) sequences detected!'%len(seq_list))

    return seq_list


def seq_extract_direct(pdb_path):

    seq_dict = {}
    with open(pdb_path, 'r') as rf:
        for line in rf:
            if line.startswith('ATOM'):
                chain = line[21]
                index = line[22:27]
                resi = ''.join(line[17:20].split(' '))
                if resi in RESIDUE_dict_reverse:
                    token = RESIDUE_dict_reverse[resi]
                else:
                    token = 'X'

                if not chain in seq_dict:
                    seq_dict[chain] = {'seq': token, 'last_idx': index}
                elif index != seq_dict[chain]['last_idx']:
                    seq_dict[chain]['seq'] += token
                    seq_dict[chain]['last_idx'] = index

    seq_list = [seq_dict[chain]['seq'] for chain in seq_dict]


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
    #angles = ic_chain_bound.dihedraICr
    angles = ic_chain_bound.dihedraAngle  # updated for the new biopython version (02/06/24)

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

        #out_dict[kind][resi_idx] = angles[dihedra_dict[k]]
        out_dict[kind][resi_idx] = angles[dihedra_dict[k]] * np.pi / 180 # updated for the new biopython version (02/06/24)

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

def TM_score(pdb_1, pdb_2, with_RMSD = False):
    ###### preprocess the files ######
    if '(' in pdb_1 or ')' in pdb_1:
        pdb_1 = '\('.join(pdb_1.split('('))
        pdb_1 = '\)'.join(pdb_1.split(')'))
    if '(' in pdb_2 or ')' in pdb_2:
        pdb_2 = '\('.join(pdb_2.split('('))
        pdb_2 = '\)'.join(pdb_2.split(')'))

    ###### first align ######
    command_1 = './TMalign ' + pdb_1 + ' ' + pdb_2 + ' -a'
    output_1 = os.popen(command_1)
    out_1 = output_1.read()
    ### TMscore
    if "(if normalized by length of Chain_1)" in out_1:
        k_1 = out_1.index("(if normalized by length of Chain_1)")
        tms_1 = float(out_1[k_1-8:k_1-1])
    else:
        return None
    ### RMSD
    if with_RMSD and 'RMSD=' in out_1:
        rmsd_1 = float(out_1.split('RMSD=')[1].split(',')[0].split(' ')[-1])
    else:
        rmsd_1 = None

    ###### second_align ######
    command_2 = './TMalign ' + pdb_2 + ' ' + pdb_1 + ' -a'
    output_2 = os.popen(command_2)
    out_2 = output_2.read()
    ### TMscore
    if "(if normalized by length of Chain_1)" in out_2:
        k_2 = out_2.index("(if normalized by length of Chain_1)")
        tms_2 = float(out_2[k_2-8:k_2-1])
    else:
        tms_2 = None
    ### RMSD
    if with_RMSD and 'RMSD=' in out_2:
        rmsd_2 = float(out_2.split('RMSD=')[1].split(',')[0].split(' ')[-1])
    else:
        rmsd_2 = None

    ###### summary ######
    ### TMscore
    if tms_1 is not None and tms_2 is not None:
        tmscore = (tms_1 + tms_2) / 2
    else:
        print('TMalign error for %s and %s!' % (pdb_1, pdb_2))
        tmscore = None
   
    if not with_RMSD:
        return tmscore

    ### RMSD
    if rmsd_1 is not None and rmsd_2 is not None:
        rmsd = (rmsd_1 + rmsd_2) / 2
    else:
        print('RMSD error for %s and %s!' % (pdb_1, pdb_2))
        rmsd = None

    return tmscore, rmsd


def TM_score_asym(pdb_1, pdb_2, with_RMSD = False):
    ### preprocess the files
    if '(' in pdb_1 or ')' in pdb_1:
        pdb_1 = '\('.join(pdb_1.split('('))
        pdb_1 = '\)'.join(pdb_1.split(')'))
    if '(' in pdb_2 or ')' in pdb_2:
        pdb_2 = '\('.join(pdb_2.split('('))
        pdb_2 = '\)'.join(pdb_2.split(')'))

    command = './TMalign ' + pdb_1 + ' ' + pdb_2
    output = os.popen(command)
    out = output.read()

    ### tmscore
    if "(if normalized by length of Chain_1)" in out:
        k_1 = out.index("(if normalized by length of Chain_1)")
        tms = float(out[k_1-8:k_1-1])
    else:
        tms = None

    if not with_RMSD:
        return tms

    ### rmsd
    if 'RMSD=' in out: 
        rmsd = float(out.split('RMSD=')[1].split(',')[0].split(' ')[-1])
    else:
        rmsd = None

    return tms, rmsd


##################################### Sequence Identity #########################################

def SeqRecovery(s1, s2):
    if len(s1) != len(s2):
        print('Unmatched sequence!')
        print(s1)
        print(s2)
        return None
    else:
        l = len(s1)
        num_match = 0
        for i in range(l):
            if s1[i] == s2[i]:
                num_match += 1
        return num_match / l


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

##################################### distribution check #########################################

def info_collect(path, token = ''):
    out_dict = {'phi':[], 'psi':[], 'omega': [], 'clash':[], 'size': []}

    dict_list = [d for d in os.listdir(path) if d.endswith('.pkl') and token in d]
    print('%d sample dictionaries in all.' % len(dict_list)) 

    for d in dict_list:

        dp = os.path.join(path, d)
        d_temp = dict_load(dp)

        out_dict['clash'].append(d_temp['clash'])
        
        for idx in d_temp['phi'].keys():
            if idx in d_temp['psi'].keys() and idx in d_temp['omega'].keys():
                out_dict['phi'].append(d_temp['phi'][idx])
                out_dict['psi'].append(d_temp['psi'][idx])
                out_dict['omega'].append(d_temp['omega'][idx])
                out_dict['size'].append(len(d_temp['seq'][0]))

    return out_dict


def bins_process(data_list, bins):
    y = [0] * len(bins)
    for data in data_list:
        for idx,val in enumerate(bins):
            if data <= val:
                y[idx] += 1
                break
    
    denorm = len(data_list) * (bins[1] - bins[0])
    y = [i / denorm for i in y]
    return np.array(y)


def KL_divergency(data_list_tar, data_list_ref, bins):
    data_list_tar = bins_process(data_list_tar, bins)
    data_list_ref = bins_process(data_list_ref, bins)
    
    return sum(kl_div(data_list_tar, data_list_ref))
    

def JS_divergency(data_list_tar, data_list_ref, bins):
    data_list_tar = bins_process(data_list_tar, bins)
    data_list_ref = bins_process(data_list_ref, bins)
    
    middle_list = 0.5 * (data_list_tar + data_list_ref)
    
    return 0.5 * (sum(kl_div(data_list_tar, middle_list)) + sum(kl_div(data_list_ref, middle_list)))
