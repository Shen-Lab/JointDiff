import math
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide

############################################################
# Data Transformation
############################################################

def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def _aa_tensor_to_sequence(aa):
    return ''.join([
        Polypeptide.index_to_one(a.item()) for a in aa.flatten()
    ])


############################################################
# Padding
############################################################

DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_id': ' ', 
    'icode': ' ',
}

DEFAULT_NO_PADDING = {
    'origin',
    'id',  # add by SZ
    'chain',  # add by SZ
    'region',  # add by SZ
    'seqmap',  # add by SZ
}

class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        if 'seqmap' in keys:  # a patch added by SZ for the dataloader of single domain proteins
            keys.remove('seqmap')
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])  # max sequence length in this batch
        for data in data_list: # a patch added by SZ for the dataloader of single domain proteins
            if not 'chain_nb' in data.keys():
                data['chain_nb'] = torch.ones(data['res_nb'].shape).long() 

        keys = self._get_common_keys(data_list)  # get the common items over all the samples in this batch
        #print(keys, self.no_padding)
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def apply_patch_to_tensor(x_full, x_patch, patch_idx):
    """
    Args:
        x_full:  (N, ...)
        x_patch: (M, ...)
        patch_idx:  (M, )
    Returns:
        (N, ...)
    """
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full


############################################################
# Monomer Process
############################################################

def _label_single_chain(data, seq_map, max_seq_length = None):
    """
    data, seq_map = parsers.parse_biopython_structure(*)
    data: dictionary
        chain_id: list of length l; chain id for each residue
        resseq: 1D tensor; pdb idx of each residue
        icode: list; insertion code of each residue
        res_nb: 1D tensor; relaive residue idx of each residue, e.g. 1,2,...
        aa: 1D tensor; aa idx of each residue, represent the sequence
        pos_heavyatom: tensor, [l, atom num (15), 3]; atom-wise coordinates of each residue
        mask_heavyatom: bool tensor, [l, 15]; mask of each atom
    seq_map: dictionary; (chain, pdb resi_idx, icode): relative idx
    """

    if data is None or seq_map is None:
        print('None found for the inputs.')
        return data, seq_map

    ### tensor to string sequence
    data['seq'] = _aa_tensor_to_sequence(data['aa'])
    length = len(data['seq'])

    ### Remove too long sequences or empty sequences
    if length <= 0:
        logging.warning('Empty sequence found. Removed')
        return None, None

    if max_seq_length is not None and length >= max_seq_length:
        logging.warning(f'Sequence too long {length}. Removed.')
        return None, None

    return data, seq_map


def preprocess_SingleChain_structure(task):
    entry = task['entry']
    pdb_path = task['pdb_path']

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdb_path)[0]

    parsed = {
        'id': entry['id'],
        'chain': entry['chain'],
        'region': entry['region'],
        #'seqmap': None,
    }
    try:
        if entry['chain'] is not None:
            (
                data_info, # parsed['data'], 
                seqmap # parsed['seqmap']
            ) = _label_single_chain(*parsers.parse_biopython_structure(
                model[entry['chain']],
                ##max_resseq = 106    # Chothia, end of Light chain Fv
                max_resseq = float('inf') # SZ: do not worry about the absolute index
            ))

            ### extract the necessary data for the batch
            for key in data_info.keys():
                if key != 'seq' and key != 'seqmap':
                    parsed[key] = data_info[key]

        else:
            raise ValueError('Chain error for %s.'%entry['id'])
    except (
        PDBExceptions.PDBConstructionException,
        parsers.ParsingException,
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'],
            e.__class__.__name__,
            str(e)
        ))
        return None

    return parsed

############################################################
# Multimer
############################################################

def mpnn_data_load(dict_path, atom_num = 4):
    """Load the information from *.pt file of ProteinMPNN."""
    chain_info = torch.load(dict_path)

    L = len(chain_info['seq'])
    aa = []
    pos_heavyatom = []
    mask_heavyatom = []
    for idx in range(L):
        coor = chain_info['xyz'][idx][:atom_num]  # (4, 3)
        resi_mask = (~coor[:, 0].isnan()).int() # (4,)

        if resi_mask.any():
            resi = chain_info['seq'][idx].upper()
            if resi in aa_idx_dict:
                aa.append(aa_idx_dict[resi])
            else:
                aa.append(20)
            pos_heavyatom.append(coor.unsqueeze(0))
            mask_heavyatom.append(resi_mask.unsqueeze(0))

    size = len(aa)
    if size == 0:
        return None, size

    out_dict = {}
    out_dict['aa'] = torch.tensor(aa)
    out_dict['pos_heavyatom'] = torch.cat(pos_heavyatom)
    out_dict['mask_heavyatom'] = torch.cat(mask_heavyatom)
    out_dict['resi_nb'] = torch.arange(size)

    return out_dict, size


def preprocess_multimer_structure(entry, atom_num = 4):
    """Data process for a complex.
    
    Args:
        entry:
            id: pdb id
            chains: {chain: chain size}
            size: complex size
            interface: interface region
            path: path of the PDB file
            cluster: cluster
    """
    pdb_path = entry['path']
    if pdb_path.endswith('.pdb'):
        parser = PDB.PDBParser(QUIET=True)
        model = parser.get_structure(id, pdb_path)[0]
    else:
        pdb_path = '/'.join(pdb_path.split('/')[:-1])

    chain_list = list(entry['chains'].keys())

    parsed = {
        'id': entry['id'],
        'chains': chain_list,
        'interface': entry['interface'],
        'feat': {}
    }

    ##################################################################
    # data process 
    ##################################################################

    try:
        size_list = []
        for chain in parsed['chains']:
            ### required features: 'aa', 'resi_nb', 'pos_heavyatom', 'mask_heavyatom'

            ###### pdb file ######
            if pdb_path.endswith('.pdb'):
                (data_info, # parsed['data'], 
                 seqmap     # parsed['seqmap']
                ) = _label_single_chain(*parsers.parse_biopython_structure(
                    model[chain],
                    max_resseq = float('inf') # SZ: do not worry about the absolute index
                ))

                parsed['feat'][chain] = {}
                for key in data_info.keys():
                    if key != 'seq' and key != 'seqmap':
                        parsed['feat'][chain][key] = data_info[key]

            ###### *.pt file (from proteinMPNN) ######
            else:
                chain_path = os.path.join(pdb_path, '%s_%s.pt' % (entry['id'], chain))
                out_dict, size_sele = mpnn_data_load(chain_path, atom_num = atom_num)
                if size_sele == 0:
                    print('Empty chain %s for %s!' % (chain, entry['id']))
                    continue

                size_list.append(size_sele)
                parsed['feat'][chain] = out_dict

        parsed['size_list'] = size_list
        parsed['size'] = sum(size_list)
        print('Done for %s.' % entry['id'])

    except (
        PDBExceptions.PDBConstructionException,
        parsers.ParsingException,
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            entry['id'],
            e.__class__.__name__,
            str(e)
        ))
        return None

    return parsed


############################################################
# Data for Confidence Net
############################################################

def preprocess_confidence_structure(task):
    entry = task['entry']
    pdb_path = task['pdb_path']

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdb_path)[0]

    parsed = {
        'id': entry['id'],
        'chain': entry['chain'],
        'region': entry['region'],
    }

    ###### label ######
    label = []
    weight = []
    for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:
        label.append(entry[metric])
        if ('%s-weight' % metric) in entry:
            weight.append(entry['%s-weight' % metric])

    parsed['label'] = np.array(label)
    if weight:
        parsed['weight'] = np.array(weight)

    try:
        if entry['chain'] is not None:
            (
                data_info, # parsed['data'], 
                seqmap # parsed['seqmap']
            ) = _label_single_chain(*parsers.parse_biopython_structure(
                model[entry['chain']],
                ##max_resseq = 106    # Chothia, end of Light chain Fv
                max_resseq = float('inf') # SZ: do not worry about the absolute index
            ))

            ### extract the necessary data for the batch
            for key in data_info.keys():
                if key != 'seq' and key != 'seqmap':
                    parsed[key] = data_info[key]

        else:
            raise ValueError('Chain error for %s.'%entry['id'])

    except (
        PDBExceptions.PDBConstructionException,
        parsers.ParsingException,
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'],
            e.__class__.__name__,
            str(e)
        ))
        return None

    return parsed

