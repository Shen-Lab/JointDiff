################################################
# dataloader for the DIPS entries
################################################

import os
import random
import logging
import datetime
import pandas as pd
import joblib
import pickle
import lmdb
import subprocess
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..utils.protein import parsers, constants
from diffab.utils.protein.constants import ressymb_order
from ._base import register_dataset
import numpy as np

def dict_save(dictionary,path):
    with open(path,'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0

def dict_load(path):
    with open(path,'rb') as handle:
        result = pickle.load(handle)
    return result


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0

TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]

ressymb_set = set(ressymb_order)
aa_idx_dict = {}
for idx, char in enumerate(ressymb_order):
    aa_idx_dict[char] = idx


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
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _label_single_chain(data, seq_map, max_seq_length = None):
    """
    data, seq_map = parsers.parse_biopython_structure(*)
    data: dictionary
        chain_id: list of length l; chain id for each residue
        resseq: 1D tensor; pdb idx of each residue
        icode: list; insertion code of each residue
        resi_nb: 1D tensor; relaive residue idx of each residue, e.g. 1,2,...
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

            # required features: 'aa', 'resi_nb', 'pos_heavyatom', 'mask_heavyatom'

            #print(pdb_path)
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
                #print(chain_path)

                out_dict, size_sele = mpnn_data_load(chain_path, atom_num = atom_num)
                if size_sele == 0:
                    print('Empty chain %s for %s!' % (chain, entry['id']))
                    continue

                size_list.append(size_sele)
                parsed['feat'][chain] = out_dict

        #else:
        #    raise ValueError('Chain error for %s.'%entry['id'])

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

################################################################################
# 
################################################################################

class BinderProcess(object):

    def __init__(self, 
        interface_dict = None, with_epitope = False, with_bindingsite = False, 
        with_scaffold = True, random_masking = False, mask_threshold = 80
    ):
        super().__init__()
        self.interface_dict = interface_dict
        self.with_epitope = with_epitope
        self.with_bindingsite = with_bindingsite
        self.with_scaffold = with_scaffold
        self.random_masking = random_masking
        self.mask_threshold = mask_threshold


    def _data_attr(self, data, name):
        if name in ('generate_flag', 'anchor_flag') and name not in data:
            return torch.zeros(data['aa'].shape, dtype=torch.bool)
        else:
            return data[name]


    def __call__(self, structure, sample_ratio = 0.5, max_aa = 80):
        """Get the input data.
        
        Args: 
            structure:
                ****** ProteinMPNN version ******
                'interface'
                'chains'
                'size_list'
                'feat':
                    chain:
                        'aa'
                        'resi_nb'
                        'pos_heavyatom'
                        'mask_heavyatom'
                ****** Fintunning version ****** 
                'antigen': antigen chain
                'interface':
                    chain_id: range(),
                    ...
                'chains': <list of chains (str)>
                'size_list': <list of sizes (int)>
                'feat': 
                    chain_id:
                        'aa': <array of char>
                        'res_nb': <tensor of aa tokens>
                        'pos_heavyatom': <heavy atom coordinates>
                        'mask_heavyatom': <mask of valid heavychains>
                    ...
                'epitope':
                    chain_id: <list of binding sites (int)>
                    ...
        """

        data_list = []

        ##############################################################
        # select the interface region
        ##############################################################

        if ('epitope' not in structure) or (not self.with_epitope):
            structure['epitope'] = {}

        ###################### Fintuning Data ########################
        if 'antigen' in structure:
            ag_chain = structure['antigen']
            chain_list = [chain for chain in structure['interface'] if chain != ag_chain]
            chain_sele = np.random.choice(chain_list)

            ###### design region ######
            for chain in structure['chains']:
                # only retrain
                if chain != chain_sele:
                    del structure['interface'][chain]
                    continue

                binder_size = len(structure['interface'][chain])
                idx_min = min(structure['interface'][chain])
                idx_max = max(structure['interface'][chain])
                if self.random_masking and binder_size > self.mask_threshold:
                    # maximum start: idx_max - self.mask_threshold + 1
                    start_idx = np.random.choice(range(idx_min, idx_max - self.mask_threshold + 2))
                    structure['interface'][chain] = [start_idx, start_idx + self.mask_threshold]
                else:
                    structure['interface'][chain] = [idx_min, idx_max + 1]
             
            ###### epitopes ######
            if self.with_epitope and (not self.with_bindingsite):
                for chain in structure['chains']:
                    if chain == chain_sele:
                        del structure['epitope'][chain]
                        continue
                    structure['epitope'][chain] = range(
                        min(structure['epitope'][chain]),
                        max(structure['epitope'][chain]) + 1
                    )

        ###################### MPNN processed Data ##################
        elif self.interface_dict is not None \
        and structure['id'] in self.interface_dict \
        and self.interface_dict[structure['id']]:

            interface = self.interface_dict[structure['id']]
            ## {chain_1: {chain_2: idx on chain_1}}

            ### select the chain for design
            chain_sele = np.random.choice(list(interface.keys()))
            idx_min = 1000
            idx_max = 0

            ###### design region ######

            for chain_sub in interface[chain_sele]:
                idx_min = min(idx_min, interface[chain_sele][chain_sub][0])
                idx_max = max(idx_max, interface[chain_sele][chain_sub][-1])
            binder_size = idx_max - idx_min + 1

            if self.random_masking and binder_size > self.mask_threshold:
                start_idx = np.random.choice(range(idx_min, idx_max - self.mask_threshold + 2)) 
                structure['interface'] = {chain_sele: [start_idx, start_idx + self.mask_threshold]}
                # print(structure['interface'][chain_sele][1] - structure['interface'][chain_sele][0])
            else:
                structure['interface'] = {chain_sele: [idx_min, idx_max + 1]}

            ###### epitope ######

            if self.with_epitope:
                for chain_sub in interface[chain_sele]:
                    ### point out the binding site
                    if self.with_bindingsite:
                        structure['epitope'][chain_sub] = interface[chain_sub][chain_sele]
                    ### point out the binding region
                    else:
                        structure['epitope'][chain_sub] = range(
                            interface[chain_sub][chain_sele][0],
                            interface[chain_sub][chain_sele][-1] + 1,
                        )
                    
        ################# monomer ###############################
        elif structure['interface'] is None and len(structure['chains']) == 1:
            ###### monomer: mask part of the tokens ######
            chain_sele = structure['chains'][0]
            size_sele = structure['size_list'][0]
            mask_region = min(size_sele * sample_ratio, max_aa)
            start_idx = np.random.choice(range(int(size_sele - mask_region)))
            structure['interface'] = {
                #chain_sele: range(int(start_idx), int(start_idx + mask_region))
                chain_sele: [int(start_idx), int(start_idx + mask_region)]
            }

        ################# select the interface ####################
        elif structure['interface'] is None:
            ###### multimer: mask a chain ######
            chain_sele = np.random.choice(structure['chains'])
            size_sele = structure['size_list'][structure['chains'].index(chain_sele)]
            #structure['interface'] = {chain_sele: range(size_sele)}
            structure['interface'] = {chain_sele: [0, size_sele]}

        ##############################################################
        # Feature process
        ##############################################################
       
        ###### fragment type ######
        data_list = []
        for i, chain in enumerate(structure['chains']):
            generate_flag = torch.full_like(
                structure['feat'][chain]['aa'], fill_value = 0,
            )
            mask = torch.full_like(
                structure['feat'][chain]['aa'], fill_value = 1,
            )
            chain_nb = torch.full_like(
                structure['feat'][chain]['aa'], fill_value = i,
            )
            L = structure['feat'][chain]['aa'].shape[-1]
  
            ## flagment token: 1 for antigen, 2 for target, 3 for scaffold, 4 for epitope

            #### design chain
            if chain in structure['interface']:
                fragment_map = torch.full_like(
                    structure['feat'][chain]['aa'], fill_value = 3,
                )  # (N,L), 3 for scaffold
                start_idx = int(structure['interface'][chain][0])
                end_idx = int(structure['interface'][chain][1])
                #print(start_idx, end_idx)

                fragment_map[start_idx : end_idx] = 2  # design region
                generate_flag[start_idx : end_idx] = 1

                if not self.with_scaffold:
                    generate_flag = generate_flag[start_idx : end_idx]
                    mask = mask[start_idx : end_idx]
                    chain_nb = chain_nb[start_idx : end_idx]
                    fragment_map = fragment_map[start_idx : end_idx]
 
                    structure['feat'][chain]['aa'] = structure['feat'][chain]['aa'][start_idx : end_idx]
                    structure['feat'][chain]['resi_nb'] = structure['feat'][chain]['resi_nb'][start_idx : end_idx]
                    structure['feat'][chain]['pos_heavyatom'] = structure['feat'][chain]['pos_heavyatom'][start_idx : end_idx]
                    structure['feat'][chain]['mask_heavyatom'] = structure['feat'][chain]['mask_heavyatom'][start_idx : end_idx]

            ### epitope
            elif chain in structure['epitope']:
                fragment_map = torch.full_like(
                    structure['feat'][chain]['aa'], fill_value = 1,
                )  # (N,L), target protein
                for idx in structure['epitope'][chain]:
                    if idx >= L:
                        continue
                    fragment_map[idx] = 4  # epitope

            ### others 
            else:
                fragment_map = torch.full_like(
                    structure['feat'][chain]['aa'], fill_value = 1,
                )

            structure['feat'][chain]['fragment_type'] = fragment_map 
            structure['feat'][chain]['generate_flag'] = generate_flag 
            structure['feat'][chain]['mask'] = mask 
            structure['feat'][chain]['chain_id'] = chain 
            structure['feat'][chain]['chain_nb'] = chain_nb
            data_list.append(structure['feat'][chain])

        ###### chain index ######
        #self.assign_chain_number_(data_list)

        list_props = {
            'chain_id': [],
            #'icode': [],
        }
        tensor_props = {
            'chain_nb': [],
            'resi_nb': [],
            'aa': [],
            'mask': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'generate_flag': [],
            'fragment_type': [],
        }

        for data in data_list:
            for k in list_props.keys():
                list_props[k].append(self._data_attr(data, k))
            for k in tensor_props.keys():
                tensor_props[k].append(self._data_attr(data, k))

        ## list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        data_out = {
            **list_props,
            **tensor_props,
        }
        return data_out


################################################################################
# 
################################################################################

class ProteinMPNNDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self, 
        summary_path: str='../data/Protein_MPNN/mpnn_data_info.pkl', 
        pdb_dir: str='../data/Protein_MPNN/pdb_2021aug02/pdb/', 
        processed_dir: str='../data/Protein_MPNN/',
        interface_path: str='../data/Protein_MPNN/interface_dict_all.pt',
        dset: str='train', 
        transform = 'default',
        reset = False,
        reso_threshold = 3.0,
        length_min = 20,
        length_max = 800,
        with_monomer = False,
        load_interface = True,
        with_epitope = False,
        with_bindingsite = False,
        with_scaffold = True,
        random_masking = False, 
        mask_threshold = 80
    ):
        """
        Args:
            summary_path: info list. 
            pdb_dir: path of the pdb files.
            processed_dir: path of the processed data. 
            split: dataset.
            random_split: whether split the data based on the sequence clusters.
            val_ratio: ratio of the validation set.
            test_ratio: ratio of the test set.
            split_seed: shuffling seed.
            transform: data transformation function.
            reset: whether reprocess the data (e.g. lmdb process, clustering) 
                if it already exists.
        """
        super().__init__()

        self.summary_path = summary_path
        self.pdb_dir = pdb_dir
        self.processed_dir = processed_dir
        self.dset = dset
        self.reso_threshold = reso_threshold
        self.length_min = length_min
        self.length_max = length_max
        self.with_monomer = with_monomer

        if with_monomer:
            self.structure_data_path = os.path.join(
                processed_dir, 'structures.%s.withMono.lmdb' % dset
            )
            self.structure_id_path = os.path.join(
                processed_dir, 'structures.%s.withMono.lmdb-ids' % dset
            )
        else:
            self.structure_data_path = os.path.join(
                processed_dir, 'structures.%s.lmdb' % dset
            )
            self.structure_id_path = os.path.join(
                processed_dir, 'structures.%s.lmdb-ids' % dset
            )

        if load_interface and interface_path is not None \
        and os.path.exists(interface_path):
            self.interface_dict = torch.load(interface_path)
            print('Interface loaded from %s.' % interface_path)
            self.load_interface = True
        else:
            self.interface_dict = None
            self.load_interface = False

        ##############################################################
        # check the input paths 
        ##############################################################

        if not (os.path.exists(self.structure_data_path) or os.path.exists(pdb_dir)):
            raise FileNotFoundError(
                f"PDB structures not found in {pdb_dir}. "
            )

        ###### check the output paths ######
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        ############################################################## 
        # prepare the sample information
        ##############################################################

        self.protein_entries = None
        self._load_protein_entries()

        self.db_conn = None
        self.db_ids = None
       
        ### Load the structure information
        self._load_structures(reset)

        ############################################################## 
        # load the data
        ##############################################################

        self._load_dataset(dset) 

        ############################################################## 
        # data transformation
        ##############################################################

        if transform == 'default':
            transform = BinderProcess(
                interface_dict = self.interface_dict, 
                with_epitope = with_epitope,
                with_bindingsite = with_bindingsite,
                with_scaffold = with_scaffold,
                random_masking = random_masking, 
                mask_threshold = mask_threshold
            )
        self.transform = transform


    ########################################################################### 
    # utility functions
    ###########################################################################

    ###################### overall data process ###############################

    def _load_protein_entries(self):
        """
        Load the sample basic information in the *.tsv file.
        """
        self.info_dict = dict_load(self.summary_path)
        entries_all = []

        for clus in tqdm(self.info_dict[self.dset]):

            ######################### cluster-wise ############################

            for pdb in self.info_dict['all'][clus]:
                 
                ##################### complex (entry) wise ####################
 
                ###### Filtering ######
                if self.reso_threshold is not None \
                and  self.reso_threshold < self.info_dict['all'][clus][pdb]['reso']:
                    continue

                if self.length_min is not None \
                and  self.length_min > self.info_dict['all'][clus][pdb]['size']:
                    continue

                if self.length_max is not None \
                and  self.length_max < self.info_dict['all'][clus][pdb]['size']:
                    continue

                if not self.with_monomer \
                and len(self.info_dict['all'][clus][pdb]['chains']) < 2:
                    continue

                if self.load_interface and (not self.with_monomer) and pdb not in self.interface_dict:
                    continue

                ###### selected entry ######
                entry = {
                    'id': pdb,
                    'chains': self.info_dict['all'][clus][pdb]['chains'],
                    'size': self.info_dict['all'][clus][pdb]['size'],
                    'path': os.path.join(
                        self.pdb_dir, self.info_dict['all'][clus][pdb]['folder'], 
                        '%s.pt' % pdb
                    ),
                    'cluster': clus,
                    'interface': None,
                }
                entries_all.append(entry)

        ######################## selected samples #############################
        self.protein_entries = entries_all


    def _load_structures(self, reset):
        """
        Load the structure information and do the filtering.
        """
        ### check whether the *.lmdb file exists or whether need to process again
        if not os.path.exists(self.structure_data_path) or reset:

            if os.path.exists(self.structure_data_path):
                ### remove the processed file for the new one
                os.unlink(self.structure_data_path)

            ### Prepare the *.lmdb and the *.lmdb-ids files
            self._preprocess_structures()

        with open(self.structure_id_path, 'rb') as f:
            self.db_ids = pickle.load(f)  # list of the sample ids

        self.protein_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids, self.protein_entries
            )
        )

        
    def _preprocess_structures(self):
        """
        Prepare the *.lmdb and the *.lmdb-ids files
        """

        data_list = []
        for entry in tqdm(self.protein_entries):
            data_list.append(preprocess_multimer_structure(entry))

        ### prepare the *.lmdb files
        db_conn = lmdb.open(
            self.structure_data_path,  # the lmdb file
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

        with open(self.structure_id_path, 'wb') as f:
            pickle.dump(ids, f)


    def _load_dataset(self, dset):
        """
        Load the preprocessed (split) dataset.
        """
        assert dset in ('train', 'val', 'test', 'all')

        # self.ids_in_split = [entry['id'] for entry in self.protein_entries 
        #     if os.path.exists(os.path.join(self.pdb_dir, '{}.pt'.format(entry['id'])))
        # ]
        self.ids_in_split = [entry['id'] for entry in self.protein_entries] 
        print('%d samples loaded for the %s set.'%(len(self.ids_in_split), dset))


    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self.structure_data_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )


    def get_structure(self, id):
        self._connect_db()
        with self.db_conn.begin() as txn:
            return pickle.loads(txn.get(id.encode()))


    def __len__(self):
        return len(self.ids_in_split)


    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        if self.transform is not None:
            data = self.transform(data)
        data['idx'] = index
        data['name'] = id
        return data


class FineTuningDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self,
        data_path: str='../data/FineTuning/data_list_new.pkl',
        length_min = 20,
        length_max = 800,
        with_monomer = False,
        with_epitope = False,
        with_bindingsite = False,
        with_scaffold = True,
        random_masking = False, 
        mask_threshold = 80
    ):
        """
        Args:
            data_path: info list. 
            pdb_dir: path of the pdb files.
            processed_dir: path of the processed data. 
            split: dataset.
            random_split: whether split the data based on the sequence clusters.
            val_ratio: ratio of the validation set.
            test_ratio: ratio of the test set.
            split_seed: shuffling seed.
            transform: data transformation function.
            reset: whether reprocess the data (e.g. lmdb process, clustering) 
                if it already exists.
        """
        super().__init__()

        self.data_list_all = dict_load(data_path)
        self.length_min = length_min
        self.length_max = length_max
        self.with_monomer = with_monomer
        self.transform = BinderProcess(
            interface_dict = None,
            with_epitope = with_epitope,
            with_bindingsite = with_bindingsite,
            with_scaffold = with_scaffold,
            random_masking = random_masking,
            mask_threshold = mask_threshold
        )

        ########################################################################
        # Data Process
        ########################################################################

        self.data_list = []

        for sample in self.data_list_all:
            ################################
            # sample:
            #     'interface':
            #         chain_id: range(),
            #         ...
            #     'chains': <list of chains (str)>
            #     'size_list': <list of sizes (int)>
            #     'feat': 
            #         chain_id:
            #             'aa': <array of char>
            #             'res_nb': <tensor of aa tokens>
            #             'pos_heavyatom': <heavy atom coordinates>
            #             'mask_heavyatom': <mask of valid heavychains>
            #         ...
            #     'cd20_chain': antigen chain
            #     'epitope':
            #         chain_id: <list of binding sites (int)>
            #         ...
            #     'ID': name
            ################################

            size_all = sum(sample['size_list'])

            ####################### Filtering ##################################

            ###### length filtering ######
            if size_all < self.length_min or size_all > self.length_max:
                continue
            ###### monomer filtering ######
            if not with_monomer and len(sample['chains']) == 1:
                continue

            ####################### feature process ############################
            ignore = False
            sample_processed = {
                'antigen': set(sample['cd20_chain']),
                'interface': sample['interface'],
                'epitope': sample['epitope'],
                'name': sample['ID'].split('/')[-1],
                'chains': sample['chains'],
                'size_list': sample['size_list'],
                'feat': dict(),
            }

            ###### chain-wise features ######
            for chain in sample['feat']:
                aa = []
                size = 0

                ### sequence
                for resi in sample['feat'][chain]['aa']:
                    if resi == 20 or resi == 'X':
                        break
                    elif resi in ressymb_set:
                        aa.append(aa_idx_dict[resi])
                    else:
                        aa.append(resi)
                    size += 1

                if size == 0:
                    ignore = True
                    break

                sample_processed['feat'][chain] = dict()
                sample_processed['feat'][chain]['aa'] = torch.tensor(aa)
                sample_processed['feat'][chain]['resi_nb'] = torch.arange(size)

                ### coordnates 
                for key in ['pos_heavyatom', 'mask_heavyatom']:
                    sample_processed['feat'][chain][key] = sample['feat'][chain][key][:size]

            ###### add the samples to the list ######
            if not ignore:
                self.data_list.append(sample_processed)


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        data = self.transform(self.data_list[index])
        data['idx'] = index
        data['name'] = self.data_list[index]['name']

        return data


################################################################################
# Test
################################################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SingleChainDataset(
        processed_dir=args.processed_dir,
        split=args.split, 
        reset=args.reset
    )
    print(dataset[0])
    print(len(dataset), len(dataset.clusters))
