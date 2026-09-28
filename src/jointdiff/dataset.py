################################################
# dataloader for the single chain entries (by SZ on Apr. 19, 2023)
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

import numpy as np
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from jointdiff.modules.data import parsers, constants

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


### add by SZ
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


class SingleChainDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self, 
        summary_path = '../../Data/Processed/CATH_forDiffAb/cath_summary_all.tsv', 
        pdb_dir = '../../Data/Origin/CATH/pdb_all/', 
        processed_dir = '../../Data/Processed/CATH_forDiffAb/',
        split = 'train',  # data set
        random_split = False, # by SZ, whether split the data based on the sequence clusters
        val_ratio = 0.1,  # by SZ, ratio of the validation set 
        test_ratio = 0.1,  # by SZ, ratio of the test set
        split_seed = 2022, # shuffling seed
        transform = None,  # data transformation function
        reset = False,  # whether reprocess the data (e.g. lmdb process, clustering) if it already exists
    ):
        super().__init__()
        ### check the input paths
        self.summary_path = summary_path
        self.pdb_dir = pdb_dir
        if not os.path.exists(pdb_dir):
            raise FileNotFoundError(
                f"PDB structures not found in {pdb_dir}. "
                #"Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )

        ### check the output paths
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        ### prepare the single sample information
        self.SingleChain_entries = None
        self._load_SingleChain_entries()

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset) # Load the structure information

        self.random_split = random_split
        if random_split and split != 'all': # do the data clustering and spliting
            ### clustering
            self.clusters = None
            self.id_to_cluster = None
            self._load_clusters(reset)

            ### data spliting
            self.ids_in_split = None
            self.val_ratio = val_ratio  # by SZ
            self.test_ratio = test_ratio  # by SZ
            self._load_split(split, split_seed)

        else:  # load the preprocessed datasets
            self._load_dataset(split) 

        ### data transformation
        self.transform = transform


    def _load_SingleChain_entries(self):
        """
        Load the sample basic information in the *.tsv file.
        """
        df = pd.read_csv(self.summary_path, sep='\t')
        entries_all = []
        for i, row in tqdm(
            df.iterrows(), 
            dynamic_ncols=True, 
            desc='Loading entries',
            total=len(df),
        ):
            entry_id = "{pdbcode}_{chain}{region}".format(
                pdbcode = row['pdb'],
                chain = nan_to_empty_string(row['chain']),
                region = nan_to_empty_string(row['region']),
            )
            entry = {
                'id': entry_id,
                'pdbcode': row['pdb'],
                'chain': row['chain'],
                'region': row['region'],
            }

            ### Filtering (could add filter here)
            entries_all.append(entry)

        self.SingleChain_entries = entries_all


    def _load_structures(self, reset):
        """
        Load the structure information and do the filtering.
        """
        ### check whether the *.lmdb file exists or whether need to process again
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                ### remove the processed file for the new one
                os.unlink(self._structure_cache_path)
            ### Prepare the *.lmdb and the *.lmdb-ids files
            self._preprocess_structures()

        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)  # list of the sample ids
        self.SingleChain_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids,
                self.SingleChain_entries
            )
        )

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')
        
    def _preprocess_structures(self):
        """
        Prepare the *.lmdb and the *.lmdb-ids files
        """
        ### sample wise data loading and prepare the id, paths for sample-wise data process
        tasks = []
        for entry in self.SingleChain_entries:
            pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(entry['id']))
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            tasks.append({
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
            })

        ### sample-wise data process: load the sample-wise information (e.g. sequence, coordinates)
        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_SingleChain_structure)(task) 
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )

        ### prepare the *.lmdb files
        db_conn = lmdb.open(
            self._structure_cache_path,  # the imdb file
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

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)


    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')


    def _load_clusters(self, reset):
        """
        Load the sequence clustering information.
        """
        ### Do the sequence clustering if the cluster files cannot be found. 
        if not os.path.exists(self._cluster_path) or reset:
            self._create_clusters()

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster


    def _create_clusters(self):
        """
        Sequence clustering.
        """
        cdr_records = []
        for id in self.db_ids:
            structure = self.get_structure(id)
            if structure['chain'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
        fasta_path = os.path.join(self.processed_dir, 'sequences.fasta')
        SeqIO.write(cdr_records, fasta_path, 'fasta')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',
            '-c', '0.8',
            '--cov-mode', '1',
        ])
        subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)


    def _load_dataset(self, split):
        """
        Load the preprocessed (split) dataset.
        """
        assert split in ('train', 'val', 'test', 'all')

        if split == 'all':
            self.ids_in_split = [entry['id'] for entry in self.SingleChain_entries 
                                     if os.path.exists(os.path.join(self.pdb_dir, '{}.pdb'.format(entry['id'])))]
        else:
            if not os.path.exists(self.processed_dir + '%s_data_list.pkl'%split):
                print('The data id file %s cannot be found!'%(self.processed_dir + '%s_data_list.pkl'%split))
                quit()

            id_list = dict_load(self.processed_dir + '%s_data_list.pkl'%split)
            self.ids_in_split = [
                entry['id'] for entry in self.SingleChain_entries if entry['id'] in id_list
            ]

        print('%d samples loaded for the %s set.'%(len(self.ids_in_split), split))


    def _load_split(self, split, split_seed):
        """
        Data spliting based on the clustering results.
        """
        assert split in ('train', 'val', 'test')
        ids_train_val_test = [
            entry['id']
            for entry in self.SingleChain_entries
        ]
        random.Random(split_seed).shuffle(ids_train_val_test)
        if split == 'test':
            self.ids_in_split = ids_train_val_test[self.val_ratio : self.val_ratio + self.test_ratio]
        elif split == 'val':
            self.ids_in_split = ids_train_val_test[:self.val_ratio]
        else:
            self.ids_in_split = ids_train_val_test[self.val_ratio + self.test_ratio:]


    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
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
        return data


def get_SingleChain_dataset(cfg, transform):
    return SingleChainDataset(
        summary_path = cfg.summary_path,
        pdb_dir = cfg.chothia_dir,
        processed_dir = cfg.processed_dir,
        split = cfg.split,
        split_seed = cfg.get('split_seed', 2022),
        transform = transform,
    )


################################################################################
# dataset for confidence model
################################################################################

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


class ConfidenceDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self,
        args,
        transform = None,  # data transformation function
        reset = False,  # whether reprocess the data (e.g. lmdb process, clustering) if it already exists
    ):
        super().__init__()

        self.args = args

        ####################################################
        # Path check
        ####################################################

        ### check the input paths
        self.summary_path = args.summary_path
        self.pdb_dir = args.pdb_dir
        if not os.path.exists(self.pdb_dir):
            raise FileNotFoundError(
                f"PDB structures not found in {pdb_dir}. "
                #"Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )

        ###### check the output paths ######
        self.processed_dir = args.processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        ####################################################
        # data loading
        ####################################################

        ###### entry list ######
        self._load_SingleChain_entries()
        entry_dset = set(dict_load(args.data_list_path))
        self.ids_in_split = [
            entry['id'] for entry in self.SingleChain_entries
            if entry['id'] in entry_dset
        ]
        if args.debug:
            self.ids_in_split = self.ids_in_split[:10]

        ###### dataset ######
        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset) # Load the structure information

        ### data transformation
        self.transform = transform

        print('%d samples loaded.' % self.__len__())


    #######################################################
    # entry list
    #######################################################

    def _load_SingleChain_entries(self):
        """
        Load the sample basic information in the *.tsv file.
        """
        df = pd.read_csv(self.summary_path, sep='\t')
        entries_all = []

        ### for label normalization
        if (not self.args.binary) and self.args.label_norm:
            min_max_dict = {}
            for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:
                min_max_dict[metric] = [float('inf'), -float('inf')]

        ### for balance
        if self.args.balance:
            sample_size_dict = {}
            for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:
                sample_size_dict[metric] = [0, 0]

        ###### sample-wise process ######
        for i, row in tqdm(
            df.iterrows(), 
            dynamic_ncols=True, 
            desc='Loading entries',
            total=len(df),
        ):
            entry_id = "{pdbcode}_{chain}{region}".format(
                pdbcode = row['pdb'],
                chain = nan_to_empty_string(row['chain']),
                region = nan_to_empty_string(row['region']),
            )
            entry = {
                'id': entry_id,
                'pdbcode': row['pdb'],
                'chain': row['chain'],
                'region': row['region'],
            }
            if self.args.balance:
                entry['binary_label'] = {}

            ###### labels ######
            for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:

                ### binary labels
                if self.args.binary or self.args.balance:
                    if metric == 'consist-seq':
                        label = 1 if row[metric] >= self.args.consist_seq_thre else 0
                    elif metric == 'consist-stru':
                        label = 1 if row[metric] <= self.args.consist_stru_thre else 0
                    elif metric == 'foldability':
                        label = 1 if row[metric] >= self.args.foldability_thre else 0
                    elif metric == 'designability':
                        label = 1 if row[metric] <= self.args.designability_thre else 0

                    if self.args.balance:
                        sample_size_dict[metric][label] += 1
                        entry['binary_label'][metric] = label

                ### binary classification
                if self.args.binary:
                    val = label
                ### regression
                else:
                    val = row[metric]
                    if self.args.label_norm:
                        min_max_dict[metric][0] = min(
                            min_max_dict[metric][0], row[metric]
                        )
                        min_max_dict[metric][1] = max(
                            min_max_dict[metric][1], row[metric]
                        )
 
                entry[metric] = val

            ### Filtering (could add filter here)
            entries_all.append(entry)

        ###### label process ######
        if (not self.args.binary) and self.args.label_norm:
            self.min_max_dict = min_max_dict
            print('Metric range:')
            for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:
                print(metric, min_max_dict[metric])

            for entry in entries_all:
                for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:
                    min_val = min_max_dict[metric][0]
                    max_val = min_max_dict[metric][1]
                    val = entry[metric]
                    val_new = (2 * (val - max_val)) / (max_val - min_val) + 1
                    entry[metric] = val_new 

        self.label_dict = {}
        for entry in entries_all:
            self.label_dict[entry['id']] = []
            for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:
                self.label_dict[entry['id']].append(entry[metric])
            self.label_dict[entry['id']] = np.array(self.label_dict[entry['id']])

        ###### label process ######
        if self.args.balance:
            print('Binary distribution / weights:')
            weight_dict = {}
            for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:
                posi_size = sample_size_dict[metric][1]
                nega_size = sample_size_dict[metric][0]
                posi_weight = (posi_size + nega_size) / (2 * posi_size)
                nega_weight = (posi_size + nega_size) / (2 * nega_size)
                weight_dict[metric] = [nega_weight, posi_weight]

                print(metric, sample_size_dict[metric], weight_dict[metric])
                
            self.sample_size_dict = sample_size_dict

            for entry in entries_all:
                for metric in ['consist-seq', 'consist-stru', 'foldability', 'designability']:
                    label = entry['binary_label'][metric] 
                    weight = weight_dict[metric][label]
                    entry['%s-weight' % metric] = weight

        self.SingleChain_entries = entries_all

    #######################################################
    # load the data
    #######################################################

    def _load_structures(self, reset):
        """
        Load the structure information and do the filtering.
        """
        ### check whether the *.lmdb file exists or whether need to process again
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                ### remove the processed file for the new one
                os.unlink(self._structure_cache_path)
            ### Prepare the *.lmdb and the *.lmdb-ids files
            self._preprocess_structures()

        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)  # list of the sample ids
        self.SingleChain_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids,
                self.SingleChain_entries
            )
        )

    #######################################################
    # data process
    #######################################################

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')
        
    def _preprocess_structures(self):
        """
        Prepare the *.lmdb and the *.lmdb-ids files
        """
        ###### sample wise data loading and prepare the id, paths for sample-wise data process ######
        tasks = []
        for entry in self.SingleChain_entries:
            pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(entry['id']))
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            tasks.append({
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
            })

        ### sample-wise data process: load the sample-wise information (e.g. sequence, coordinates)
        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_confidence_structure)(task) 
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )

        ### prepare the *.lmdb files
        db_conn = lmdb.open(
            self._structure_cache_path,  # the imdb file
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

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)


    #######################################################
    # entry query
    #######################################################

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
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

    #######################################################
    # entry query
    #######################################################

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        if self.transform is not None:
            data = self.transform(data)
        data['idx'] = index
        data['label'] = self.label_dict[id]

        return data


################################################################################
# for multimer dataloader 
################################################################################

###### data processor ######
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
                    if chain in structure['interface']:
                        del structure['interface'][chain]
                    continue

                binder_size = len(structure['interface'][chain])
                #idx_min = min(structure['interface'][chain])
                idx_min = min(structure['interface'][chain]) - 1
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
                        if chain in structure['epitope']:
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
                    #fragment_map[idx] = 4  # epitope
                    fragment_map[idx-1] = 4  # epitope

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


###### ProteinMPNN dataloader ######
class ProteinMPNNDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self, 
        summary_path: str='../data/ProteinMPNN/mpnn_data_info.pkl', 
        pdb_dir: str='../data/ProteinMPNN/pdb_2021aug02/pdb/', 
        processed_dir: str='../data/ProteinMPNN/',
        interface_path: str='../data/ProteinMPNN/interface_dict_all.pt',
        dset: str='train', 
        transform = 'default',
        reset = False,
        reso_threshold = 3.0,
        length_min = 20,
        length_max = 800,
        with_monomer = True,
        load_interface = True,
        with_epitope = True,
        with_bindingsite = False,
        with_scaffold = False,
        random_masking = False, 
        mask_threshold = 80,
        dimer_only = False
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
        self.dimer_only = dimer_only

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

                if self.dimer_only and len(self.info_dict['all'][clus][pdb]['chains']) != 2:
                    continue

                ###### selected entry ######
                entry = {
                    'id': pdb,
                    'chains': self.info_dict['all'][clus][pdb]['chains'],
                    'size': self.info_dict['all'][clus][pdb]['size'],
                    'path': None if self.pdb_dir is None else os.path.join(
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
                'antigen': sample['cd20_chain'],
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
# Debug                                                                        #
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

