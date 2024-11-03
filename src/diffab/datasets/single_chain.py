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
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..utils.protein import parsers, constants
from ._base import register_dataset

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
        # if not os.path.exists(pdb_dir):
        #     raise FileNotFoundError(
        #         f"PDB structures not found in {pdb_dir}. "
        #         #"Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
        #     )

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


@register_dataset('single_chain')
def get_SingleChain_dataset(cfg, transform):
    return SingleChainDataset(
        summary_path = cfg.summary_path,
        pdb_dir = cfg.chothia_dir,
        processed_dir = cfg.processed_dir,
        split = cfg.split,
        split_seed = cfg.get('split_seed', 2022),
        transform = transform,
    )


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
