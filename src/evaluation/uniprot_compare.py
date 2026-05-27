import os
import argparse
import time
from tqdm.auto import tqdm
import numpy as np

from Bio import ExPASy
from Bio import SwissProt
from utils_eval import dict_save

###################################### utility function #####################################

def get_uniprot_entry(uniprot_id):
    handle = ExPASy.get_sprot_raw(uniprot_id)
    record = SwissProt.read(handle)
    return record

def GO_query(uniprot_id):
    out = set()
    try:
        record = get_uniprot_entry(uniprot_id)
        for reference in record.cross_references:
            if reference[0] == 'GO':
                out.add(reference[1])
    except Exception as e:
        print(uniprot_id, e)
    return out

# uniprot_id = "P12345"
# uniprot_id = "Q64663"
# record = get_uniprot_entry(uniprot_id)

# #Extract GO annotations
# for reference in record.cross_references:
#     if reference[0] == 'GO':
#         print(reference)

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_pred', type=str, default='../../Results/Chroma/go_pred_foldseek-seq.txt')
    parser.add_argument('--struc_pred', type=str, default='../../Results/Chroma/go_pred_foldseek-stru.txt')
    parser.add_argument('--out_path', type=str, default='../../Results/Chroma/uniprot_consist.txt')
    parser.add_argument('--threshold', type=float, default=1000.)

    args = parser.parse_args()

    ########################## load the input #################################

    ###### sequence ######
    seq_uniprot = {}
   
    with open(args.seq_pred, 'r') as rf:
        for line in rf:
            # query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits 
            line = line.strip('\n').split('\t')
            name = line[0]
            target = line[1]
            e_value = float(line[-2])

            if e_value > args.threshold:
                continue

            if name not in seq_uniprot:
                seq_uniprot[name] = set()
            seq_uniprot[name].add(target)

    print('Query and target loaded for sequence.')

    ###### structure ######
    struc_uniprot = {}

    with open(args.struc_pred, 'r') as rf:
        for line in rf:
            # query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits 
            line = line.strip('\n').split('\t')
            name = line[0]
            target = line[1]
            e_value = float(line[-2])

            if e_value > args.threshold:
                continue

            if name not in struc_uniprot:
                struc_uniprot[name] = set()
            struc_uniprot[name].add(target)

    print('Query and target loaded for structure.')


    name_set = set(seq_uniprot.keys()).intersection(set(struc_uniprot.keys()))
    print('%d samples in all.' % len(name_set))
    if not name_set:
        quit()

    ########################## comparism #################################

    with open(args.out_path, 'w') as wf:

        wf.write('Query\tseq_target\tstru_target\tinter\tunion\tscore\n')
        score_list = []

        for name in tqdm(name_set):
            seq_set = seq_uniprot[name]
            struc_set = struc_uniprot[name]
            inter_set = seq_set.intersection(struc_set)
            union_set = seq_set.union(struc_set)
 
            inter_size = len(inter_set)
            union_size = len(union_set)             
            if union_size > 0:
                score = inter_size / union_size
                info = '%s\t%d\t%d\t%d\t%d\t%f\n' % (
                    name, len(seq_set), len(struc_set), inter_size, union_size, score
                )
                score_list.append(score)
            else:
                info = '%s\t%d\t%d\t%d\t%d\tnan\n' % (
                    name, len(seq_set), len(struc_set), inter_size, union_size, 
                )
            wf.write(info)

        if score_list:
            print('ave-score=%f for %d samples.' % (np.mean(score_list), len(score_list)))
            wf.write('# ave-score=%f for %d samples.\n' % (np.mean(score_list), len(score_list)))



