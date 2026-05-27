import os
import argparse
import time
from tqdm.auto import tqdm

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

    parser.add_argument('--in_path', type=str, default='../../Results/Chroma/go_pred_foldseek-seq.txt')
    parser.add_argument('--out_path', type=str, default='../../Results/Chroma/go_foldseek-seq.pkl')
    parser.add_argument('--threshold', type=float, default=30.)

    args = parser.parse_args()

    output_dict = {}

    ###### load the input ######
   
    with open(args.in_path, 'r') as rf:
        for line in rf:
            # query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits 
            line = line.strip('\n').split('\t')
            name = line[0]
            target = line[1]
            e_value = float(line[-2])

            if e_value > args.threshold:
                continue

            if name not in output_dict:
                output_dict[name] = {'uniprot':[]}
            output_dict[name]['uniprot'].append((e_value, target))

    for name in output_dict:
        output_dict[name]['uniprot'] = sorted(output_dict[name]['uniprot'])

    print('Query and target loaded.')

    ###### GO mapping ######
    #for name in tqdm(output_dict):
    for name in output_dict:
        start = time.time()      
        # output_dict[name]['GO'] = set()
  
        # for target in output_dict[name]['uniprot']:
        #     if not target.startswith('AF'):
        #         continue
        #     uniprot_id = target.split('-')[1]
        #     go_set = GO_query(uniprot_id)
        #     output_dict[name]['GO'] = output_dict[name]['GO'].union(go_set)

        target = output_dict[name]['uniprot'][0][1] # select the top-1 uniprot
        e_value = output_dict[name]['uniprot'][0][0] # smallest e-value 
        if not (target.startswith('AF') or target.startswith('af')):
            continue

        if target.startswith('AF'):
            uniprot_id = target.split('-')[1]
        else:
            uniprot_id = target.split('_')[1]

        output_dict[name]['GO'] = GO_query(uniprot_id)

        print(
            name, 
            len(output_dict[name]['uniprot']), e_value,
            len(output_dict[name]['GO']), 
            '%.3fs cost' % (time.time() - start)
        )

    _ = dict_save(output_dict, args.out_path)




