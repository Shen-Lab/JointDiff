######################################################################
# Calulate the consistency of the designed strcuture and sequences
# by SZ; 6/6/2023
######################################################################

import os
import numpy as np
import pickle
import argparse

from utils_eval import dict_save, TM_score, TM_score_asym
### for frozen cases
import signal

####################################### auxiliary function #######################################

def consistency_stat(tms_dict, name = 'TM-score', stat = np.mean):
    val_micro = []
    val_macro = []

    for sample in tms_dict:
        ### protein-wise
        val_sample = []

        for pred in tms_dict[sample]:
            ### prediction-wise
            if name == 'TM-score':
                val = tms_dict[sample][pred][0]
            else:
                val = tms_dict[sample][pred][1]

            if val is not None: 
                val_sample.append(val)
                val_micro.append(val)
                                
        if val_sample:
            val_macro.append(stat(val_sample))
            
    print(name)
    print('%s: macro=%4f; micro=%4f' %(
        name, np.mean(val_macro), np.mean(val_micro)
    ))


### based on https://stackoverflow.com/questions/25027122/break-the-function-after-certain-time
class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default='../../Results/Chroma/samples/')
    parser.add_argument('--pred_path', type=str, default='../../Results/Chroma/struc_pred_esmfold/')
    parser.add_argument('--out_path', type=str, default='../../Results/Chroma/consistency_TMscore_dict.pkl')

    args = parser.parse_args()

    ###### sample collect ######
    sample_set = {p[:-4] for p in os.listdir(args.ref_path) if p.endswith('.pdb')}
    pred_list = [p for p in os.listdir(args.pred_path) if p.endswith('.pdb')]
    sample_dict = {}
    pred_num = 0 

    for p in pred_list:
        p_name = p[:-4]
        flag = True
        
        if p_name in sample_set:
            name = p_name
            flag = False
        else:
            p_name = p_name.split('_')
            l = len(p_name)
            for i in range(l-1, 0, -1):
                name = '_'.join(p_name[:i]) 
                if name in sample_set:
                    flag = False
                    break

        if flag:
             print('Reference not found for prediction %s!' % p)
             continue

        if not name in sample_dict:
            sample_dict[name] = []
        sample_dict[name].append(p)
        pred_num += 1
                
    print('%d predictions covering %d samples out of %d.' % (
        pred_num, len(sample_dict), len(sample_set)
    ))

    ###### TMscore cal ######

    out_dict = {}
    TMs_all = []
    RMSD_all = []

    for sample in sample_dict:
        ref_pdb = os.path.join(args.ref_path, '%s.pdb' % sample)
        out_dict[sample] = {}

        for pred in sample_dict[sample]:
            pred_pdb = os.path.join(args.pred_path, pred)
            pred_name = pred[:-4]
            try:
                tms, rmsd = TM_score(pred_pdb, ref_pdb, with_RMSD = True) 
                if tms is not None and rmsd is not None:
                    out_dict[sample][pred_name] = (tms, rmsd)
                    
                print(pred_name, tms, rmsd)

            except TimeoutException:
                print('Out of time for %s and %s!' % (gt_name, sample_idx))
                continue # continue the for loop if TMalign takes more than 5 second
            else:
                # Reset the alarm
                signal.alarm(0)

            # except Exception as e:
            #     print('Failed for %s!' % pred_name, e)

    _ = dict_save(out_dict, args.out_path)

    print('Joint-consistancy:')
    consistency_stat(out_dict, name = 'TM-score', stat = np.mean)
    consistency_stat(out_dict, name = 'RMSD', stat = np.mean)
