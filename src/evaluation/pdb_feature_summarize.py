#####################################################
# statistic over the features
# * dihedral torsion angles (phi, psi and omega)
# * clash
#####################################################

import os
import numpy as np
import argparse

from utils_eval import dict_load, dict_save
from utils_eval import info_collect, KL_divergency, JS_divergency

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_path', type=str, default='../../Results/Chroma/Features/')
    parser.add_argument('--out_path', type=str, default='../../Results/Chroma/features_summary.pkl')
    parser.add_argument('--nature_path', type=str, default='../../Results/Nature/features_summary.pkl')
    parser.add_argument('--token', type=str, default='none')
    parser.add_argument('--reset', type=int, default=1)

    args = parser.parse_args()

    if args.token is None or args.token.lower() == 'none':
        args.token = ''

    args.reset = bool(args.reset)

    ##################### statistics on the features ##########################

    if os.path.exists(args.out_path) and (not args.reset):
        out_dict = dict_load(args.out_path)

    else:
        out_dict = info_collect(args.feat_path, token = args.token)
        _ = dict_save(out_dict, args.out_path)

    print('%d angles tuples in all.' % len(out_dict['phi']))
    print('Clash: mean=%.4f, min=%d, median=%d, max=%d, std=%.4f' % (
        np.mean(out_dict['clash']),
        min(out_dict['clash']),
        np.median(out_dict['clash']),
        max(out_dict['clash']),
        np.std(out_dict['clash'])
    ))
    print('Size: mean=%.4f, min=%d, median=%d, max=%d, std=%.4f' % (
        np.mean(out_dict['size']),
        min(out_dict['size']),
        np.median(out_dict['size']),
        max(out_dict['size']),
        np.std(out_dict['size'])
    ))

    ##################### compare with nature #################################

    if (args.nature_path is not None) and (args.nature_path.upper() != 'NONE'):

        nature_dict = dict_load(args.nature_path)
        bins = [i * 0.05 for i in range(-64, 65)]

        for key in ['phi', 'psi', 'omega']:
            kld = KL_divergency(out_dict[key], nature_dict[key], bins = bins)
            jsd = JS_divergency(out_dict[key], nature_dict[key], bins = bins)

            print('%s: JSD=%.4f; KLD=%.4f' % (key, jsd, kld))

 
