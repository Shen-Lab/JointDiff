import os
import shutil
import argparse
from tqdm.auto import tqdm
from easydict import EasyDict
import torch

######################################################################################
# Arguments                                                                          #
######################################################################################

def arguments():

    parser = argparse.ArgumentParser()

    ##################### paths and name ######################################
    ### config (if not None, use the hyper-parameters in the config file)
    parser.add_argument('--config', type=str, default=None)
    ### data
    parser.add_argument('--train_summary', type=str,
        #default='../../Documents/Data/Processed/CATH_forDiffAb/cath_summary_all.tsv'
        default='../Data/examples/summary_debug.tsv'
    )
    parser.add_argument('--train_pdb', type=str,
        #default='../../Documents/Data/Origin/CATH/pdb_all/'
        default='../Data/examples/pdbs/'
    )
    parser.add_argument('--train_processed', type=str,
        #default='../../Documents/Data/Processed/CATH_forDiffAb/'
        default='../Data/examples/'
    )
    parser.add_argument('--val_summary', type=str,
        #default='../../Documents/Data/Processed/CATH_forDiffAb/cath_summary_all.tsv'
        default='../Data/examples/summary_debug.tsv'
    )
    parser.add_argument('--val_pdb', type=str,
        #default='../../Documents/Data/Origin/CATH/pdb_all/'
        default='../Data/examples/pdbs/'
    )
    parser.add_argument('--val_processed', type=str,
        #default='../../Documents/Data/Processed/CATH_forDiffAb/'
        #default=None
        default='../Data/examples/'
    )
    ### save dir
    parser.add_argument('--logdir', type=str,
        default='../Debug/checkpoints/'
    )
    parser.add_argument('--tag', type=str, default=None,
        help='tag of the saved files'
    )
    parser.add_argument('--args_name', type=str, default=None)
    ### warm start
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)

    ############################## model setting ###############################
    parser.add_argument('--train_version', type=str, default='jointdiff-x')
    parser.add_argument('--train_structure', type=int, default=1)
    parser.add_argument('--train_sequence', type=int, default=1)
    parser.add_argument('--encode_share', type=int, default=1)
    parser.add_argument('--modality', type=str, default='joint')
    parser.add_argument('--res_feat_dim', type=int, default=128)
    parser.add_argument('--pair_feat_dim', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--position_scale', type=float, nargs='*', default=[50.0])
    parser.add_argument('--seq_diff_version', type=str, default='multinomial')
    parser.add_argument('--remember_padding', type=int, default=0)
    parser.add_argument('--max_relpos', type=int, default=32)
    parser.add_argument('--all_bb_atom', type=int, default=0)

    ###################### other development setting ###########################
    ### centralization
    parser.add_argument('--centralize', type=int, default=1)
    ### random masking
    parser.add_argument('--random_mask', type=int, default=1)
    parser.add_argument('--motif_factor', type=float, default=0.0)
    parser.add_argument('--sele_ratio', type=float, default=0.7)
    parser.add_argument('--min_mask_ratio', type=float, default=0.2)
    parser.add_argument('--max_mask_ratio', type=float, default=1.)
    parser.add_argument('--consecutive_prob', type=float, default=0.5)
    parser.add_argument('--separate_mask', type=int, default=0)

    ###################### optimizer & scheduler ################################
    ### optimizer
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--lr', type=float, default= 1.e-4)
    parser.add_argument('--weight_decay', type=float, default= 0.0)
    parser.add_argument('--beta1', type=float, default= 0.9)
    parser.add_argument('--beta2', type=float, default= 0.999)
    ### scheduler
    parser.add_argument('--scheduler_type', type=str, default='plateau')
    parser.add_argument('--scheduler_factor', type=float, default=0.8)
    parser.add_argument('--scheduler_patience', type=float, default=10.)
    parser.add_argument('--min_lr', type=float, default=5.e-6) 
    ### grad norm   
    parser.add_argument('--max_grad_norm', type=int, default=100.0)
 
    ########################### for losses #####################################
    ###### basic loss ######
    parser.add_argument('--micro', type=int, default=1)
    parser.add_argument('--unnorm_first', type=int, default=1)
    parser.add_argument('--posi_loss_version', type=str, default='mse-align')
    ###### distance loss ######
    parser.add_argument('--with_dist_loss', type=int, default=1)
    parser.add_argument('--dist_loss_version', type=str, default='mse')
    parser.add_argument('--threshold_dist', type=float, default=15.)
    parser.add_argument('--dist_clamp', type=float, default=20.)
    ###### distance loss ######
    parser.add_argument('--with_distogram', type=int, default=1)
    ###### clash loss ######
    parser.add_argument('--with_clash', type=int, default=1)
    parser.add_argument('--threshold_clash', type=float, default=3.6)
    ###### gap loss ######
    parser.add_argument('--with_gap', type=int, default=1)
    parser.add_argument('--threshold_gap', type=float, default=3.9)
    ###### loss weights
    parser.add_argument('--rot_weight', type=float, default=1.0)
    parser.add_argument('--pos_weight', type=float, default=1.0)
    parser.add_argument('--seq_weight', type=float, default=1.0)
    parser.add_argument('--dist_weight', type=float, default=1.0)
    parser.add_argument('--distogram_weight', type=float, default=1.0)
    parser.add_argument('--clash_weight', type=float, default=1.0)
    parser.add_argument('--gap_weight', type=float, default=1.0)

    ########################### training #######################################
    parser.add_argument('--debug', action='store_true', default=False)
    ### device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    ### dataloader
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    ### training & val
    parser.add_argument('--max_iters', type=int, default=1000000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)

    ########################### arguments summary ##############################
    args = parser.parse_args()

    ###### configuations ######
    if args.config is None or args.config.upper() == 'NONE':
        args.config = None
    if args.config is not None:
        args_dict, args_name = load_config(args.config)
        return args_dict, args_name

    ###### paths and lables ######
    if args.tag is None or args.tag.upper() == 'NONE':
        args.tag = ''
    if args.resume is not None and args.resume.upper() == 'NONE':
        args.resume = None
    if args.finetune is not None and args.finetune.upper() == 'NONE':
        args.finetune = None

    ###### device ######
    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if args.position_scale[0] == -1:
        args.position_scale == 'adapt'
        position_scale_token = args.position_scale
    else:
        position_scale_token = '-'.join([str(val) for val in args.position_scale])

    ###### arguments summarization ######
    args_dict = {
        ###### paths ######
        'path': {
            'logdir': args.logdir,
            'tag': args.tag,
            'resume': args.resume, 
            'finetune': args.finetune, 
        },
        ###### model ######
        'model': {
            'train_version': args.train_version,
            'train_structure': bool(args.train_structure),
            'train_sequence': bool(args.train_sequence),
            'encode_share': bool(args.encode_share),
            'res_feat_dim': args.res_feat_dim,
            'pair_feat_dim': args.pair_feat_dim,
            'max_relpos': args.max_relpos,
            'all_bb_atom': bool(args.all_bb_atom),
            'with_distogram': bool(args.with_distogram),
            'diffusion': {
                'modality': args.modality,
                'position_scale': args.position_scale,
                'seq_diff_version': args.seq_diff_version,
                'num_steps': args.num_steps,
                'eps_net_opt': {
                    'num_layers': args.num_layers,
                },
                'remember_padding': bool(args.remember_padding),
            }
         },
        ###### data ######
        'dataset': {
            'train': {
                'summary_path': args.train_summary,
                'pdb_dir': args.train_pdb,
                'processed_dir': args.train_processed,
                'split': 'train',
            },
        },
        'dataloader': {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
        },
        ###### training ######
        'train': {
            ### optimizer
            'optimizer': {
                'type': args.optimizer_type,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'beta1': args.beta1,
                'beta2': args.beta2,
            },
            'max_grad_norm': args.max_grad_norm,
            ### scheduler
            'scheduler': {
                'type': args.scheduler_type,
                'factor': args.scheduler_factor,
                'patience': args.scheduler_patience,
                'min_lr': args.min_lr,
            },
            ### loss
            'micro': bool(args.micro),
            'unnorm_first': bool(args.unnorm_first),
            'posi_loss_version': args.posi_loss_version,
            'with_dist_loss': bool(args.with_dist_loss),
            'dist_loss_version': args.dist_loss_version,
            'threshold_dist': args.threshold_dist,
            'dist_clamp': args.dist_clamp,
            'with_clash': bool(args.with_clash),
            'threshold_clash': args.threshold_clash,
            'with_gap': bool(args.with_gap),
            'threshold_gap': args.threshold_gap,
            'loss_weights': {
                'rot': args.rot_weight,
                'pos': args.pos_weight,
                'seq': args.seq_weight,
                'dist': args.dist_weight,
                'distogram': args.seq_weight,
                'clash': args.clash_weight,
                'gap': args.gap_weight,
            },
            ### data process
            'centralize': bool(args.centralize),
            'random_mask': bool(args.random_mask),
            'motif_factor': args.motif_factor,
            'sele_ratio': args.sele_ratio,
            'min_mask_ratio': args.min_mask_ratio,
            'max_mask_ratio': args.max_mask_ratio,
            'consecutive_prob': bool(args.consecutive_prob),
            'separate_mask': bool(args.separate_mask),
            ### training setting
            'max_iters': args.max_iters,
            'val_freq': args.val_freq,
            'device': args.device,
            'multi_gpu': args.multi_gpu,
            'seed': args.seed,
            'debug': args.debug,
        },
    }

    ### validation dataset
    if args.val_summary is not None and args.val_summary.upper() != 'NONE' \
    and args.val_pdb is not None and args.val_pdb.upper() != 'NONE' \
    and args.val_processed is not None and args.val_processed.upper() != 'NONE':
        args_dict['dataset']['val'] = {
            'summary_path': args.val_summary,
            'pdb_dir': args.val_pdb,
            'processed_dir': args.val_processed,
            'split': 'val',
        }
 
    ###### model name ######
    if args.args_name is None or args.args_name.upper() == 'NONE':
        args.args_name = '%s_%s_%s_model%d-%d-%d-step%d_posi-scale-%s' % (
            args.train_version, args.modality, args.seq_diff_version,
            args.num_layers, args.res_feat_dim, args.pair_feat_dim, args.num_steps,
            position_scale_token
            #'-'.join([str(val) for val in args.position_scale])
        )
        ### random masking
        if args.random_mask:
            args.args_name += '_rm-%s' % str(args.motif_factor)
        ### atom version
        if args.all_bb_atom:
            args.args_name += '_allbbatom'
        ### loss
        if args.micro:
            loss_tag = '_micro'
        else:
            loss_tag = '_macro'
        loss_tag += '-posi+%s' % args.posi_loss_version

        if args.with_dist_loss:
            loss_tag += '-dist+%s' % args.dist_loss_version
        if args.with_distogram:
            loss_tag += '-distogram'
        if args.with_clash:
            loss_tag += '-clash'
        if args.with_gap:
            loss_tag += '-gap'

        args.args_name += loss_tag

        ### other tag
        args.args_name += args.tag

    return EasyDict(args_dict), args.args_name


######################################################################################
# Main Function                                                                      #
######################################################################################

def main(config, config_name):

    if config.path.resume is not None:
        ckpt_dir = os.path.dirname(config.path.resume)
        ckpt_dir_list = [ckpt_dir]
    else:
        ckpt_dir_list_all = [
            d for d in os.listdir(config.path.logdir)
            if os.path.isdir(os.path.join(config.path.logdir, d))
        ]
        ckpt_dir_list = []
        for d in ckpt_dir_list_all:
            if d.startswith(config_name):
                ckpt_dir_list.append(
                    os.path.join(config.path.logdir, d, 'checkpoints')
                )

    if not ckpt_dir_list:
        print('No model for %s.' % config_name)

    else:

        for ckpt_dir in ckpt_dir_list:
            if not os.path.isdir(ckpt_dir):
                print(f"{ckpt_dir} does not exist!")
                continue 

            ckpt_list = [int(p.split('.')[0])
                for p in os.listdir(ckpt_dir) if p.endswith('.pt') and (not p.startswith('point'))
            ]
            if not ckpt_list:
                print(f"{ckpt_dir} is empty!")
                continue

            last_ckpt = max(ckpt_list)
            #print(os.path.join(ckpt_dir, f'{last_ckpt}.pt'))
            model = ckpt_dir.split('/checkpoints')[0].split('/')[-1]
            print('%s_%d' % (model, last_ckpt))
            

######################################################################################
# Running the Script                                                                 #
######################################################################################

if __name__ == '__main__':
    config, config_name = arguments()
    main(config, config_name)

