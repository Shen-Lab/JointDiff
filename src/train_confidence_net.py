import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jointdiff.modules.utils.train import *
from jointdiff.modules.utils.misc import *
from jointdiff.modules.data.data_utils import *
from jointdiff.dataset import ConfidenceDataset
from jointdiff.model import ConfidenceNet 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

######################################################################################
# Train                                                                              #
######################################################################################

def train(
    model, dataloader, optimizer, scheduler, 
    logger, writer, args, 
):

    time_start = current_milli_time()
    model.train()

    ############################################################################
    # Training
    ############################################################################

    for epoch in range(args.start_epoch, args.max_epoch + 1): 

        ########################### epoch-wise process #########################
        loss_epo = []

        for i, batch in enumerate(dataloader):

            batch = recursive_to(batch, args.device)
            batch['chain_nb'] = batch['mask']
            batch['pos_heavyatom'] = batch['pos_heavyatom'][:, :, :args.num_atoms, :] 
            batch['mask_heavyatom'] = batch['mask_heavyatom'][:, :, :args.num_atoms] 

            ###### centralize the data ######
            if args.centralize:
                mean = batch['pos_heavyatom'].sum(dim = (1, 2))   # (N, 3)
                mean = mean / batch['mask_heavyatom'].sum(dim = (1, 2)).unsqueeze(1)  # (N, 3)
                batch['pos_heavyatom'] -= mean.unsqueeze(1).unsqueeze(1)  # (N, L, 4, 3)
                batch['pos_heavyatom'][batch['mask_heavyatom'] == 0] = 0

            ############################################################################
            # Inference & Loss
            ############################################################################

            pred = model(batch)  # (B, 4) 

            if args.binary:
                loss = F.binary_cross_entropy(
                    pred, batch['label'].float(),
                    reduction = 'none'
                )  # (B, 4)
            else:
                loss = F.mse_loss(
                    pred, batch['label'].float(),
                    reduction = 'none'
                )  # (B, 4)

            if args.balance:
                loss = loss * batch['weight']  # (B, 4)
            loss = loss.mean()

            time_forward_end = current_milli_time()

            ############################################################################
            # Backpropogate 
            ############################################################################

            loss.backward()
            orig_grad_norm = clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()

            time_backward_end = current_milli_time()
            loss_epo.append(float(loss))

            ############################################################################
            # Record 
            ############################################################################

            logger.info('Iter%d: %.6f' % (i+1, float(loss)))

            ### initial point
            if epoch == 1 and i == 0: 
                torch.save({
                    'args': args,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': 0,
                    'batch': recursive_to(batch, 'cpu'),
                }, os.path.join(args.ckpt_dir, 'checkpoint_0.pt'))
            elif i % args.save_iter == 0:
                torch.save({
                    'args': args,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'iter': i,
                    'batch': recursive_to(batch, 'cpu'),
                }, os.path.join(args.ckpt_dir, 'checkpoint_%d-%d.pt' % (epoch, i)))


            if not torch.isfinite(loss):
                logger.error('NaN or Inf detected.')
                torch.save({
                    'args': args,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'batch': recursive_to(batch, 'cpu'),
                }, os.path.join(args.ckpt_dir, 'checkpoint_nan_%d.pt' % epoch))
                raise KeyboardInterrupt()

            torch.cuda.empty_cache()

        ############################################################################
        # epoch done & save the model 
        ############################################################################

        print('Epoch %d done, loss_ave=%.4f' % (epoch, np.mean(loss_epo)))

        if epoch % args.save_period == 0: 
             torch.save({
                 'args': args,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'epoch': epoch,
                 'batch': recursive_to(batch, 'cpu'),
             }, os.path.join(args.ckpt_dir, 'checkpoint_%d.pt' % epoch))


######################################################################################
# Main Function                                                                      #
######################################################################################

def main(args):
    #######################################################################
    # Configuration and Logger
    #######################################################################

    if args.resume:
        args.log_dir = os.path.dirname(os.path.dirname(args.resume))
    else:
        name = 'confidence_%d-%d-%d_posiscale%.4f' % (
            args.res_feat_dim,
            args.pair_feat_dim,
            args.num_layers,
            args.posiscale,
        )
        if args.binary:
            name = '%s_binary' % name
        elif args.label_norm:
            name = '%s_norm-regression' % name
        else:
            name = '%s_regression' % name

        if args.tag and (not args.tag.startswith('_')):
            args.tag = '_%s' % args.tag
        elif args.tag is None:
            args.tag = ''
        name = '%s_%s' % (name, args.tag)

        args.log_dir = os.path.join(args.logdir, name)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    args.ckpt_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    logger = get_logger('train', args.log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(args.log_dir)
    tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(
        args.log_dir
    )
    logger.info(args)

    #######################################################################
    # Model and Optimizer
    #######################################################################

    model = ConfidenceNet(
        res_feat_dim = args.res_feat_dim,
        pair_feat_dim = args.pair_feat_dim,
        num_layers = args.num_layers,
        num_atoms = args.num_atoms,
        max_relpos = args.max_relpos,
        binary = args.binary
    ).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    ################# Optimizer & scheduler #################
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr = args.lr,
        weight_decay=args.weight_decay,
        betas= (args.beta1, args.beta2, )
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.factor,
        patience=args.patience,
        min_lr=args.min_lr,
    )
    optimizer.zero_grad()

    ################# Resume #################
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device) #, weights_only = True)
        args.start_epoch = ckpt['epoch'] + 1
        ckpt['model'] = {''.join(key.split('module.')[:]) : ckpt['model'][key]
            for key in ckpt['model']
        }
        model.load_state_dict(ckpt['model'])

        if args.resume is not None:
            logger.info('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
    else:
        args.start_epoch = 1
        logger.info('Starting from scratch...')

    ################## Parallel  ##################
    if args.multi_gpu:
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)
        logger.info('%d GPUs detected. Applying parallel training and a batch size of %d.' % 
            (torch.cuda.device_count(), args.batch_size)
        )

    elif args.device == 'cuda':
        logger.info('Applying single GPU training with a batch size of %d.' % 
            (args.batch_size)
        )

    else:
        logger.info('Applying CPU training with a batch size of %d.' %
            (args.batch_size)
        )

    #######################################################################
    # Data Loading
    #######################################################################

    train_dataset = ConfidenceDataset(args, reset = args.data_reset)
    train_dataloader = DataLoader(
        train_dataset, batch_size = args.batch_size,
        collate_fn = PaddingCollate(), shuffle=True, 
        num_workers = args.num_workers
    )
    logger.info('%d training samples loaded.'  % (len(train_dataset)))

    #######################################################################
    # Training
    #######################################################################
    
    train(
        model, train_dataloader, optimizer, scheduler, 
        logger, writer, args, 
    )


######################################################################################
# Running the Script                                                                 #
######################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##################### paths and name ######################################
    parser.add_argument('--summary_path',
        type=str, default='../../Documents/Data/Distillation/Confidence/confidence_summary_all.tsv'
    )
    parser.add_argument('--pdb_dir', type=str,
        default='../../Documents/Data/Distillation/Confidence/pdbs/'
    )
    parser.add_argument('--data_list_path', type=str,
        default='../../Documents/Data/Distillation/Confidence/train_data_list.pkl'
    )
    parser.add_argument('--processed_dir', type=str,
        default='../../Documents/Data/Distillation/Confidence/'
    )
    parser.add_argument('--logdir', type=str,
        default='../../Documents/TrainedModels/JointDiff/logs_debug/'
    )
    parser.add_argument('--tag', type=str, default=None,
        help='tag of the saved files'
    )
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)

    ########################### setting #######################################
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=int, default=1)
    ###### data ######
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--centralize', type=int, default=1)
    parser.add_argument('--data_reset', type=int, default=0)
    ###### model ######
    parser.add_argument('--res_feat_dim', type=int, default=128)
    parser.add_argument('--pair_feat_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_atoms', type=int, default=4)
    parser.add_argument('--max_relpos', type=int, default=30)
    parser.add_argument('--posiscale', type=float, default=50.)
    ###### optimizer ######
    parser.add_argument('--lr', type=float, default=1.e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    ###### scheduler ######
    parser.add_argument('--factor', type=float, default=0.8)
    parser.add_argument('--patience', type=float, default=10.)
    parser.add_argument('--min_lr', type=float, default=5.e-6)
    ###### loss ######
    parser.add_argument('--binary', type=int, default=1)
    parser.add_argument('--consist_seq_thre', type=float, default=0.3)
    parser.add_argument('--consist_stru_thre', type=float, default=2.0)
    parser.add_argument('--foldability_thre', type=float, default=0.3)
    parser.add_argument('--designability_thre', type=float, default=2.0)
    parser.add_argument('--label_norm', type=int, default=0)
    parser.add_argument('--balance', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=100.)
    ###### training ######
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--save_iter', type=int, default=2000)

    args = parser.parse_args()

    ###### labels ######
    if args.tag is None or args.tag.upper() == 'NONE':
        args.tag = ''
    if args.resume is not None and args.resume.upper() == 'NONE':
        args.resume = None
    if args.finetune is not None and args.finetune.upper() == 'NONE':
        args.finetune = None

    ###### arguments setting ######
    args.multi_gpu = bool(args.multi_gpu)
    if torch.cuda.device_count() > 1 and args.device == 'cuda' and args.multi_gpu:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    args.centralize = bool(args.centralize)
    args.data_reset = bool(args.data_reset)
    args.binary = bool(args.binary)
    args.label_norm = bool(args.label_norm)
    args.balance = bool(args.balance)

    ###### running ######
    main(args)
