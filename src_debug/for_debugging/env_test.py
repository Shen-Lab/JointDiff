import os
import shutil
import argparse
from tqdm.auto import tqdm
from easydict import EasyDict

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn  
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.multiprocessing

import sys
sys.path.append("../")
from jointdiff.model import DiffusionSingleChainDesign
from jointdiff.modules.utils.train import get_optimizer, get_scheduler
from jointdiff.modules.utils.misc import (
    BlackHole, seed_all, 
    get_new_log_dir, get_logger, inf_iterator,
)
from jointdiff.modules.data.data_utils import PaddingCollate
from jointdiff.trainer import (
    train, validate, load_config, get_dataset, count_parameters
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


