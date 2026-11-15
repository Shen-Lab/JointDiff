######################################################
# Transfer the torch entries into numpy
# by SZ; 9/25/2023
######################################################

import numpy as np
import torch
import argparse
from utils_eval import dict_load, dict_save
import pickle
import copy

def torch2numpy(input_entry):
    if torch.is_tensor(input_entry):
        return input_entry.detach().cpu().numpy()
    elif type(input_entry) == dict:
        out_dict = {}
        for key in input_entry.keys():
            out_dict[key] = torch2numpy(input_entry[key])
        return out_dict
    else:
        return copy.deepcopy(input_entry)


parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, default='../../Results/diffab/traj_energy.pkl')
parser.add_argument('--output_path', type=str, default='../../Results/diffab/traj_energy_numpy.pkl')

args = parser.parse_args()

in_dict = dict_load(args.input_path) 
out_dict = torch2numpy(in_dict)
_ = dict_save(out_dict, args.output_path)















