### for energy score ###
import sbmOpenMM
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
### for fitness ###
import esm

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# for parallel computing
from threading import Thread
import time

###################################################################################
# Constants
###################################################################################

try:
    from diffab.utils.protein.constants import ressymb_order
    # diffab restypes: 'ACDEFGHIKLMNPQRSTVWYX'
    print('Apply ressymb_order loaded from diffab.')
except:
    ressymb_order = 'ACDEFGHIKLMNPQRSTVWYX'
    print('Directly set diffab ressymb_order ("ACDEFGHIKLMNPQRSTVWYX").')

ESM_RESTYPES = ['<cls>', '<pad>', '<eos>', '<unk>',
                'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

ESMFOLD_RESTYPES = 'ARNDCQEGHILKMFPSTWYVX'

RESIDUE_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU',
                'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
                'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'B': 'ASX',
                'Z': 'GLX', 'X': 'UNK'}

ELEMENT_dict = {'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O'}

###################################################################################
# Auxiliary Functions
###################################################################################


def empty_file(path):
    with open(path, 'r') as rf:
        lines = rf.readlines()
    return len(lines) == 0


def remove(string, char):
    """Remove the character in a string."""
    #string_char = [i for i in string.split(char) if i != '']
    # return string_char[0]
    string_char = string.split(char)
    return ''.join(string_char)


def pdb_write(coor, path, seq=None, chain='A',
              atom_list=['N', 'CA', 'C', 'O']):
    """
    Args: 
        coor: (L, atom_num, 3)
        seq: str of length L
    """
    if seq is not None and coor.shape[0] != len(seq):
        raise Exception('Error! The size of the strutctue and the sequence do not match! (%d and %d)' % (
            coor.shape[0], len(seq)))
    elif coor.shape[1] != len(atom_list):
        raise Exception('Error! The size of the resi-wise coor and the atom_num do not match! (%d and %d)' %
                        (coor.shape[1], len(atom_list)))
    else:
        with open(path, 'w') as wf:
            a_idx = 1
            for i, resi in enumerate(seq):
                # residue-wise info
                r_idx = i + 1
                aa = RESIDUE_dict[resi]

                for j, vec in enumerate(coor[i]):
                    # atom-wise info
                    atom = atom_list[j]
                    element = ELEMENT_dict[atom]

                    line = 'ATOM  '
                    line += '{:>5} '.format(a_idx)
                    line += '{:<4} '.format(atom)
                    line += aa + ' '
                    line += chain
                    line += '{:>4}    '.format(r_idx)  # residue index
                    line += '{:>8}'.format('%.3f' % vec[0])  # x
                    line += '{:>8}'.format('%.3f' % vec[1])  # y
                    line += '{:>8}'.format('%.3f' % vec[2])  # z
                    line += '{:>6}'.format('1.00')  # occupancy
                    line += '{:>6}'.format('0.00')  # temperature
                    line += ' ' * 10
                    line += '{:>2}'.format(element)  # element

                    wf.write(line + '\n')
                    a_idx += 1


def sequence_transform(restypes_in=ressymb_order,
                       restypes_tar=ESM_RESTYPES):
    """Prepare the matrix to transform the sequence of restypes_in to
    restypes_tar.

    Args:
        restypes_in: default='ACDEFGHIKLMNPQRSTVWYX'
        restypes_tar: default=['<cls>', '<pad>', '<eos>', '<unk>',
                               'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                               'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                               'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
    Output:
        trans_mat: (len_in, len_tar)

    """

    len_in = len(restypes_in)
    len_tar = len(restypes_tar)
    trans_mat = torch.zeros(len_in, len_tar)
    trans_mat[-1][-1] = 1

    for in_idx, token in enumerate(restypes_in):
        tar_idx = restypes_tar.index(token)
        trans_mat[in_idx][tar_idx] = 1

    return trans_mat


class ThreadWithReturnValue(Thread):
    """
    Multi-thread computing.
    """

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return

###################################################################################
# For fitness calculation (with esm-1b)
###################################################################################


class FitnessGrad(nn.Module):
    def __init__(self, version='ESM-1b', input_voc_set=ressymb_order):
        super(FitnessGrad, self).__init__()

        self.version = version
        if version == 'ESM-1b':
            self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        elif version == 'ESM-2':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            raise Exception('The version %s is unsupported.' % (version))
        # for gradients calculation (by SZ)
        emb_W = self.model.embed_tokens.weight
        # self.model.emb_inv = nn.Parameter(torch.matmul(torch.inverse(torch.matmul(emb_W.T, emb_W)), emb_W.T),
        #                            requires_grad=False)
        # rather than psedo-inverse, directly apply the transposed mat
        self.model.emb_inv = nn.Parameter(emb_W.T, requires_grad=False)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()

        # for sequence transformation
        self.input_voc_set = input_voc_set
        self.input_voc_size = len(input_voc_set)
        self.esm_voc_set = self.batch_converter.alphabet.all_toks  # list (33,)
        self.esm_voc_size = len(self.esm_voc_set)

        self.seq_trans_mat = sequence_transform(restypes_in=self.input_voc_set,
                                                restypes_tar=self.esm_voc_set)  # (input_voc_size, esm_voc_size)
        self.seq_trans_mat = nn.Parameter(
            self.seq_trans_mat, requires_grad=False)
        # (esm_voc_size, input_voc_size)
        self.grad_sele_mat = nn.Parameter(
            self.seq_trans_mat.T, requires_grad=False)

    def forward(self, seq, mask=None, with_grad=True, seq_transform=True, with_padding=False):
        """
        Args:
            seq: torch.tensor (B, L_max) or list of sequences
            mask: (B, L_max) or None, 1 for valid tokens and 0 for others
        Output:
            fitness: (B,)
            grad: (B, L_max)
        """
        if type(seq) == list:  # strings to tensors
            # transform sequences into sequence tuples (ESM format: [("name","seq"),...])
            seq = [('protein_%d' % i, s) for i, s in enumerate(seq)]
            # transform sequence tuples into tensors
            batch_labels, batch_strs, seq = self.batch_converter(seq)
            # seq: (B, L_max + 1)
        elif seq_transform:
            seq = self.seq_trans_mat[seq].max(dim=-1).indices  # (B, L_max)
            if mask is not None:
                seq[mask == 0] = 1  # <pad>
            # add <cls>
            seq = F.pad(seq, (1, 0), 'constant', 0)  # (B, L_max + 1)

        seq = seq.to(self.grad_sele_mat.device)
        #print(seq.device, self.grad_sele_mat.device)
        fitness, grad = self.model.fitness_cal(
            seq, with_grad=with_grad, with_padding=with_padding, grad_sele=self.grad_sele_mat)
        # fitness: (B,)
        # grad: (B, L_max, 21)
        if not with_padding and with_grad:
            grad = grad[:, :, :-1]  # (B, L_max, 20)
        return fitness, grad

###################################################################################
# For energy and Force calculation (with sbm-openmm)
###################################################################################


def CA_contact_gen(pdb_path, out_path, threshold=12):
    """Generate the CA contact file given a pdb_file."""
    ###### load the coordinates information ######
    resi_num = 0

    with open(pdb_path, 'r') as p_file:
        lines = p_file.readlines()

        coor_dict = {}
        chain_pre = None
        chain_idx = 0

        for line in lines:
            if line[0:4] == 'ATOM' and ''.join(line[12:16].split(' ')) == 'CA':
                resi_num += 1
                ### atom-wise info ###
                x = float(remove(line[30:38], ' '))
                y = float(remove(line[38:46], ' '))
                z = float(remove(line[46:54], ' '))
                ### chain info ###
                chain = line[21]
                if chain_pre is None or chain != chain_pre:
                    chain_idx += 1
                    coor_dict[chain_idx] = np.array([[x, y, z]])
                    chain_pre = chain
                else:
                    coor_dict[chain_idx] = np.vstack(
                        [coor_dict[chain_idx], np.array([[x, y, z]])])

    # ordered list of chains
    chain_list = sorted(list(coor_dict.keys()))

    ###### start_idx of each chain ######
    chain_idx_dict = {}
    chain_start_idx = 0
    for chain_idx in chain_list:
        chain_idx_dict[chain_idx] = chain_start_idx
        chain_start_idx += coor_dict[chain_idx].shape[0]

    ###### contact cal ######
    dist_mat = np.zeros([resi_num, resi_num])
    contact_list = []

    # chain-wise
    for chain_idx_1 in chain_list:
        start_idx_1 = chain_idx_dict[chain_idx_1]

        for chain_idx_2 in chain_list:
            start_idx_2 = chain_idx_dict[chain_idx_2]

            # residue-wise
            for resi_1, coor_1 in enumerate(coor_dict[chain_idx_1]):
                idx_1 = start_idx_1 + resi_1  # absolute idx of the residue 1 on the whole protein

                for resi_2, coor_2 in enumerate(coor_dict[chain_idx_2]):
                    idx_2 = start_idx_2 + resi_2  # absolute idx of the residue 2 on the whole protein

                    # distance
                    dist = np.linalg.norm(coor_1 - coor_2)
                    dist_mat[idx_1][idx_2], dist_mat[idx_2][idx_1] = dist, dist

                    # contact check
                    if (chain_idx_1 < chain_idx_2 or resi_1 < resi_2) and dist <= threshold:
                        contact_list.append(
                            (chain_idx_1, resi_1 + 1, chain_idx_2, resi_2 + 1))

    # contact map
    contact_map = (dist_mat <= threshold) * 1

    ###### Save the contact infomation. ######
    with open(out_path, 'w') as wf:
        for ctat in contact_list:
            wf.write('%d\t%d\t%d\t%d\n' % ctat)

    return dist_mat, contact_map


def contact_gen_torch(pdb_path, out_path, threshold=12, atom_sele='CA'):
    """Generate the contact file for arbitrary atom given a pdb_file with
    PyTorch."""
    ###### load the coordinates information ######
    resi_num = 0

    with open(pdb_path, 'r') as p_file:
        lines = p_file.readlines()

        coor_dict = {}
        chain_pre = None
        chain_idx_dict = {}
        chain_idx = 0
        resi_pre = None
        resi_order = {}

        for line in lines:
            if line.startswith('ATOM'):
                ### chain info ###
                chain = line[21]
                chain_change = chain != chain_pre  # True for start a new chain
                if chain_change:  # start a new chain
                    if not chain in chain_idx_dict:
                        chain_idx += 1
                        chain_idx_dict[chain] = chain_idx
                        coor_dict[chain] = []
                        resi_order[chain] = -1
                    chain_pre = chain

                ### residue info ###
                resi_idx = line[22:27]
                resi_type = line[17:20]
                if resi_idx != resi_pre or chain_change:  # start a new residue
                    coor_dict[chain].append(torch.nan * torch.ones(3))
                    resi_order[chain] += 1
                    resi_pre = resi_idx

                ### atom info ###
                atom_name = ''.join(line[12:16].split(' '))

                if atom_name == atom_sele or (atom_sele == 'CB' and atom_name == 'CA' and resi_type == 'GLY'):
                    # coordinates
                    x = float(remove(line[30:38], ' '))
                    y = float(remove(line[38:46], ' '))
                    z = float(remove(line[46:54], ' '))
                    coor = torch.tensor([x, y, z])
                    # update
                    coor_dict[chain][resi_order[chain]] = coor

    # arange the chain_index and coordinates
    chain_start_dict = {}
    start_idx = 0
    coor_all = None

    for i, chain in enumerate(coor_dict.keys()):
        chain_start_dict[start_idx] = chain
        start_idx += len(coor_dict[chain])

        if i == 0:
            coor_all = torch.stack(coor_dict[chain])
        else:
            coor_all = torch.cat([coor_all, coor_dict[chain]], dim=0)

    ###### contact cal ######
    dist_mat = torch.cdist(coor_all, coor_all)
    contact_map = (dist_mat <= threshold).int()
    contact_map = contact_map - torch.eye(contact_map.shape[-1]).int()

    ###### contact list ######
    contact_list = []
    # second residue
    for i in range(start_idx):
        if i in chain_start_dict.keys():  # update the chain index
            chain_idx_2 = chain_idx_dict[chain_start_dict[i]]
            resi_idx_2 = 0
        else:
            resi_idx_2 += 1

        # first residue
        for j in range(i):
            if j in chain_start_dict.keys():  # update the chain index
                chain_idx_1 = chain_idx_dict[chain_start_dict[j]]
                resi_idx_1 = 1
            else:
                resi_idx_1 += 1

            if contact_map[i, j] == 1:
                contact_list.append(
                    (chain_idx_1, resi_idx_1, chain_idx_2, resi_idx_2))

    ###### Save the contact infomation. ######
    with open(out_path, 'w') as wf:
        for ctat in contact_list:
            wf.write('%d\t%d\t%d\t%d\n' % ctat)

    return dist_mat, contact_map


def quantity_to_tensor(quantity):
    """
    Args:
        List of sbmopenmm quantities.
    Output: 
        pytorch tensor
    """
    return torch.stack([torch.tensor(q._value) for q in quantity])


def sbm_getCAModel(
        structure_file, 
        contact_file = None,
        default_parameters=True,
        default_forces=True,
        create_system=True,
        contact_force=None,
        torsion_energy=1.0,
        contact_energy=1.0,
        minimize=False,
        residue_masses=False,
        residue_radii=False,
        forcefield_file=None,
        check_large_forces=False,
        RepulsionOnly=True,
    ):
    """Set up the openmm object with sbmopenmm."""

    sbm = sbmOpenMM.system(structure_file)
    sbm.getCAlphaOnly()
    sbm.getAtoms()
    # add forces
    if forcefield_file is None:
        if residue_masses:
            sbm.setCAMassPerResidueType()
        sbm.getBonds()
        sbm.getAngles()
        sbm.getProperTorsions()

        if contact_file is not None:
            sbm.readContactFile(contact_file)

    elif forcefield_file is not None:
        print('Forcefield file given. Bonds, angles, torsions and native contacts definitions will be read from it!')

    # Add default parameters to each interaction term
    if default_parameters and forcefield_file is None:
        sbm.setBondParameters(20000.0)
        sbm.setAngleParameters(40.0)
        sbm.setProperTorsionParameters(torsion_energy)
        sbm.setNativeContactParameters(contact_energy)
        if residue_radii:
            sbm.setCARadiusPerResidueType()
        else:
            sbm.setParticlesRadii(0.4)
        sbm.rf_epsilon = 0.1

    elif forcefield_file is not None:
        sbm.rf_epsilon = 0.1
        sbm.loadForcefieldFromFile(forcefield_file)

    # Create default system force objects
    if default_parameters and default_forces:

        if not RepulsionOnly:
            sbm.addHarmonicBondForces()
            sbm.addHarmonicAngleForces()
            sbm.addPeriodicTorsionForces()

            if contact_force == '12-10':
                sbm.addLJ12_10ContactForces()
            elif contact_force == '12-10-6':
                sbm.addLJ12_10_6ContactForces()
            elif (contact_file is not None) and (contact_force is not None):
                raise ValueError(
                    'Wrong contact_force option, valid options are "12-10" and "12-10-6"')

        sbm.addLJRepulsionForces(cutoff=1.5)

    # Generate the system object and add previously generated forces
    if default_parameters and default_forces and create_system:
        sbm.createSystemObject(
            minimize=minimize, check_bond_distances=False, check_large_forces=check_large_forces)

    return sbm


def force_and_energy(
        pdb_path=None, 
        with_contact=True,
        contact_path=None, 
        temp_dir='./',
        contact_thre=12, 
        get_force=True, 
        get_energy=True,
        name_tag=None, 
        sum_result=False, 
        force_pad=None, 
        device='cuda',
        with_pdb_write=False, 
        coor=None, 
        seq=None, 
        mask=None, 
        atom_list=['CA'], 
        pdb_remove=True,
        RepulsionOnly=True,
        with_resi=False,
    ):
    """
    Args:
        pdb_path: str, path of the pdb file; if with_pdb_write, then will reassign this term for coor 
        temp_dir: str, directory to save the temperary file
        contact_thre: threshold for the contact map
        sum_result: whether summarize all the terms
        force_pad: int or None; the output length to be padded

        <for pdb writting; when with_pdb_write is True> 
        with_pdb_write: bool; whether apply the coordinates coor and transform it into pdb files
        coor: torch.Tensor or None; (L, atom_num, N), coordinates
        seq: str, torch.Tensor or None; sequence
        mask: torch.Tensor or None; 1 for valid tokens and 0 for invalid ones
        atom_list: list or None, atoms to be considered in the pdb file; default = ['CA']
        pdb_remove: bool; whether remove the temperary pdb file

    Output: 
        force_dict (key: (L, 3)): 
            if sum_result
                all
            else
                Harmonic Bond Force
                Harmonic Angle Force
                Periodic Torsion Force
                LJ 12-10 Contact Force
                LJ 12 Repulsion Force 
        energy_dict (key: float):
            if sum_result
                all
            else
                Harmonic Bond Energy 
                Harmonic Angle Energy 
                Periodic Torsion Energy 
                LJ 12-10 Contact Energy: energy = epsilon*(5*(sigma/r)^12-6*(sigma/r)^10)
                LJ 12 Repulsion Energy
        Potential Energy = Harmonic Bond Energy + Harmonic Angle Energy + Periodic Torsion Energy + LJ 12-10 Contact Energy + LJ 12 Repulsion Energy
    """
    if not (get_force or get_energy):
        print('Warning! Neither force nor energy calculation is required!')
        return None, None

    ############### prepare pdb file and the contact file #####################
    if name_tag is not None:
        name_tag = '_' + name_tag
    else:
        name_tag = ''

    ###### for PDB ######
    if with_pdb_write:
        if coor is None:
            print('No coordinate is provided!')
            return None, None

        ### sequence prepare
        if mask is not None:
            coor = coor[mask == 1]
            length = coor.shape[0]
            seq = seq[:length]
            if torch.is_tensor(seq):
                seq = [ressymb_order[res_idx] for res_idx in seq]
        elif torch.is_tensor(seq):
            aa = ''
            for res_idx in seq:
                if res_idx >= 20:
                    break
                aa += ressymb_order[res_idx]
            seq = aa
            length = len(seq)
            coor = coor[:length]

        ### pdb path
        path_idx = 1
        pdb_path = os.path.join(
            temp_dir,
            'temp%s_%d.pdb' % (name_tag, path_idx)
        )  
        while os.path.exists(pdb_path): # to avoid overwrite existing files
            path_idx += 1
            pdb_path = os.path.join(
                temp_dir,
                'temp%s_%d.pdb' % (name_tag, path_idx)
            )
        ### pdb write
        pdb_write(coor=coor, path=pdb_path, seq=seq, atom_list=atom_list)
        print('PDB saved at %s.' % pdb_path)
        pdb_remove_flag = pdb_remove

    else:
        pdb_remove_flag = False

    ###### for contact ######
    contact_remove_flag = False

    if with_contact and contact_path is None:
        ### contact prepare
        ### contact path
        path_idx = 1
        contact_path = os.path.join(
            temp_dir,
            'temp%s_%d.contact' % (name_tag, path_idx)
        )
        while os.path.exists(contact_path_path): # to avoid overwrite existing files
            path_idx += 1
            contact_path_path = os.path.join(
                temp_dir,
                'temp%s_%d.contact_path' % (name_tag, path_idx)
            )
        ### contact write
        dist_mat, contact_map = CA_contact_gen(
            pdb_path, contact_path, threshold=contact_thre)
        print('Contact saved at %s.' % contact_path)
        contact_remove_flag = pdb_remove 

    elif not with_contact:
        contact_path = None

    ######################## sbmOpenMM simulation model #######################
    sbmCAModel = sbm_getCAModel(
        structure_file = pdb_path,
        contact_file = contact_path,
        contact_force = None,  # currently do not consider contact energy
        residue_masses = with_resi,
        residue_radii = with_resi,
        RepulsionOnly = RepulsionOnly,     
    )
    #### remove temperary files
    if pdb_remove_flag:
        os.remove(pdb_path)
    if contact_remove_flag:
        os.remove(contact_path)

    ### get simulation modules
    integrator = LangevinIntegrator(
        1*unit.kelvin, 1/unit.picosecond, 0.0005*unit.picoseconds)
    simulation = Simulation(sbmCAModel.topology, sbmCAModel.system, integrator)
    simulation.context.setPositions(sbmCAModel.positions)

    ###################### calculate the force and energy #####################
    # Harmonic Bond Energy
    # Harmonic Angle Energy
    # Periodic Torsion Energy
    # LJ 12-10 Contact Energy (given the contact file)
    # LJ 12 Repulsion Energy
    if get_force:
        force_dict = {}
    else:
        force_dict = None

    if get_energy:
        energy_dict = {}
    else:
        energy_dict = None

    ###### value calculation ######
    for i, n in enumerate(sbmCAModel.forceGroups):
        if RepulsionOnly and (not n.startswith('LJ 12 Repulsion')):
            continue

        state = simulation.context.getState(
            getEnergy=get_energy, getForces=True, groups={i})

        ### force
        if get_force:
            # snm unit: kilojoule/(nanometer*mole); 10 kilojoule/(nanometer*mole) = 1 kilojoule/(A*mole)
            force_dict[n] = (quantity_to_tensor(
                state.getForces()) * 0.1).to(device)
        ### energy
        if get_energy:
            energy_dict[n] = (torch.tensor(state.getPotentialEnergy(
            ).value_in_unit(unit.kilojoules_per_mole))).to(device)

    ###### add up the results and add the padding ######

    ### force
    if get_force:
        ### add up forces
        if sum_result:
            force_dict = {'all': torch.sum(torch.stack(
                [force_dict[key] for key in force_dict.keys()]), dim=0)}

        ### add padding
        # shape match; force_dict: {key: (L_single, 3)} -> {key: (1, L, 3)}
        if force_pad is not None:
            force_dict = {key: F.pad(
                                   force_dict[key], 
                                   (0, 0, 0, force_pad - force_dict[key].shape[0]), 
                                   'constant', 
                                   0
                               ).unsqueeze(0).to(device)
                for key in force_dict.keys()}

    ### energy 
    if get_energy and sum_result:
        energy_dict = {'all': torch.sum(torch.stack(
            [energy_dict[key] for key in energy_dict.keys()]), dim=0)}

    return force_dict, energy_dict


def force_and_energy_multi(
        pdb_path_list=None,
        with_contact=True,
        contact_path_list=None,
        temp_dir='./',
        contact_thre=12, 
        name_tag=None, 
        get_force=True, 
        get_energy=True, 
        sum_result=True, 
        force_pad=200,
        multithread=True, 
        device='cuda',
        with_pdb_write=False, 
        coor_list=None, 
        seq_list=None, 
        mask_list=None, 
        atom_list=['CA'], 
        pdb_remove=True,
        RepulsionOnly=True,
        with_resi=False,
    ):
    """
    Get the force or energy prediction for multiple PDB files (or coordinates)
    if with_pdb_write:
        apply coor: torch.Tensor (N, L, atom_num, 3)
    else:
        apply pdb_path_list: list of str; paths pf pdb_files
    """
    force_dict_all = {}
    energy_dict_all = {}

    ### batch size
    if with_pdb_write and coor_list is None:
        print('Error! The coordinates list is empty!')
        return None, None
    elif with_pdb_write:
        N = coor_list.shape[0]
    else:
        N = len(pdb_path_list)

    ####################### energy and force computing ########################
    force_dict_all = {}
    energy_dict_all = {}
    if multithread:
        Thread_list = []

    ###### sample-wise calculation ######
    for i in range(N):
        ### paths
        pdb_path = None if with_pdb_write else pdb_path_list[i]
        contact_path = None if (contact_path_list is None) or (not with_contact) else contact_path_list[i]

        if name_tag is None:
            name_tag_thread = '%s-%d' % (device, i)
        else:
            name_tag_thread = '%s-%s-%d' % (name_tag, device, i)

        ### inputs
        coor = coor_list[i] if with_pdb_write else None
        seq = seq_list[i] if (with_pdb_write and seq_list is not None) else None
        mask = mask_list[i] if (with_pdb_write and mask_list is not None) else None

        ###### multithread computing ######
        if multithread:
            ### create thread
            Thread_list.append(ThreadWithReturnValue(
                                   target=force_and_energy, 
                                   args=(
                                       pdb_path,
                                       with_contact,
                                       contact_path,
                                       temp_dir,
                                       contact_thre,
                                       get_force,
                                       get_energy,
                                       name_tag,
                                       sum_result,
                                       force_pad,
                                       device,
                                       with_pdb_write,
                                       coor,
                                       seq,
                                       mask,
                                       atom_list,
                                       pdb_remove,
                                       RepulsionOnly,
                                       with_resi,
                                   )
                               )
            )
            ### start thread
            Thread_list[-1].start()
   
        ###### single-thread sequential computing ######     
        else:
            force_dict_all[i], energy_dict_all[i] = force_and_energy(
                pdb_path = pdb_path,
                with_contact = with_contact,
                contact_path = contact_path, 
                temp_dir = temp_dir,
                contact_thre = contact_thre, 
                get_force = get_force, 
                get_energy = get_energy, 
                name_tag = name_tag_thread,
                sum_result = sum_result, 
                force_pad = force_pad, 
                device = device,
                with_pdb_write = with_pdb_write, 
                coor = coor, 
                seq = seq, 
                mask = mask,
                atom_list = atom_list, 
                pdb_remove = pdb_remove,
                RepulsionOnly = RepulsionOnly,
                with_resi =  with_resi
            )

    ####### read the results ######
    if multithread:
        for i, thread in enumerate(Thread_list):
            force_dict_all[i], energy_dict_all[i] = thread.join()

    ######################## results summarization ############################
    if get_force and force_dict_all:
        key_set = force_dict_all[0].keys()
        force_dict_all = {key: torch.cat([force_dict_all[i][key] for i in range(N)], dim=0)
                          for key in key_set}
    else:
        force_dict_all = None

    if get_energy and energy_dict_all:
        key_set = energy_dict_all[0].keys()
        energy_dict_all = {key: torch.stack([energy_dict_all[i][key] for i in range(N)])
                           for key in key_set}
    else:
        energy_dict_all = None

    return force_dict_all, energy_dict_all

###################################################################################
# Guidance Calculation
###################################################################################


class Guidance_cal(nn.Module):
    def __init__(self,
                 with_energy_guide=True, openmm_version='CA',
                 with_fitness_guide=True, esm_version='ESM-1b',
                 input_voc_set=ressymb_order, device='cuda'):
        """Calculate the energy score or its gradient (force), and fitness
        score or its gradient.

        openmm_verion: str, 'CA' or 'all'
        esm_version: str, 'ESM-1b' or 'ESM2'

        """
        super(Guidance_cal, self).__init__()

        self.with_energy_guide = with_energy_guide
        self.openmm_version = openmm_version
        self.with_fitness_guide = with_fitness_guide
        self.esm_version = esm_version
        self.device = device

        if with_fitness_guide:
            self.fitness_module = FitnessGrad(
                version=esm_version, input_voc_set=input_voc_set).to(device)

    def energy_guide(self, 
        coor, seq, mask=None, atom_list=['CA'],
        with_contact=True, contact_path_list=None, temp_dir='./', contact_thre=12,
        get_force=True, get_energy=True, sum_result=True, multithread=False,
        RepulsionOnly = False, with_resi = False,                 
    ):
        """
        Args:
            coor: (N, L, 3) for 'CA' version and (N, L, K = atom_num, 3) for 'all' version
            seq: (N, L)
            mask: (N, L), 1 for valid tokens and 0 for others
            with_contact: list of str or bool; list of str for predefined contact path, bool for whether include the contact energy
            get_force: whether calculate the force
            get_energy: whether calculate the energy
            sum_result: whether use the sumarized result 
                        (i.e. if True then only return the sum of all energies)
            multithread: whether apply the multithread calculating
        """
        if not (get_force or get_energy):
            print('Warning! Neither force nor energy calculation is required!')
            return None, None

        ###### shape ######
        N, L = seq.shape
        # print(N,L)
        coor = coor.reshape(N, L, -1, 3)  # (N, L, K = atom_num, 3)
        # atom_num = coor.shape[2]

        # energy cal
        force_dict_all, energy_dict_all = force_and_energy_multi(
            pdb_path_list=None,
            with_contact=with_contact,
            contact_path_list=contact_path_list,
            contact_thre=contact_thre,
            get_force=get_force,
            get_energy=get_energy,
            sum_result=sum_result,
            force_pad=L,
            temp_dir=temp_dir,
            name_tag=None,
            multithread=multithread,
            device=coor.device,
            with_pdb_write=True,
            coor_list=coor,
            seq_list=seq,
            mask_list=mask,
            atom_list=atom_list,
            pdb_remove=True,
            RepulsionOnly=RepulsionOnly,
            with_resi=with_resi,
        )

        return force_dict_all, energy_dict_all

    def fitness_guide(self, seq, mask=None, with_grad=True, seq_transform=True, with_padding=False):
        if not self.with_fitness_guide:
            print('Fitness module is not defined!')
            return None, None
        else:
            fitness, grad = self.fitness_module(seq, mask=mask,
                                                with_grad=with_grad, seq_transform=seq_transform, with_padding=with_padding)
            # fitness: (N,) or n
            # grad: (N, L, 20) or None (if with_grad=False)
            return fitness, grad

###################################################################################
# For debugging
###################################################################################


if __name__ == '__main__':
    ########################### for contact ###################################

    # pdb_path = 'temp/1YPA_I.pdb'
    # out_path = 'temp/1YPA_I.contact'
    # pdb_path = '../../Results/diffab/forward-diff_struc/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_2023_05_01__17_05_48/sample84_0.pdb'
    # pdb_path = '../../../Tools/PDB_test/107L_A1-162.pdb'
    # out_path = 'temp.contact'
    # out_path_2 = 'temp2.contact'
    # dist_mat, contact_map = CA_contact_gen(pdb_path, out_path, threshold=12)
    # dist_mat_2, contact_map_2 = contact_gen_torch(
    #     pdb_path, out_path_2, threshold=12, atom_sele='CA')
    # print(contact_map == contact_map_2)


    ########################### for sbmopenmm ###################################

    # ### groundtruth ###
    # sbmCAModel = sbmOpenMM.models.getCAModel(pdb_path, out_path)

    # ### test ###
    # print('Test...')
    # force_dict, energy_dict = force_and_energy(pdb_path, temp_dir = './', contact_thre = 12, get_energy = True)
    # for k in energy_dict.keys():
    #     print(k, force_dict[k].shape, energy_dict[k])

    ########################### for multithread ###################################

    #pdb_path = 'pdb_for_test/'
    pdb_path = '../Data/Origin/CATH/pdb_all/'
    contact_path = '../Data/Processed/CATH_forDiffAb/ContactMap_CA/'
    #pdb_list = ['107L_A1-162.pdb', '109L_A1-162.pdb', '111L_A1-162.pdb', '113L_A1-162.pdb', '115L_A1-162.pdb',
    #            '108L_A1-162.pdb', '110L_A1-162.pdb', '112L_A1-162.pdb'] #, '114L_A1-162.pdb', '118L_A1-162.pdb']
    pdb_list = [
        '107L_A1-162',
        '108L_A1-162',
        '109L_A1-162',
        '110L_A1-162',
        '111L_A1-162',
        '112L_A1-162',
        '115L_A1-162',
        '118L_A1-162',
        '120L_A1-162',
        '122L_A1-162',
        '125L_A1-162',
        '126L_A1-162',
        '128L_A1-162',
        '139L_A1-162',
        '140L_A1-162',
        '142L_A1-162'
    ]
    pdb_path_list = [pdb_path + p + '.pdb' for p in pdb_list]
    contact_path_list = [contact_path + p + '.contact' for p in pdb_list]

    print('single_thread:')
    start_time = time.time()

    force_dict, energy_dict = force_and_energy_multi(pdb_path_list, with_contact=True,
                              contact_path_list = contact_path_list,
                              contact_thre = 12, get_force = True, get_energy = True, sum_result = True, force_pad = 200,
                              temp_dir = './', name_tag = None, multithread = False, device = 'cuda')
    for k in energy_dict.keys():
        print(k)
        print(force_dict[k].shape, force_dict[k][0,0,:])
        print(energy_dict[k])
    print('%.4fs cost.'%(time.time() - start_time))
    print('')

    print('multi_thread:')
    start_time = time.time()

    force_dict, energy_dict = force_and_energy_multi(pdb_path_list,
                           with_contact=True,
                           contact_path_list = contact_path_list,
                           contact_thre = 12, get_force = True, get_energy = True, sum_result = True, force_pad = 200,
                           temp_dir = './', name_tag = None, multithread = True, device = 'cuda')
    for k in energy_dict.keys():
        print(k)
        print(force_dict[k].shape, force_dict[k][0,0,:])
        print(energy_dict[k])
    print('%.4fs cost.'%(time.time() - start_time))
    print('')
