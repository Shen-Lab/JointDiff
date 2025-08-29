#####################################################
# test the energy scores with sbm-openmm
#####################################################

import os
import math
import numpy as np
import argparse
import time
from tqdm.auto import tqdm
import logging

import sbmOpenMM
from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit

import utils_eval

####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--struc_path', type=str, default='../../Results/originDiff/forward-diff_struc/struc/Step100_posiscale10.0/')
    parser.add_argument('--contact_path', type=str, default='../../Data/Processed/CATH_forDiffAb/ContactMap_test_rearanged/')
    parser.add_argument('--out_path', type=str, default='../../Results/originDiff/sbmopenmm_check/forward_Step100_posiscale10.0_100-1.pkl')

    parser.add_argument('--with_seq', type=int, default=1)
    parser.add_argument('--CA_only', type=int, default=1)
    parser.add_argument('--torsion_energy', type=float, default=1.0)
    parser.add_argument('--contact_energy', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--job_num', type=int, default=100)
    parser.add_argument('--job_idx', type=int, default=1)

    args = parser.parse_args()

    if args.contact_path is not None and args.contact_path.upper() == 'NONE':
        args.contact_path = None

    with_seq = bool(args.with_seq)
    CA_only = bool(args.CA_only)

    if (not with_seq) and (not CA_only):
        print('Can only apply the CA-only version when the sequence is not given!')
        quit()

    ###########################################################################
    # sample list 
    ###########################################################################

    title_list = ['.pdb'.join(pdb.split('.pdb')[:-1]) 
        for pdb in os.listdir(args.struc_path) if pdb.endswith('.pdb')]

    if title_list:
        title_list = sorted(title_list)

    num_all = len(title_list)
    interval = math.ceil(num_all / args.job_num)
   
    title_list_sele = title_list[(args.job_idx - 1) * interval : args.job_idx * interval]
    print('%d samples selected out of %d.' % (len(title_list_sele), num_all))

    ###########################################################################
    # Path 
    ###########################################################################

    if os.path.exists(args.out_path):
        out_dict = utils_eval.dict_load(args.out_path)
    else:
        out_dict = {}

    ###########################################################################
    # Energy Calculation 
    ###########################################################################

    for idx, title in tqdm(enumerate(title_list_sele)): 

        if title in out_dict:
            continue
            
        out_dict[title] = {}
        pdb_path = os.path.join(args.struc_path, title + '.pdb')

        ###### sbm-set ######
       
        sbm = sbmOpenMM.system(pdb_path)
        if args.CA_only:
            sbm.getCAlphaOnly()
        sbm.getAtoms()
        if args.with_seq:
            sbm.setCAMassPerResidueType()
        sbm.getBonds()
        sbm.getAngles()
        sbm.getProperTorsions()

        if args.contact_path is not None:
            name = '_'.join(title.split('sample-')[-1].split('_')[:2])
            contact_file = os.path.join(args.contact_path, name + '.contact')
            sbm.readContactFile(contact_file)

        sbm.setBondParameters(20000.0)
        sbm.setAngleParameters(40.0)
        sbm.setProperTorsionParameters(args.torsion_energy)
        sbm.setNativeContactParameters(args.contact_energy)
        if args.with_seq:
            sbm.setCARadiusPerResidueType()
        else:
            sbm.setParticlesRadii(0.4)
        sbm.rf_epsilon = 0.1
        sbm.addHarmonicBondForces()
        sbm.addHarmonicAngleForces()
        sbm.addPeriodicTorsionForces()
        sbm.addLJ12_10ContactForces()

        sbm.addLJRepulsionForces(cutoff=1.5)

        sbm.createSystemObject(
            minimize=False, 
            check_bond_distances = False, 
            check_large_forces = False
        )

        ###### energy cal ######

        integrator = LangevinIntegrator(
            args.temperature * unit.kelvin, 
            1/unit.picosecond, 
            0.0005*unit.picoseconds
        )
        simulation = Simulation(sbm.topology, sbm.system, integrator)
        simulation.context.setPositions(sbm.positions)

        out_dict[title] = {}
        for i,n in enumerate(sbm.forceGroups):
            state = simulation.context.getState(getEnergy=True, getForces=True, groups={i})
            out_dict[title][n] = state.getPotentialEnergy()

        ###### record the result ######

        if (idx + 1) % 100 == 0:
            _ = utils_eval.dict_save(out_dict, args.out_path)

    ###########################################################################
    # save the results
    ###########################################################################

    _ = utils_eval.dict_save(out_dict, args.out_path)

