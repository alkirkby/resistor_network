# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:06:33 2025

@author: alisonk
"""
import numpy as np

from rnpy.core.resistornetwork import Rock_volume
from rnpy.functions.assignfaults_new import get_Nf2D, add_random_fault_sticks_to_arrays
from rnpy.functions.readoutputs import read_fault_params
import time
import os


def setup_and_solve_fault_sticks(Nf, fault_lengths_m, fault_widths, hydraulic_apertures, pz, resistivity,
                                 **kwargs):
    
    # initialise rock volume
    Rv = Rock_volume(**kwargs)
    Rv.aperture[np.isfinite(Rv.aperture)] = 2e-50
    Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = 2e-50
    Rv.initialise_electrical_resistance()
    
    # initialise hydraulic aperture to reflect matrix permeability
    Rv.aperture_hydraulic[np.isfinite(Rv.aperture_hydraulic)]  = (12*Rv.permeability_matrix)**0.5

    # initialise Nf_update, additional faults to be updated each round
    Nf_update = np.copy(Nf)
    count = 0
    
    t0 = time.time()
    # assign faults to array:
    while np.any(Nf_update) > 0:
        for ii,Nfval in enumerate(Nf_update):
            Rv = add_random_fault_sticks_to_arrays(Rv, Nfval, fault_lengths_m[ii], fault_widths[ii], hydraulic_apertures[ii],
                                           resistivity[ii],pz)
            
        Nf_update_new = ((np.array(Nf) - np.array([np.sum(Rv.aperture[:,:,1,1:,1:]==fault_widths[i])/(fault_lengths_m[i]/cellsize) for i in range(len(Nf))]))).astype(int)

        if np.all(np.array(Nf_update_new) == np.array(Nf_update)):
            count += 1
        else:
            Nf_update = Nf_update_new
        # print(Nf_update)
        # if we have had 10 rounds with no change, break out of array.
        if count == 10:
            break
    t1 = time.time()
    print('Nf_update, took %.3f s'%(t1-t0))
    
    Rv.initialise_permeability()
    
    t2 = time.time()
    print('initialise permeability took %.1f s'%(t2-t1))
    
    Rv.solve_resistor_network2(method='pardiso')
    t3 = time.time()
    
    print('solve with pardiso solver took %.1f s'%(t3-t2))
    Rv.compute_conductive_fraction()    
    
        
    return Rv



if __name__ == '__main__':
    # load fault widths from modelling
    t0 = time.time()
    if os.name == 'nt':
        savepath = r'I:\Geophysics\Research\Energy_Futures_Project_2_Geophysics\Rock_property_modelling\summary_data_from_models'
    else:
        savepath = '/share/scratch/alisonk/Rock_property_modelling/Fault_stick_models'
    a = 3.0
    R = 14
    
    matrix_k, matrix_res = 1e-18,1000
    rfluid = 0.5
    
    
    porosity_target = 0.02
    
    # construct output filename
    output_fn = 'FaultSticks_a%.1f_R%1im_rm%1i_rf%0.1f_por%1ipc.dat'%(a,R,matrix_res,rfluid,porosity_target*100)
    
    
    lvals_center, fw, aph, resistivity = read_fault_params(os.path.join(savepath,'fault_k_aperture_rfluid%s.json'%(rfluid)))

    t1a = time.time()
    print('get lvals, %.2fs'%(t1a-t0))
    
    Nf, alpha, lvals_range = get_Nf2D(a,R,lvals_center,fw,porosity_target,alpha_start = 1.0)
    
    t1 = time.time()
    print('get Nf, %.2fs'%(t1-t1a))
    
    # build aperture array
    cellsize = 0.01
    ncells = int(R/cellsize)
    Rv_inputs = dict(ncells=[0,ncells,ncells],
                     cellsize=cellsize,
                     resistivity_fluid=rfluid,
                     resistivity_matrix=matrix_res,
                     permeability_matrix=1e-18)
    
    # write fixed variables to file
    with open(os.path.join(savepath,output_fn),'w') as openfile:
        openfile.write('# a %.1f\n'%a)
        openfile.write('# total_size_m %.1f\n'%R)
        openfile.write('# matrix_permeability %.1e\n'%matrix_k)
        openfile.write('# matrix_resistivity %.1e\n'%matrix_res)
        openfile.write('# porosity_target %.3f\n'%porosity_target)
        openfile.write('# alpha %.2f\n'%alpha)
        openfile.write('# fault_lengths_center '+' '.join([str(val) for val in lvals_center])+'\n')
        openfile.write('# fault_lengths_bin_ranges '+' '.join([str(val) for val in lvals_range])+'\n')
        openfile.write('# Num_fractures_per_bin '+' '.join(['%.1i'%val for val in Nf])+'\n')
    
    
    kbulk = []
    rbulk = []
    cf = []
    t2 = time.time()
    print('initialise inputs, %.2fs'%(t2-t1))
    for pz in [0.85]:#,0.9,0.95, 0.9, 0.95, 0.98, 0.99
        kbulk.append([])
        rbulk.append([])
        cf.append([])
        for rpt in range(10):
            t3 = time.time()
            
            Rv = setup_and_solve_fault_sticks(Nf, lvals_center,fw, aph, pz, resistivity,
                                              # log_fn = os.path.join(savepath,output_fn[:-4]+'.log'),
                                              **Rv_inputs)
            t4 = time.time()
            print('setup and solve, %.2fs'%(t4-t3))
            line = '%.2f %.4f' + ' %.3e'*4 +'\n'
            line = line%(pz,Rv.conductive_fraction,
                         Rv.permeability_bulk[1],Rv.permeability_bulk[2],
                         Rv.resistivity_bulk[1],Rv.resistivity_bulk[2])
            with open(os.path.join(savepath,output_fn),'a') as openfile:
                openfile.write(line)
            print(pz)
            print(Rv.conductive_fraction)
            print(Rv.permeability_bulk)
            print(Rv.resistivity_bulk)
            kbulk[-1].append(Rv.permeability_bulk)
            rbulk[-1].append(Rv.resistivity_bulk)
            cf[-1].append(Rv.conductive_fraction)
            t5 = time.time()
            print('write to file, %.2fs'%(t5-t4))
