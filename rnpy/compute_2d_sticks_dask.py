# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:06:33 2025

@author: alisonk
"""
import numpy as np
import sys
import os

if os.name == 'nt':
    sys.path.append(r'C:\git\resistor_network')
    print("appended to path")

from rnpy.core.resistornetwork import Rock_volume
from rnpy.functions.assignfaults_new import get_Nf2D, get_alpha, add_random_fault_sticks_to_arrays
from rnpy.functions.readoutputs import read_fault_params, get_equivalent_rho
from rnpy.functions.utils import get_bin_ranges_from_centers
import time
import argparse

from dask.distributed import Client
from dask import delayed, compute



def parse_arguments(arguments):
    """
    parse arguments from command line
    """
    
    argument_names = [['target_porosity','p','target porosity',1,float],
                      ['width','R','model width',1,float],
                      ['cellsize','c','cellsize for fault stick model',1,float],
                      ['a',None,'density exponent a',1,float],
                      ['probability_z','pz','probability of a fault in z direction',1,float],
                      ['direction','d','direction of flow through fractures, y (parallel to offset) or z (perpendicular)',1,str],
                      ['repeats','r','number of repeats to run',1,int],
                      ['resistivity_fluid','rf','resistivity of fluid',1,float],
                      ['resistivity_matrix','rm','resistivity of matrix','*',float],
                      ['permeability_matrix','km','permeability of matrix','*',float],
                      ['lmin',None,'minimum fault length',1,float],
                      ['lmax',None,'maximum fault length',1,float],
                      ['maximum_y_fault_length',None,'maximum fault length for cross-faults',1,float],
                      ['working_directory','wd','working directory for inputs and outputs',1,str],
                      ['n_workers','nw','number of workers to parallelise by',1,int],
                      ['threads_per_worker',None,'number of threads per worker',1,int],
                      ]
    
    parser = argparse.ArgumentParser()
    
    
    for i in range(len(argument_names)):
        
        longname,shortname,helpmsg,nargs,vtype = argument_names[i]
        action = 'store'
        longname = '--'+longname
        
        if shortname is None:
            parser.add_argument(longname,help=helpmsg,
                                nargs=nargs,type=vtype,action=action)
        else:
            shortname = '-'+shortname
            parser.add_argument(shortname,longname,help=helpmsg,
                                nargs=nargs,type=vtype,action=action)
    
    args = parser.parse_args(arguments[1:])

    input_parameters = {}
    # assign parameters to correct dictionaries. Only allowing fault separation
    # as a loop parameter at this point.
    for at in args._get_kwargs():
        if at[1] is not None:
            value = at[1]
            if len(value) == 1:
                input_parameters[at[0]] = value[0]
            else:
                input_parameters[at[0]] = np.array(value)
    
    return input_parameters

@delayed
def setup_and_solve_fault_sticks(Nf, fault_lengths_m, fault_widths, hydraulic_apertures, pz, resistivity, 
                                 kwargs, max_y_fault_length=None):
    print("setting up fault sticks")
    # initialise rock volume
    Rv = Rock_volume(**kwargs)
    Rv.aperture[np.isfinite(Rv.aperture)] = 2e-50
    Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = 2e-50
    Rv.initialise_electrical_resistance()
    
    # initialise hydraulic aperture to reflect matrix permeability
    for i in range(3):
        Rv.aperture_hydraulic[:,:,:,i][np.isfinite(Rv.aperture_hydraulic[:,:,:,i])]  = \
            (12*Rv.permeability_matrix[i])**0.5

    # initialise Nf_update, additional faults to be updated each round
    Nf_update = np.copy(Nf)
    count = 0
    
    # split up hydraulic_apertures, fault lengths, widths, resistivity by fault length
    if np.iterable(fault_lengths_m):
        fault_widths, hydraulic_apertures, resistivity = \
            [[arr[fault_lengths_m==length] for length in np.unique(fault_lengths_m)]\
             for arr in [fault_widths, hydraulic_apertures, resistivity]]
        fault_lengths_m = np.unique(fault_lengths_m)
        fault_widths = [np.mean(ww) for ww in fault_widths]
    
    t0 = time.time()
    # assign faults to array:
    count1 = 0
    fault_lengths_assigned = None
    while np.any(Nf_update) > 0:
        for ii,Nfval in enumerate(Nf_update):
            pz_input = pz
            if max_y_fault_length is not None:
                if fault_lengths_m[ii] > max_y_fault_length:
                    pz_input = 1.
            # print("fault_lengths_m[ii]",fault_lengths_m[ii],pz_input,pz)
            Rv, fault_lengths_assigned = add_random_fault_sticks_to_arrays(Rv, Nfval, fault_lengths_m[ii], fault_widths[ii], hydraulic_apertures[ii],
                                           resistivity[ii],pz_input,fault_lengths_assigned=fault_lengths_assigned)
        Nf_update_new = ((np.array(Nf) - np.array([np.sum(fault_lengths_assigned==fault_lengths_m[i])/(fault_lengths_m[i]/Rv.cellsize[1]) for i in range(len(Nf))]))).astype(int)
        
        if np.all(np.array(Nf_update_new) == np.array(Nf_update)):
            count += 1
        else:
            Nf_update = Nf_update_new
        # print(Nf_update)
        # if we have had 10 rounds with no change, break out of array.
        if count == 10:
            break
        count1 += 1
    t1 = time.time()
    print('Nf_update, took %.3f s'%(t1-t0))
    print('Number of updates, ',count1)
    
    Rv.initialise_permeability()
    
    t2 = time.time()
    print('initialise permeability took %.1f s'%(t2-t1))
    
    Rv.solve_resistor_network2(method='pardiso')
    t3 = time.time()
    
    print('solve with pardiso solver took %.1f s'%(t3-t2))
    Rv.compute_conductive_fraction()    
    
        
    return Rv




if __name__ == '__main__':


    
    # %%
    t0 = time.time()
    inputs = parse_arguments(sys.argv)
    
    n_workers = inputs['n_workers']
    threads_per_worker = inputs['threads_per_worker']
    # %%
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        # local_directory=local_dir,
        processes=True if os.name == 'nt' else False,
        dashboard_address=None,
        memory_limit="auto",
    )
    print("n_workers",n_workers)
    # goto http://localhost:8787 for progress
    # %%    


    wd = inputs['working_directory']
    a = float(inputs['a'])
    porosity_target = inputs['target_porosity']
    rfluid = inputs['resistivity_fluid']
    R = inputs['width']
    pz = inputs['probability_z']
    repeats = inputs['repeats']
    cellsize = inputs['cellsize']
    direction = inputs['direction']
    max_y_fault_length = inputs['maximum_y_fault_length'] if \
        'maximum_y_fault_length' in inputs.keys() else None
    
    lmin, lmax = None, None
    if 'lmin' in inputs.keys():
        lmin = inputs['lmin']
    if 'lmax' in inputs.keys():
        lmax = inputs['lmax']
    
    matrix_res = np.array([1000]*3)
    matrix_k = np.array([1e-18]*3)
    
    if 'resistivity_matrix' in inputs.keys():
        matrix_res = np.array(inputs['resistivity_matrix'])
    if 'permeability_matrix' in inputs.keys():
        matrix_k = np.array(inputs['permeability_matrix'])
    if 'alpha' in inputs.keys():
        alpha = inputs['alpha']
    
    # load fault widths from modelling
    t0a = time.time()
    print("loading inputs, took %.2fs"%(t0a-t0))

    # construct output filename
    if rfluid >= 0.1:
        rfstr = '%.1f'%rfluid
    else:
        rfstr = '%.2f'%rfluid
        
    if R >= 1:
        Rstr = '%1i'%R
    else:
        Rstr = '%.1f'%R
    rfstr = '%.1f'%rfluid if rfluid >= 0.1 else '%.2f'%rfluid
    Rstr = '%1i'%R if R >= 1 else '%.1f'%R
    cs_str = '%1i'%(cellsize*1000) if cellsize >= 1e-3 else '%.1f'%(cellsize*1000)
        
    prop_suffix = 'a%.1f_R%sm_rf%s_por%.1fpc_pz%.2f_cs%smm_%s'%(a,Rstr,rfstr,porosity_target*100,pz,cs_str,direction)
    # prop_suffix = f'a{a:1f}_R%1im_rm%1i_rf%0.1f_por%.1fpc_pz%.2f_cs%1imm_%s'%(a,R,matrix_res,rfluid,porosity_target*100,pz,cellsize*1000,direction)

    output_fn = f'FaultSticks_{prop_suffix}.dat'
    
    # read fault lengths (center of bin) + pre-computed properties from json file
    lvals_center, lvals_range, fw, aph, resistivity = read_fault_params(os.path.join(wd,
                                                  'fault_k_aperture_rf%s_%s.npy'%(rfluid,direction)),
                                                           None)
    
    
    if len(np.unique(lvals_center)) < len(lvals_center):
            
        lvals_center_unique = np.unique(lvals_center)
        lvals_range_unique = np.unique(lvals_range)
        # get mean fault width by 
        fw_mean = np.array([np.mean(fw[lvals_center==ll]) for ll in lvals_center_unique])
    else:
        lvals_center_unique, lvals_range_unique, fw_mean = lvals_center, lvals_range, fw

    R_for_alpha_calc = 10*lvals_center_unique.max()

    t1a = time.time()
    print('get lvals, %.2fs'%(t1a-t0))
    
    # determine numbers of faults in each length bin using alpha and target porosity
    print(a,R,lvals_center_unique,fw_mean,porosity_target)
    
    # use big R-squared (R2) to compute alpha
    alpha = get_alpha(a,R_for_alpha_calc,lvals_center_unique,lvals_range_unique,fw_mean,porosity_target,alpha_start = 0.1)

    Nf = get_Nf2D(a, alpha, R, lvals_range_unique)

    if max_y_fault_length is not None:
        Nf_precise = get_Nf2D(a, alpha, R_for_alpha_calc, lvals_range_unique)
        Nf_lw = Nf_precise * lvals_center_unique
        filt_ylen = lvals_center_unique >= max_y_fault_length
        pz_short = (pz * Nf_lw.sum() - Nf_lw[filt_ylen].sum())/Nf_lw[~filt_ylen].sum()
    else:
        pz_short = pz
    
    if lmax is None:
        lmax, idx1 = np.amax(lvals_range_unique)+1, len(lvals_range_unique)+1
    else:
        idx1 = np.where(lvals_center_unique <= lmax)[-1][-1] - \
            len(lvals_center_unique) + 1 # relative to the end
    if lmin is None:
        lmin, idx0 = 0, 0
    else:
        idx0 = np.where(lvals_center_unique > lmin)[0][0]
    
    filt = np.all([lvals_center > lmin, lvals_center <= lmax],axis=0)
    
    # for arr in [aph, resistivity, fw, lvals_center]:
    #     arr = arr[filt]
    aph = aph[filt]
    resistivity = resistivity[filt]
    fw = fw[filt]
    lvals_center = lvals_center[filt]
    Nf = Nf[idx0:idx1]
    lvals_range_unique = lvals_range_unique[idx0:idx1]
    lvals_center_unique = lvals_center_unique[idx0:idx1]
    
    
    t1 = time.time()
    print('get Nf, %.2fs'%(t1-t1a))
    
    # build aperture array
    ncells = int(R/cellsize)
    Rv_inputs = dict(ncells=[0,ncells,ncells],
                     cellsize=cellsize,
                     resistivity_fluid=rfluid,
                     resistivity_matrix=matrix_res,
                     permeability_matrix=matrix_k)
    
    # write fixed variables to file
    with open(os.path.join(wd,output_fn),'w') as openfile:
        openfile.write('# a %.1f\n'%a)
        openfile.write('# total_size_m %.1f\n'%R)
        openfile.write('# cellsize_mm %.1f\n'%(cellsize*1000))
        openfile.write('# matrix_permeability %.1e %.1e %.1e\n'%tuple(matrix_k))
        openfile.write('# matrix_resistivity %.1e %.1e %.1e\n'%tuple(matrix_res))
        openfile.write('# porosity_target %.3f\n'%porosity_target)
        openfile.write('# alpha %.2f\n'%alpha)
        openfile.write('# pz %.5f\n'%pz)
        openfile.write('# pz_short %.5f\n'%pz_short)
        openfile.write('# fault_lengths_center '+' '.join([str(val) for val in lvals_center_unique])+'\n')
        openfile.write('# fault_lengths_bin_ranges '+' '.join([str(val) for val in lvals_range_unique])+'\n')
        openfile.write('# Num_fractures_per_bin '+' '.join(['%.1i'%val for val in Nf])+'\n')
    
    
    kbulk = []
    rbulk = []
    cf = []
    t2 = time.time()
    print('initialise inputs, %.2fs'%(t2-t1))
    
    if np.iterable(matrix_res):
        resistivity_ij = np.column_stack([get_equivalent_rho(resistivity,fw,cellsize,rho_matrix=matrix_res[i])\
                                          for i in [1,2]])
    t3 = time.time()
    simulations = [setup_and_solve_fault_sticks(Nf, lvals_center,fw, aph, 
                                                pz_short, 
                                                resistivity_ij,
                                                Rv_inputs,
                                                max_y_fault_length=max_y_fault_length) \
                   for _ in range(repeats)]
    
    Rv_list = compute(*simulations)
    
    
    
    # Rv_list = [setup_and_solve_fault_sticks(Nf, lvals_center,fw, aph, 
    #                                             pz_short, 
    #                                             resistivity_ij,
    #                                             Rv_inputs,
    #                                             max_y_fault_length=max_y_fault_length)]
    t4 = time.time()
    print('setup and solve, %.2fs'%(t4-t3))
    kbulk = np.stack([Rv.permeability_bulk for Rv in Rv_list])
    rbulk = np.stack([Rv.resistivity_bulk for Rv in Rv_list])
    cf = np.stack([Rv.conductive_fraction for Rv in Rv_list])
    
    for Rv in Rv_list:
        line = '%.2f %.4f' + ' %.4e'*4 +'\n'
        line = line%(pz,
                     Rv.conductive_fraction,
                     Rv.permeability_bulk[1],
                     Rv.permeability_bulk[2],
                     Rv.resistivity_bulk[1],
                     Rv.resistivity_bulk[2])       
        with open(os.path.join(wd,output_fn),'a') as openfile:
            openfile.write(line)
            
    array_savedir = os.path.join(wd,'arrays_%s'%prop_suffix)
    if not os.path.exists(array_savedir):
        os.mkdir(array_savedir)
    # np.save(os.path.join(array_savedir, 'resistivity'), Rv.resistivity)
    # np.save(os.path.join(array_savedir, 'aperture'),Rv.aperture)
    # np.save(os.path.join(array_savedir, 'hydraulic_aperture'),Rv.aperture_hydraulic)
    # np.save(os.path.join(array_savedir, 'permeability'),Rv.permeability)
    np.savez_compressed(os.path.join(os.path.join(array_savedir,'arrays')),
                        resistivity=Rv.resistivity,
                        aperture=Rv.aperture,
                        hydraulic_aperture=Rv.aperture_hydraulic,
                        permeability=Rv.permeability,
                        current=Rv.current,
                        flowrate=Rv.flowrate)
    t6 = time.time()
    print('total time, %.1fs'%(t6-t0))
    