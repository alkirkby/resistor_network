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
from rnpy.functions.assignfaults_new import get_Nf2D, add_random_fault_sticks_to_arrays
from rnpy.functions.readoutputs import read_fault_params, get_equivalent_rho
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
                                 **kwargs):
    
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
    while np.any(Nf_update) > 0:
        for ii,Nfval in enumerate(Nf_update):
            Rv = add_random_fault_sticks_to_arrays(Rv, Nfval, fault_lengths_m[ii], fault_widths[ii], hydraulic_apertures[ii],
                                           resistivity[ii],pz)
        Nf_update_new = ((np.array(Nf) - np.array([np.sum(Rv.aperture[:,:,1,1:,1:]==fault_widths[i])/(fault_lengths_m[i]/Rv.cellsize[1]) for i in range(len(Nf))]))).astype(int)
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
        processes=True,
        memory_limit="auto",
    )
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
    print(inputs)
    
    # load fault widths from modelling
    t0a = time.time()
    print("loading inputs, took %.2fs"%(t0a-t0))

    # construct output filename
    prop_suffix = 'a%.1f_R%1im_rf%0.1f_por%.1fpc_pz%.2f_cs%1imm_%s'%(a,R,rfluid,porosity_target*100,pz,cellsize*1000,direction)
    # prop_suffix = f'a{a:1f}_R%1im_rm%1i_rf%0.1f_por%.1fpc_pz%.2f_cs%1imm_%s'%(a,R,matrix_res,rfluid,porosity_target*100,pz,cellsize*1000,direction)

    output_fn = f'FaultSticks_{prop_suffix}.dat'
    
    # read fault lengths (center of bin) + pre-computed properties from json file
    lvals_center, fw, aph, resistivity = read_fault_params(os.path.join(wd,
                                                  'fault_k_aperture_rf%s_%s.npy'%(rfluid,direction)),
                                                           None)
    if len(np.unique(lvals_center)) < len(lvals_center):
            
        lvals_center_unique = np.unique(lvals_center)
        # get mean fault width by 
        fw_mean = np.array([np.mean(fw[lvals_center==ll]) for ll in lvals_center_unique])
    else:
        lvals_center_unique, fw_mean = lvals_center, fw

    t1a = time.time()
    print('get lvals, %.2fs'%(t1a-t0))
    
    # determine numbers of faults in each length bin using alpha and target porosity
    Nf, alpha, lvals_range = get_Nf2D(a,R,lvals_center_unique,fw_mean,porosity_target,alpha_start = 1.0)
    
    
    if lmax is None:
        lmax, idx1 = max(lvals_range)+1, len(lvals_range)+1
    else:
        idx1 = np.where(lvals_center_unique <= lmax)[-1][-1] - \
            len(lvals_center_unique) + 1
    if lmin is None:
        lmin, idx0 = 0, 0
    else:
        idx0 = np.where(lvals_center_unique > lmin)[0][0]
    
    filt = np.all([lvals_center > lmin, lvals_center <= lmax],axis=0)
    
    print("len(aph)",len(aph))
    
    # for arr in [aph, resistivity, fw, lvals_center]:
    #     arr = arr[filt]
    aph = aph[filt]
    resistivity = resistivity[filt]
    fw = fw[filt]
    lvals_center = lvals_center[filt]
    print("len(aph)",len(aph))
    print(aph)
        
    Nf = Nf[idx0:idx1]
    lvals_range = lvals_range[idx0:idx1]
    lvals_center_unique = lvals_center_unique[idx0:idx1]
    

    print(Nf)
    print(lvals_center_unique)
    print(lvals_range)
    
    t1 = time.time()
    print('get Nf, %.2fs'%(t1-t1a))
    
    # build aperture array
    ncells = int(R/cellsize)
    Rv_inputs = dict(ncells=[0,ncells,ncells],
                     cellsize=cellsize,
                     resistivity_fluid=rfluid,
                     resistivity_matrix=matrix_res,
                     permeability_matrix=1e-18)
    
    # write fixed variables to file
    with open(os.path.join(wd,output_fn),'w') as openfile:
        openfile.write('# a %.1f\n'%a)
        openfile.write('# total_size_m %.1f\n'%R)
        openfile.write('# cellsize_mm %.1f\n'%(cellsize*1000))
        openfile.write('# matrix_permeability %.1e %.1e %.1e\n'%tuple(matrix_k))
        openfile.write('# matrix_resistivity %.1e %.1e %.1e\n'%tuple(matrix_res))
        openfile.write('# porosity_target %.3f\n'%porosity_target)
        openfile.write('# alpha %.2f\n'%alpha)
        openfile.write('# fault_lengths_center '+' '.join([str(val) for val in lvals_center_unique])+'\n')
        openfile.write('# fault_lengths_bin_ranges '+' '.join([str(val) for val in lvals_range])+'\n')
        openfile.write('# Num_fractures_per_bin '+' '.join(['%.1i'%val for val in Nf])+'\n')
    
    
    kbulk = []
    rbulk = []
    cf = []
    t2 = time.time()
    print('initialise inputs, %.2fs'%(t2-t1))
    
    if np.iterable(matrix_res):
        resistivity_ij = np.column_stack([get_equivalent_rho(resistivity,fw,cellsize,rho_matrix=matrix_res[i])\
                                          for i in [1,2]])
    simulations = [setup_and_solve_fault_sticks(Nf, lvals_center,fw, aph, pz, resistivity_ij,
                                      **Rv_inputs) for _ in range(repeats)]
    t3 = time.time()
    Rv_list = compute(*simulations)
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
    