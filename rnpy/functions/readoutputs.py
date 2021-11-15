# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:12:34 2021

@author: alisonk
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import sem


def update_idx_dict(idx_dict):
    if idx_dict is None:
        idx_dict = {'fs':0,'cf':1,'res':2,'k':3,'xcs':5}
    else:
        idx_dict2 = {'fs':0,'cf':1,'res':2,'k':3,'xcs':5}
        idx_dict2.update(idx_dict)
        idx_dict = idx_dict2
        
    return idx_dict
    

def interpolate_to_all_fs(outputs,fs_list=None,idx_dict=None):
    """
    Interpolate outputs from simulations to all fault separations

    Parameters
    ----------
    outputs : TYPE
        DESCRIPTION.

    Returns
    -------
    data_dict1 : TYPE
        DESCRIPTION.

    """
    data_dict1 = {}
    if fs_list is None:
        data_dict1['fs'] = np.unique(outputs[:,:,0])
    else:
        data_dict1['fs'] = fs_list
    nrpts = outputs.shape[0]
    
    idx_dict = update_idx_dict(idx_dict)

    for pname in idx_dict.keys():
        if pname != 'fs':
            i = idx_dict[pname]
            data_dict1[pname] = np.zeros((nrpts,data_dict1['fs'].shape[0]))
            for r in range(nrpts):
                if pname in ['res','k']:
                    # interpolate in log space
                    func = interp1d(outputs[r,:,idx_dict['fs']],np.log10(outputs[r,:,i]),bounds_error=False)
                    data_dict1[pname][r] = 10**func(data_dict1['fs'])
                else:
                    # interpolate in linear space
                    func = interp1d(outputs[r,:,idx_dict['fs']],outputs[r,:,i],bounds_error=False)
                    data_dict1[pname][r] = func(data_dict1['fs'])   
            
    return data_dict1


def interpolate_to_permeability_values(outputs,permeability_values,idx_dict=None):
    """
    Interpolate outputs from simulations to all fault separations

    Parameters
    ----------
    outputs : TYPE
        DESCRIPTION.

    Returns
    -------
    data_dict1 : TYPE
        DESCRIPTION.

    """
    data_dict1 = {}
    data_dict1['k'] = permeability_values
    nrpts = outputs.shape[0]
    
    idx_dict = update_idx_dict(idx_dict)


    for pname in idx_dict.keys():
        if pname != 'k':
            i = idx_dict[pname]
            data_dict1[pname] = np.zeros((nrpts,data_dict1['k'].shape[0]))
            for r in range(nrpts):
                if pname in ['res','cf']:
                    # interpolate in log space
                    func = interp1d(np.log10(outputs[r,:,idx_dict['k']]),np.log10(outputs[r,:,i]),bounds_error=False)
                    data_dict1[pname][r] = 10**func(np.log10(data_dict1['k']))
                else:
                    # interpolate in linear space (but log permeability)
                    # func = interp1d(outputs[r,:,idx_dict['k']],outputs[r,:,i],bounds_error=False)
                    # data_dict1[pname][r] = func(data_dict1['k'])     
                    func = interp1d(np.log10(outputs[r,:,idx_dict['k']]),outputs[r,:,i],bounds_error=False)
                    data_dict1[pname][r] = func(np.log10(data_dict1['k']))              
            
    return data_dict1


def bulk_permeability(permeability,x_cellsize,cellsize_max,permeability_matrix=1e-18):
    """
    Correct permeability results to a constant fault size

    Parameters
    ----------
    permeability : float or numpy array, shape (n,)
        permeability values from simulation.
    x_cellsize : float or numpy array, shape (n,)
        x cellsize for each permeability (or a single value).
    cellsize_max : float
        maximum cell size to correct to.

    Returns
    -------
    None.

    """
    
    return (permeability*x_cellsize + \
             permeability_matrix*(cellsize_max - x_cellsize))/cellsize_max
        
        
def bulk_resistivity(resistivity,x_cellsize,cellsize_max,
                     res_matrix=1000):
    """
    Correct resistivity results to a constant fault size

    Parameters
    ----------
    resistivity : float or numpy array, shape (n,)
        resistivity values from simulation.
    x_cellsize : float or numpy array, shape (n,)
        x cellsize for each permeability (or a single value).
    cellsize_max : float
        maximum cell size to correct to.

    Returns
    -------
    None.

    """
    
    return cellsize_max/((cellsize_max - x_cellsize)/res_matrix +\
                                     x_cellsize/resistivity)

def bulk_cfraction(cfraction,x_cellsize,cellsize_max):
    return cfraction*x_cellsize/cellsize_max

def hydraulic_aperture(permeability,cellsize_max,permeability_matrix = 1e-18):
    
    # old approach
    # aph = np.sqrt(12*(permeability-(1-conductive_fraction)*permeability_matrix)/conductive_fraction)
    # aph[aph < np.sqrt(12*permeability_matrix)] = np.sqrt(12*permeability_matrix)
    aperture = np.zeros_like(permeability)
    
    with np.nditer(aperture, flags=['multi_index']) as it:
        for x in it:
            i,j = it.multi_index
            # find solutions to the equation d**3/(12*csmax) - km*d/csmax = kb - km
            # where d is hydraulic aperture, csmax is cellsize_max, km is permeability_matrix 
            # and kb is permeability, the bulk permeability of the fracture
            roots = np.roots([1./(12.*cellsize_max),0.00,
                              permeability_matrix/cellsize_max,
                              permeability_matrix-permeability[i,j]])
            
            # we want the real root
            idx = np.where(np.iscomplex(roots)==False)[0][0]  
            
            aperture[i,j] = np.real(roots[idx])
    
    
    return aperture

def getmean(vals,mtype='meanlog10',stdtype='sem',semm=0):
    
    vals = np.log10(vals)
    
    if stdtype =='sem':
        std = sem(vals,axis=0)
    elif stdtype == 'std':
        std = np.std(vals,axis=0)
    
    if mtype=='meanlog10':
        mean = np.nanmean(vals,axis=0)

    elif mtype == 'median':
        mean =  np.nanmedian(vals,axis=0)
                             
    return 10**(mean+semm*std)
