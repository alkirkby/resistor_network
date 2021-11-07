# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:12:34 2021

@author: alisonk
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import sem

def interpolate_to_all_fs(outputs,idx_dict=None):
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
    data_dict1['fs'] = np.unique(outputs[:,:,0])
    nrpts = outputs.shape[0]
    
    if idx_dict is None:
        idx_dict = {'cf':1,'res':2,'k':3,'xcs':5}

    for pname in idx_dict.keys():
        i = idx_dict[pname]
        data_dict1[pname] = np.zeros((nrpts,data_dict1['fs'].shape[0]))
        for r in range(nrpts):
            func = interp1d(outputs[r,:,0],np.log10(outputs[r,:,i]),bounds_error=False)
            data_dict1[pname][r] = 10**func(data_dict1['fs'])
            
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

def getmean(vals,mtype='meanlog10',semm=0):
    if mtype=='meanlog10':
        return 10**(np.mean(np.log10(vals),axis=0)+semm*sem(np.log10(vals),axis=0))
    elif mtype == 'median':
        return np.median(np.log10(vals)) + semm*sem(np.log10(vals),axis=0)
