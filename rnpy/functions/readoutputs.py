# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:12:34 2021

@author: alisonk
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import sem


def get_perp(plane):
    return [val for val in 'xyz' if val not in plane][0]

def update_idx_dict(idx_dict):
    if idx_dict is None:
        idx_dict = {'fs':0,'cf':1,'res':2,'k':3,'xcs':5}
    else:
        idx_dict2 = {'fs':0,'cf':1,'res':2,'k':3,'xcs':5}
        idx_dict2.update(idx_dict)
        idx_dict = idx_dict2
        
    return idx_dict
    
def get_idx_list(outputs,idx_dict,key_param='fs',plane='yz', direction='z'):
        
    if outputs.dtype.names is None:
        idx_dict = update_idx_dict(idx_dict)
        idx_list = list(idx_dict.keys())
        idx_list.remove(key_param)
        
    else:
        idx_list = ['fault_separation','conductive_fraction']+\
                   ['resistivity_bulk_'+direction for direction in plane]+\
                   ['permeability_bulk_'+direction for direction in plane]+\
                   ['contact_area']
        perp_direction = get_perp(plane)
        idx_list += ['cellsize_'+perp_direction]
        idx_list.remove(key_param)
            
        
    return idx_list, idx_dict


def interpolate_to_all(outputs,value_list=None,idx_dict=None, plane = 'yz', key_param = 'fs'):
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

    # get number of repeats
    nrpts = outputs.shape[0]

    # get index list to loop through
    idx_list, idx_dict = get_idx_list(outputs,idx_dict,key_param=key_param)

    # assign fixed values that we are interpolating variables to
    # get from outputs if not provided
    if value_list is None:
        
        if outputs.dtype.names is None:
            ifs = idx_dict[key_param]
            data_dict1[key_param] = np.unique(outputs[:,:,ifs])
        else:
            ifs = key_param
            data_dict1[key_param] = np.unique(outputs[ifs])      
    else:
        data_dict1[key_param] = value_list
    
    
    for pname in idx_list:
        if outputs.dtype.names is None:
            interp_x = outputs[:,:,idx_dict[key_param]]
            interp_y = outputs[:,:,idx_dict[pname]]
        else:
            if pname in outputs.dtype.names:
                interp_x = outputs[key_param]
                interp_y = outputs[pname]
                
                
        # if pname in outputs.dtype.names:
        data_dict1[pname] = np.zeros((nrpts,data_dict1[key_param].shape[0]))
        for r in range(nrpts):
            if pname.split('_')[0] in ['res','k','permeability','resistivity']:
                # interpolate resistivity and permeability in log space
                func = interp1d(interp_x[r],np.log10(interp_y[r]),bounds_error=False)
                data_dict1[pname][r] = 10**func(data_dict1[key_param])

            else:
                # interpolate other parametersin linear space
                func = interp1d(interp_x[r],interp_y[r],bounds_error=False)
                data_dict1[pname][r] = func(data_dict1[key_param])   
            
    return data_dict1

def interpolate_to_all_fs(outputs,fs_list=None,idx_dict = None, plane='yz'):
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
    if outputs.dtype.names is None:
        key_param = 'fs'
    else:
        key_param = 'fault_separation'
        
    return interpolate_to_all(outputs,
                              value_list=fs_list,
                              key_param=key_param,
                              idx_dict=idx_dict,
                              plane=plane)    


def interpolate_to_permeability_values(outputs,permeability_values,idx_dict=None,direction='z',plane='yz'):
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
    if outputs.dtype.names is None:
        key_param = 'k'
    else:
        key_param = 'permeability_bulk_'+direction
    
    return interpolate_to_all(outputs,
                              value_list=permeability_values,
                              key_param=key_param,
                              idx_dict=idx_dict,
                              plane=plane)


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



def resistivity_fault(resistivity_bulk, resistivity_matrix, porosity):
    return porosity/(1./resistivity_bulk - (1.-porosity)/resistivity_matrix)


def permeability_fault(permeability_bulk, permeability_matrix, porosity):
    return (permeability_bulk - (1.-porosity)*permeability_matrix)/porosity


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


def get_param(fn,param_name):
    """
    

    Parameters
    ----------
    fn : str
        full path to output file.
    param_name : str
        name of parameter to get

    Returns
    -------
    Parameter value used in simulation file provided

    """

    with open(fn) as openfile:
        for line in openfile:
            if param_name in line:
                paramlist = line.strip().split()[2:]
                try:
                    paramlist = [float(val) for val in paramlist]
                except:
                    pass
    
                if len(paramlist) == 1:
                    paramlist = paramlist[0]
                if param_name == 'ncells':
                    paramlist = [int(val) for val in paramlist]
                    
                return paramlist
    
def _get_ncols(fn,delimiter=' '):
    with open(fn) as openfile:
        for line in openfile:
            if not line.startswith('#'):
                line = openfile.readline()
                return len(line.split(delimiter))

def read_header(fn):
    ncols = _get_ncols(fn)
    with open(fn) as openfile:
        for line in openfile:
            if len(line.split()) == ncols + 1:
                if 'fault_separation' in line:
                    header_names = line.strip().split()[1:]
                    return header_names


def load_outputs(fn,clip=0):
    """
    

    Parameters
    ----------
    fn : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    outputs = np.genfromtxt(fn,names=read_header(fn))

    nr = len(np.unique(outputs['repeat']))
    nfs = int(len(outputs)/nr)
    outputs = outputs.reshape(nr,nfs)
    # clip data
    return outputs[:,:outputs.shape[1]-clip]

