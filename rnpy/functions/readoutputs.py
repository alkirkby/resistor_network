# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:12:34 2021

@author: alisonk
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import sem
import time
from rnpy.functions.utils import roundsf


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
                   ['permeability_bulk_'+direction for direction in plane]+\
                   ['contact_area','aperture_mean_x','repeat gouge_fraction',
                    'gouge_area_fraction']
        idx_list += [val for val in outputs.dtype.names if val.startswith('resistivity_bulk_')]
        perp_direction = get_perp(plane)
        idx_list += ['cellsize_'+perp_direction]
        idx_list.remove(key_param)
            
        
    return idx_list, idx_dict


def interpolate_to_all(outputs,value_list=None,idx_dict=None, plane = 'yz', 
                       key_param = 'fs'):
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
            else:
                interp_x,interp_y = None,None
                
                
        # if pname in outputs.dtype.names:
        if interp_x is not None:
            data_dict1[pname] = np.zeros((nrpts,data_dict1[key_param].shape[0]))
            for r in range(nrpts):
                if pname.split('_')[0] in ['res','k','permeability','resistivity']:
                    # interpolate resistivity and permeability in log space
                    func = interp1d(interp_x[r],np.log10(interp_y[r]),bounds_error=False)
                    data_dict1[pname][r] = 10**func(data_dict1[key_param])
    
                else:
                    # interpolate other parameters in linear space
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



def interpolate_to_all_apertures(outputs, rounding=9, rounding_sf=2):
    '''
    

    Parameters
    ----------
    outputs : structured numpy array, outputs from a resistor network simulation, 
                produced by using load_outputs function.
                must have columns named 'cellsize_x' and 'conductive_fraction'.
            Apertures are calculated from cellsize_x and conductive_fraction
            then rounded to a specified number of both significant figures
            and decimal places.
    rounding : integer, optional
        number of decimal places to round to. The default is 9.
    rounding_sf : integer, optional
        number of significant figures to round to. The default is 2.

    Returns
    -------
    new_outputs : structured numpy array
        Array containing the outputs in input array, interpolated to a common
        set of apertures.

    '''

    repeats = outputs.shape[0]
    aperture = outputs['cellsize_x']*outputs['conductive_fraction']
    aperture_i = np.unique([roundsf(val,3) for val in np.unique(np.around(aperture,9))])
    
    new_outputs = np.zeros((repeats, len(aperture_i)),
                           dtype=outputs.dtype.descr + [('aperture','<f8')])
    t0 = time.time()
    for i in range(repeats):
        
        new_outputs['aperture'] = aperture_i
        for name in new_outputs.dtype.names:
            if name != 'aperture':
                func = interp1d(aperture[i],outputs[name][i],bounds_error=False)
                new_outputs[name][i] = func(aperture_i)
    t1 = time.time()
    print('interpolation took %.1f s'%(t1-t0))
    return new_outputs

def compute_average_parameters(outputs, average_type='mean', percentile=16):
    new_outputs = np.zeros(outputs.shape[1],
                           dtype = outputs.dtype)
    for name in outputs.dtype.names:
        args = dict(axis=0)
        function = getattr(np, 'nan' + average_type)
        
        if average_type == 'percentile':
            args['q'] = percentile
        new_outputs[name] = function(outputs[name], **args)
    return new_outputs


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
                     resistivity_matrix=1000):
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
    
    return cellsize_max/((cellsize_max - x_cellsize)/resistivity_matrix +\
                                     x_cellsize/resistivity)

def bulk_cfraction(cfraction,x_cellsize,cellsize_max):
    return cfraction*x_cellsize/cellsize_max



def resistivity_fault(resistivity_bulk, resistivity_matrix, porosity):
    return porosity/(1./resistivity_bulk - (1.-porosity)/resistivity_matrix)


def permeability_fault(permeability_bulk, permeability_matrix, porosity):
    
    kf = (permeability_bulk - (1.-porosity)*permeability_matrix)/porosity

    filt = np.abs(permeability_bulk - permeability_matrix)/permeability_matrix < 1e-6
    
    
    if np.iterable(filt):
        kf = np.array(kf)
        if np.iterable(permeability_matrix):
            for i in range(3):
                kf[:,i][filt[:,i]] = permeability_matrix[i]
        else:
            kf[filt] = permeability_matrix
    else:
        if filt:
            kf = permeability_matrix * 1.0
        
    return kf


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
        std = sem(vals,axis=0,nan_policy='omit')
    elif stdtype == 'std':
        std = np.nanstd(vals,axis=0)
    
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
                if len(line) > 100:
                    header_names = line.strip().split()[1:]
                    return header_names


def load_outputs(fn,clip=0,**genfromtxt_args):
    """
    

    Parameters
    ----------
    fn : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    outputs = np.genfromtxt(fn,names=read_header(fn),**genfromtxt_args)
    
    nr = len(np.unique(outputs['repeat']))
    nfs = int(len(outputs)/nr)
    outputs = outputs.reshape(nr,nfs)

    # clip data
    return outputs[:,:outputs.shape[1]-clip]



def add_aperture_2d(outputs):
    # add mean aperture
    new_dtype = outputs.dtype.descr + [('aperture_mean_x','<f8')]
    outputs_new = np.zeros(outputs.shape,dtype=new_dtype)
    for ff in outputs.dtype.names:
        outputs_new[ff] = outputs[ff]
    outputs_new['aperture_mean_x'] = outputs['cellsize_x']*outputs['conductive_fraction']

    return outputs_new

def get_cellsize_suffix(cellsize):
    if cellsize >= 0.01:
        cs_suffix = '%1icm'%(cellsize * 100)
    else:
        cs_suffix = '%1imm'%(cellsize * 1000)
        
    return cs_suffix

def read_fault_params_json(jsonfn, cellsize):
 
    import json
    with open(jsonfn) as infile: 
        fault_k_ap = json.load(infile)

    lvals_center = np.array(list(fault_k_ap.keys())).astype(float)
    
    fw = np.array([fault_k_ap[key]['mean_aperture'] for key in fault_k_ap.keys()])
    
    resistivity = np.array([fault_k_ap[key]['mean_resistivity_bulk_%s'%get_cellsize_suffix(cellsize)]\
                            for key in fault_k_ap.keys()])
    fault_k = np.array([fault_k_ap[key]['mean_fault_k'] for key in fault_k_ap.keys()])
    aph = (12*fault_k)**0.5
    
    return lvals_center, fw, aph, resistivity

def read_fault_params_npy(npyfile, cellsize):
    fault_k_ap = np.load(npyfile)
    print(npyfile)
    print(fault_k_ap.dtype.names)
    if cellsize is None:
        resistivity = fault_k_ap['resistivity_fault']
    else:
        resistivity = fault_k_ap['resistivity_bulk_%s'%get_cellsize_suffix(cellsize)]
        
    return fault_k_ap['length_m'],  fault_k_ap['mean_aperture'],\
        (12*fault_k_ap['permeability_fault'])**0.5, \
        resistivity
    
    

def read_fault_params(fn, cellsize):
    if fn.endswith('.npy'):
        return read_fault_params_npy(fn,cellsize)
    else:
        return read_fault_params_json(fn,cellsize)
    
    
def get_equivalent_rho(rho,width,equivalent_width, rho_matrix=1000):
    '''
    Get equivalent resistivity over a defined width

    Parameters
    ----------
    rho : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    equivalent_width : TYPE
        DESCRIPTION.
    rho_matrix : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    None.

    '''
    pad = equivalent_width - width
    
    return equivalent_width/(pad/rho_matrix + width/rho)