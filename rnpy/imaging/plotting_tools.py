# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:18:56 2021

@author: alisonk
"""

from rnpy.functions.readoutputs import get_param,load_outputs,interpolate_to_all_fs,\
    get_perp, resistivity_fault, permeability_fault, bulk_resistivity, bulk_permeability,\
    bulk_cfraction,interpolate_to_permeability_values
from rnpy.functions.utils import get_logspace_array
import numpy as np

def prepare_data_dict(fn_list,plot_by,plane,direction,fs_list=None,clip=0,interpolate_to='fs'):
    
    data_dict = {}
    
    # direction in plane of fault but perpendicular to flow
    other_direction= [dd for dd in plane if dd != direction][0]
    odi = 'xyz'.index(other_direction)
    
    # load data and interpolate to all fault separation values
    for fn in fn_list:
        outputs = load_outputs(fn,clip=clip)
        # get value of parameter to divide up data by e.g. offset
        param = get_param(fn, plot_by)
        
        if plot_by == 'offset':
            # convert offset (as a fraction of fault size) to offset in mm
            # obtain number of cells and cellsize in direction perpendicular to flow
            cs = get_param(fn,'cellsize')[odi]
            nc = get_param(fn,'ncells')[odi]
            # multiply, rounding and converting to mm
            param = np.around(param*cs*nc*1e3,2)
        
        if plot_by in ['ncells','cellsize']:
            param = param[odi]
            if plot_by == 'cellsize':
                param = np.round(param*1e3,2)
        
        if interpolate_to == 'fs':
            data_dict[param] = interpolate_to_all_fs(outputs,fs_list=fs_list)
        elif interpolate_to == 'k':
            kmin = np.array([np.nanmin(outputs['permeability_bulk_'+dd]) for dd in 'xyz'])
            kmax = np.array([np.nanmax(outputs['permeability_bulk_'+dd]) for dd in 'xyz'])
            kvalues = get_logspace_array(np.amin(kmin[kmin!=0]),np.amax(kmax),4)
            data_dict[param] = interpolate_to_permeability_values(outputs,kvalues,direction=direction)
            
        for pp in ['matrix_flow','matrix_current','resistivity_matrix','permeability_matrix']:
            data_dict[param][pp] = get_param(fn,pp)
        if data_dict[param]['permeability_matrix'] is None:
            data_dict[param]['permeability_matrix'] = 1e-18
        

        
    
    return data_dict, outputs.dtype.names

def prepare_plotdata(data,xparam,yparam,csmax,plane,direction,output_dtype_names,interpolate_to='fs'):
    
    perp_direction = get_perp(plane)
    xkey_dict = {'ca':'contact_area','cf':'conductive_fraction',
                 'fs':'fault_separation','xcs':'cellsize_'+perp_direction,
                 'apm':'mean_aperture',
                 'k':'permeability_bulk_'+direction,
                 'res':'resistivity_bulk_'+direction}

    
    
    # get x and y values to plot
    if output_dtype_names is None:
        cf = data['cf']
        xcs = data['xcs']
        if xparam == 'apm':
            plotx = cf*xcs
        else:
            plotx = data[xparam]   
        yvals = data[yparam]
    else:
        cf = data['conductive_fraction']
        xcs = data['cellsize_'+perp_direction]
        if xparam == 'apm':
            plotx = cf*xcs
        else:
            plotx = data[xkey_dict[xparam]] 
        yvals = data[xkey_dict[yparam]]
    if csmax == 'max':
        csmax = np.nanmax(xcs)
                
    # apply any correction required to plot bulk or fault properties
    if csmax is None:
        if interpolate_to == 'k':
            cf = np.mean(cf,axis=0)
        if yparam.startswith('res'):
            # check if simulations included current in the matrix
            if data['matrix_current'] != 'False': 
                yvals = resistivity_fault(yvals,data['resistivity_matrix'],cf)
            xkey_dict['res'] = xkey_dict['res'].replace('bulk_','fault_')
        else:
            # check if simulations included flow in the matrix
            if data['matrix_flow'] != 'False':
                yvals = permeability_fault(yvals,data['permeability_matrix'],cf)
            # else:
            #     print(cf)
            #     yvals[yvals > data['permeability_matrix']] = \
            #         permeability_fault(yvals[yvals > data['permeability_matrix']],
            #                            data['permeability_matrix'],
            #                            cf[yvals > data['permeability_matrix']])
            #     yvals = permeability_fault(yvals,data['permeability_matrix'],cf)
            xkey_dict['k'] = xkey_dict['k'].replace('bulk_','fault_')
    else:
        if interpolate_to == 'k':
            xcs = np.mean(xcs,axis=0)
        if xparam in ['cf','conductive_fraction']:
            plotx = bulk_cfraction(plotx,xcs,csmax)
        if yparam.startswith('res'):
            yvals = bulk_resistivity(yvals,xcs,csmax,resistivity_matrix=data['resistivity_matrix'])
        else:
            yvals = bulk_permeability(yvals,xcs,csmax,permeability_matrix=data['permeability_matrix'])       

    if interpolate_to == 'fs':
        if xparam != 'fs':
            plotx = np.nanmean(plotx,axis=0)

    xlabel = str.capitalize(xkey_dict[xparam]).replace('_',' ')
    ylabel = str.capitalize(xkey_dict[yparam]).replace('_',' ')
        
    return plotx, yvals, xlabel, ylabel


def clip_by_ca(plotz,ca,ca_threshold):
    
    
    ca1 = None
    if np.iterable(ca_threshold):
        if len(ca_threshold) == 2:
            ca0,ca1 = ca_threshold
        else:
            ca0 = ca_threshold[0]
            
    else:
        ca0 = ca_threshold
        
        
    plotz[ca <= ca0] = np.nan
    if ca1 is not None:
        plotz[ca >= ca1] = np.nan
        
        
    return plotz