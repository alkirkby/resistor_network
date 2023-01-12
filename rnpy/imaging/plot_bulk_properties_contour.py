# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:14:27 2021

@author: alisonk
"""

from rnpy.functions.readoutputs import get_param, load_outputs
from rnpy.imaging.plotting_tools import prepare_data_dict, prepare_plotdata,\
    clip_by_ca
from rnpy.functions.utils import roundsf
import numpy as np
import matplotlib.pyplot as plt

def plot_contour(fn_list,xparam = 'apm',yparam='offset',zparam='k',clip=0,csmax=None,
            direction='z', plane='yz', label_prefix = '', ca_threshold=None):
    
    fslist= []
    for fn in fn_list:
        outputs = load_outputs(fn)
        
        fslist = np.unique(np.append(fslist,outputs['fault_separation'].flatten()))
    fslist = np.unique([roundsf(val,2) for val in fslist[np.isfinite(fslist)]])
    
    data_dict, output_dtype_names = prepare_data_dict(fn_list,yparam,plane,
                                                      direction,clip=clip,
                                                      fs_list=fslist)

        
    
    data_keys = np.array(list(data_dict.keys()))
    data_keys.sort()
    
    plotx_list = []
    plotz_list = []
    
    # loop through data
    for i, val in enumerate(data_keys):
        # assume all runs used the same matrix permeability/resistivity values
        km = get_param(fn_list[0], 'permeability_matrix')
        if km is None:
            km = 1e-18
        rm = get_param(fn_list[0], 'resistivity_matrix')
        
        
        plotx, zvals, xlabel, zlabel = prepare_plotdata(data_dict[val],xparam,zparam,csmax,
                                        plane,direction,output_dtype_names)
        plotx_list.append(plotx)
        
        plotz = np.nanmedian(zvals,axis=0)

        if ca_threshold is not None:
            ca = data_dict[val]['contact_area'][list('xyz').index(direction)]
            clip_by_ca(plotz,
                       ca,
                       ca_threshold)
                
        plotz_list.append(plotz)
        
    plotx_list = np.array(plotx_list)
    plotz_list = np.array(plotz_list)

    ykey_dict = {'ca':'Contact area','cf':'Conductive fraction',
                 'fs':'fault separation', 'apm':'mean_aperture'}
    if yparam in ykey_dict.keys():
        ylabel= str.capitalize(ykey_dict[yparam])
    else:
        ylabel = str.capitalize(yparam)
    
    
    _,ploty_list = np.meshgrid(plotx,data_keys)

    plotx_list[np.isnan(plotx_list)] = 0.
    
    cmap = 'viridis'
    if zparam == 'k':
        # print(plotz_list)
        levels = np.arange(np.log10(km),np.log10(np.nanmax(plotz_list))+0.1,0.5)
    else:
        levels = np.arange(np.log10(np.nanmin(plotz_list))+0.1,np.log10(rm),0.1)
        cmap += '_r'
        # levels = np.arange(1,2,0.1)
    plt.contour(plotx_list,ploty_list,np.log10(plotz_list),levels=levels,
                cmap=cmap)
    
    logscaleprops = ['apm','cf','offset','elevation_scalefactor']
    if xparam in logscaleprops:
        plt.xscale('log')
    if yparam in logscaleprops:
        plt.yscale('log')
    plt.xlim(np.amin(plotx_list),np.amax(plotx_list))
    plt.ylim(np.amin(ploty_list),np.amax(ploty_list))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    cbar = plt.colorbar()
    cbar.set_label("log10("+zlabel+")")
    