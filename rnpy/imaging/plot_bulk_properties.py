# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:33:59 2021

@author: alisonk
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from rnpy.functions.readoutputs import get_param, getmean
    
from rnpy.imaging.plotting_tools import prepare_data_dict, prepare_plotdata


def _get_colors():
    # get default matplotlib plotting cycle
    colors= plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # add randomly picked colors from matplotlib named colors (dark only)
    css4_colors_dark = [colorval for colorval in list(mcolors.CSS4_COLORS.values())\
                    if not np.all(np.array(hex2rgb(colorval)) > 150)]
    rseeds = [14, 75, 58, 17, 44, 81, 3, 26, 99, 89, 47, 60, 1, 88, 101, 50, 22]
    colors += list(np.array(css4_colors_dark)[np.array(rseeds)])
    
    return colors

def hex2rgb(hexstr):
    return tuple(int(hexstr.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))


def plot_xy(fn_list,xparam = 'apm',yparam='k',clip=0,plot_by='offset',csmax=None,
            direction='z', plane='yz', range_type='percentile',range_num=None,
            label_prefix = ''):
    """
    

    Parameters
    ----------
    fn_list : list
        List of files to plot.
    xparam : str
        Parameter to plot on x axis. Options are cf (porosity or conductive
        fraction), apm (mean aperture), ca (contact area) or fs (fault 
        separation).
    yparam : str
        Parameter to plot on y axis. Options are k (permeability) or res 
        (resistivity)
    plot_by : str, optional
        Parameter that is varied in the file list to sort outputs by. The 
        default is 'offset'.
    csmax : str, NoneType, or float
        Width in direction perpendicular to plot averaged resistivity or
        permeability values for. If None, plots the resistivity/permeability
        of the fault. If set to max, gets the maximum x cellsize from all the
        data.
    direction : str, optional
        Direction of permeability simulation to plot. The default is 'z'.
    range_type : what sort of range interval to plot. Options are percentile
        (to plot the median and a certain percentile either side of the mean)
        or sem (to plot mean and a certain number of standared errors either
        side of the mean)
    range_num : percentile value or number of standard deviations to show

    Returns
    -------
    None.

    """

    colors = _get_colors()
    
    
    if range_num is None:
        if range_type == 'percentile':
            range_num = 34
        else:
            range_num = 1
    
    data_dict, output_dtype_names = prepare_data_dict(fn_list,plot_by,plane,direction,clip=clip)
        
    
    data_keys = np.array(list(data_dict.keys()))
    data_keys.sort()
    
    # loop through data
    for i, val in enumerate(data_keys):
        # assume all runs used the same matrix permeability/resistivity values
        km = get_param(fn_list[0], 'permeability_matrix')
        if km is None:
            km = 1e-18
        rm = get_param(fn_list[0], 'resistivity_matrix')
        
        
        plotx, yvals, xlabel, ylabel = prepare_plotdata(data_dict[val],xparam,yparam,csmax,rm,km,
                                        plane,direction,output_dtype_names)
            

        

            
        # compute min, max and mean/median values to plot
        # print(yvals[:,0],yvals[:,-1])
        if range_type == 'percentile':
            y0,y,y1 = [np.percentile(yvals,perc,axis=0) for perc in \
                       [50-range_num, 50, 50+range_num]]
        elif range_type == 'sem':
            y0,y,y1 = [getmean(yvals,mtype='meanlog10',stdtype='sem',semm=i) \
                       for i in [-range_num,0,range_num]]
                
        # print(y)

        label = label_prefix + '%s = %s'%(plot_by,val)
        if plot_by in ['offset','cellsize']:
            label += 'mm'

        plt.plot(plotx, y, color=colors[i], label=label)
        plt.fill_between(plotx, y0, y1, alpha=0.5, color=colors[i])
        plt.yscale('log')
        
        if xparam not in ['ca','fs','contact_area','fault_separation']:
            plt.xscale('log')
            
        if xparam == 'cf':
            plt.xlim(1e-5,1e0)
        if xparam == 'apm':
            plt.xlim(1e-8,1e-2)
            
        plt.legend(fontsize=8)
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    return
        