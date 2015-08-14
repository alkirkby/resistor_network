# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:05:42 2014

@author: Alison Kirkby

Modelling random resistor networks using python.

"""
import rnpy.functions.readoutputs as rnro
import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as mplfm


def plot_data(data, input_params, pnames, rnos, parameter_names, 
              plot_bold = None, reference_width = None, direction = 'z',plot_params = {},
              ):
    """
    iterate through all input permutations and repeats and plot some x y points
    
    
    
    """
    pp = dict(c='k',lw=0.1)
        
    pp.update(plot_params)
    


    for vals in itertools.product(*(input_params.values())):
        
        # get the x and y parameters to get the percolation threshold on
        data1,xall,yall,rrat,kbulk,rbulk = rnro.get_xy(data,input_params.keys(),
                                                       vals,parameter_names[0],
                                                       parameter_names[1],
                                                       pnames,
                                                       direction=direction, 
                                                       reference_width=reference_width)
        # cycle through repeat numbers and plot
        for rno in rnos:
            if rno in data1['repeat']:
                x,y,fsep,csx = [arr[data1['repeat']==rno] for arr in [xall,yall,data1['fault_separation'],data1['cellsizex']]]
                x,y,indices = rnro.sort_xy(x,y,fsep)
                plt.plot(x,y,**pp)#
                        
            if plot_bold is not None:
                if rno == plot_bold:
                    plt.plot(x,y,'k-')
            
            
            
                
                
def plot_rk_aperture(wd,filelist,reference_width = None, direction = 'z',
                     input_params = {}):
    """
    plot resistivity and permeability vs aperture
    plotxy: option to plot some xy data (format [[x0,x1,..,xn],[y10,y11,..,y1n],
    [y20,y21,..,y2n]] - two sets of y data for permeability and resistivity axes)
    
    
    """
    
    variables = ['permeability_matrix','resistivity_matrix','resistivity_fluid']
    
    data, input_params2, fixed_params, pnames, rnos = rnro.read_data(wd,filelist,variables)
    input_params2.update(input_params)

    plst = [['aperture','permeability'],['aperture','resistivity']]
    apmin,apmax = np.amin(data['aperture_meanz']), np.amax(data['aperture_meanz'])
    xlim = [aparray for aparray in [apmin,apmax]]
    # calculate y limits so that they are plotted on same scale in log space for
    # resistivity and permeability
    ylim = dict(permeability=[1e-18,1e-8])
    ylim['resistivity'] = [1,10**(np.log10(ylim['permeability'][1])-np.log10(ylim['permeability'][0]))]
    print xlim,ylim
    ylabels = {'resistivity':'Resistivity ratio $\mathrm{\mathsf{R_{matrix}/R_{fracture}}}$',
               'permeability':'Permeability, m$\mathrm{\mathsf{^2}}$'}
    
    plot_params = dict(resistivity = dict(c='b',lw=0.1),
                       permeability = dict(c='k',lw=0.1))
    
    firstplot = True
    ax1 = plt.gca()
    for parameter_names in plst:
        ppdict = plot_params[parameter_names[1]]
        plot_data(data,input_params2,pnames,rnos,parameter_names,
                  reference_width = reference_width, direction = direction,
                  plot_params = ppdict,
                  plot_bold = 0)
        
        plt.xlim(*xlim)
        plt.xscale('log')
        plt.ylim(*ylim[parameter_names[1]])
        plt.yscale('log')
        plt.ylabel(ylabels[parameter_names[1]], color=ppdict['c'])
        plt.gca().tick_params(axis='y', colors=ppdict['c'])
        if firstplot:
            plt.xlabel('Mean aperture, mm')  
            ax2 = plt.twinx()
            firstplot = False
          

    return ax1,ax2

def plot_r_vs_k(wd,filelist,reference_width = None, direction = 'z',
                input_params = {}, limits = None):
    
    variables = ['permeability_matrix','resistivity_matrix','resistivity_fluid']
    
    data, input_params2, fixed_params, pnames, rnos = rnro.read_data(wd,filelist,variables)
    input_params2.update(input_params)

    parameter_names = ['resistivity','permeability']
    xlim = [1e0,1e3]
    ylim = [1e-18,1e-8]

    labels = {'resistivity':'Resistivity ratio $\mathrm{\mathsf{R_{matrix}/R_{fracture}}}$',
               'permeability':'Permeability, m$\mathrm{\mathsf{^2}}$'}
    
    plot_params = dict(c='k',lw=0.1)

    plot_data(data,input_params2,pnames,rnos,parameter_names,
              reference_width = reference_width, direction = direction,
              plot_params = plot_params,
              plot_bold = 0)
    ax1 = plt.gca()
    plt.xlim(*xlim)
    plt.xscale('log')
    plt.ylim(*ylim)
    plt.yscale('log')
    plt.xlabel(labels['resistivity'])     
    plt.ylabel(labels['permeability'])    

    return ax1 
