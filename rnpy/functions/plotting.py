# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681
"""

import numpy as np

def get_meshlocations(d,n):
    """
    get locations of nodes for plotting
    
    n = list containing number of cells in x and z directions [nx,nz]
    d = list containing cell size in x and z directions [dx,dz]    
    
    """

    
    plotx = np.linspace(-d[0],d[0]*n[0],n[0]+2)
    plotz = np.linspace(0.,d[1]*(n[1]+1),n[1]+2)
    
    return np.meshgrid(plotx,plotz)


    
def get_direction(property_array):
    """
    get the direction of the current and fluid flow for plotting.
    
    1 means flow/current is down/right or zero
    -1 means flow/current is up/left     
    
    property_array =numpy array containing property to be evaluated
    
    """
    parray = 1.*property_array
    parray[parray==0.] = 1.
    direction = parray/(np.abs(parray))
    
    return direction


def get_quiver_origins(d,plotxz,parameter):
    """
    get the locations of the origins of the quiver plot arrows.
    These are slightly different from the mesh nodes because where arrows are 
    negative, the arrow goes in the opposite direction so needs to be 
    shifted by dx/dy
    
    d = list containing cell size in x and z directions [dx,dz]    
    plotxz = list containing mesh locations from get_meshlocations
    parameter = array containing current or fluid flow values, dimensions
    [nx+2,nz+2,2]
    
    """
    
    # initialise qplotx
    qplotxz = np.zeros_like(plotxz)
    
    # calculate shifts first
    # if x arrows point left (negative), shift starting location right by dx
    qplotxz[0][get_direction(parameter[:,:,0]) < 0.] += d[0]
    # if z arrows point down (positive), shift starting location up by dz
    qplotxz[1][-get_direction(parameter[:,:,1]) > 0.] -= d[1]       
    # add plotxz to qplotxz to give absolute starting locations for arrows
    for i in range(2):
        qplotxz[i] += plotxz[i]

    return [[qplotxz[0],plotxz[0]],[plotxz[1],qplotxz[1]]]
    

def get_quiver_UW(parameter,plot_range=None):
    """
    take an array containing inputs/outputs and prepare it for input into
    a quiver plot with separate horizontal and vertical components.
    Includes removal of data outside given range
    
    d = list containing cell size in x and z directions [dx,dz]
    qplotxz = length 2 list of arrays containing quiver plot origins for arrows,
    shape of each array is (nx+2,nz+2)
    parameter = array containing current or fluid flow values, dimensions
    [nx+2,nz+2,2]
    
    """
    
    # get U, W and C. All length 2 lists containing plot data for horizontal 
    # and vertical arrows respectively
    U = [get_direction(parameter[:,:,0]),np.zeros_like(parameter[:,:,1])]
    W = [np.zeros_like(parameter[:,:,0]),-get_direction(parameter[:,:,1])]
    C = [np.abs(parameter[:,:,0]),np.abs(parameter[:,:,1])]
    
    
    
    # remove arrows outside of specified plot range
    
    for i in range(2):
        if plot_range is not None:
            C[i][C[i]<plot_range[0]] = np.nan
            C[i][C[i]>plot_range[1]] = np.nan

    return U,W,C


def get_faultlengths(parameter,d,tolerance=0.05):
    """
    gets "fault" lengths for a conductivity/permeability array
    returns a list of 2 arrays with fault lengths in the x and z directions
    
    parameter = array (shape nx,nz,2) containing values in x and z directions
    values can be anything but highest values are considered faults
    d = list containing dx,dz values (cell size in x and z direction)
    tolerance = how close value has to be compared to minimum array value to be
    considered a fault
    
    """
    parameter[np.isnan(parameter)] = 0.
    cx,cz = [parameter[:,:,i] for i in [0,1]]

    faultlengths = []
    
    fault_value = np.amax(np.isfinite(parameter))
    
    for ii,conductivity in enumerate([cx,cz.T]):
        faultlengths.append([])
        # turn it into a true/false array
        faults = np.zeros_like(conductivity)
        faults[conductivity > (1.-tolerance)*fault_value] = 1.
        faults = faults.astype(bool)
        
        for line in faults:
            # initialise a new fault at the beginning of each line
            newfault = True
            for j in range(len(line)):
                # check if faulted cell
                if line[j]:
                    if newfault:
                        fl = d[ii]
                        newfault = False
                    else:
                        fl += d[ii]
                    # check if we are at the end of a fault
                    if (j == len(line)-1) or (not line[j+1]):      
                        faultlengths[ii].append(fl)
                        newfault = True               
        faultlengths[ii] = np.around(faultlengths[ii],
                                     decimals=int(np.ceil(-np.log10(d[ii]))))

    return faultlengths
