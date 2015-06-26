# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681

functions dealing with assigning properties to arrays, including faults, 
fault apertures, permeability, resistivity, etc

- adding nulls to correct edges of a fault, resistivity, permeability or aperture array 
- adding fault to an array

"""
from __future__ import division, print_function
import numpy as np
import rnpy.functions.faultaperture as rnfa 
    
def get_electrical_resistivity(aperture_array,r_matrix,r_fluid,cs):
    """
    
    returns a numpy array containing resistance values 
    
    =================================inputs====================================
    aperture_array = array containing fault apertures
    r_matrix, r_fluid = resistivity of matrix and fluid
    cs = list containing cell size (length of connector) in x,y and z directions 
    [dx,dy,dz] or float/integer if d is the same in all directions
    
    ===========================================================================
    """
    
    res_array = np.zeros_like(aperture_array)
    if type(cs) in [float,int]:
        cs = [float(cs)]*3

    # ln, the width normal to the cell
    ln = [cs[2],cs[0],cs[1]]

    for i in range(3):
        res_array[:,:,:,i] = ln[i]/((ln[i]-aperture_array[:,:,:,i])/r_matrix +\
                                           aperture_array[:,:,:,i]/r_fluid)    

    return res_array    
    

def get_electrical_resistance(aperture_array,r_matrix,r_fluid,d):
    """
    
    returns a numpy array containing resistance values 
    
    =================================inputs====================================
    aperture_array = array containing fault apertures
    r_matrix, r_fluid = resistivity of matrix and fluid
    d = list containing cell size (length of connector) in x,y and z directions 
    [dx,dy,dz] or float/integer if d is the same in all directions
    
    ===========================================================================
    """
    
    res_array = np.zeros_like(aperture_array)
    if type(d) in [float,int]:
        d = [float(d)]*3
        
    # ly, the width of each cell
    ly = [d[1],d[2],d[0]]
    # ln, the width normal to the cell
    ln = [d[2],d[0],d[1]]

    for i in range(3):
        res_array[:,:,:,i] = 1./((ln[i]-aperture_array[:,:,:,i])/r_matrix +\
                                        aperture_array[:,:,:,i]/r_fluid)*d[i]/ly[i]

    return res_array


def get_permeability(aperture_array,k_matrix,d):
    """
    calculate permeability based on an aperture array
    
    =================================inputs====================================
    aperture_array = array containing fault apertures
    k_matrix = permeability of matrix
    d = list containing cell size (length of connector) in x,y and z directions 
    [dx,dy,dz] or float/integer if d is the same in all directions
    ===========================================================================    
    """
    permeability_array = np.ones_like(aperture_array)*k_matrix        
    if type(d) in [float,int]:
        d = [float(d)]*3

    # ln, the width normal to the cell
    ln = [d[2],d[0],d[1]]

    for i in range(3):
        permeability_array[:,:,:,i] = aperture_array[:,:,:,i]**2/12. + \
                                      (ln[i]-aperture_array[:,:,:,i])*k_matrix
   
    
    return permeability_array


def get_hydraulic_resistance(aperture_array,k_matrix,d,mu=1e-3):
    """
    calculate hydraulic resistance based on a hydraulic permeability array
    
    =================================inputs====================================
    aperture_array = array containing fault apertures
    k_matrix = permeability of matrix
    d = list containing cell size (length of connector) in x,y and z directions 
    [dx,dy,dz] or float/integer if d is the same in all directions
    mu = viscosity of fluid
    ===========================================================================
    
    """
    hydraulic_resistance = np.ones_like(aperture_array)
   
    if type(d) in [float,int]:
        d = [float(d)]*3

    # ly, the width of each cell
    ly = [d[1],d[2],d[0]]
    # ln, the width normal to the cell
    ln = [d[2],d[0],d[1]]


    aperture_array[(np.isfinite(aperture_array))&(aperture_array < 1e-50)] = 1e-50

    for i in range(3):
        hydraulic_resistance[:,:,:,i] = mu*d[i]/(ly[i]*(aperture_array[:,:,:,i]**3/12.\
                                        +k_matrix*(ln[i]-aperture_array[:,:,:,i])))
    
    aperture_array[(np.isfinite(aperture_array))&(aperture_array <= 1e-50)] = 0

    return hydraulic_resistance


def get_geometry_factor(output_array,cellsize):
    """
    
    """
    if type(cellsize) in [int,float]:
        dx,dy,dz = [cellsize]*3
    elif type(cellsize) in [list,np.ndarray]:
        if len(cellsize)==3:
            dx,dy,dz = cellsize
        else:
            dx,dy,dz = [cellsize[0]]*3
            
    nz,ny,nx = np.array(np.shape(output_array))[:3] - 2

    
    return np.array([dz*dy*(ny+1)*(nz+1)/(dx*nx),
                     dz*dx*(nx+1)*(nz+1)/(dy*ny),
                     dy*dx*(nx+1)*(ny+1)/(dz*nz)])


def get_flow(output_array):
    print(np.array([np.sum(output_array[:,:,-1,0,0]),
                     np.sum(output_array[:,-1,:,1,1]),
                     np.sum(output_array[-1,:,:,2,2])]))
    return np.array([np.sum(output_array[:,:,-1,0,0]),
                     np.sum(output_array[:,-1,:,1,1]),
                     np.sum(output_array[-1,:,:,2,2])])


def get_bulk_resistivity(current_array,cellsize):
    
    factor = get_geometry_factor(current_array,cellsize)
    flow = get_flow(current_array)
    
    resistance = 1./flow
    
    return factor*resistance, resistance 


def get_bulk_permeability(flowrate_array,cellsize,fluid_viscosity):

    factor = get_geometry_factor(flowrate_array,cellsize)
    flow = get_flow(flowrate_array)   

    resistance = 1./flow
    
    return fluid_viscosity/(resistance*factor),resistance
