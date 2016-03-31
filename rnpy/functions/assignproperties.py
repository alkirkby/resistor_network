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
    
    

def get_electrical_resistance(aperture_array,r_matrix,r_fluid,d):
    """
    
    returns a numpy array containing resistance values and an array containing 
    resistivities 
    
    =================================inputs====================================
    aperture_array = array containing fault apertures
    r_matrix, r_fluid = resistivity of matrix and fluid
    d = list containing cell size (length of connector) in x,y and z directions 
    [dx,dy,dz] or float/integer if d is the same in all directions
    
    ===========================================================================
    """
    
    resistance_array = np.zeros(np.shape(aperture_array)[:-1])
    resistivity_array = resistance_array.copy()
    if type(d) in [float,int]:
        d = [float(d)]*3

    for i in range(3):
        # the two directions perpendicular to direction of flow, indices and cell sizes
        dpi = [dd for dd in range(3) if dd != i]
        dp = [d[dd] for dd in dpi]
        # cross sectional area of the cell perpendicular to flow
        area_matrix = np.product(dp)
        area_fracture = np.zeros_like(aperture_array[:,:,:,0,0])
        for ii in range(2):
            # define the area taken up by the fracture
            area_fracture += aperture_array[:,:,:,i,dpi[ii]]*d[dpi[1-ii]]
        # subtract the overlapping bit if there is any
        area_fracture-= aperture_array[:,:,:,i,0]*aperture_array[:,:,:,i,1]
        
        # subtract fracture area from matrix area and remove any negative matrix area
        area_matrix -= area_fracture
        area_matrix[area_matrix<0.] = 0.
        # resistance is the weighted harmonic mean of the fractured bit (in the two
        # directions along flow) and the matrix bit
#        if len(area_fracture[np.isfinite(area_fracture)]) >0:
#            print(np.amax(area_fracture[np.isfinite(area_fracture)]))
#        else:
#            print('\n')
        resistance_array[:,:,:,i] = d[i]/(area_matrix/r_matrix + area_fracture/r_fluid)
        resistivity_array[:,:,:,i] = (area_fracture + area_matrix)/(area_matrix/r_matrix + area_fracture/r_fluid)

        
    return resistance_array,resistivity_array


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
    permeability_array = np.ones(np.shape(aperture_array)[:-1])*k_matrix        
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
    hresistance = np.zeros(np.shape(aperture_array)[:-1])
    permeability = hresistance.copy()
    
    if type(d) in [float,int]:
        d = [float(d)]*3
        
        
    for i in range(3):
        # the two directions perpendicular to direction of flow, indices and values
        dpi = [dd for dd in range(3) if dd != i]
        dp = [d[dd] for dd in dpi]
        # cross sectional area of the cell perpendicular to flow
        area_matrix = np.product(dp)
        area_fracture = np.zeros_like(aperture_array[:,:,:,0,0])
        for ii in range(2):
            # subtract the area taken up by the fracture
            area_fracture += aperture_array[:,:,:,i,dpi[ii]]*d[dpi[1-ii]]
        # subtract the overlapping bit if there is any
        area_fracture-= aperture_array[:,:,:,i,0]*aperture_array[:,:,:,i,1]
        
        # subtract fracture area from matrix area and remove any negative matrix area
        area_matrix -= area_fracture
        area_matrix[area_matrix<0.] = 0.    
        # permeability is the weighted mean of the fractured bit (in the two
        # directions along flow) and the matrix bit
        hresistance[:,:,:,i] = mu*d[i]/(d[dpi[1]]*aperture_array[:,:,:,i,dpi[0]]**3/12. +\
                                        d[dpi[0]]*aperture_array[:,:,:,i,dpi[1]]**3/12. +\
                                        area_matrix*k_matrix)
        permeability[:,:,:,i] = mu*d[i]/(hresistance[:,:,:,i]*(area_fracture + area_matrix))


    return hresistance,permeability


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
    #print(np.array([np.sum(output_array[:,:,-1,0,0]),
    #                 np.sum(output_array[:,-1,:,1,1]),
    #                 np.sum(output_array[-1,:,:,2,2])]))

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
