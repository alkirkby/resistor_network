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
import scipy.optimize as so
    
    

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
    ncells = (np.array(aperture_array.shape)[:-2] - 2)[::-1]
    
    if type(d) in [float,int]:
        d = [float(d)]*3
    
    
    # connectors
    for i in range(3):
        # the two directions perpendicular to direction of flow, indices and cell sizes
        dpi = [dd for dd in range(3) if dd != i]
        dp = [d[dd] for dd in dpi]
    
        # first need to correct apertures to spread out any that are wider than the cellsize
        # direction of opening, cycle through faults opening in x, y and z directions
        
        for od in dpi:
    #        print i,od,len(aperture_array[:,:,:,i,od][aperture_array[:,:,:,i,od] > d[od]])
            for k1,j1,i1 in np.array(np.where(aperture_array[:,:,:,i,od] > d[od])).T:
                ncf = 3
                apval = aperture_array[k1,j1,i1,i,od]
                while ncf < ncells[od]:
                    # check if aperture is less than ncf * cellsize, if not add 2 (spread the fault further)
                    if apval < ncf*d[od]:
                        break
                    ncf += 2
    
                # aperture value for the side bits
                apedge = (apval - (ncf-2)*d[od])/2.
    
                # set the extra bits, first the internal points just set to minres
                ind = int(ncf/2) - 1
                
    
                if od == 0:
                    ind0,ind2 = max(1,i1-ind),min(ncells[0]+1,i1+1+ind)
                    aperture_array[k1,j1,ind0:i1,i,od] = d[od]
                    aperture_array[k1,j1,i1+1:ind2,i,od] = d[od]
                    aperture_array[k1,j1,ind0-1,i,od] += apedge
                    aperture_array[k1,j1,ind2,i,od] += apedge
                    
                    # update the connectors perpendicular to plane
                    aperture_array[k1,j1,ind0-1:ind2,0,0] = d[od]
                    if i == 1:
                        aperture_array[k1,j1+1,ind0-1:ind2,0,0] = d[od]
                    elif i == 2:
                        aperture_array[k1+1,j1,ind0-1:ind2,0,0] = d[od] 
                    
                elif od == 1:
                    ind0,ind2 = max(1,j1-ind),min(ncells[0]+1,j1+1+ind)
                    aperture_array[k1,ind0:j1,i1,i,od] = d[od]
                    aperture_array[k1,j1+1:ind2,i1,i,od] = d[od]
                    aperture_array[k1,ind0-1,i1,i,od] += apedge
                    aperture_array[k1,ind2,i1,i,od] += apedge
                    
                    # update the connectors perpendicular to plane
                    aperture_array[k1,ind0-1:ind2,i1,1,1] = d[od]
                    if i == 0:
                        aperture_array[k1,ind0-1:ind2,i1+1,1,1] = d[od]
                    elif i == 2:
                        aperture_array[k1+1,ind0-1:ind2,i1,1,1] = d[od]
    
                elif od == 2:
                    ind0,ind2 = max(1,k1-ind),min(ncells[0]+1,k1+1+ind)
                    aperture_array[ind0:k1,j1,i1,i,od] = d[od]
                    aperture_array[k1+1:ind2,j1,i1,i,od] = d[od]  
                    aperture_array[ind0-1,k1,i1,i,od] += apedge
                    aperture_array[ind2,k1,i1,i,od] += apedge 
                    
                    # update the connectors perpendicular to plane
                    aperture_array[ind0-1:ind2,j1,i1,2,2] = d[od]
                    if i == 0:
                        aperture_array[ind0-1:ind2,j1,i1+1,2,2] = d[od]
                    elif i == 1:
                        aperture_array[ind0-1:ind2,j1+1,i1,2,2] = d[od]
            
            
            aperture_array[:,:,:,i,od][aperture_array[:,:,:,i,od] > d[od]] = d[od]
    
        # cross sectional area of the cell perpendicular to flow
        area_matrix = np.product(dp)
        area_fracture = np.zeros_like(aperture_array[:,:,:,0,0])
        for ii in range(2):
            # define the area taken up by the fracture
            area_fracture += aperture_array[:,:,:,i,dpi[ii]]*d[dpi[1-ii]]
        # subtract the overlapping bit if there is any
        area_fracture-= aperture_array[:,:,:,i,dpi[0]]*aperture_array[:,:,:,i,dpi[1]]
        
        # subtract fracture area from matrix area and remove any negative matrix area
        area_matrix -= area_fracture
        area_matrix[area_matrix<0.] = 0.
        
        # resistance is the weighted harmonic mean of the fractured bit (in the two
        # directions along flow) and the matrix bit
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
        permeability_array[:,:,:,i] = (aperture_array[:,:,:,i]**3/12. + \
                                      (ln[i]-aperture_array[:,:,:,i])*k_matrix)/ln[i]
   
    
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
        area_fracture-= aperture_array[:,:,:,i,dpi[ii]]*aperture_array[:,:,:,i,dpi[ii-1]]
        
        # subtract fracture area from matrix area and remove any negative matrix area
        area_matrix -= area_fracture
        area_matrix[area_matrix<0.] = 0.    
        # permeability is the weighted mean of the fractured bit (in the two
        # directions along flow) and the matrix bit
        # first calculate the hydraulic resistance
        hresistance[:,:,:,i] = mu*d[i]/(d[dpi[1]]*aperture_array[:,:,:,i,dpi[0]]**3/12. +\
                                        d[dpi[0]]*aperture_array[:,:,:,i,dpi[1]]**3/12. +\
                                        area_matrix*k_matrix)
        permeability[:,:,:,i] = mu*d[i]/(hresistance[:,:,:,i]*(area_fracture + area_matrix))


    return hresistance,permeability


def get_hydraulic_resistivity(hresistance,cellsize):
    """
    get hydraulic resistivity (equivalent to electrical resistivity) for
    putting into solver
    
    hresistance = hydraulic resistance array
    cellsize = tuple,list or array containing cellsize in x, y and z direction
    
    """
    # initialise new array, keeping nulls in correct spots    
    hresistivity = hresistance*0.
    
    for i in range(3):
        dpi = [cellsize[dd] for dd in range(3) if dd != i]
        hresistivity[:,:,:,i] = hresistance[:,:,:,i]*np.product(dpi)/cellsize[i]

        
    return hresistivity


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


def get_bulk_resistivity(current_array,cellsize,deltaV):
    
    factor = get_geometry_factor(current_array,cellsize)
    flow = get_flow(current_array)
    resistance = deltaV/flow
    
    return factor*resistance, resistance 


def get_bulk_permeability(flowrate_array,cellsize,fluid_viscosity,deltaP):

    factor = get_geometry_factor(flowrate_array,cellsize)
    flow = get_flow(flowrate_array)   

    resistance = deltaP/flow
    
    return fluid_viscosity/(resistance*factor),resistance

def effectiveres(b,rhoeff,rhof,rhom,width):
    """
    function defining effective resistivity as a function of the matrix and fluid
    resistivities, rhom and rhof, the fault width b, and the fault volume width.
    """
    return 1./rhoeff - (b/rhof+(width-b)/rhom)/width  
  
def get_electric_aperture(width,rhoeff,rhof,rhom):
    """
    calculate effective aperture of a volume with effective resistivity
    rhoeff, of width = width, and resistivity of fluid and matrix, rhof and rhom
    in terms of a single planar fault through the centre of the volume
    
    """
    if rhof > rhoeff:
        print("can't calculate effective aperture, rhof must be < rhoeff")
        return
    elif ((rhoeff == 0) or np.isinf(rhoeff)):
        print("can't calculate effective aperture, rhoeff must be finite and > 0")
        return
    
    return so.newton(effectiveres,0.0,args=(rhoeff,rhof,rhom,width),maxiter=100)
    
def effectivek(bh,keff,km,width):
    """
    function defining effective permeability of a volume with a planar fracture
    through it (flat plates) with separation bh, width of volume =width, 
    matrix permeability km
    """
    return keff - (bh**3/12. - (width - bh)*km)/width

def get_hydraulic_aperture(width,keff,km):
    # need to set a threshold because python is retarded and thinks that 1e-18 < 1e-18
    
    if keff <= km:
        print("keff is %.3e which is < km (%.3e), setting effective aperture to 0.0"%(keff,km))
        return 0.0
    if np.isinf(keff):
        print("can't calculate effective aperture, keff must be finite, setting to 0.0")
        return 0.0
    else:
        # to get a starting value for bh, approximate bh << width
        bhstart = (width*12*(keff-km))**(1./3)
        return so.newton(effectivek,bhstart,args=(keff,km,width),maxiter=100)
        

