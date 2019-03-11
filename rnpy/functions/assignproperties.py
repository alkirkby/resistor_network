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
    
def update_apertures(aperture_array,i,k1,j1,i1,ind,od,d,apedge):
    """
    
    updated now to include fat faults with width greater than cell size.
    this is achieved by finding all the cells with aperture > cell size and
    expanding the fault at this point, in a direction perpendicular to the
    fault, in both directions. Therefore, the fault is always an odd number of
    cells wide, i.e. 1,3,5,7... cells wide, depending on the aperture.
    
    the resistance in the outer cells of the fault is calculated as a weighted 
    mean of the matrix and fluid resistivity based on the leftover portion of
    fault that hasn't been assigned to a full cell.
    """
    
    ncells = (np.array(aperture_array.shape)[:-2] - 2)[::-1]   
    
    if od == 0:
        ind0,ind2 = max(1,i1-ind),min(ncells[0]+1,i1+1+ind)
        aperture_array[k1,j1,ind0:i1,i,od] = d[od]
        aperture_array[k1,j1,i1+1:ind2,i,od] = d[od]
        aperture_array[k1,j1,ind0-1,i,od] += apedge
        if ind2 < aperture_array.shape[2]:
            aperture_array[k1,j1,ind2,i,od] += apedge
        
        # update the connectors perpendicular to plane
        aperture_array[k1,j1,ind0-1:ind2,0,0] = d[od]
#        aperture_array[k1,j1,ind0-1,0,0] = d[od]/2.
#        aperture_array[k1,j1,ind2-1,0,0] = d[od]/2.
        if i == 1:
            aperture_array[k1,j1+1,ind0-1:ind2,0,0] = d[od]
        elif i == 2:
            aperture_array[k1+1,j1,ind0-1:ind2,0,0] = d[od] 
        
    elif od == 1:
        ind0,ind2 = max(1,j1-ind),min(ncells[0]+1,j1+1+ind)
        aperture_array[k1,ind0:j1,i1,i,od] = d[od]
        aperture_array[k1,j1+1:ind2,i1,i,od] = d[od]
        aperture_array[k1,ind0-1,i1,i,od] += apedge
        if ind2 < aperture_array.shape[1]:
            aperture_array[k1,ind2,i1,i,od] += apedge
        
        # update the connectors perpendicular to plane
        aperture_array[k1,ind0-1:ind2,i1,1,1] = d[od]
#        aperture_array[k1,ind0-1,i1,1,1] = d[od]/2.
#        aperture_array[k1,ind2-1,i1,1,1] = d[od]/2.
        if i == 0:
            aperture_array[k1,ind0-1:ind2,i1+1,1,1] = d[od]
        elif i == 2:
            aperture_array[k1+1,ind0-1:ind2,i1,1,1] = d[od]

    elif od == 2:
        ind0,ind2 = max(1,k1-ind),min(ncells[0]+1,k1+1+ind)
#        print("ind2,j1,i1,i,od",ind2,j1,i1,i,od)
        aperture_array[ind0:k1,j1,i1,i,od] = d[od]
        aperture_array[k1+1:ind2,j1,i1,i,od] = d[od]  
        aperture_array[ind0-1,j1,i1,i,od] += apedge
        if ind2 < aperture_array.shape[0]:
            aperture_array[ind2,j1,i1,i,od] += apedge 
        
        # update the connectors perpendicular to plane
        aperture_array[ind0-1:ind2,j1,i1,2,2] = d[od]
#        aperture_array[ind0-1,j1,i1,2,2] = d[od]/2.
#        aperture_array[ind2-1,j1,i1,2,2] = d[od]/2.
        
        if i == 0:
            aperture_array[ind0-1:ind2,j1,i1+1,2,2] = d[od]
        elif i == 1:
            aperture_array[ind0-1:ind2,j1+1,i1,2,2] = d[od]

    return aperture_array,ind0,ind2
    

def update_all_apertures(aperture_array,d):
    
    aperture_array = np.copy(aperture_array)
    for i in range(3):
        # the two directions perpendicular to direction of flow, indices and cell sizes
        dpi = [dd for dd in range(3) if dd != i]
    
        ncells = (np.array(aperture_array.shape)[:-2] - 2)[::-1]
        # first need to correct apertures to spread out any that are wider than the cellsize
        # direction of opening, cycle through faults opening in x, y and z directions
        
        for od in dpi:
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
                
                aperture_array,ind0,ind2 = update_apertures(aperture_array,i,k1,j1,i1,ind,od,d,apedge)
    
            
            aperture_array[:,:,:,i,od][np.where(aperture_array[:,:,:,i,od] > d[od])] = d[od]
    
            
    return aperture_array


def get_electrical_resistance(aperture_array,r_matrix,r_fluid,d):
    """
    
    returns a numpy array containing resistance values and an array containing 
    resistivities 
    
    updated now to include fat faults with width greater than cell size.
    this is achieved by finding all the cells with aperture > cell size and
    expanding the fault at this point, in a direction perpendicular to the
    fault, in both directions. Therefore, the fault is always an odd number of
    cells wide, i.e. 1,3,5,7... cells wide, depending on the aperture.
    
    the resistance in the outer cells of the fault is calculated as a weighted 
    mean of the matrix and fluid resistivity based on the leftover portion of
    fault that hasn't been assigned to a full cell.
    
    =================================inputs====================================
    aperture_array = array containing fault apertures
    r_matrix, r_fluid = resistivity of matrix and fluid
    d = list containing cell size (length of connector) in x,y and z directions 
    [dx,dy,dz] or float/integer if d is the same in all directions
    
    ===========================================================================
    """

    # initialise new arrays to contain resistance and resistivity
    resistance_array = np.zeros(np.shape(aperture_array)[:-1])
    resistivity_array = resistance_array.copy()
    # get the number of cells
    ncells = (np.array(aperture_array.shape)[:-2] - 2)[::-1]

    # correct d, if a single value is passed in
    if type(d) in [float,int]:
        d = [float(d)]*3


    for i in range(3):
        dp = [d[dd] for dd in range(3) if dd != i]
        resistance_array[:,:,:,i] = d[i]*r_matrix/np.product(dp)   

    for i in range(3):
        # the two directions perpendicular to direction of flow, indices and cell sizes
        dpi = [dd for dd in range(3) if dd != i]
        dp = [d[dd] for dd in dpi]
    
        # first need to correct apertures to spread out any that are wider than the cellsize
        # direction of opening, cycle through faults opening in x, y and z directions
        
        for od in dpi:
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
                
                aperture_array,ind0,ind2 = update_apertures(aperture_array,i,k1,j1,i1,ind,od,d,apedge)

            
            aperture_array[:,:,:,i,od][aperture_array[:,:,:,i,od] > d[od]] = d[od]
    
        # cross sectional area of the cell perpendicular to flow
        area_matrix = np.product(dp)
        area_fracture = np.zeros_like(aperture_array[:,:,:,0,0])
        for ii in range(2):
            # define the area taken up by the fracture
            area_fracture += aperture_array[:,:,:,i,dpi[ii]]*d[dpi[1-ii]]
        # subtract the overlapping bit if there is any
        area_fracture-= aperture_array[:,:,:,i,dpi[0]]*aperture_array[:,:,:,i,dpi[1]]
        area_fracture[area_fracture<0.] = 0.
        
        # subtract fracture area from matrix area and remove any negative matrix area
        area_matrix -= area_fracture
        area_matrix[area_matrix<0.] = 0.
        
        
        # resistance is the weighted harmonic mean of the fractured bit (in the two
        # directions along flow) and the matrix bit, but only assign if less than existing value
        resistance_array[:,:,:,i] = np.amin([resistance_array[:,:,:,i],
                                             d[i]/(area_matrix/r_matrix + area_fracture/r_fluid)],
                                             axis=0)
        
    for i in range(3):
        # assign connectors in direction of opening (in the case where faults 
        # are more than one cell width)
        cond = aperture_array[:,:,:,i,i] > 0
        resistance_array[:,:,:,i][cond] = \
        np.amin([r_fluid*aperture_array[:,:,:,i,i][cond]/np.product(dp) +\
                 r_matrix*(d[i] - aperture_array[:,:,:,i,i][cond])/np.product(dp),
                 resistance_array[:,:,:,i][cond]],axis=0)

        resistivity_array[:,:,:,i] = resistance_array[:,:,:,i]*np.product(dp)/d[i]
        
    return resistance_array,resistivity_array,aperture_array


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


def get_hydraulic_resistance_old(aperture_array,k_matrix,d,mu=1e-3):
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


def get_hydraulic_resistance(aperture_array,k_matrix,d,mu=1e-3):
    """
    new calculation of hydraulic resistance, incorporating apertures that are
    wider than one cell width
    
    Approach: similar to the electrical resistivity assignment but modified
    based on the concept of 'hydraulic resistivity'. Assign all cells
    a hydraulic resistivity based on the aperture. Where the aperture is > cell
    size, the adjacent cells are given the same hydraulic resistivity of the 
    central cell. The resistance is then calculated as a weighted mean of 
    matrix 'hydraulic resistivity' and the fracture resistivity, given by the
    hydraulic resistivity array.
    
    =================================inputs====================================
    aperture_array = hydraulic aperture, array of shape (nz+2,ny+2,nx+2,3,3)
    k_matrix = permeability of matrix, float
    d = tuple,list or array containing cellsize in x, y and z direction
    mu = fluid viscosity
    ===========================================================================
    
    """
    if type(d) in [float,int]:
        d = [float(d)]*3
        
    hresistance = np.ones(np.shape(aperture_array)[:-1])
    
    for i in range(3):
        dp = [d[dd] for dd in range(3) if dd != i]
        hresistance[:,:,:,i] = d[i]*mu/(np.product(dp)*k_matrix)
    
    permeability = hresistance.copy()
    ncells = (np.array(aperture_array.shape)[:-2] - 2)[::-1]    
    
    # array to contain "hydraulic resistivity" multiplier values (to be multiplied 
    # by dz/(dy*dx) where dz is the direction of flow)
    hydres = 12.*mu/aperture_array**2.
    hydres[hydres > mu/k_matrix] = mu/k_matrix

        
    for i in range(3):
        # the two directions perpendicular to direction of flow, indices and values
        dpi = [dd for dd in range(3) if dd != i]
        dp = [d[dd] for dd in dpi]
    
        for od in dpi:
            # find all indices where the aperture is bigger than the cellsize
            for k1,j1,i1 in np.array(np.where(aperture_array[:,:,:,i,od] > d[od])).T:
                apval = aperture_array[k1,j1,i1,i,od]
                
                # start with a 3 cell width fault            
                ncf = 3
                # first, check how many cells the fault needs to occupy            
                while ncf < ncells[od]:
                    # check if aperture is less than ncf * cellsize, if not add 2 (spread the fault further)
                    if apval < ncf*d[od]:
                        break
                    ncf += 2            
                
                # the length by which the cell encroaches on edge cells
                apedge = (apval - (ncf-2.)*d[od])/2.
                # set the extra bits, first the internal points just set to minres
                ind = int(ncf/2) - 1

  
                
                # update the apertures based on the location of the cell and the fault width at that point
                aperture_array,ind0,ind2 = update_apertures(aperture_array,i,k1,j1,i1,ind,od,d,apedge)
                
                # hydraulic resistivity for the aperture in question
                rhoh = 12.*mu/apval**2.
                # update the hydraulic resistivity values in a similar way to the aperture
                if od == 0:
                    hydres[k1:k1+2,j1:j1+2,ind0-1:ind2+1,i][hydres[k1:k1+2,j1:j1+2,ind0-1:ind2+1,i,od] > rhoh] = rhoh
                    # update the cells perpendicular to the fault, so fluid is able to flow within the fault
                    hydres[k1,j1,ind0-1:ind2,0,0] = rhoh
                    if i == 1:
                        hydres[k1,j1+1,ind0-1:ind2,0,0] = rhoh
                    elif i == 2:
                        hydres[k1+1,j1,ind0-1:ind2,0,0] = rhoh 
                elif od == 1:
                    hydres[k1:k1+2,ind0-1:ind2+1,i1:i1+2,i][hydres[k1:k1+2,ind0-1:ind2+1,i1:i1+2,i,od] > rhoh] = rhoh
                    # update the cells perpendicular to the fault, so fluid is able to flow within the fault
                    hydres[k1,ind0-1:ind2,i1,1,1] = rhoh
                    if i == 0:
                        hydres[k1,ind0-1:ind2,i1+1,1,1] = rhoh
                    elif i == 2:
                        hydres[k1+1,ind0-1:ind2,i1,1,1] = rhoh               
                elif od == 2:
                    hydres[ind0-1:ind2+1,j1:j1+2,i1:i1+2,i][hydres[ind0-1:ind2+1,j1:j1+2,i1:i1+2,i,od] > rhoh] = rhoh
                    # update the cells perpendicular to the fault, so fluid is able to flow within the fault
                    hydres[ind0-1:ind2,j1,i1,2,2] = rhoh
                    if i == 0:
                        hydres[ind0-1:ind2,j1,i1+1,2,2] = rhoh
                    elif i == 1:
                        hydres[ind0-1:ind2,j1+1,i1,2,2] = rhoh
            aperture_array[:,:,:,i,od][aperture_array[:,:,:,i,od] > d[od]] = d[od]
    for i in range(3):
        # the two directions perpendicular to direction of flow, indices and values
        dpi = [dd for dd in range(3) if dd != i]
        dp = [d[dd] for dd in dpi]

        hr0,hr1 = hydres[:,:,:,i,dpi[0]],hydres[:,:,:,i,dpi[1]]
        ap0,ap1 = aperture_array[:,:,:,i,dpi[0]],aperture_array[:,:,:,i,dpi[1]]

        # need to subtract the aperture from one of these so the overlapping area isn't counted twice
        d0,d1 = dp[0],dp[1]

        # subtract fracture area from matrix area and remove any negative matrix area
        area_matrix = np.product(dp) - (ap0*d1 + ap1*d0 - ap0*ap1)
        
        # permeability is the weighted mean of the fractured bit (in the two
        # directions along flow) and the matrix bit
        # first calculate the hydraulic resistance, subtracting the overlap bit
        hrnew = d[i]/(area_matrix*k_matrix/mu + d0*ap1/hr1 + d1*ap0/hr0 - ap0*ap1/np.amax([hr0,hr1],axis=0))
        # only assign new values if they are lower than existing values
        hresistance[:,:,:,i] = np.amin([hrnew,hresistance[:,:,:,i]],axis=0)
        # assign connectors in direction of opening (in the case where faults 
        # are more than one cell width)
        cond = aperture_array[:,:,:,i,i] > 0
        hresistance[:,:,:,i][cond] = \
        np.amin([hydres[:,:,:,i,i][cond]*aperture_array[:,:,:,i,i][cond]/np.product(dp)+\
                 mu*(d[i] - aperture_array[:,:,:,i,i][cond])/(np.product(dp)*k_matrix),
                 hresistance[:,:,:,i][cond]],axis=0)
        # calculate permeability
        permeability[:,:,:,i] = mu*d[i]/(hresistance[:,:,:,i]*np.product(dp))
        
    return hresistance, permeability


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
        
def permeability2hydraulic_resistance(permeability,cellsize,fluid_viscosity):
    """
    calculate hydraulic resistance from permeability
    
    inputs:
    permeability = array with dimensions nz,ny,nx,3 (representing x, y and z 
                   directions), can contain nans
    fluid_viscosity = float
    dimensions = [x,y,z] dimensions of volume or individual cells in array    
    
    returns:
    hydraulic resistance array of same dimensions as permeability
    
    """

    dx,dy,dz = np.array(cellsize).astype(float)
    gf = np.array([dy*dz/dx,dx*dz/dy,dx*dy/dz])
    
    return fluid_viscosity/(gf*permeability)