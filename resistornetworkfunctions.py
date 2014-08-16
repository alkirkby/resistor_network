# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:05:42 2014

@author: Alison Kirkby

Modelling random resistor networks using python.

"""
from __future__ import division, print_function
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

"""
===================Functions relating to class definition======================
"""


def assign_random_resistivity(n,p,r_matrix,r_fluid,linearity_factor):
    """
    returns a numpy array containing resistivity values for the direction pz
    note, to calculate other directions can use this function, just need to 
    transpose later.
    
    =================================inputs====================================
    n = list containing number of cells in x and z directions [nx,nz]
    p = list containing probability of connection in x and z directions [px,pz]
    r_matrix, r_fluid = resistivity of matrix and fluid
    linearity_factor =  factor to adjust probabilities according to value in 
                        previous row to make linear structures
                        e.g. if linearity_factor == 2: a given cell is twice as 
                        likely to be connected if the corresponding cell in the 
                        previous row is connected. Probabilities are 
                        normalised so that overall probability in each row = pz
    ===========================================================================
    """
    # initialise array with resistivity r_fluid
    resz = np.ones((n[1],n[0]+1))*r_fluid
    # initialise first row
    resz0 = np.random.random(size=(n[0]+1))*r_matrix
    resz[0,resz0>=p[1]*r_matrix] = r_matrix

    
    for i in range(1,n[1]):
        # figure out number of fractures in previous row
        nf = len(resz[i,resz[i-1]==r_fluid])
        # number of matrix cells in previous row
        nm = len(resz[i])-nf
        # multiplication factor to apply to matrix probabilities
        f = float(len(resz[i]))/float(linearity_factor*nf+nm)
        # probability of fracture if the above cell is a matrix cell
        pmz = f*p[1]
        # probability of fracture if the above cell is a fracture cell
        pfz = linearity_factor*pmz
        # make a new row containing values between 0 and r_matrix
        reszi = np.random.random(size=(n[0]+1))*r_matrix 
        # if adjacent cell on previous row had a fracture, 
        # assign matrix cell to this row with probability 1-pfz
        resz[i,(reszi>=pfz*r_matrix)&(resz[i-1]==r_fluid)] = r_matrix
        # if adjacent cell on previous row had no fracture, 
        # assign matrix cell to this row with probability 1-pmz
        resz[i,(reszi>=pmz*r_matrix)&(resz[i-1]==r_matrix)] = r_matrix
    # assign nan value to final row
    resz_final = np.ones((n[1]+2,n[0]+2))*np.nan
    resz_final[1:-1,1:] = resz

    return resz_final

def get_phi(dx,fracture_diameter):
    """
    returns fracture porosity resulting from fractures in z direction
    
    =================================inputs====================================
    dx = cell size in direction orthogonal to that being calculated
    fracture_diameter = diameter of fracture in model, float
    ===========================================================================
    """
    
    # use fracture diameter and cellsize dx and dz to define fracture
    # volume percent in z directions
    return fracture_diameter/dx
    

def get_electrical_resistance(d,fracture_diameter,resz,r_matrix,r_fluid):
    """
    returns a numpy array containing resistance values for z direction
    works for x as well, just need to swap the order of d
    
    =================================inputs====================================
    d = list containing cell size (length of connector) in x and z directions [dx,dz]
    where z is the direction 
    
    phiz = porosity from fractures in z direction
    res_array = array containing resistivity values
    r_matrix, r_fluid = resistivity of matrix and fluid
    ===========================================================================
    """
    
    phiz = get_phi(d[0],fracture_diameter)
    # convert resistivities to resistances
    # first, resistance in an open node: harmonic mean of resistance in
    # fracture and resistance in surrounding rockmass. 
    resz[(resz!=r_matrix)&(np.isfinite(resz))] = 1./(phiz/r_fluid)# + (1.-phiz)/r_matrix)
    # second, resistance in a closed node, r = rho*length/width
    resz[resz==r_matrix] = r_matrix*d[1]/d[0]

    return resz


def get_permeability(res_array,r_fluid,k_matrix,fracture_diameter):
    """
    calculate permeability based on a resistivity array
    
    =================================inputs====================================
    res_array = numpy array containing resistivity
    r_fluid = resistivity of fluid, in ohm-m
    k_matrix = permeability of matrix, in m^2
    fracture_diameter = fracture diameter
    ===========================================================================    
    """

    permeability = np.ones_like(res_array)*k_matrix        
    permeability[res_array==r_fluid] = fracture_diameter**2/12.
    permeability[np.isnan(res_array)] = np.nan
    
    return permeability


def get_hydraulic_resistance(d,k_array,k_matrix,fracture_diameter):
    """
    calculate hydraulic resistance based on a hydraulic permeability array
    
    =================================inputs====================================
    d = list containing cell size (length of connector) in x and z directions [dx,dz]
    where z is the direction
    k_array = numpy array containing permeability
    k_matrix = permeability of matrix, in m^2
    fracture_diameter = fracture diameter    
    ===========================================================================
    
    """

    hresistance = (1./k_array)*(d[1]/d[0])
    hresistance[(k_array != k_matrix)&(np.isfinite(k_array))]\
    = 12.*d[1]/(fracture_diameter**3) 

    return hresistance

  
def build_matrix(propertyx,propertyz):
    """
    build a matrix to solve for current or permeability
    
    """
    
    nx,nz = len(propertyx[0]),len(propertyz)

    # 1. construct part of matrix dealing with voltages (sum zero in each cell)
    #    a. construct x parts of loop
    #       define top part of loop - positive (positive right)
    d1v = propertyx[:-1].flatten()
    #       define bottom part of loop - negative (positive right)
    d2v = -propertyx[1:].flatten()
    #       construct matrix using above diagonals        
    xblock_v = sparse.diags([d1v,d2v],[0,nx],shape=(nx*nz,nx*(nz+1)))
    
    #    b. construct z parts of loop
    blocks = []
  
    for j in range(nz):
    #       construct dia1 - summing z voltage drops on lhs of loop, negative (positive down)
        dia1 = -propertyz[j,:nx]
    #       construct dia2 - summing z voltage drops on rhs of loop, positive (positive down)
        dia2 = propertyz[j,1:]
    #       construct diagonal matrix containing the above diagonals
        blocks.append(sparse.diags([dia1,dia2],[0,1],shape=(nx,nx+1)))
    #       construct matrix using above blocks
    yblock_v = sparse.block_diag(blocks)

    #    c. construct parts of matrix dealing with top and bottom z currents (zero voltage)
    yblock2_v = sparse.coo_matrix((nx*nz,nx+1))

    #    d. combine parts together to make voltage part of matrix
    m_voltage = sparse.coo_matrix(sparse.bmat([[xblock_v,yblock2_v,yblock_v,yblock2_v]]))

    
    # 2. construct part of matrix dealing with currents (sum zero in each node)
    #    need to skip a node in the middle of the matrix
    #    a. part dealing with x currents
    #       top and bottom parts of matrix
    
    onx = np.ones(nx)
    onx2 = np.ones(nx/2)

    xblock1 = sparse.diags([-onx,onx],offsets=[0,-1],shape=(nx+1,nx))
    
    #       middle part of matrix - one node is skipped in the middle so different pattern
    xblock2_s1 = sparse.diags([onx2,-onx2[:-1]],offsets=[0,1])
    xblock2_s2 = sparse.diags([-onx2,onx2[:-1]],offsets=[0,-1])
    #xblock2 = sparse.block_diag([xblock2_s1,xblock2_s2])

    #       build matrix from xblock1 and xblock2
    xblock = sparse.block_diag([xblock1]*int(nz/2)+[xblock2_s2]+[xblock2_s1]+[xblock1]*int(nz/2))
    
    #    b. part dealing with y currents
    #       block above skipped node (same as below)

    yblock1 = sparse.diags([np.ones(((nz/2)*(nx+1)+nx/2)),-np.ones(((nz/2)*(nx+1)+nx/2))],
                            offsets=[0,nx+1],
                            shape = (((nz/2)*(nx+1)+nx/2),((nz/2)*(nx+1)+nx/2)+nx+1))
    #       empty block to fill in the gap                  
    yblock2 = sparse.coo_matrix(((nz/2)*(nx+1)+nx/2,(nz/2)*(nx+1)+nx/2+1))
    
    #    c. combine the blocks together
    yblock = sparse.bmat([[sparse.bmat([[yblock1,yblock2]])],[sparse.bmat([[yblock2,yblock1]])]])
    
    #    d. combine x and y blocks together
    m_current = sparse.hstack([xblock,yblock])
    
    # 3. current in = current out
    m_cicu = np.hstack([np.zeros(nx*(nz+1)),np.ones(nx+1),np.zeros((nx+1)*nz),-np.ones(nx+1)])
    
    # 4. normalisation
    norm1a = sparse.coo_matrix((nx+1,(nz+1)*nx+nx+1))
    norm1b_sub = []
    for i in range(nz):
        norm1b_sub.append(sparse.diags(propertyz[i],0))
    norm1b = sparse.hstack(norm1b_sub)
    norm1c = sparse.coo_matrix((nx+1,nx+1))
    norm1 = sparse.hstack([norm1a,norm1b,norm1c])
    
    norm2a = sparse.diags(propertyx[-1],nx*nz,shape=(nx,nx*(nz+1)))
    norm2b_sub = []
    for i in range(nz):
        norm2b_sub.append(sparse.diags(propertyz[i,:nx],0,shape=(nx,nx+1)))
    norm2b = sparse.hstack(norm2b_sub)    
    
    norm2c = sparse.coo_matrix((nx,nx+1))
    norm2 = sparse.hstack([norm2a,norm2c,norm2b,norm2c])
    
    m_norm = sparse.vstack([norm1,norm2])
    
    # 5. combine all matrices together.
    m = sparse.csr_matrix(sparse.vstack([m_voltage,m_current,m_cicu,m_norm]))

    return m



def build_sums(nfree,n):
    """
    
    builds the matrix b to solve the matrix equation Ab = C
    where A is the matrix defined in build_matrix
    and C is the electrical current values.
    
    nfree = length of C (equal to length of each axis in A)
    n = list containing number of nodes in x and z direction [nx,nz]
    
    """
    
    b_dense = np.zeros(nfree)
    # apply a unit voltage
    b_dense[-(2*n[0] + 1):] = 1.
    
    return sparse.csr_matrix(b_dense)


    
def solve_matrix(A,b):
    """
    solve the matrix equation Ab = C
    
    """
   
    return linalg.spsolve(A,b)
    

"""
===================Functions relating to plotting==============================
"""


def get_meshlocations(d,n):
    """
    get locations of nodes for plotting
    
    n = list containing number of cells in x and z directions [nx,nz]
    d = list containing cell size in x and z directions [dx,dz]    
    
    """

    
    plotx = np.linspace(-d[0],d[0]*(n[0]+1),n[0]+2)
    plotz = np.linspace(-d[1],d[1]*(n[1]+1),n[1]+2)
    
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
    qplotxz[1][get_direction(parameter[:,:,1]) > 0.] += d[1]       
    
    # add plotxz to qplotxz to give absolute starting locations for arrows
    for i in range(2):
        qplotxz[i] += plotxz[i]

    return [[qplotxz[0],plotxz[0]],[plotxz[1],qplotxz[1]]]
    

def get_quiver_UW(parameter,plot_range):
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
    W = [np.zeros_like(parameter[:,:,0]),get_direction(parameter[:,:,1])]
    C = [np.abs(parameter[:,:,0]),np.abs(parameter[:,:,1])]

    
    
    # remove arrows outside of specified plot range
    for i in range(2):
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