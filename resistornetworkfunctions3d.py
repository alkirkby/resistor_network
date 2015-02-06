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

def initialise_faults(n, p, faultlengthmax = None, decayfactor=5.):
    """ 
    
    Initialising faults from a pool - random location, orientation (i.e. in the 
    xz, xy or zy plane), length and width. Translate these onto an array.
    returns an array with values of 1 indicating faults, and a list of fault 
    locations.
    
    =================================inputs====================================
    n = list containing number of cells in x, y and z directions [nx,ny,nz]
    p = probability of connection in yz, xz and xy directions [pyz,pxz,pxy]
    faultlengthmax =  maximum fault length for network
    decayfactor = defines the shape of the fault length distribution. 
                  Fault length follows a decaying distribution so that longer
                  faults are more probable than shorter faults:
                  fl = faultlengthmax*e^(-ax) where x ranges from 0 to 1 and
                  a is the decay factor
    ===========================================================================
    """
    
    n = np.array(n)
    ptot = np.sum(p)/3.
    pnorm = np.array(p)/np.sum(p)
    # initialise an array of zeros
    fault_array = np.zeros([n[2]+1,n[1]+1,n[0]+1,3])

    if faultlengthmax is None:
        faultlengthmax = float(max(n))
    faults = []
    
    while float(np.size(fault_array[fault_array==1.]))/float(np.size(fault_array)) < ptot:
        # pick a random location for the fault x,y,z
        centre = [np.random.randint(0,nn) for nn in n]
        # pick random x,y,z extents for the fault
        d = faultlengthmax*np.exp(-decayfactor*np.random.random(3))
        # no faults smaller than one cell
        d[d<1.] = 1.
        # pick orientation for the fault according to relative probability p
        fo = np.random.choice(np.arange(3),p=pnorm)
        # reset width (normal to plane of fault)
        # when assigned to array this translates to 1 cell width
        d[fo] = 0.5
        # define locations of edges of fault
        mm = np.array([[np.ceil(centre[0]-d[0]),np.ceil(centre[0]+d[0])],
                       [np.ceil(centre[1]-d[1]),np.ceil(centre[1]+d[1])],
                       [np.ceil(centre[2]-d[2]),np.ceil(centre[2]+d[2])]])
        # remove any negative numbers and numbers larger than network bounds
        mm[mm < 0] = 0
        for m in range(len(mm)):
            if mm[m,1] > n[m]:
                mm[m,1] = n[m]
        # assign fault to fault array
        for i in range(3):
            if i != fo:
                fvals = np.zeros(3)
                fvals[-(fo+i)] = 1.
                fault_array[mm[2,0]:mm[2,1]+fvals[2],
                            mm[1,0]:mm[1,1]+fvals[1],
                            mm[0,0]:mm[0,1]+fvals[0],i] = 1.
        
        # get uv extents of fault in local plane (i.e. exclude width normal to plane)
        u,v_ = [mm[i] for i in range(3) if i != fo]
        v = np.array([[v_[0]]*2,[v_[1]]*2])
        faultuvw = [np.array([u,u[::-1]]),v]
        faultuvw.insert(fo,np.array([[mm[fo,0]]*2]*2))
        faults.append(faultuvw)
        
    # deal with edges
    fault_array[:,:,-1,0] = np.nan
    fault_array[:,-1,:,1] = np.nan
    fault_array[-1,:,:,2] = np.nan
    # make a new larger array of nans
    fault_array_final = np.ones(list(n[::-1]+2)+[3])*np.nan
    # put the fault array into this array in the correct position.
    fault_array_final[1:,1:,1:] = fault_array
    
    return fault_array_final,np.array(faults)


def assign_fault_aperture(fault_array,faults = None,
                          fault_dz=0.,separation=1e-3,
                          offset=[0.,0.]):
    """
    take a fault array and assign aperture values. This is done by creating two
    identical fault surfaces then separating them (normal to fault surface) and 
    offsetting them (parallel to fault surface). The aperture is then
    calculated as the difference between the two surfaces, and negative values
    are set to zero.
    To get a planar fault, set fault_dz to zero.
    Returns: numpy array containing x,y,z aperture values, numpy array
             containing geometry corrected aperture values for hydraulic flow
             simulation [after Brush and Thomson 2003, Water Resources Research]
    
    =================================inputs====================================

    fault_array = array containing 1 (fault), 0 (matrix), or nan (outside array)
                  shape (nx,ny,nz,3), created using initialise_faults
    faults = array or list containing u,v,w extents of faults, optional but 
             speeds up process.
    fault_dz, float = adjacent points on fault surface are separated by a 
                      random value between +fault_dz/2 and -fault_dz/2, in metres
    separation, float = fault separation normal to fault surface, in metres
    offset, list of 2 integers = number of cells horizontal offset between 
                                 surfaces [offset_x,offset_y]
    
    ===========================================================================    
    """
    nx,ny,nz = np.array(np.shape(fault_array))[:3][::-1] - 2
    aperture_array = fault_array.copy()

    # if no fault indices provided, initialise a fault index array. For now,
    # just assume everywhere is faulted (this gets cancelled out when
    # multiplied by fault array, just takes longer)
    if faults is None:
        faults = []
        faults += [[[[2,nx+2]]*2,[[2,2],[ny+2,ny+2]],[[i,i]]*2] for i in range(2,nz+2)]
        faults += [[[[i,i]]*2,[[2,ny+2]]*2,[[2,2],[nz+2,nz+2]]] for i in range(2,nx+2)]
        faults += [[[[2,2],[nx+2,nx+2]],[[i,i]]*2,[[2,nz+2]]*2] for i in range(2,ny+2)]
        
    for ib, bounds in enumerate(np.amax(faults)-np.amin(faults)):
        # i and j define horizontal extents of the fault
        imax,jmax = bounds[bounds!=0] + offset
        # initialise a fault surface of the correct size
        fs = np.zeros([imax,jmax])
        for i in range(imax):
            for j in range(jmax):
                # get a random value between -dz/2 and +dz/2 to add to base value
                rv = fault_dz*(np.random.rand() - 0.5)
                if i == 0:
                    if j == 0:
                        # start the fault elevation at zero for first cell
                        base = 0.
                    else:
                        # for all other first-row cells, take the previous cell
                        base = fs[i,j-1]
                elif j == 0:
                    # starting (base) elevation for beginning of each row taken
                    # from previous row
                    base = fs[i-1,j]
                else:
                    # start (base) elevation for middle cells taken as 
                    # an average of three adjacent cells in previous row
                    base = np.mean(fs[i-1,j-1:j+2])

                fs[i,j] = base + rv
        # level the fault so there's no tilt
        subtract = np.mean(fs,axis=0)
        for i in range(len(fs)):
            fs[i] -= subtract                
        
        
        b = fs[offset[]:,offset[]:]-fs[:-offset[],:-offset[]]
        b[b<0.] = 0.
        
        
    

def assign_random_resistivity(aperture_array,r_matrix,r_fluid):
    """ 
    
    Initialising faults from a pool - random location, orientation (i.e. in the 
    xz, xy or zy plane), length and width. Translate these onto an array.
    
    =================================inputs====================================
    n = list containing number of cells in x, y and z directions [nx,ny,nz]
    p = probability of connection in yz, xz and xy directions [pyz,pxz,pxy]
    r_matrix, r_fluid = resistivity of matrix and fluid
    faultlengthmax =  maximum fault length for network
    decayfactor = defines the shape of the fault length distribution. 
                  Fault length follows a decaying distribution so that longer
                  faults are more probable than shorter faults:
                  fl = faultlengthmax*e^(-ax) where x ranges from 0 to 1 and
                  a is the decay factor
    ===========================================================================
    """
    
    res_array = np.ones_like(fault_array)*r_matrix
    


def get_unique_values(inarray,log=False):
    
    
    inarray= 1.*inarray
    if log:
        inarray = np.log10(inarray)
    
    r = 6
    while len(np.unique(inarray[np.isfinite(inarray)])) > 2:
        inarray = np.around(inarray,decimals=r)
        r -= 1

    if log:
        inarray = 10**inarray

    return np.unique(inarray[np.isfinite(inarray)])

def get_phi(d,fracture_diameter):
    """

    
    returns fracture porosity
    
    =================================inputs====================================
    res = resistivity arry containing x,y,z resistivities.
          shape [nz+2,ny+2,nx+2,3]
    d = cell size in the x,y,z directions [dx,dy,dz]
    
    fracture_diameter = diameter of fracture in model, float
    ===========================================================================
    """

    d = np.array(d)
    a = np.array([d[1]*d[2],d[0]*d[2],d[0]*d[1]])
    
    return (np.pi/4.)*fracture_diameter**2/a
    

def get_electrical_resistance(res,r_matrix,r_fluid,d,fracture_diameter):
    """
    
    returns a numpy array containing resistance values 
    
    =================================inputs====================================
    d = list containing cell size (length of connector) in x,y and z directions 
    [dx,dy,dz]
    
    phiz = porosity from fractures in z direction
    res = resistivity arry containing x,y,z resistivities.
          shape [nz+2,ny+2,nx+2,3]
    r_matrix, r_fluid = resistivity of matrix and fluid
    ===========================================================================
    """
    
    phi = get_phi(d,fracture_diameter)
    res = 1.*res
    
    for i in range(3):
        # area of the cell
        acell = np.product([d[ii] for ii in range(3) if ii != i])
        # convert resistivities to resistances
        # first, resistance in an open node: harmonic mean of resistance in
        # fracture and resistance in surrounding rockmass. 
        res[:,:,:,i][(res[:,:,:,i]==r_fluid)&(np.isfinite(res[:,:,:,i]))] = \
        d[i]/(acell*((phi[i]/r_fluid) + (1.-phi[i])/r_matrix))
        # second, resistance in a closed node, r = rho*length/width
        res[:,:,:,i][res[:,:,:,i]==r_matrix] = d[i]*r_matrix/acell

    return res


def get_permeability(res,r_fluid,k_matrix,fracture_diameter):
    """



    calculate permeability based on a resistivity array
    
    =================================inputs====================================
    res_array = numpy array containing resistivity
    r_fluid = resistivity of fluid, in ohm-m
    k_matrix = permeability of matrix, in m^2
    fracture_diameter = fracture diameter
    ===========================================================================    
    """
    permeability = np.ones_like(res)*k_matrix        
    permeability[res==r_fluid] = fracture_diameter**2/32.
    permeability[np.isnan(res)] = np.nan
    
    return permeability


def get_hydraulic_resistance(k,k_matrix,d,fracture_diameter,mu=1e-3):
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
    hresistance = np.ones_like(k)
    for i in range(3):
        # area of the cell
        acell = np.product([d[ii] for ii in range(3) if ii != i])
        hresistance[:,:,:,i] = (mu/k[:,:,:,i])*(d[i]/acell)
        hresistance[:,:,:,i][(k[:,:,:,i] != k_matrix)&(np.isfinite(k[:,:,:,i]))]\
        = mu*128.*d[i]/(np.pi*fracture_diameter**4)

    return hresistance


def embed_network(prop,embedfiles,pembed_m,pembed_f,vmatrix,vfluid):
    """
    embed a network with resistivity values from smaller networks
    
    prop = array containing property values, 2 unique values vmatrix and vfluid
    embedfiles = list containing full path to text file containing x, y and z 
                 property values for the embedment
    pembed_m = numpy array or list containing x,y,z probability of embedment 
               for matrix (closed) cells in prop
    pembed_f = numpy array or list containing x,y,z probability of embedment 
               for open cells in prop
    vmatrix,vfluid = property values for matrix and fluid in prop
    
    """
    
    # get values to embed
    embedvals = [np.loadtxt(f) for f in embedfiles]
    
    # get number of fluid and matrix cells in prop array
    nf = len(prop[prop==vmatrix])
    nm = len(prop[prop==vfluid])
    
    # get number of of fluid and matrix cells to embed
    pf = float(nf)*pembed_f
    pm = float(nm)*pembed_m
    
    # embed fluid cells with probability pembed_f
    count = 0
    n = 0
    ii = 0
    
    for p,val in [[pf,vfluid],[pm,vmatrix]]:
        while count < p:
            if n >= len(embedvals[ii]):
                n = 0
            k,j,i = [np.random.randint(1,int(nn)-1) for nn in np.shape(prop)]
            if np.all(prop[np.isfinite(prop[k,j,i])] == val):
                prop[k,j,i] = embedvals[ii][n]
                n += 1
                count += 1
        ii += 1


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

def get_nfree(n):
    nx,ny,nz = n
    return [nx*(ny+1)*(nz+1),ny*(nx+1)*(nz+1),(nx+1)*(ny+1)*(nz+2)]
    
def get_nnodes(n):
    nx,ny,nz = n
    return (nx+1)*(ny+1)*(nz+1)

def get_ncells(n):
    nx,ny,nz = n
    ncxz = nx*(ny+1)*nz # number of cells in the xz plane
    ncyz = (nx+1)*ny*nz # number of cells in the yz plane
    return [ncxz,ncyz] # number of cells
    
def buildmatrix3d_kirchhoff(n):
    """
    calculate numbers to populate matrix and their row and column, relating
    to kirchhoff's law for electrical current and equivalent for fluid flow
    (i.e., sum of total current in and out of each node is zero)
    
    ==============================inputs=======================================
    n = number of cells in the x (horizontal), y (into the plane)
        and z (vertical) directions [nx,ny,nz]
    ===========================================================================
    """
    nx,ny,nz = n
    nfx,nfy,nfz = get_nfree(n)
    nn = get_nnodes(n)
    
    #   a. x connectors
    data1a = np.hstack([-np.ones(nfx),np.ones(nfx)])
    rows1as = np.hstack([np.arange(nx)]*(ny+1)*(nz+1)) \
            + np.hstack([np.ones(nx)*(nx+1)*i for i in range((ny+1)*(nz+1))])
    rows1a = np.hstack([rows1as,rows1as + 1])
    cols1a = np.hstack([np.arange(nfx)]*2)
    
    #   b. y connectors
    data1b = np.hstack([-np.ones(nfy),np.ones(nfy)])
    rows1bs = np.hstack([np.arange(ny*(nx+1))]*(nz+1)) \
            + np.hstack([np.ones(ny*(nx+1))*(nx+1)*(ny+1)*i for i in range(nz+1)])
    rows1b = np.hstack([rows1bs,rows1bs + nx + 1])
    cols1b = np.hstack([np.arange(nfy)]*2)+nfx
    
    #   c. z connectors
    data1c = np.hstack([np.ones(nn),-np.ones(nn)])
    cols1cs = np.arange(nn) + nfx + nfy
    cols1c = np.hstack([cols1cs,cols1cs + (nx+1)*(ny+1)])
    rows1c = np.hstack([np.arange(nn)]*2)    
    
    return np.hstack([data1a,data1b,data1c]),np.hstack([rows1a,rows1b,rows1c]),\
           np.hstack([cols1a,cols1b,cols1c])
      
      
def buildmatrix3d_potential(resistance):
    """
    calculate numbers to populate matrix and their row and column, relating
    to conservation of potential and equivalent for fluid flow
    (i.e., potential is conservative in each elementary cell)
    
    ==============================inputs=======================================
    resistivity = array containing resistivities in the x,y,z directions
     
    ===========================================================================
    """

    nz,ny,nx = [int(i-2) for i in np.shape(resistance)[:3]]
    n = [nx,ny,nz]
    nfx,nfy,nfz = get_nfree(n)
    nn = get_nnodes(n)
    ncxz,ncyz = get_ncells(n)
    nc = ncxz + ncyz # number of cells

    resx = resistance[1:,1:,1:-1,0]
    resy = resistance[1:,1:-1,1:,1]
    resz = resistance[1:-1,1:,1:,2]    

    #    a. x connectors
    data2a = np.hstack([np.ones(ncxz)*resx.flatten()[:ncxz], 
                        np.ones(ncxz)*(-resx.flatten()[-ncxz:])])
    rows2a = np.hstack([np.arange(ncxz)+nn]*2)
    cols2a = np.hstack([np.arange(ncxz), np.arange(ncxz) + nx*(ny+1)])
    
    #    b. y connectors
    data2b = np.hstack([np.ones(ncyz)*resy.flatten()[:ncyz], 
                        np.ones(ncyz)*(-resy.flatten()[-ncyz:])])
    rows2b = np.hstack([np.arange(ncyz) + nn + ncxz]*2)
    cols2b = np.hstack([np.arange(ncyz) + nx*(ny+1)*(nz+1),
                        np.arange(ncyz) + nx*(ny+1)*(nz+1) + ny*(nx+1)])
    
    #    c. z connectors
    data2c = np.hstack([np.ones(nc)*np.hstack([-resz[:,:,:-1].flatten(),-resz[:,:-1,:].flatten()]),
                        np.ones(nc)*np.hstack([resz[:,:,1:].flatten(),resz[:,1:,:].flatten()])])
                        
    rows2c = np.hstack([np.arange(nc) + nn]*2)#nfx + nfy + (nx+1)*(ny+1)]*2)
    cols2c1 = np.hstack([np.arange(nx)]*(ny+1)*nz) \
            + np.hstack([np.ones(nx)*(nx+1)*i for i in range((ny+1)*nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
    cols2c2 = np.hstack([np.arange((nx+1)*ny)]*nz) \
            + np.hstack([np.ones((nx+1)*ny)*(nx+1)*(ny+1)*i for i in range(nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
    cols2c = np.hstack([cols2c1,cols2c2,cols2c1+1,cols2c2+nx+1])
    
    return np.hstack([data2a,data2b,data2c]),np.hstack([rows2a,rows2b,rows2c]),\
           np.hstack([cols2a,cols2b,cols2c])
    
    
def buildmatrix3d_normalisation(resistance):
    """
    calculate numbers to populate matrix and their row and column, relating
    to normalisation across the network (i.e., total voltage drop across
    entry and exit nodes), also add one row that forces currents flowing
    into the network to equal currents exiting the network
    
    ==============================inputs=======================================
    resistivity = array containing resistivities in the x,y,z directions as for
    buildmatrix3d_potential
    ===========================================================================
    """
    
    nz,ny,nx = [int(i-2) for i in np.shape(resistance)[:3]]
    n = [nx,ny,nz]
    nfx,nfy,nfz = get_nfree(n)
    nfree = sum([nfx,nfy,nfz])
    nn = get_nnodes(n)
  
    resx = resistance[1:,1:,1:-1,0]
    resy = resistance[1:,1:-1,1:,1]
    resz = resistance[1:-1,1:,1:,2]    
    
    ncxz,ncyz = get_ncells([nx,ny,nz])
    nc = ncxz + ncyz # number of cells
 
    #    a. x connectors
    data3a = np.ones(nx*(ny+1))*resx[-1].flatten()
    rows3a = np.arange(nx*(ny+1)) + nn + nc + (nx+1)*(ny+1)
    cols3a = np.arange(nx*(ny+1)) + ncxz
    
    #    b. y connectors
    data3b = np.ones((nx+1)*ny)*resy[-1].flatten()
    rows3b = np.arange((nx+1)*ny) + nn + nc + (2*nx+1)*(ny+1)
    cols3b = np.arange((nx+1)*ny) + nfx + (nx+1)*ny*nz
    
    #    c. z connectors
    data3c1 = np.ones((nx+1)*(ny+1)*nz)*resz.flatten()
    rows3c1 = np.hstack([np.arange((nx+1)*(ny+1))]*nz) + nn + nc
    cols3c1 = np.hstack([np.arange((nx+1)*(ny+1))]*nz) \
            + np.hstack([np.ones((nx+1)*(ny+1))*(nx+1)*(ny+1)*i for i in range(nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
    
    data3c2 = np.ones(ncxz)*resz[:,:,:-1].flatten()
    cols3c2 = np.hstack([np.arange(nx)]*(ny+1)*nz) \
            + np.hstack([np.ones(nx)*(nx+1)*i for i in range(ny+1)*nz]) \
            + np.hstack([np.ones(nx*(ny+1))*(nx+1)*(ny+1)*i for i in range(nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
            
    rows3c2 = np.hstack([np.arange(nx*(ny+1))]*nz) \
            + nn + nc + (nx+1)*(ny+1)
    
    data3c3 = np.ones((nx+1)*ny*nz)*resz[:,:-1,:].flatten()
    rows3c3 = np.hstack([np.arange((nx+1)*ny)]*nz) \
            + nn + nc + (2*nx+1)*(ny+1)     
    cols3c3 = np.hstack([np.arange((nx+1)*ny)]*nz) \
            + np.hstack([np.ones((nx+1)*ny)*(nx+1)*(ny+1)*i for i in range(nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
    
    data3c = np.hstack([data3c1,data3c2,data3c3])
    rows3c = np.hstack([rows3c1,rows3c2,rows3c3])
    cols3c = np.hstack([cols3c1,cols3c2,cols3c3]) 
 
    # 4. Current in = current out
    data4 = np.hstack([np.ones((nx+1)*(ny+1)),-np.ones((nx+1)*(ny+1))])
    rows4 = np.ones((nx+1)*(ny+1)*2)*(nfree)
    cols4 = np.hstack([np.arange((nx+1)*(ny+1)) + nfx + nfy,
                       np.arange((nx+1)*(ny+1)) + nfx + nfy + (nx+1)*(ny+1)*(nz+1)])
                       
    return np.hstack([data3a,data3b,data3c,data4]),\
           np.hstack([rows3a,rows3b,rows3c,rows4]),\
           np.hstack([cols3a,cols3b,cols3c,cols4])  

def build_matrix3d(resistance):
    """
    """
    nx,ny,nz = np.array(np.shape(resistance)[:-1][::-1])-2
    n = [nx,ny,nz]
    nn = get_nnodes(n)
    nc = sum(get_ncells(n))
    nfree = sum(get_nfree(n))
    data1,rows1,cols1 = buildmatrix3d_kirchhoff([nx,ny,nz])
    data2,rows2,cols2 = buildmatrix3d_potential(resistance)
    data3,rows3,cols3 = buildmatrix3d_normalisation(resistance)
    
    data,rows,cols = np.hstack([data1,data2,data3]),\
                     np.hstack([rows1,rows2,rows3]),\
                     np.hstack([cols1,cols2,cols3])
    m = sparse.coo_matrix((data,(rows,cols)), shape=(nfree+1,nfree))
    mc = sparse.bmat([[m.tocsr()[:int(nn/2)]],[m.tocsr()[int(nn/2)+1:]]]).tocsr()
    b = np.zeros(nfree)
    b[nn+nc-1:-1] = 1.
    
    return mc,b

"""
===================Functions relating to plotting==============================
"""

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


"""
=================Functions relating to set up of multiple runs=================
"""


def divide_inputs(work_to_do,size):
    """
    divide list of inputs into chunks to send to each processor
    
    """
    chunks = [[] for _ in range(size)]
    for i,d in enumerate(work_to_do):
        chunks[i%size].append(d)

    return chunks
