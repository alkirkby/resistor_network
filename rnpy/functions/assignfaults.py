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
import rnpy.functions.array as rna



def get_normal(minmax):
    """
    get axis normal to plane. Returns integer 0, 1 or 2 representing x, y or z
    direction.
    
    ================================inputs=====================================
    minmax = array or list (shape (3,2)) containing min and max indices for fault
               in x, y and z directions, [[xmin,xmax],[ymin,ymax],[zmin,zmax]]. 
               As a fault is a planar feature, start index must equal finish 
               index in one of the three directions       
    ===========================================================================    
    """
    direction = None
    
    for i, mm in enumerate(minmax):
#        print(mm)
        if mm[1]-mm[0] == 0:
            direction = int(i)
            break
    
    return direction


def minmax2uvw(minmax, direction=None):
    """
    convert array containing extents in x y and z direction (shape (3,2)) to
    an array containing the x, y, z coordinates of the corners.
    
    Returns:
    array (shape (3,2,2) containing x, y and z coordinates of corners.
    
    ================================inputs=====================================
    minmax = array or list (shape (3,2)) containing min and max indices for fault
               in x, y and z directions, [[xmin,xmax],[ymin,ymax],[zmin,zmax]]. 
               Start index must equal finish index in one of the three directions.       
    ===========================================================================    
    """

    if direction is None:
        direction = get_normal(minmax)

    # get uv extents of fault in local plane (i.e. exclude width normal to plane)
    u,v_ = [minmax[i] for i in range(3) if i != direction]
    v = np.array([[v_[0]]*2,[v_[1]]*2])
    faultuvw = [np.array([u,u[::-1]]),v]
    # insert the index of plane, min=max
    faultuvw.insert(direction,np.array([[minmax[direction][0]]*2]*2))

    return np.array(faultuvw)

    

def add_fault_to_array(fault_mm,fault_array,direction=None):
    """self.solve_
    add a fault to an array based on extents provided. 
    Returns:
    - the updated fault array
    - x,y,z positions of the edge of the fault plane (shape (3,2,2))
    
    ================================inputs=====================================
    fault_mm = array or list (shape (3,2)) containing min and max indices for fault
               in x, y and z directions. As a fault is a planar feature, start 
               index must equal finish index in one of the three directions or
               specify the direction normal to the plane (0,1 or 2 for x,y z)
    fault_array = array containing 0's and 1's - 1 indicates fault, 0 indicates
                  no fault
                  
    ===========================================================================    
    
    
    """
    fault_array = fault_array.copy()    
    
    if type(fault_mm) == list:
        fault_mm = np.array(fault_mm)
    
    if direction is None:
        direction = get_normal(fault_mm)

    if direction is None:
        print("invalid fault minmax values, minimum must be same as max in one direction")
        return
        
    for i in range(3):
        if i != direction:
            fvals = np.zeros(3)
            fvals[-(direction+i)] = 1.
            fvals[direction] = 1.

            fault_array[fault_mm[2,0]:fault_mm[2,1]+fvals[2],
                        fault_mm[1,0]:fault_mm[1,1]+fvals[1],
                        fault_mm[0,0]:fault_mm[0,1]+fvals[0],i] = 1.
    
    return fault_array
    
    
    
def build_random_faults(n, p, faultlengthmax = None, decayfactor=5.):
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
    
    while True:
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
        if float(np.size(fault_array[fault_array==1.])+np.product(mm[:,1]-mm[:,0]))/\
           float(np.size(fault_array[np.isfinite(fault_array)])) < ptot:
            mm[:,1] -= 1.
            fault_array = add_fault_to_array(mm,fault_array,direction=fo)
            faultuvw = minmax2uvw(mm,direction=fo)
            faults.append(faultuvw+1.)
        else:
            break
    # make a new larger array of nans
    fault_array_final = np.zeros(list(np.array(np.shape(fault_array))[:-1]+1)+[3])
    
    # put the fault array into this array in the correct position.
    fault_array_final[1:,1:,1:] = fault_array
    # deal with edges
    fault_array_final = rna.add_nulls(fault_array_final)

    
    return fault_array_final,np.array(faults)

def get_duvw(ncells,ftype = 'single_yz'):
    
    nx, ny, nz = ncells
    
    
    ix = int(nx/2) + 1
    iy0, iy1 = 1, ny + 1
    iz0, iz1 = 1, nz + 1
    
    return [[[ix,ix],[iy0,iy1],[iz0,iz1]]]


def get_faultsize(duvw,offset):
    """
    get fault size based on the u,v,w extents of the fault
    
    """
    
    size = int(np.amax(duvw) + 2*(max(0.2*np.amax(duvw),4)) + offset)
    size += size % 2
    
    return size


def get_faultpair_inputs(fractal_dimension,elevation_scalefactor,
                         mismatch_wavelength_cutoff,cellsize):

    faultpair_inputs = dict(D=fractal_dimension,
                            scalefactor=elevation_scalefactor)
    if mismatch_wavelength_cutoff is not None:
        faultpair_inputs['fcw'] = mismatch_wavelength_cutoff
    if cellsize is not None:
        faultpair_inputs['cs'] = cellsize

    return faultpair_inputs
    

def assign_fault_aperture(fault_array,fault_uvw, 
                          cs = 0.25e-3,
                          fault_separation=1e-4, 
                          fault_surfaces = None,
                          offset=0, 
                          fractal_dimension = 2.5, 
                          mismatch_wavelength_cutoff = None, 
                          elevation_scalefactor = None,
                          correct_aperture_for_geometry = True
                          ):
    """
    take a fault array and assign aperture values. This is done by creating two
    identical fault surfaces then separating them (normal to fault surface) and 
    offsetting them (parallel to fault surface). The aperture is then
    calculated as the difference between the two surfaces, and negative values
    are set to zero.
    To get a planar fault, set fault_dz to zero.
    Returns: numpy array containing aperture values, numpy array
             containing geometry corrected aperture values for hydraulic flow
             simulation [after Brush and Thomson 2003, Water Resources Research],
             and numpy array containing corrected aperture values for electric
             current. different in x, y and z directions.
             
    
    =================================inputs====================================

    fault_array = array containing 1 (fault), 0 (matrix), or nan (outside array)
                  shape (nx,ny,nz,3), created using initialise_faults
    fault_uvw = array or list containing u,v,w extents of faults
    cs = cellsize in metres, has to be same in x and y directions
    fault_separation, float = fault separation normal to fault surface, in metres
    fault_surfaces = list or array of length the same as fault_uvw, each item containing 
                     2 numpy arrays, containing fault surface elevations, if 
                     None then random fault aperture is built
    offset, integer = number of cells horizontal offset between surfaces.
    fractal_dimension, integer = fractal dimension of surface, recommended in 
                                 range [2.,2.5]
    mismatch_wavelength_cutoff, integer = cutoff frequency for matching of 
                                         surfaces, default 3% of fault plane 
                                         size
    elevation_scalefactor, integer = scale for the standard deviation of the height 
                                     of the fault surface; multiplied by 
                                     (size * cellsize)**0.5 to ensure rock surface
                                     scales correctly.
    correct_aperture_for_geometry, True/False, whether or not to correct aperture for
                                      geometry
    cellsize = size in metres of the cells, used to calculate a sensible default
               for mismatch cutoff frequency, only needed if
               mismatch_wavelength_cutoff not provided
    ===========================================================================    
    """
    fault_array = rna.add_nulls(fault_array)
    
    nx,ny,nz = np.array(np.shape(fault_array))[:3][::-1]
    
    # aperture array, first axis length 3 if correcting for geometry, 1 if not
    ap_array = np.array([np.ones_like(fault_array)*1e-50]*3) # yz, xz and xy directions
#    ap_array[0] *= 1e-50
    bvals = []
    faultheights = []

    for i, nn in enumerate(fault_uvw):
        bvals.append([])
        u0,v0,w0 = np.amin(nn, axis=(1,2))
        u1,v1,w1 = np.amax(nn, axis=(1,2))
        duvw = np.array([u1-u0,v1-v0,w1-w0])
        du,dv,dw = (duvw*0.5).astype(int)

        # define size, add some padding to account for edge effects and make 
        # the fault square as I am not sure if fft is working properly for non-
        # square geometries
#        print("getting fault size")
        size = get_faultsize(duvw,offset)
        
        # define direction normal to fault
#        print("getting direction")
        direction = list(duvw).index(0)
#        print("getting_faultpair inputs")
        faultpair_inputs = get_faultpair_inputs(fractal_dimension,
                                                elevation_scalefactor,
                                                mismatch_wavelength_cutoff,
                                                cs)
            
            
            
        build = False
        if fault_surfaces is None:
            build = True
#            print("fault surfaces none")
        else:
            try:
                h1,h2 = fault_surfaces[i]
                if type(h1) != np.ndarray:
                    try:
                        h1 = np.array(h1)
                    except:
                        raise
                if type(h2) != np.ndarray:
                    try:
                        h2 = np.array(h2)
                    except:
                        raise                
            except:
                build = True
#                print("fault surfaces wrong type")
            
        if build:
            print("building new fault surfaces")
            h1,h2 = rnfa.build_fault_pair(size, **faultpair_inputs)

#        print("i have some fault surfaces")
        if offset > 0:
            b = h1[offset:,offset:] - h2[:-offset,:-offset] + fault_separation
        else:
            b = h1 - h2 + fault_separation
            
        # set zero values to really low value to allow averaging
        b[b <= 1e-50] = 1e-50
        # centre indices of array b
        cb = (np.array(np.shape(b))*0.5).astype(int)
        
        if correct_aperture_for_geometry:
#            print("correcting aperture for geometry")
            bf, bc = rnfa.correct_aperture_geometry(h1[offset:,offset:],b,cs)
        else:
#            print("no aperture corrections")
            bf, bc = [np.ones_like(b[:-1,:-1])]*2
#        print("assigning faults to array")
        for i,bb in enumerate([[b[:-1,:-1]]*2,bf,bc]):
            b0,b1 = bb
#            print(b0,b1)
            if direction == 0:
                ap_array[i,w0:w1+1,v0:v1,u0,1] += b0[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-dv:cb[1]+dv+duvw[1]%2]
                ap_array[i,w0:w1,v0:v1+1,u0,2] += b1[cb[0]-dw:cb[0]+dw+duvw[2]%2,cb[1]-dv:cb[1]+dv+duvw[1]%2+1]
            elif direction == 1:
                ap_array[i,w0:w1+1,v0,u0:u1,0] += b0[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-du:cb[1]+du+duvw[0]%2]
                ap_array[i,w0:w1,v0,u0:u1+1,2] += b1[cb[0]-dw:cb[0]+dw+duvw[2]%2,cb[1]-du:cb[1]+du+duvw[0]%2+1]
            elif direction == 2:
                ap_array[i,w0,v0:v1+1,u0:u1,0] += b0[cb[0]-dv:cb[0]+dv+duvw[1]%2+1,cb[1]-du:cb[1]+du+duvw[0]%2]
                ap_array[i,w0,v0:v1,u0:u1+1,1] += b1[cb[0]-dv:cb[0]+dv+duvw[1]%2,cb[1]-du:cb[1]+du+duvw[0]%2+1]
            bvals[-1].append([bb,b0,b1])
        faultheights.append([h1,h2])
    for i in range(len(ap_array)):
        ap_array[i] *= fault_array
    ap_array[(np.isfinite(ap_array))&(ap_array < 1e-50)] = 1e-50
    corr_c = ap_array[2]/ap_array[0]
    corr_f = ap_array[1]/ap_array[0]
    aperture_array = ap_array[0]
    aperture_array[(np.isfinite(aperture_array))&(aperture_array <= 2e-50)] = 0.
    
    return aperture_array,corr_f,corr_c,faultheights
    
