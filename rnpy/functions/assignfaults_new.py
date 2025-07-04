# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:55:34 2016

@author: a1655681
"""

import numpy as np
import rnpy.functions.array as rna
import rnpy.functions.faultaperture as rnfa
from scipy.interpolate import interp1d
from rnpy.functions.utils import get_bin_ranges_from_centers


def get_faultlength_distribution(lvals,volume,alpha=10,a=3.5):
    """
    get number of faults in each length range given by lvals in metres
    returns an array (Nf) containing the number of fractures for each length range
    fractional numbers are allowed and are dealt with by assigning an additional
    fault with probability given by the leftover fraction    
    
    lvals = array containing fault lengths
    volume = volume in m^3 that will contain the faults
    alpha = constant used in the equation
    a = exponent (fractal dimension)
    
    """
    
    Nf = np.zeros(len(lvals)-1)

    for i in range(len(lvals) - 1):
        lmin,lmax = lvals[i:i+2]
        Nf[i] = (alpha/(1.-a))*lmax**(1.-a)*volume - (alpha/(1.-a))*lmin**(1.-a)*volume
    
    return Nf
    
def get_Nf2D(a, alpha, R, lvals_range):
    '''
    Get Number of faults within area R within bin ranges provided by lvals_range

    Parameters
    ----------
    a : TYPE
        density exponent.
    alpha : TYPE
        density constant.
    R2 : float
        Area in metres squared
    lvals_range : TYPE
        Bin ranges of fault lengths.

    Returns
    -------
    Number of faults in each bin range (len(lvals_range)-1).

    '''
    Nf = []
    for i in range(len(lvals_range)-1):
        lmin,lmax = lvals_range[i:i+2]
        Nf = np.append(Nf, round(alpha/(a-1.)*lmin**(1.-a)*R**2 - alpha/(a-1.)*lmax**(1.-a)*R**2)).astype(int)
        # Nf = np.append(Nf, alpha/(a-1.)*lmin**(1.-a)*R2 - alpha/(a-1.)*lmax**(1.-a)*R2)

    return Nf

def get_alpha(a,R,lvals_center,fw,porosity_target,alpha_start=0.0):
    '''
    

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    R : TYPE
        DESCRIPTION.
    lvals_center : TYPE
        DESCRIPTION.
    fw : TYPE
        DESCRIPTION.
    porosity_target : TYPE
        DESCRIPTION.
    alpha_start : TYPE, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    Nf : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    lvals_range : TYPE
        DESCRIPTION.

    '''
    lvals_range = get_bin_ranges_from_centers(lvals_center)

    alpha = alpha_start * 1.0
    Nf = get_Nf2D(a, alpha, R, lvals_range)
        
    while np.sum(fw * Nf * lvals_center)/(R**2) < porosity_target:
        Nf = get_Nf2D(a, alpha, R, lvals_range)
        alpha += 0.01
        
        
    return alpha, lvals_range

def create_random_fault(network_size,faultsizerange,plane):
    """
    create a fault in random location along specified plane. Fault will be
    truncated if it extends out of the network.
    
    
    network_size = list,tuple or array containing size in x, y and z directions
    faultsizerange = list,tuple or array containing minimum and maximum size
    plane = yz, xz, or xy plane
    
    """
    network_size = np.array(network_size)
    lmin,lmax = faultsizerange
    # random position and random size within range
    x,y,z = np.random.random(3)*network_size
    size = (np.random.random()*(lmax - lmin) + lmin)

    
    if 'x' not in plane:
        fracturecoords = [[[x,y-size/2.,z-size/2.],[x,y+size/2.,z-size/2.]],
                          [[x,y-size/2.,z+size/2.],[x,y+size/2.,z+size/2.]]]
    elif 'y' not in plane:
        fracturecoords = [[[x-size/2.,y,z-size/2.],[x+size/2.,y,z-size/2.]],
                          [[x-size/2.,y,z+size/2.],[x+size/2.,y,z+size/2.]]]
    elif 'z' not in plane:
        fracturecoords = [[[x-size/2.,y-size/2.,z],[x+size/2.,y-size/2.,z]],
                          [[x-size/2.,y+size/2.,z],[x+size/2.,y+size/2.,z]]]
    fracturecoords = np.array(fracturecoords)
    fracturecoords[fracturecoords < 0.] = 0.
#    for i in range(3):
#        fracturecoords[fracturecoords[:,:,i] > network_size[i]] = network_size[i]

    return fracturecoords

def get_random_plane(pxyz):
    """
    select a plane (yz,xz, or xy) according to relative probability pxyz
    
    """
    planeind = int(np.random.choice(3,p=pxyz/sum(pxyz)))
    plane = 'xyz'.replace('xyz'[planeind],'')    
    
    return plane

def getplane(fracturecoord):
    """
    get the plane that the fracture lies in
    
    """
    planeind = [len(np.unique(fracturecoord[:,:,i])) for i in range(3)].index(1)
    
    return 'xyz'.replace('xyz'[planeind],'')
    

def checkintersection(fracturecoord,plane,elevation,return_2dcoords = False):
    """
    """
    fractureplane = getplane(fracturecoord)
    length = 0.
    
    # ensure plane is written in a consistent order
    if plane in ['yx','zy','zx']:
        plane = plane[::-1]
    
    # if fracture is in the same plane as the test plane then there is no
    # intersection (ignore coinciding planes)
    if plane == fractureplane:
        if return_2dcoords:
            coords2d = np.zeros((2,2))
            return length,coords2d
        else:
            return length

    if plane == 'xy':
        fmin,fmax = np.amin(fracturecoord[:,:,2]),np.amax(fracturecoord[:,:,2])
        # i, index with along fracture information
        i = ['xz','yz'].index(fractureplane)
        coords2d = np.array([[fracturecoord[:,:,0].min(),fracturecoord[:,:,0].max()],
                             [fracturecoord[:,:,1].min(),fracturecoord[:,:,1].max()]])
    elif plane == 'yz':
        fmin,fmax = np.amin(fracturecoord[:,:,0]),np.amax(fracturecoord[:,:,0])
        i = ['xy','xz'].index(fractureplane) + 1
        coords2d = np.array([[fracturecoord[:,:,1].min(),fracturecoord[:,:,1].max()],
                             [fracturecoord[:,:,2].min(),fracturecoord[:,:,2].max()]])
    elif plane == 'xz':
        fmin,fmax = np.amin(fracturecoord[:,:,1]),np.amax(fracturecoord[:,:,1])
        i = ['xy','yz'].index(fractureplane)*2
        coords2d = np.array([[fracturecoord[:,:,0].min(),fracturecoord[:,:,0].max()],
                             [fracturecoord[:,:,2].min(),fracturecoord[:,:,2].max()]])
   
    if ((elevation >= fmin) and (elevation <= fmax)):
        lmin,lmax = np.amin(fracturecoord[:,:,i]), np.amax(fracturecoord[:,:,i])
        length = lmax-lmin
    else:
        coords2d = np.zeros((2,2))

    if return_2dcoords:
        return length, coords2d
    else:
        return length



def get_fracture_coords(lvals,networksize,pxyz,return_Nf = False,a=3.5,alpha=10.):
    """
    get the coordinates of fractures to assign to a resistivity volume
    returns an array containing fracture coordinates (shape (nf,2,2,3) where
    nf is the number of fractures)
    
    inputs:
    lvals = 1D array containing fault length bins to calculate number of fractures
            for. The array defines the bin intervals so, for example [0.01,0.02,0.04]
            will calculate Nf for 1-2cm and 2-4cm fractures.
    networksize = tuple, array or list containing network size in x,y and z 
                  direction
    pxyz = tuple, array or list containing relative probability of a fault in 
           yz,xz and yz planes
    
    """
    # get the total volume in metres**3
    volume = np.prod(networksize)
    
    # get number of faults for each bin given by lvals
    Nf = get_faultlength_distribution(lvals,volume,alpha=alpha,a=a)
    
    fracturecoords = []

    
    for ii in range(len(Nf)):
        # loop through number of whole samples
        ni = 0
        while ni < Nf[ii]:
            # choose plane:
            plane = get_random_plane(pxyz)
            # create a fracture along the plane
            fracturecoords.append(create_random_fault(networksize,lvals[ii:ii+2],plane))
    
            ni += 1
        # deal with decimal point (fractional number of points) by selecting with probability given by leftover fraction
        randval = np.random.random()
        if randval < Nf[ii] % 1:
            plane = get_random_plane(pxyz)
            fracturecoords.append(create_random_fault(networksize,lvals[ii:ii+2],plane))
            ni += 1

    fracturecoords = np.array(fracturecoords)

    if return_Nf:
        return fracturecoords,Nf
    else:
        return fracturecoords


def coords2indices(fracturecoords,networksize,ncells):
    """
    convert x,y,z coordinates to indices for assignment to an array.
    
    Inputs:
    faultcoords: coordinates in format [[[x,y,z],[x,y,z]],
                                        [[x,y,z],[x,y,z]],]
    networksize: tuple, list or array containing size of network in metres in
                 x, y, and z directions
    ncells: tuple, list or array containing number of cells in fault array in 
            x, y and z directions
    
    """   

    fractureind = fracturecoords.copy()
    fractureind[fractureind < 0] = 0.
    
    nz,ny,nx = ncells

    # normalise to the size of the array
    fractureind[:,:,:,0] *= nx/networksize[0]
    fractureind[:,:,:,1] *= ny/networksize[1]
    fractureind[:,:,:,2] *= nz/networksize[2]
    
    # change to int for indexing and add 1 to account for nulls on the edges
    fractureind = fractureind.astype(int) + 1
    
    return fractureind
    

def add_faults_to_array(faultarray,fractureind):
    """
    add faults to an array where 1 is a faulted cell and 0 is a non-faulted 
    cell.
    
    
    """

    faultarray *= 0.
    
    for fi in fractureind:
        u0,v0,w0 = np.amin(fi,axis=(0,1))
        u1,v1,w1 = fi.max(axis=(0,1))
        size = fi.max(axis=(0,1)) - fi.min(axis=(0,1))
        perp = list(size).index(min(size))

        if perp == 0:
            faultarray[w0:w1,v0:v1+1,u0,2,0] = 1.
            faultarray[w0:w1+1,v0:v1,u0,1,0] = 1.
        elif perp == 1:
            faultarray[w0:w1,v0,u0:u1+1,2,1] = 1.
            faultarray[w0:w1+1,v0,u0:u1,0,1] = 1.
        elif perp == 2:
            faultarray[w0,v0:v1,u0:u1+1,1,2] = 1.
            faultarray[w0,v0:v1+1,u0:u1,0,2] = 1.
    

    faultarray = rna.add_nulls(faultarray)
    
    return faultarray
    
    
    
def get_faultsize(duvw,offset):
    """
    get fault size based on the u,v,w extents of the fault
    
    """
    
    size = int(np.amax(duvw) + 6 + offset) #2*(max(0.2*np.amax(duvw),6)) 
    size += size % 2
    
    return size



def get_faultpair_inputs(fractal_dimension,elevation_scalefactor,
                         mismatch_wavelength_cutoff,cellsize):

    faultpair_inputs = dict(D=fractal_dimension,
                            scalefactor=elevation_scalefactor)
    if mismatch_wavelength_cutoff is not None:
        faultpair_inputs['lc'] = mismatch_wavelength_cutoff
    if cellsize is not None:
        faultpair_inputs['cs'] = cellsize

    return faultpair_inputs
    

def offset_faults_with_deformation(h1,h2,fs,offset):
    h1n = h1.copy()
    h2n = h2.copy()
    
    overlap_avg = 0

    
    # progressively move along fault plane
    for oo in range(offset):
        # offset fault surfaces by one cell
        if oo == 0:
            # apply fault separation on first step
            h1n = h1n[:-1,1:] + fs
        else:
            h1n = h1n[:-1,1:]
        h2n = h2n[:-1,:-1]
        # compute aperture
        ap = h1n-h2n
        
        # sum overlapping heights (i.e. height of gouge created)
        # don't include edges which will eventually be gone
        apcalc = ap[:ap.shape[0]-(offset-oo-1),:ap.shape[1]-(offset-oo-1)]
        
        # apcalc = ap
        # print(np.sum(apcalc[apcalc < 0]))
        overlap_avg -= np.sum(apcalc[apcalc < 0])/apcalc.size
        
        # remove negative apertures
        # first, compute new fault surface height (=average of height 1 and height 
        # 2, i.e. both fault surfaces have been "eroded" by an equal amount)
        newheight = np.mean([h1n,h2n],axis=0)
        # newheight = h2n[:]
        h1n[ap<0] = newheight[ap<0]
        h2n[ap<0] = newheight[ap<0]
        
        
    return ap,h1n,h2n,overlap_avg


def assign_fault_aperture(fault_uvw,
                          ncells,
                          cs = 0.25e-3,
                          fault_separation=1e-4, 
                          fault_surfaces = None,
                          offset=0, 
                          deform_fault_surface=False,
                          fractal_dimension = 2.5, 
                          mismatch_wavelength_cutoff = None, 
                          elevation_scalefactor = None,
                          elevation_prefactor = 1.,
                          correct_aperture_for_geometry = True,
                          aperture_type = 'random',
                          fill_array = True,
                          aperture_list=None,
                          aperture_list_electric = None,
                          aperture_list_hydraulic = None,
                          preserve_negative_apertures = False,
                          random_numbers_dir=None,
                          minimum_aperture=None
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
    fault_separation = array containing fault separation values normal to fault surface,
                              length same as fault_uvw
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
    aperture_type, 'random' or 'constant' - random (variable) or constant aperture
    fill_array = whether or not to create an aperture array or just return the
                 fault aperture, trimmed to the size of the fault, and the
                 indices for the aperture array
    ===========================================================================    
    """
    
    nx,ny,nz = ncells
    
    if aperture_type == 'list':
        fill_array = True
    
    
    if fill_array:
        ap_array = np.array([np.ones((nz+2,ny+2,nx+2,3,3))*1e-50]*3) # yz, xz and xy directions

    if ((aperture_type != 'list') or (aperture_list is None)):
        aperture_list = []
        
    # if either electric or hydraulic aperture is None, set them both to None
    if aperture_list_electric is None:
        aperture_list_hydraulic = None
    if aperture_list_hydraulic is None:
        aperture_list_electric = None
    aperture_list_c = []
    aperture_list_f = []

    if not np.iterable(fault_separation):
        fault_separation = np.ones(len(fault_uvw))*fault_separation
        
#    ap_array[0] *= 1e-50
    bvals = []
    faultheights = []
    overlap_vol = []
    # default value for overlap, unless it is updated by deforming fault surfaces
    overlap_avg = 0

    for i, nn in enumerate(fault_uvw):
#        print nn
        bvals.append([])
        # get minimum and maximum x,y and z coordinates
        u0,v0,w0 = np.amin(nn, axis=(0,1))
        u1,v1,w1 = np.amax(nn, axis=(0,1))
        
        # get size of original fault so we get correct scale factor
        size_noclip = max(u1-u0,v1-v0,w1-w0)

        # make sure extents are within the array
        u1 = min(u1,nx + 1)
        v1 = min(v1,ny + 1)
        w1 = min(w1,nz + 1)
        u0 = min(u0,nx + 1)
        v0 = min(v0,ny + 1)
        w0 = min(w0,nz + 1)
        # size in the x, y and z directions
        duvw = np.array([u1-u0,v1-v0,w1-w0])
        du,dv,dw = (duvw*0.5).astype(int)

        # define size, add some padding to account for edge effects and make 
        # the fault square as I am not sure if fft works properly for non-
        # square geometries


        # if list of apertures provided, assign these to the array. Because these
        # apertures have been pre-trimmed to fit the array, we assign them 
        # differently to if they were calculated from scratch.
        print("aperture type",aperture_type)
        if ((aperture_type == 'list') and (aperture_list is not None)):
            
            du1,dv1,dw1 = u1-u0,v1-v0,w1-w0
            
            # loop through true aperture, hydraulic aperture, electric aperture
            for iii,ap in enumerate(aperture_list):
                print("ap.shape",ap.shape)
                dperp=list(duvw).index(0)

                if dperp == 0:
                    try:
                        ap_array[iii,w0:w1+1,v0:v1+1,u0-1:u1+1] += ap[i][:dw1+1,:dv1+1]#np.amax([ap_array[iii,w0:w1+1,v0:v1+1,u0-1:u1+1],ap[i][:dw1+1,:dv1+1]],axis=0)
                    except:
#                            print "aperture wrong shape, resetting fault extents"
                        u1,v1,w1 = np.array([u0,v0,w0]) + ap[i][:dw1+1,:dv1+1].shape[2::-1] - np.array([2,1,1])
#                            print dperp,u0,v0,w0,u1,v1,w1
                        ap_array[iii,w0:w1+1,v0:v1+1,u0-1:u1+1] += ap[i][:dw1+1,:dv1+1]
                elif dperp == 1:
                    try:
                        ap_array[iii,w0:w1+1,v0-1:v0+1,u0:u1+1] += ap[i][:dw1+1,:,:du1+1]#np.amax([ap_array[iii,w0:w1+1,v0-1:v0+1,u0:u1+1],ap[i][:dw1+1,:,:du1+1]],axis=0)
                    except:
#                            print "aperture wrong shape, resetting fault extents"
                        u1,v1,w1 = np.array([u0,v0,w0]) + ap[i][:dw1+1,:,:du1+1].shape[2::-1] - np.array([1,2,1])
#                            print dperp,u0,v0,w0,u1,v1,w1
                        ap_array[iii,w0:w1+1,v0-1:v0+1,u0:u1+1] += ap[i][:dw1+1,:,:du1+1]
                elif dperp == 2:
                    try:
                        ap_array[iii,w0-1:w0+1,v0:v1+1,u0:u1+1] += ap[i][:,:dv1+1,:du1+1]#np.amax([ap_array[iii,w0-1:w0+1,v0:v1+1,u0:u1+1],ap[i][:,:dv1+1,:du1+1]],axis=0)
                    except:
#                            print "aperture wrong shape, resetting fault extents"
                        u1,v1,w1 = np.array([u0,v0,w0]) + ap[i][:,:dv1+1,:du1+1].shape[2::-1] - np.array([1,1,2])
#                            print dperp,u0,v0,w0,u1,v1,w1
                        ap_array[iii,w0-1:w0+1,v0:v1+1,u0:u1+1] += ap[i][:,:dv1+1,:du1+1]
        elif aperture_type not in ['random','constant']:
            aperture_type = 'random'
        
        if aperture_type in ['random','constant']:

            # if offset between 0 and 1, assume it is a fraction of fault size
            if 0 < offset < 1:
                offset = int(np.round(offset*size_noclip))
            else:
                # ensure it is an integer
                offset = int(offset)
            
            # get size of fault including padding
            size = get_faultsize(duvw,offset)
            
            # define direction normal to fault
            direction = list(duvw).index(0)
            # get fault pair inputs as a dictionary
            faultpair_inputs = get_faultpair_inputs(fractal_dimension,
                                                    elevation_scalefactor,
                                                    mismatch_wavelength_cutoff,
                                                    cs)
                
            faultpair_inputs['random_numbers_dir'] = random_numbers_dir
            faultpair_inputs['prefactor'] = elevation_prefactor
            
            build = False
            if fault_surfaces is None:
                
                build = True
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
                    print("fault surfaces wrong type")
                
            if build:
                print("building new faults"),
                if aperture_type == 'random':
                    h1,h2 = rnfa.build_fault_pair(size, size_noclip, **faultpair_inputs)
                else:
                    h1,h2 = [np.zeros((size,size))]*2


                
            h1d = h1.copy()
            h2d = h2.copy()
            if offset > 0:
                
                if deform_fault_surface:
                    print("deforming fault surface")
                    b, h1dd, h2dd, overlap_avg = offset_faults_with_deformation(h1, h2, 
                                                                   fault_separation[i], 
                                                                   offset)
                    h1d[offset:,offset:] = h1dd
                    h2d[offset:,:-offset] = h2dd
                else:
                    print("not deforming fault surface")
                    b = h1[offset:,offset:] - h2[offset:,:-offset] + fault_separation[i]
            else:
                b = h1 - h2 + fault_separation[i]
                
            # set zero values to really low value to allow averaging
            if not preserve_negative_apertures:
                if minimum_aperture is None:
                    minimum_aperture = 1e-50
                b[b <= minimum_aperture] = minimum_aperture
                
            # centre indices of array b
            cb = (np.array(np.shape(b))*0.5).astype(int)
            
            if aperture_type in ['random', 'list']:
                # opportunity to provide corrected apertures
                if aperture_list_electric is not None:
                    bc = aperture_list_electric[i]
                    bf = aperture_list_hydraulic[i]
                else:
                    if correct_aperture_for_geometry:
                        print("correcting for geometry")
                        bf, bc = rnfa.correct_aperture_for_geometry(h1d[offset:,offset:],
                                                                    b,
                                                                    fault_separation[i],
                                                                    cs)
                    else:
                        print("not correcting apertures for geometry")
                        # bf, bc = [np.array([b[:-1,:-1]]*3)]*2
                        bf, bc = [[np.mean([b[1:,1:],b[1:,:-1],
                                            b[:-1,1:],b[:-1,:-1]],axis=0)]*3 for _ in range(2)]
            else:
                print("not correcting apertures for geometry 2")
                # bf, bc = [np.array([b[:-1,:-1]]*3)]*2
                bf, bc = [[np.mean([b[1:,1:],b[1:,:-1],
                                    b[:-1,1:],b[:-1,:-1]],axis=0)]*3 for _ in range(2)]
            tmp_aplist = []
            # physical aperture
            bphy = [np.mean([b[1:,1:],b[1:,:-1],
                                b[:-1,1:],b[:-1,:-1]],axis=0)]*3
            # assign the corrected apertures to aperture array

            for ii,bb in enumerate([bphy,bf,bc]):
                b0,b1,b2 = bb
                if direction == 0:
                    b0vals,b1vals,b2vals = b0[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-dv:cb[1]+dv+duvw[1]%2],\
                                           b1[cb[0]-dw:cb[0]+dw+duvw[2]%2,cb[1]-dv:cb[1]+dv+duvw[1]%2+1],\
                                           b2[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-dv:cb[1]+dv+duvw[1]%2+1]/2.
                    if fill_array:
                        if w1-w0+1 > int(np.shape(b2vals)[0]):
                            print("indices don't match up, w0 {}, w1 {}, b2vals shape[0] {}".format(w0,w1,b2vals.shape[0]))
                            w1 = int(np.shape(b2vals)[0])+w0-1
                        elif w1-w0+1 < int(np.shape(b2vals)[0]):
                            print("indices don't match up, w0 {}, w1 {}, b2vals shape[0] {}".format(w0,w1,b2vals.shape[0]))
                            b2vals = b2vals[:w1-w0+1]
                            b1vals = b1vals[:w1-w0]
                            b0vals = b0vals[:w1-w0+1]
                        if v1-v0+1 > int(np.shape(b2vals)[1]):
                            print("indices don't match up, v0 {}, v1 {}, b2vals shape[1] {}".format(v0,v1,b2vals.shape[1]))
                            v1 = int(np.shape(b2vals)[1])+v0-1
                        elif v1-v0+1 < int(np.shape(b2vals)[1]):
                            print("indices don't match up, v0 {}, v1 {}, b2vals shape[1] {}".format(v0,v1,b2vals.shape[1]))
                            b2vals = b2vals[:,:v1-v0+1]
                            b1vals = b1vals[:,:v1-v0+1]
                            b0vals = b0vals[:,:v1-v0]
                        # faults perpendicular to x direction, i.e. yz plane
                        ap_array[ii,w0:w1+1,v0:v1+1,u0-1,0,0] += b2vals
                        ap_array[ii,w0:w1+1,v0:v1+1,u0,0,0] += b2vals
                        # y direction opening in x direction
                        ap_array[ii,w0:w1+1,v0:v1,u0,1,0] += b0vals
                        # z direction opening in x direction
                        ap_array[ii,w0:w1,v0:v1+1,u0,2,0] += b1vals
                    #print "assigning aperture to list"
                    aperture = np.zeros((w1-w0+1,v1-v0+1,2,3,3))
                    aperture[:,:,0,0,0] = b2vals
                    aperture[:,:,1,0,0] = b2vals
                    aperture[:,:-1,1,1,0] = b0vals
                    aperture[:-1,:,1,2,0] = b1vals
                        
                elif direction == 1:
                    b0vals,b1vals,b2vals = b0[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-du:cb[1]+du+duvw[0]%2],\
                                           b1[cb[0]-dw:cb[0]+dw+duvw[2]%2,cb[1]-du:cb[1]+du+duvw[0]%2+1],\
                                           b2[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-du:cb[1]+du+duvw[0]%2+1]/2.
                    if fill_array:
                        # correct for slight discrepancies in array shape
                        if w1-w0+1 > int(np.shape(b2vals)[0]):
                            print("indices don't match up, w0 {}, w1 {}, b2vals shape[0] {}".format(w0,w1,b2vals.shape[0]))
                            w1 = int(np.shape(b2vals)[0])+w0-1
                        elif w1-w0+1 < int(np.shape(b2vals)[0]):
                            print("indices don't match up, w0 {}, w1 {}, b2vals shape[0] {}".format(w0,w1,b2vals.shape[0]))
                            b2vals = b2vals[:w1-w0+1]
                            b1vals = b1vals[:w1-w0]
                            b0vals = b0vals[:w1-w0+1]
                        if u1-u0+1 > int(np.shape(b2vals)[1]):
                            print("indices don't match up, u0 {}, u1 {}, b2vals shape[1] {}".format(u0,u1,b2vals.shape[1]))
                            u1 = int(np.shape(b2vals)[1])+u0-1
                        elif u1-u0+1 < int(np.shape(b2vals)[1]):
                            print("indices don't match up, u0 {}, u1 {}, b2vals shape[1] {}".format(u0,u1,b2vals.shape[1]))
                            b2vals = b2vals[:,:u1-u0+1]
                            b1vals = b1vals[:,:u1-u0+1]
                            b0vals = b0vals[:,:u1-u0]
                        # faults perpendicular to y direction, i.e. xz plane
                        ap_array[ii,w0:w1+1,v0-1,u0:u1+1,1,1] += b2vals
                        ap_array[ii,w0:w1+1,v0,u0:u1+1,1,1] += b2vals
                        # x direction opening in y direction
                        ap_array[ii,w0:w1+1,v0,u0:u1,0,1] += b0vals
                        # z direction opening in y direction
                        ap_array[ii,w0:w1,v0,u0:u1+1,2,1] += b1vals
                    #print "assigning aperture to list"
                    aperture = np.zeros((w1+1-w0,2,u1+1-u0,3,3))
                    aperture[:,0,:,1,1] = b2vals
                    aperture[:,1,:,1,1] = b2vals
                    aperture[:,1,:-1,0,1] = b0vals
                    aperture[:-1,1,:,2,1] = b1vals
                    
                        
                elif direction == 2:
                    b0vals,b1vals,b2vals = b0[cb[0]-dv:cb[0]+dv+duvw[1]%2+1,cb[1]-du:cb[1]+du+duvw[0]%2],\
                                           b1[cb[0]-dv:cb[0]+dv+duvw[1]%2,cb[1]-du:cb[1]+du+duvw[0]%2+1],\
                                           b2[cb[0]-dv:cb[0]+dv+duvw[1]%2+1,cb[1]-du:cb[1]+du+duvw[0]%2+1]/2.
                    if fill_array:
                        # correct for slight discrepancies in array shape
                        if v1-v0+1 > int(np.shape(b2vals)[0]):
                            print("indices don't match up, v0 {}, v1 {}, b2vals shape[0] {}".format(v0,v1,b2vals.shape[0]))
                            v1 = int(np.shape(b2vals)[0])+v0-1
                        elif v1-v0+1 < int(np.shape(b2vals)[0]):
                            print("indices don't match up, v0 {}, v1 {}, b2vals shape[0] {}".format(v0,v1,b2vals.shape[0]))
                            b2vals = b2vals[:v1-v0+1]
                            b1vals = b1vals[:v1-v0]
                            b0vals = b0vals[:v1-v0+1]
                        if u1-u0+1 > int(np.shape(b2vals)[1]):
                            print("indices don't match up, u0 {}, u1 {}, b2vals shape[1] {}".format(u0,u1,b2vals.shape[1]))
                            u1 = int(np.shape(b2vals)[1])+u0-1
                        elif u1-u0+1 < int(np.shape(b2vals)[1]):
                            print("indices don't match up, u0 {}, u1 {}, b2vals shape[1] {}".format(u0,u1,b2vals.shape[1]))
                            b2vals = b2vals[:,:u1-u0+1]
                            b1vals = b1vals[:,:u1-u0+1]
                            b0vals = b0vals[:,:u1-u0]
                        # faults perpendicular to z direction, i.e. xy plane
                        ap_array[ii,w0-1,v0:v1+1,u0:u1+1,2,2] += b2vals
                        ap_array[ii,w0,v0:v1+1,u0:u1+1,2,2] += b2vals
                        # x direction opening in z direction
                        ap_array[ii,w0,v0:v1+1,u0:u1,0,2] += b0vals
                        # y direction opening in z direction
                        ap_array[ii,w0,v0:v1,u0:u1+1,1,2] += b1vals
                    #print "assigning aperture to list"
                    aperture = np.zeros((2,v1+1-v0,u1+1-u0,3,3))
                    aperture[0,:,:,2,2] = b2vals
                    aperture[1,:,:,2,2] = b2vals
                    aperture[1,:,:-1,0,2] = b0vals
                    aperture[1,:-1,:,1,2] = b1vals
                
                tmp_aplist.append(aperture)
    
                bvals[-1].append([bb,b0,b1])
                
            faultheights.append([h1,h2])
            # average overlap height per cell * cellsize ** 2 * number of cells
            overlap_vol.append(overlap_avg * \
                               cs**2 * \
                               np.prod(np.array(duvw)[np.array(duvw) > 0])) 
            aperture_list.append(tmp_aplist[0])
            aperture_list_f.append(tmp_aplist[1])
            aperture_list_c.append(tmp_aplist[2])

        
#        ap_array[i] *= fault_array

    if fill_array:
        for ii in range(3):
            rna.add_nulls(ap_array[ii])
        if aperture_type == 'list':
#            print len(aperture_list)
            aperture_list_f = aperture_list[1]
            aperture_list_c = aperture_list[2]
            aperture_list = aperture_list[0]
        if not preserve_negative_apertures:
            ap_array[(np.isfinite(ap_array))&(ap_array < 2e-50)] = 2e-50
        aperture_c = ap_array[2]
        aperture_f = ap_array[1]
        aperture_array = ap_array[0]
        return aperture_list,aperture_list_f,aperture_list_c,\
               aperture_array,aperture_f,aperture_c,faultheights,overlap_vol
    else:
        return aperture_list,aperture_list_f,aperture_list_c,faultheights,\
            overlap_vol



def update_from_precalculated(rv,effective_apertures_fn,permeability_matrix=1e-18):
    effective_apertures = np.loadtxt(effective_apertures_fn)

    # add extreme values so that we cover the entire interpolation range
    # < min; hydraulic aperture = sqrt(matrix permeability * 12); electric aperture = 1e-50
    # > max; fault sep = hydraulic aperture = electric aperture
    first_row = [-1,np.sqrt(12*permeability_matrix),1e-50]
    last_row = [1,1,1]
    effective_apertures = np.concatenate([[first_row],effective_apertures,[last_row]],axis=0)
    
    # create interpolation functions for effective apertures
    feah = interp1d(effective_apertures[:,0],effective_apertures[:,1])
    feae = interp1d(effective_apertures[:,0],effective_apertures[:,2])
    
    # update aperture values
    for i in range(len(rv.aperture)):
        for j in range(len(rv.aperture[0])):
            for k in range(len(rv.aperture[0,0])):
                for ii in range(3):
                    jjlist = [0,1,2]
                    jjlist.remove(ii)
                    for jj in jjlist:                       
                        if (rv.fault_array[i,j,k,jj,ii] and np.isfinite(rv.aperture[i,j,k,jj,ii])):
                            rv.aperture_hydraulic[i,j,k,jj,ii] = feah(rv.aperture[i,j,k,jj,ii])
                            rv.aperture_electric[i,j,k,jj,ii] = feae(rv.aperture[i,j,k,jj,ii])
    
    # initialise electrical and hydraulic resistance
    rv.initialise_electrical_resistance()
    rv.initialise_permeability()
    
    return rv



def add_random_fault_sticks_to_arrays(Rv, Nfval, fault_length_m, fault_width, 
                                      hydraulic_aperture, resistivity,pz,
                                      fault_lengths_assigned=None):

    ncells = Rv.ncells[1]
    cellsize = Rv.cellsize[1]
    Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = cellsize
    
    # array to record what fault lengths were assigned where
    if fault_lengths_assigned is None:
        fault_lengths_assigned = np.zeros_like(Rv.resistivity)
    

    if Nfval > 0:
        orientationj = np.random.choice([0,1],p=[1.0 - pz, pz],size=Nfval)
        faultsj = orientationj * int(fault_length_m/cellsize)
        orientationi = (1-orientationj).astype(int)
        faultsi = orientationi * int(fault_length_m/cellsize)
        


        # j axis (vertical faults) open in the y direction
        idxo = np.ones_like(faultsj,dtype=int)
        # i axis (horizontal faults) open in the z direction
        idxo[faultsi>0] = 2
        
        # i axis (horizontal faults) are associated with y connectors
        idxc = np.ones_like(faultsj,dtype=int)
        # j axis (vertical faults) are associated with z connectors
        idxc[faultsj>0] = 2
        
        # can only initialise a fault where there isn't one already of that size or bigger
        if np.any(Rv.aperture >= fault_width):
            i = np.zeros(Nfval,dtype=int)
            j = np.zeros(Nfval,dtype=int)
            # faults in i direction (horizontal) are y connectors opening in z direction
            available_ij_i = np.column_stack(np.where(Rv.aperture[:, :,1,1,2] < fault_width))
            # make a list of random indices to pull from available_ij_i
            random_indices_i = np.random.choice(np.arange(len(available_ij_i)),replace=False,size=sum(orientationi))
            i[orientationi==1] = available_ij_i[random_indices_i][:,1]
            j[orientationi==1] = available_ij_i[random_indices_i][:,0]
            # faults in j direction (horizontal) are y connectors opening in z direction
            available_ij_j = np.column_stack(np.where(Rv.aperture[:, :,1,2,1] < fault_width))
            # make a list of random indices to pull from available_ij_i
            random_indices_j = np.random.choice(np.arange(len(available_ij_j)),replace=False,size=sum(orientationj))
            j[orientationj==1] = available_ij_j[random_indices_j][:,0]
            i[orientationj==1] = available_ij_j[random_indices_j][:,1]
        else:
            # pick a random location for the fault centre
            i = np.random.randint(1,ncells+2,size=Nfval)
            j = np.random.randint(1,ncells+2,size=Nfval)

        j0 = (j - faultsj/2).astype(int)
        j1 = (j + faultsj/2).astype(int)
        
        # add extra length if we have an odd-cellsize length fault as cutting 
        # in half truncates the fault
        if int(fault_length_m/cellsize)%2 == 1:
            extra_bit = np.random.choice([0,1],size=Nfval)
            j0 -= (extra_bit * orientationj).astype(int)
            j1 += ((1-extra_bit) * orientationj).astype(int)
        
        # truncate so we don't go out of bounds
        j0[j0 < 1] = 1
        j1[j1 > ncells + 1] = ncells + 1
        
        
        i0 = (i - faultsi/2).astype(int)
        i1 = (i + faultsi/2).astype(int)

        # add extra length if we have an odd-cellsize length fault as cutting 
        # in half truncates the fault
        if int(fault_length_m/cellsize)%2 == 1:
            i0 -= (extra_bit * orientationi).astype(int)
            i1 += ((1-extra_bit) * orientationi).astype(int)
        
        # truncate so we don't go out of bounds
        i0[i0 < 1] = 1
        i1[i1 > ncells + 1] = ncells + 1

        
        # initialise indices to update
        idx_i = i0*1
        idx_j = j0*1

        # randomly pull list of resistivity and hydraulic aperture values from 
        # the array, if they are arrays.
        assign_dict = {}
        idxs = None
        for val, name in [[hydraulic_aperture, 'aph'],
                          [resistivity, 'resf']]:

            if np.iterable(val):
                # choose some random indices
                # only define idxs once (use same indices for all properties)
                if idxs is None:
                    idxs = np.random.choice(np.arange(len(val)), size=idx_j.shape)
                
                if len(val.shape) == 2:
                    assign_dict[name] = np.zeros(len(idx_j))
                    assign_dict[name][np.where(orientationi)] = val[:,0][idxs][np.where(orientationi)]
                    assign_dict[name][np.where(orientationj)] = val[:,1][idxs][np.where(orientationj)]
                    
                    # import os
                    # savefile_res = r'C:\tmp\resistivity_val_length%.4fm.npy'%fault_length_m
                    # savefile_idxs = r'C:\tmp\orientationj_val_length%.4fm.npy'%fault_length_m
                    # if not os.path.exists(savefile_res):
                    #     np.save(savefile_res,assign_dict['resf'])
                    # if not os.path.exists(savefile_idxs):
                    #     np.save(savefile_idxs,orientationj)                    
                else:
                    assign_dict[name] = val[idxs]
            else:
                assign_dict[name] = val

        
        for add_idx in range(int(fault_length_m/cellsize)):
            # filter to take out indices where aperture is already larger than proposed update
            # or null in the resistivity array
            filt = np.array([k for k in range(len(idx_j)) if (Rv.aperture[idx_j[k],idx_i[k],1,idxc[k],idxo[k]] < fault_width\
                                                              and not np.isnan(Rv.resistivity[idx_j[k],idx_i[k],1,idxc[k]]))])
            if len(filt) > 0:
                idx_jj = idx_j[filt]
                idx_ii = idx_i[filt]
                idxo_i = idxo[filt]
                idxc_i = idxc[filt]
                
                # apply filter to the values to assign
                values_to_assign = {}
                for key in assign_dict.keys():
                    if np.iterable(assign_dict[key]):
                        values_to_assign[key] = assign_dict[key][filt]

                Rv.aperture[idx_jj, idx_ii,1,idxc_i,idxo_i] = fault_width #
                # print(Rv.aperture[idx_j, idx_i,1,1,idxo])
                Rv.aperture_electric[idx_jj, idx_ii,1,idxc_i,idxo_i] = fault_width
                Rv.aperture_hydraulic[idx_jj, idx_ii,1,idxc_i,idxo_i] = values_to_assign['aph']
                Rv.resistance[idx_jj, idx_ii,1,idxc_i] = values_to_assign['resf']/cellsize
                Rv.resistivity[idx_jj, idx_ii,1,idxc_i] = values_to_assign['resf']
                fault_lengths_assigned[idx_jj, idx_ii, 1, idxc_i] = fault_length_m
                idx_i[np.all([faultsi>0],axis=0)] += 1
                idx_j[np.all([faultsj>0],axis=0)] += 1
                
                idx_i[idx_i > ncells + 1] = ncells + 1
                idx_j[idx_j > ncells + 1] = ncells + 1

    return Rv, fault_lengths_assigned