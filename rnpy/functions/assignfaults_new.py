# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:55:34 2016

@author: a1655681
"""

import numpy as np
import rnpy.functions.array as rna
import rnpy.functions.faultaperture as rnfa



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
    volume = np.product(networksize)
    
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
        x0,y0,z0 = np.amin(fi,axis=(0,1))
        x1,y1,z1 = fi.max(axis=(0,1)) + 1
        size = fi.max(axis=(0,1)) - fi.min(axis=(0,1))
        perp = list(size).index(min(size))
        i1 = [i for i in range(3) if i != perp]

        for ii in i1:
            faultarray[z0:z1,y0:y1,x0:x1,ii,perp] = 1.

    

    faultarray = rna.add_nulls(faultarray)
    
    return faultarray
    
    
    
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
    


def assign_fault_aperture(fault_uvw,
                          ncells,
                          cs = 0.25e-3,
                          fault_separation=1e-4, 
                          fault_surfaces = None,
                          offset=0, 
                          fractal_dimension = 2.5, 
                          mismatch_wavelength_cutoff = None, 
                          elevation_scalefactor = None,
                          correct_aperture_for_geometry = True,
                          aperture_type = 'random',
                          fill_array = True,
                          aperture_list=None
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
        aperture_list_c = []
        aperture_list_f = []

    if type(fault_separation) is float:
        fault_separation = np.ones(len(fault_uvw))*fault_separation
        
#    ap_array[0] *= 1e-50
    bvals = []
    faultheights = []

    for i, nn in enumerate(fault_uvw):
#        print nn
        bvals.append([])
        # get minimum and maximum x,y and z coordinates
        u0,v0,w0 = np.amin(nn, axis=(0,1))
        u1,v1,w1 = np.amax(nn, axis=(0,1))

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

        if aperture_type == 'list':
            if aperture_list is not None:
                du1,dv1,dw1 = u1-u0,v1-v0,w1-w0
                #print "u0,v0,w0,u1,v1,w1",u0,v0,w0,u1,v1,w1
                #print "dw1,dv1,du1",dw1,dv1,du1
                
#                print ap_array.shape
                for iii,ap in enumerate(aperture_list):
                    print "ap[i]",ap[i],len(ap[i])
#                    print np.shape(ap_array[iii,w0:w1+1,v0:v1+1,u0-1:u1+1])
                    dperp=list(duvw).index(0)
#                    print ap[i].shape

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
            else:
                aperture_type = 'random'
        
        if aperture_type in ['random','constant']:
            size = get_faultsize(duvw,offset)
            # define direction normal to fault
            direction = list(duvw).index(0)
            # get fault pair inputs as a dictionary
            faultpair_inputs = get_faultpair_inputs(fractal_dimension,
                                                    elevation_scalefactor,
                                                    mismatch_wavelength_cutoff,
                                                    cs)
            
                
                
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
                if aperture_type == 'random':
                    h1,h2 = rnfa.build_fault_pair(size, **faultpair_inputs)
                else:
                    h1,h2 = [np.zeros((size,size))]*2
    
            if offset > 0:
                b = h1[offset:,offset:] - h2[offset:,:-offset] + fault_separation[i]
            else:
                b = h1 - h2 + fault_separation[i]
                
            # set zero values to really low value to allow averaging
            b[b <= 1e-50] = 1e-50
            # centre indices of array b
            cb = (np.array(np.shape(b))*0.5).astype(int)
            
            if aperture_type == 'random':
                if correct_aperture_for_geometry:
                    bf, bc = rnfa.correct_aperture_geometry(h1[offset:,offset:],b,cs)
                else:
                    bf, bc = [np.array([b[:-1,:-1]]*3)]*2
            else:
                bf, bc = [np.array([b[:-1,:-1]]*3)]*2
    #        print np.shape(b),np.shape(bf), np.shape(bc)
            tmp_aplist = []
            # assign the corrected apertures to aperture array
            for ii,bb in enumerate([[b[:-1,:-1]]*3,bf,bc]):
                b0,b1,b2 = bb
                if direction == 0:
                    b0vals,b1vals,b2vals = b0[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-dv:cb[1]+dv+duvw[1]%2],\
                                           b1[cb[0]-dw:cb[0]+dw+duvw[2]%2,cb[1]-dv:cb[1]+dv+duvw[1]%2+1],\
                                           b2[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-dv:cb[1]+dv+duvw[1]%2+1]/2.
                    if fill_array:
                        # faults perpendicular to x direction, i.e. yz plane
                        ap_array[ii,w0:w1+1,v0:v1+1,u0-1,0,0] += b2vals
                        ap_array[ii,w0:w1+1,v0:v1+1,u0,0,0] += b2vals
                        # y direction opening in x direction
                        ap_array[ii,w0:w1+1,v0:v1,u0,1,0] += b0vals
                        # z direction opening in x direction
                        ap_array[ii,w0:w1,v0:v1+1,u0,2,0] += b1vals

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
                        # faults perpendicular to y direction, i.e. xz plane
                        ap_array[ii,w0:w1+1,v0-1,u0:u1+1,1,1] += b2vals
                        ap_array[ii,w0:w1+1,v0,u0:u1+1,1,1] += b2vals
                        # x direction opening in y direction
                        ap_array[ii,w0:w1+1,v0,u0:u1,0,1] += b0vals
                        # z direction opening in y direction
                        ap_array[ii,w0:w1,v0,u0:u1+1,2,1] += b1vals

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
                        # faults perpendicular to z direction, i.e. xy plane
                        ap_array[ii,w0-1,v0:v1+1,u0:u1+1,2,2] += b2vals
                        ap_array[ii,w0,v0:v1+1,u0:u1+1,2,2] += b2vals
                        # x direction opening in z direction
                        ap_array[ii,w0,v0:v1+1,u0:u1,0,2] += b0vals
                        # y direction opening in z direction
                        ap_array[ii,w0,v0:v1,u0:u1+1,1,2] += b1vals
                    
                    aperture = np.zeros((2,v1+1-v0,u1+1-u0,3,3))
                    aperture[0,:,:,2,2] = b2vals
                    aperture[1,:,:,2,2] = b2vals
                    aperture[1,:,:-1,0,2] = b0vals
                    aperture[1,:-1,:,1,2] = b1vals
                

                tmp_aplist.append(aperture)

                bvals[-1].append([bb,b0,b1])
                
            faultheights.append([h1,h2])
            aperture_list.append(tmp_aplist[0])
            aperture_list_f.append(tmp_aplist[1])
            aperture_list_c.append(tmp_aplist[2])
              
    #        ap_array[i] *= fault_array

    
    if fill_array:
        for ii in range(3):
            rna.add_nulls(ap_array[ii])
        if aperture_type == 'list':
            print len(aperture_list)
            aperture_list_f = aperture_list[1]
            aperture_list_c = aperture_list[2]
            aperture_list = aperture_list[0]
        ap_array[(np.isfinite(ap_array))&(ap_array < 2e-50)] = 0.
        aperture_c = ap_array[2]
        aperture_f = ap_array[1]
        aperture_array = ap_array[0]

        return aperture_list,aperture_list_f,aperture_list_c,\
               aperture_array,aperture_f,aperture_c,faultheights
    else:
        return aperture_list,aperture_list_f,aperture_list_c,faultheights
