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
import scipy.stats as stats

"""
===================Functions relating to class definition======================
"""

def _prepare_ifft_inputs(y1a):
    """
    creates an array with correct inputs for np.fft.irfftn to create a real-
    valued output. negative-frequency components are calculated as complex
    conjugates of positive-frequency components, reflected diagonally.
    
    """
    size = int(y1a.shape[0] - 1)
    y1 = np.zeros((size+1,size+1),dtype=complex)
    y1[1:,1:size/2+1] = y1a[1:,1:]
    y1[1:size/2+1,size/2+1:] = np.real(y1a[size/2+1:,1:][::-1,::-1]) \
                          - 1j*np.imag(y1a[size/2+1:,1:][::-1,::-1])
    y1[size/2+1:,size/2+1:] = np.real(y1a[1:size/2+1,1:][::-1,::-1]) \
                         - 1j*np.imag(y1a[1:size/2+1,1:][::-1,::-1])    
    
    return y1

def _build_fault_pair(size,fc=None,D=2.5,std=1e-3):
    """
    Build a fault pair by the method of Ishibashi et al 2015 JGR (and previous
    authors). Uses numpy n dimensional inverse fourier transform. Returns two
    fault surfaces
    =================================inputs====================================
    size, integer = size of fault (fault will be square)
    fc, float = cutoff frequency for matching of faults, the two fault surfaces 
                will match at frequencies greater than the cutoff frequency,
                default is 3 % of size
    D, float = fractal dimension of returned fault, recommended values in range 
               [2.,2.5]
    std, float = standard deviation of surface height of fault 1, surface 2 
                 will be scaled by the same factor as height 1 so may not have 
                 exactly the same standard deviation but needs to be scaled the same to ensure the
                 surfaces are matched properly
    ===========================================================================    
    """
    if fc is None:
        fc = size*3e-4
    # get frequency components
    pl = np.fft.fftfreq(size+1)
    pl[0] = 1.
    # define frequencies in 2d
    p,q = np.meshgrid(pl[:size/2+1],pl)
    # define f
    f = (p**2+q**2)**0.5
    # define gamma for correlation between surfaces
    gamma = f.copy()
    gamma[gamma >= fc] = 1.
    # define 2 sets of uniform random numbers
    R1 = np.random.random(size=np.shape(f))
    R2 = np.random.random(size=np.shape(f))
    # define fourier components
    y1 = _prepare_ifft_inputs((p**2+q**2)**(-(4.-D)/2.)*np.exp(1j*2.*np.pi*R1))
    y2 = _prepare_ifft_inputs((p**2+q**2)**(-(4.-D)/2.)*np.exp(1j*2.*np.pi*(R1+gamma*R2)))
    # use inverse discrete fast fourier transform to get surface heights
    h1 = np.fft.irfftn(y1,y1.shape)
    h2 = np.fft.irfftn(y2,y2.shape)
    # scale so that standard deviation is as specified
    scaling_factor = std/np.std(h1)
    h1 = h1*scaling_factor
    h2 = h2*scaling_factor
    
    return h1, h2

def _add_nulls(in_array):
    """
    initialise a fault, resistivity, permeability, aperture etc array by
    putting nulls at edges in correct spots
    
    """

    in_array[:,:,-1,0] = np.nan
    in_array[:,-1,:,1] = np.nan
    in_array[-1,:,:,2] = np.nan
    in_array[:,:,0] = np.nan
    in_array[:,0,:] = np.nan
    in_array[0,:,:] = np.nan
    
    return in_array


def _correct_aperture_geometry(faultsurface_1,aperture,dl):
    """
    correct an aperture array for geometry, e.g. tapered plates or sloping
    plates
    
    =================================inputs====================================
    faultsurface_1 = numpy array containing elevation values for the bottom
                     fault surface
    aperture = numpy array containing aperture values (shape same as fault surface)
    dl = spacing in the x and y direction
    
    ===========================================================================    
    
    
    """

    ny,nx = np.array(aperture.shape).astype(int)
    # rename faultsurface_1 to make it shorter
    s1 = faultsurface_1
    
    # midpoint at original nodes
    rz = faultsurface_1 + aperture/2.
    
    aperture[aperture<1e-50] = 1e-50    
    
    # aperture and height values at nodes
    s1n = [np.mean([s1[:-1],s1[1:]],axis=0),
           np.mean([s1[:,:-1],s1[:,1:]],axis=0)]
           
    # mean vertical hydraulic aperture, defined based on the cube of the aperture as 
    # fluid flow is defined based on the cube
    bn = [np.mean([aperture[:-1]**3.,aperture[1:]**3.],axis=0)**(1./3.),
          np.mean([aperture[:,:-1]**3.,aperture[:,1:]**3.],axis=0)**(1./3.)]
    s2n = [s1n[i] + bn[i] for i in range(2)]
    
    # rzp = z component of normal vector perpendicular to flow
    rzp = [dl/((rz[:-1]-rz[1:])**2.+dl**2.)**0.5,
           dl/((rz[:,:-1]-rz[:,1:])**2.+dl**2.)**0.5] 
           
           
    # aperture and height values at plane between nodes
    s1p = np.mean([s1[:-1,:-1],s1[1:,1:],s1[1:,:-1],s1[:-1,1:]],axis=0)
    bp = [np.mean([bn[0][:,:-1],bn[0][:,1:]],axis=0),
          np.mean([bn[1][:-1],bn[1][1:]],axis=0)]
#    s2p = [s1p[i] + bp[i] for i in range(2)]
    
    
    # midpoint elevation at nodes
    rzn = [np.mean([s1n[i],s2n[i]],axis=0) for i in range(2)]
#    rzp = np.mean([s1p[i],s2p[i]],axis=0)
    
    # distance between node points, slightly > dl
#    print(rzn)
    dr = [((dl)**2 + (rzn[0][:,:-1]-rzn[0][:,1:])**2)**0.5,
          ((dl)**2 + (rzn[1][:-1,:]-rzn[1][1:,:])**2)**0.5]
                   
    # nz - z component of unit normal vector from mid point of plates
    nz = [dl*rzp[0][:,:-1]/dr[0],#
          dl*rzp[1][:-1]/dr[1]]#
    

    
    
    # kappa - correction factor for undulation defined in x and y directions
    kappa = [nz[i]*dl/dr[i] for i in range(2)]
    
    # beta - correction factor when applying harmonic mean, for current it is equal to kappa
    betaf = [nz[i]**3.*dl/dr[i] for i in range(2)]
    betac = [nz[i]*dl/dr[i] for i in range(2)]
    
    # theta, relative angle of tapered plates for correction, defined in x and y directions

#    theta = 2*np.array([np.arctan(0.5*kappa[0]*np.abs(bn[0][:,:-1]-bn[0][:,1:])/dl),
#                        np.arctan(0.5*kappa[1]*np.abs(bn[1][:-1,:]-bn[1][1:,:])/dl)])

    theta = np.abs(np.array([np.arctan((s1n[0][:,:-1]-s1n[0][:,1:])/dl) -\
                             np.arctan((s2n[0][:,:-1]-s2n[0][:,1:])/dl),
                             np.arctan((s1n[1][:-1]-s1n[1][1:])/dl) -\
                             np.arctan((s2n[1][:-1]-s2n[1][1:])/dl)]))
                        
    # corrected b**3, defined in x and y directions, and comprising first and 
    # second half volumes
    tf = 3*(np.tan(theta)-theta)/((np.tan(theta))**3)
    # tf is undefined for theta = 0 (0/0) so need to fix this, should equal 1 
    # (i.e. parallel plates)
    tf[theta==0.] = 1.
    bf3beta = np.array([[(2.*bn[0][:,1:]**2*bp[0]**2/(bn[0][:,1:]+bp[0]))*tf[0]*betaf[0],
                         (2.*bn[0][:,:-1]**2*bp[0]**2/(bn[0][:,:-1]+bp[0]))*tf[0]*betaf[0]],
                        [(2.*bn[1][1:]**2*bp[1]**2/(bn[1][1:]+bp[1]))*tf[1]*betaf[1],
                         (2.*bn[1][:-1]**2*bp[1]**2/(bn[1][:-1]+bp[1]))*tf[1]*betaf[1]]])
    bf3beta[np.isnan(bf3beta)] = 1e-150

    # initialise an array to contain averaged bf**3 values
    bf3 = np.ones((2,ny-1,nx-1))
    
    # x values
    # first column, only one half volume
    bf3[0,:,0] = bf3beta[0,1,:,0]/rzp[0][:,0]
    # remaining columns, harmonic mean of 2 adjacent columns
    bf3[0,:,1:] = stats.hmean([bf3beta[0,1,:,1:],bf3beta[0,0,:,:-1]],axis=0)
#    print("bf3**(1./3.)x",bf3[0]**(1./3.))

    bf3[0,:,1:] /= rzp[0][:,1:-1]

    # y values
    # first row, only one half volume (second half volume)
    bf3[1,0] = bf3beta[1,1,0]/rzp[1][0]
#    print(bf3**(1./3.))
    # remaining rows, harmonic mean of 2 adjacent rows
    bf3[1,1:] = stats.hmean([bf3beta[1,1,1:],bf3beta[1,0,:-1]],axis=0)
#    print(bf3**(1./3.))
#    print("bf3**(1./3.)y",bf3[1]**(1./3.))
#    print(np.average([rzp[1][1:],rzp[1][:-1]],axis=0))
#    print(bf3[1]**(1./3.))
    bf3[1,1:] = bf3[1,1:]/rzp[1][1:-1]
#    print(bf3[1]**(1./3.))
    b = bf3**(1./3.)

    return b
    

def add_fault_to_array(fault_mm,fault_array,direction=None):
    """
    
    """
    if type(fault_mm) == list:
        fault_mm = np.array(fault_mm)
    if direction is None:
        for i, mm in enumerate(fault_mm):
            if mm[1]-mm[0] == 0:
                direction = int(i)
                break
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
    
    # get uv extents of fault in local plane (i.e. exclude width normal to plane)
    u,v_ = [fault_mm[i] for i in range(3) if i != direction]
    v = np.array([[v_[0]]*2,[v_[1]]*2])
    faultuvw = [np.array([u,u[::-1]]),v]
    faultuvw.insert(direction,np.array([[fault_mm[direction,0]]*2]*2))

    return fault_array,np.array(faultuvw)

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
            fault_array,faultuvw = add_fault_to_array(mm,fault_array,direction=fo)
            faults.append(faultuvw+1.)
        else:
            break
    # make a new larger array of nans
    fault_array_final = np.zeros(list(np.array(np.shape(fault_array))[:-1]+1)+[3])
    
    # put the fault array into this array in the correct position.
    fault_array_final[1:,1:,1:] = fault_array
    # deal with edges
    fault_array_final = _add_nulls(fault_array_final)

    
    return fault_array_final,np.array(faults)


def assign_fault_aperture(fault_array,fault_uvw, 
                          dl = 1e-3,
                          fault_separation=1e-4, 
                          offset=0, 
                          fractal_dimension = 2.5, 
                          mismatch_frequency_cutoff = None, 
                          elevation_standard_deviation = 1e-4,
                          correct_for_geometry = True):
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
    dl = cellsize in metres, has to be same in x and y directions
    separation, float = fault separation normal to fault surface, in metres
    offset, integer = number of cells horizontal offset between surfaces.
    fractal_dimension, integer = fractal dimension of surface, recommended in 
                                 range [2.,2.5]
    mismatch_frequency_cutoff, integer = cutoff frequency for matching of 
                                         surfaces, default 3% of fault plane 
                                         size
    elevation_standard_deviation, integer = standard deviation of the height 
                                            of the fault surface
    correct_for_geometry, True/False, whether or not to correct aperture for
                                      geometry
       
    ===========================================================================    
    """
    fault_array = _add_nulls(fault_array)
    
    nx,ny,nz = np.array(np.shape(fault_array))[:3][::-1] 
    aperture_array = np.zeros_like(fault_array) # yz, xz and xy directions
    bvals = []

    for i, nn in enumerate(fault_uvw):
        u0,v0,w0 = np.amin(nn, axis=(1,2))
        u1,v1,w1 = np.amax(nn, axis=(1,2))
        duvw = np.array([u1-u0,v1-v0,w1-w0])
        du,dv,dw = (duvw*0.5).astype(int)

        # define size, add some padding to account for edge effects and make 
        # the fault square as I am not sure if fft is working properly for non-
        # square geometries
        size = int(np.amax(duvw)(1.+max(0.2*np.amax(duvw),4)) + offset)
        size += size%2
        
        # define direction normal to fault
        direction = list(duvw).index(0)
        
        # define cutoff frequency for correlation
        if mismatch_frequency_cutoff is None:
            mismatch_frequency_cutoff = size*3e-4
        h1,h2 = _build_fault_pair(size,
                                  mismatch_frequency_cutoff,
                                  fractal_dimension,
                                  elevation_standard_deviation)
        
        if offset > 0:
            b = h1[offset:,offset:] - h2[:-offset,:-offset] + fault_separation
        else:
            b = h1 - h2 + fault_separation
            
        # set zero values to really low value to allow averaging
        b[b <= 1e-50] = 1e-50
        cb = np.array(np.shape(b))*0.5
        
        if correct_for_geometry:
            bc = b.copy()
            bf,betaf,betac = _correct_aperture_geometry(h1[offset:,offset:],b,dl)
            
        else:
            bf,bc,betaf,betac = b.copy(),b.copy(),np.ones(2),np.ones(2)
        
        for bb, beta in [[bf,betaf],[bc,betac]]:
            bvals.append([])
            b0 = stats.hmean(np.array([beta[0]*bb[:,1:],beta[0]*bb[:,:-1]]),axis=0)
            b1 = stats.hmean(np.array([beta[1]*bb[1:],beta[1]*bb[:-1]]),axis=0)
            
            if direction == 0:
                aperture_array[w0:w1+1,v0:v1,u0,1] += b0[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-dv:cb[1]+dv+duvw[1]%2]
                aperture_array[w0:w1,v0:v1+1,u0,2] += b1[cb[0]-dw:cb[0]+dw+duvw[2]%2,cb[1]-dv:cb[1]+dv+duvw[1]%2+1]
            elif direction == 1:
                aperture_array[w0:w1+1,v0,u0:u1,0] += b0[cb[0]-dw:cb[0]+dw+duvw[2]%2+1,cb[1]-du:cb[1]+du+duvw[0]%2]
                aperture_array[w0:w1,v0,u0:u1+1,2] += b1[cb[0]-dw:cb[0]+dw+duvw[2]%2,cb[1]-du:cb[1]+du+duvw[0]%2+1]
            elif direction == 2:
                aperture_array[w0,v0:v1+1,u0:u1,0] += b0[cb[0]-dv:cb[0]+dv+duvw[1]%2+1,cb[1]-du:cb[1]+du+duvw[0]%2]
                aperture_array[w0,v0:v1,u0:u1+1,1] += b1[cb[0]-dv:cb[0]+dv+duvw[1]%2,cb[1]-du:cb[1]+du+duvw[0]%2+1]
            bvals[-1].append([b,bb,b0,b1])
    aperture_array *= fault_array
    
    return aperture_array,bvals




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

    for i in range(3):
        hydraulic_resistance[:,:,:,i] = mu*d[i]/(ly[i]*(aperture_array[:,:,:,i]**3/12.\
                                        +k_matrix*(ln[i]-aperture_array[:,:,:,i])))

    return hydraulic_resistance



def get_hydraulic_resistance_old(k,k_matrix,d,fracture_diameter,mu=1e-3):
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
    # take out central row to make the matrix square
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
    print(chunks)
    return chunks
