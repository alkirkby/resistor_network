# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681

functions relating to creation of a fractal aperture geometry

"""

import numpy as np
import scipy.stats as stats

def prepare_ifft_inputs(y1a):
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


def get_faultpair_defaults(cs, std, lc, fcw):
    """
    get sensible defaults for fault height elevation based on cellsize.    
    returns std, lc, fc, fcw
    
    """   
    
    
    if std is None:
        std =  cs*2.
    if lc is None:
        lc = cs*4.

    # can't have a frequency cutoff equivalent to a wavelength of less than 2 cells
    fc = min(0.5,cs/lc)
    # update lc to reflect any changes in fc
    lc = cs/fc

    return std, lc, fc, min(0.25,fc)



def build_fault_pair(size,D=2.5,cs=2.5e-4,std=None,lc=None,fcw=None):
    """
    Build a fault pair by the method of Ishibashi et al 2015 JGR (and previous
    authors). Uses numpy n dimensional inverse fourier transform. Returns two
    fault surfaces
    =================================inputs====================================
    size, integer = size of fault (fault will be square)
    D, float = fractal dimension of returned fault, recommended values in range 
               [2.,2.5]
    std, float = standard deviation of surface height of fault 1, surface 2 
                 will be scaled by the same factor as height 1 so may not have 
                 exactly the same standard deviation but needs to be scaled the same to ensure the
                 surfaces are matched properly
    cs, float = cellsize, used to calculate defaults for lc,lcw and std
    lc, float = cutoff wavelength in metres for matching of faults, the two 
                fault surfaces will match at wavelengths greater than the 
                cutoff frequency, default is 1mm (1e-3)
    fcw, float = window to include for tapering of wavelengths above cutoff.

    ===========================================================================    
    """
    
    std, lc, fc, fcw = get_faultpair_defaults(cs, std, lc, fcw)    
    
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
    gamma[gamma < fc-fcw] = 0.
    gamma[(gamma < 1)&(gamma > 0)] -= (fc-fcw)
    gamma[(gamma < 1)&(gamma > 0)] /= fcw
    
    # define 2 sets of uniform random numbers
    R1 = np.random.random(size=np.shape(f))
    R2 = np.random.random(size=np.shape(f))
    # define fourier components
    y1 = prepare_ifft_inputs((p**2+q**2)**(-(4.-D)/2.)*np.exp(1j*2.*np.pi*R1))
    y2 = prepare_ifft_inputs((p**2+q**2)**(-(4.-D)/2.)*np.exp(1j*2.*np.pi*(R1+gamma*R2)))
   
    
    # use inverse discrete fast fourier transform to get surface heights
    h1 = np.fft.irfftn(y1,y1.shape)
    h2 = np.fft.irfftn(y2,y2.shape)
    # scale so that standard deviation is as specified
    if std is not None:
        scaling_factor = std/np.std(h1)
        h1 = h1*scaling_factor
        h2 = h2*scaling_factor
    
    return h1, h2


def correct_aperture_geometry(faultsurface_1,aperture,dl):
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
    bnf = [np.mean([aperture[:-1]**3.,aperture[1:]**3.],axis=0)**(1./3.),
          np.mean([aperture[:,:-1]**3.,aperture[:,1:]**3.],axis=0)**(1./3.)]
    bnc = [np.mean([aperture[:-1],aperture[1:]],axis=0),
          np.mean([aperture[:,:-1],aperture[:,1:]],axis=0)]
    s2n = [s1n[i] + bnf[i] for i in range(2)]
    s2nc = [s1n[i] + bnc[i] for i in range(2)]

    # midpoint elevation at nodes
    rzn = [np.mean([s1n[i],s2n[i]],axis=0) for i in range(2)]
    rznc = [np.mean([s1n[i],s2nc[i]],axis=0) for i in range(2)]
   
    # rzp = z component of normal vector perpendicular to flow
    rzp = [dl/((np.average([rz[1:,:-1],rz[1:,1:]],axis=0)-np.average([rz[:-1,:-1],rz[:-1,1:]],axis=0))**2.+dl**2.)**0.5,
           dl/((np.average([rz[1:,:-1],rz[:-1,:-1]],axis=0)-np.average([rz[1:,1:],rz[:-1,1:]],axis=0))**2.+dl**2.)**0.5]    
    
    # aperture at plane between nodes
    bpf = [np.mean([bnf[0][:,:-1],bnf[0][:,1:]],axis=0),
          np.mean([bnf[1][:-1],bnf[1][1:]],axis=0)]
    bpc = [np.mean([bnc[0][:,:-1],bnc[0][:,1:]],axis=0),
          np.mean([bnc[1][:-1],bnc[1][1:]],axis=0)]
    
    # distance between node points, slightly > dl
#    print(rzn)
    dr = [((dl)**2 + (rzn[0][:,:-1]-rzn[0][:,1:])**2)**0.5,
          ((dl)**2 + (rzn[1][:-1,:]-rzn[1][1:,:])**2)**0.5]
    drc = [((dl)**2 + (rznc[0][:,:-1]-rznc[0][:,1:])**2)**0.5,
          ((dl)**2 + (rznc[1][:-1,:]-rznc[1][1:,:])**2)**0.5]          
    
    # nz - z component of unit normal vector from mid point of plates
    nz = [dl*rzp[0]/dr[0],
          dl*rzp[1]/dr[1]]
    nzc = [dl/drc[0],
           dl/drc[1]]
    
    # beta - correction factor when applying harmonic mean, for current it is equal to kappa
    betaf = [nz[i]**3.*dl/dr[i] for i in range(2)]
    betac = [nzc[i]*dl/drc[i] for i in range(2)]

    # calculate bc, or b corrected for geometry for current, for each of the first
    # and second half volumes associated with each node.
    bchv = np.zeros((2,2,ny-1,nx-1))

    for j in range(ny-1):
        for i in range(nx-1):
            for hv in range(2):
                if np.abs((bnc[0][j,i+1-hv] - bpc[0][j,i])/(bnc[0][j,i+1-hv] + bpc[0][j,i])) > 1e-8:
                    bchv[0,hv,j,i] = ((bnc[0][j,i+1-hv] - bpc[0][j,i])/\
                    np.log(bnc[0][j,i+1-hv]/bpc[0][j,i]))*betac[0][j,i]
                else:
                    bchv[0,hv,j,i] = bpc[0][j,i]*betac[0][j,i]
                if np.abs((bnc[1][j+1-hv,i] - bpc[1][j,i])/(bnc[1][j+1-hv,i] + bpc[1][j,i])) > 1e-8:
                    bchv[1,hv,j,i] = ((bnc[1][j+1-hv,i] - bpc[1][j,i])/\
                    np.log(bnc[1][j+1-hv,i]/bpc[1][j,i]))*betac[1][j,i]
                else:
                    bchv[1,hv,j,i] = bpc[1][j,i]*betac[1][j,i]
    bc = np.ones((2,ny-1,nx-1))
    # x values
    # first column, only one half volume
    bc[0,:,0] = bchv[0,1,:,0]#/rzp[0][:,0]
    # remaining columns, harmonic mean of 2 adjacent columns
    bc[0,:,1:] = stats.hmean([bchv[0,1,:,1:],bchv[0,0,:,:-1]],axis=0)

    # y values
    # first row, only one half volume (second half volume)
    bc[1,0] = bchv[1,1,0]#/rzp[1][0]
    
    bc[bc < 1e-50] = 1e-50

    # remaining rows, harmonic mean of 2 adjacent rows
    bc[1,1:] = stats.hmean([bchv[1,1,1:],bchv[1,0,:-1]],axis=0)    

    # theta, relative angle of tapered plates for correction, defined in x and y directions
    theta = np.abs(np.array([np.arctan((s1n[0][:,:-1]-s1n[0][:,1:])/dl) -\
                             np.arctan((s2n[0][:,:-1]-s2n[0][:,1:])/dl),
                             np.arctan((s1n[1][:-1]-s1n[1][1:])/dl) -\
                             np.arctan((s2n[1][:-1]-s2n[1][1:])/dl)]))
    # set zero theta values to small number to prevent silly error messages (makes no difference to result)
    theta[np.abs(theta) < 1e-50] = 1e-50
    # corrected b**3, defined in x and y directions, and comprising first and 
    # second half volumes
    tf = 3*(np.tan(theta)-theta)/((np.tan(theta))**3)
    # tf is undefined for theta = 0 (0/0) so need to fix this, should equal 1 
    # (i.e. parallel plates)
    tf[theta==0.] = 1.
    # calculate product bf**3 * beta, average for first and second half volume
    # associated with each node. Need to divide by rzp to account for any
    # longer cross-sectional length if the fracture is tilted perpendicular
    # to flow direction.
    bf3beta = np.array([[(2.*bnf[0][:,1:]**2*bpf[0]**2/(bnf[0][:,1:]+bpf[0]))*tf[0]*betaf[0]/rzp[0],
                         (2.*bnf[0][:,:-1]**2*bpf[0]**2/(bnf[0][:,:-1]+bpf[0]))*tf[0]*betaf[0]/rzp[0]],
                        [(2.*bnf[1][1:]**2*bpf[1]**2/(bnf[1][1:]+bpf[1]))*tf[1]*betaf[1]/rzp[1],
                         (2.*bnf[1][:-1]**2*bpf[1]**2/(bnf[1][:-1]+bpf[1]))*tf[1]*betaf[1]/rzp[1]]])
    bf3beta[np.isnan(bf3beta)] = 1e-150
    bf3beta[bf3beta < 1e-50] = 1e-50

    # initialise array to contain averaged bf**3 and bc values
    bf3 = np.ones((2,ny-1,nx-1))
    
    # x values
    # first column, only one half volume
    bf3[0,:,0] = bf3beta[0,1,:,0]#/rzp[0][:,0]
    # remaining columns, harmonic mean of 2 adjacent columns
    bf3[0,:,1:] = stats.hmean([bf3beta[0,1,:,1:],bf3beta[0,0,:,:-1]],axis=0)

    # y values
    # first row, only one half volume (second half volume)
    bf3[1,0] = bf3beta[1,1,0]#/rzp[1][0]

    # remaining rows, harmonic mean of 2 adjacent rows
    bf3[1,1:] = stats.hmean([bf3beta[1,1,1:],bf3beta[1,0,:-1]],axis=0)

    bf = bf3**(1./3.)

    return bf, bc