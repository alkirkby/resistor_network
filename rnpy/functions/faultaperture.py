# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681

functions relating to creation of a fractal aperture geometry

"""

import numpy as np
import scipy.stats as stats
import scipy.optimize as so


def R(freq):
#    return freq**(-0.5*np.log10(freq))
    lfreq = np.log10(freq)
    poly = [-0.00914017, -0.07034244, -0.4616544 ,  0.53296253,  0.0834078 ]
#    poly = [-0.00914017, -0.07034244, -0.4616544 ,  0.53296253,  0.1834078 ]
#    poly = np.array([-0.01828034, -0.14068488, -0.9233088 ,  1.06592505,  0.16681561])
    powers=np.arange(len(poly))[::-1]
#    return 10**(-0.31506034*lfreq**2 + 0.5665134*lfreq + 0.02426884)
    value = 0.
    for weight, power in np.vstack([poly,powers]).T:
#        print weight,power
        value += weight*lfreq**power
    
    return 10**value


def func(gammaf,freq,var):
    return 2*(1.-(np.sin(2.*np.pi*gammaf)/(2.*np.pi*gammaf))) - R(freq)
    

  


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


def get_faultpair_defaults(cs, lc):
    """
    get sensible defaults for fault height elevation based on cellsize.    
    returns std, lc, fc, fcw
    
    """   
    
    if lc is None:
        lc = 1e-3

    fc = cs/lc
    

    return lc, fc



def build_fault_pair(size,D=2.5,cs=2.5e-4,scalefactor=None,lc=None,fcw=None):
    """
    Build a fault pair by the method of Ishibashi et al 2015 JGR (and previous
    authors). Uses numpy n dimensional inverse fourier transform. Returns two
    fault surfaces
    =================================inputs====================================
    size, integer = dimensions (number of cells across) for fault (fault will be square)
    D, float = fractal dimension of returned fault, recommended values in range 
               [2.,2.5]
    std, float = scaling factor for heights, heights are adjusted so their 
                 standard deviation equals scalefactor * (size * cs)**0.5
                 multiplied by size (=ncells in one direction, the surface is
                 square). Surface 2 will be scaled by the same factor as surface
                 1 
    cs, float = cellsize, used to calculate defaults for lc,lcw and std
    lc, float = cutoff wavelength in metres for matching of faults, the two 
                fault surfaces will match at wavelengths greater than the 
                cutoff frequency, default is 1mm (1e-3)
    fcw, float = window to include for tapering of wavelengths above cutoff.

    ===========================================================================    
    """
    
    lc, fc = get_faultpair_defaults(cs, lc)
    
    if scalefactor is None:
        scalefactor = 1e-3
        
    std = scalefactor*(cs*size)**(3.-D)
    
    # get frequency components
    pl = np.fft.fftfreq(size+1)#*1e-3/cs
    pl[0] = 1.
    # define frequencies in 2d
    p,q = np.meshgrid(pl[:size/2+1],pl)
    # define f
    
    f = 1./(1./p**2+1./q**2)**0.5#*(2**.5)

    # define gamma for correlation between surfaces
#    gamma = np.ones_like(f)
#    for fi in range(len(f)):
#        for fj in range(len(f[fi])):
#            # get spatial frequency in millimetres
#            freq = np.abs(f[fi,fj])*1e-3/cs
##            print freq
#            if freq < fc:
#                gamma[fi,fj] = so.newton(func,0.5,args=(freq,1.))
##    if np.amax(gamma) == 1.:
##        gamma /= np.amax(gamma[gamma<1.])
#    gamma[gamma<0] = 0.
#    gamma[gamma>1] = 1.
#    print gamma
    gamma = f.copy()
#    k = 1./f
#    kc = 1./fc
    fc = fc#*1e-3/cs
    gamma = f/fc
    gamma[f > fc] = 1.
#    gamma[gamma>1] = 1.
#    gamma[gamma<0] = 0.
#    gamma *= 3
#    gamma = 1.-10**(-f/fc)
#
#    gamma[gamma >= fc] = 1.
#    gamma[gamma < (fc-fcw)] = 0.
#    gamma[(gamma < 1)&(gamma > 0)] -= (fc-fcw)
#    gamma[(gamma < 1)&(gamma > 0)] /= fcw
    
#    gamma[f < 0.1] /= 2.
    
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
        meanstd = np.average([np.std(line) for line in h1])
        scaling_factor = std/meanstd
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
