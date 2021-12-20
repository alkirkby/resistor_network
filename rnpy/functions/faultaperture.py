# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681

functions relating to creation of a fractal aperture geometry

"""
import os
import numpy as np
import scipy.stats as stats
import scipy.optimize as so
from scipy.ndimage import median_filter


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
    y1[1:,1:int(size/2)+1] = y1a[1:,1:]
    y1[1:int(size/2)+1,int(size/2)+1:] = np.real(y1a[int(size/2)+1:,1:][::-1,::-1]) \
                          - 1j*np.imag(y1a[int(size/2)+1:,1:][::-1,::-1])
    y1[int(size/2)+1:,int(size/2)+1:] = np.real(y1a[1:int(size/2)+1,1:][::-1,::-1]) \
                         - 1j*np.imag(y1a[1:int(size/2)+1,1:][::-1,::-1])    
    
    return y1


def get_faultpair_defaults(cs, lc):
    """
    get sensible defaults for fault height elevation based on cellsize.    
    returns std, lc, fc, fcw
    
    """   
    
    if lc is None:
        lc = 1e-3

    fc = 1./lc
    

    return lc, fc



def build_fault_pair(size,size_noclip,D=2.4,cs=2.5e-4,scalefactor=1e-3,
                     lc=None,fcw=None,matchingmethod='me',beta=0.6,
                     random_numbers_dir=None,prefactor=1.):
    """
    Build a fault pair by the method of Ishibashi et al 2015 JGR (and previous
    authors). Uses numpy n dimensional inverse fourier transform. Returns two
    fault surfaces
    =================================inputs====================================
    size, integer = dimensions (number of cells across) for fault (fault will be square)
    size_noclip = size of fault prior to clipping to size of volume (used
                  to calculate scaling of elevation)
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
    random_numbers_dir = directory containing random numbers to use to generate
                fault surfaces to use for testing purposes

    ===========================================================================    
    """
    
    if size % 2 != 0:
        size += 1
    
    lc, fc = get_faultpair_defaults(cs, lc)

    
    
    # get frequency components
    pl = np.fft.fftfreq(size+1,d=cs)#*1e-3/cs
    pl[0] = 1.
    # define frequencies in 2d
    p,q = np.meshgrid(pl[:int(size/2)+1],pl)
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
    f2 = f.copy()
    # take an average of the x and y frequency magnitudes, as matching parameters measured on profiles
    f2 = 2./(1./np.abs(p)+1./np.abs(q))
#    k = 1./f
#    kc = 1./fc
    fc = fc#*1e-3/cs
    gamma = f2/fc
    gamma[f2 > fc] = 1.
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
    if random_numbers_dir:
        R1 = np.loadtxt(os.path.join(random_numbers_dir,'R1.dat'))
        R2 = np.loadtxt(os.path.join(random_numbers_dir,'R2.dat'))
    # define 2 sets of uniform random numbers
    else:
        R1 = np.random.random(size=np.shape(f))
        R2 = np.random.random(size=np.shape(f))
    # np.savetxt(os.path.join(r'C:\tmp\R1.dat'),R1,fmt='%.4f')
    # np.savetxt(os.path.join(r'C:\tmp\R2.dat'),R2,fmt='%.4f')
    # define fourier components
    y1 = prepare_ifft_inputs(prefactor*(p**2+q**2)**(-(4.-D)/2.)*np.exp(1j*2.*np.pi*R1))
    
    if matchingmethod == 'Glover':
        gamma[f2>2.*fc] = 0.
        gamma[f2<2.*fc] = beta*(1.-(f2[f2<2.*fc]/(2.*fc)))
        y2 = prepare_ifft_inputs(prefactor*(p**2+q**2)**(-(4.-D)/2.)*np.exp(1j*2.*np.pi*(R1*gamma+R2*(1.-gamma))))
    else:
        gamma = f2/fc
        gamma[f2 > fc] = 1.
        y2 = prepare_ifft_inputs(prefactor*(p**2+q**2)**(-(4.-D)/2.)*np.exp(1j*2.*np.pi*(R1+gamma*R2)))
   
    
    # use inverse discrete fast fourier transform to get surface heights
    h1 = np.fft.irfftn(y1,y1.shape)
    h2 = np.fft.irfftn(y2,y2.shape)
    
    # scale so that standard deviation is as specified
    if scalefactor is not None:
        std = scalefactor*(cs*size_noclip)**(3.-D)
        ic = int(h1.shape[1]/2)
        i0 = int(ic-size_noclip/2)
        i1 = int(ic+size_noclip/2)
        meanstd = np.average([np.std(line[i0:i1]) for line in h1])
        scaling_factor = std/meanstd
        h1 = h1*scaling_factor
        h2 = h2*scaling_factor
        
    return h1, h2


def old_correct_aperture_geometry(faultsurface_1,aperture,dl):
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
    
    aperture = np.copy(aperture)
    aperture[aperture<1e-50] = 1e-50    
    
    # height values at nodes
    s1n = [np.mean([s1[:-1],s1[1:]],axis=0),
           np.mean([s1[:,:-1],s1[:,1:]],axis=0)]
           
    # mean vertical hydraulic aperture, defined based on the cube of the aperture as 
    # fluid flow is defined based on the cube, in x and y directions
    bnf = [np.mean([aperture[:-1]**3.,aperture[1:]**3.],axis=0)**(1./3.),
          np.mean([aperture[:,:-1]**3.,aperture[:,1:]**3.],axis=0)**(1./3.)]
    # mean vertical electric aperture
    bnc = [np.mean([aperture[:-1],aperture[1:]],axis=0),
          np.mean([aperture[:,:-1],aperture[:,1:]],axis=0)]
    
    # height of surface 2 at nodes
    s2n = [s1n[i] + bnf[i] for i in range(2)]
    s2nc = [s1n[i] + bnc[i] for i in range(2)]

    # midpoint elevation at nodes
    rzn = [np.mean([s1n[i],s2n[i]],axis=0) for i in range(2)]
    rznc = [np.mean([s1n[i],s2nc[i]],axis=0) for i in range(2)]
   
    # rzp = z component of normal vector perpendicular to flow
    rzp = [dl/((np.average([rz[1:,:-1],rz[1:,1:]],axis=0)-np.average([rz[:-1,:-1],rz[:-1,1:]],axis=0))**2.+dl**2.)**0.5,
           dl/((np.average([rz[1:,:-1],rz[:-1,:-1]],axis=0)-np.average([rz[1:,1:],rz[:-1,1:]],axis=0))**2.+dl**2.)**0.5]    

    # aperture at centre of plane between nodes, defined for flow in x and y directions
    bpf = [np.mean([bnf[0][:,:-1],bnf[0][:,1:]],axis=0),
          np.mean([bnf[1][:-1],bnf[1][1:]],axis=0)]
    bpc = [np.mean([bnc[0][:,:-1],bnc[0][:,1:]],axis=0),
          np.mean([bnc[1][:-1],bnc[1][1:]],axis=0)]

    # distance between node points, slightly > dl if the plates are sloping
    dr = [((dl)**2 + (rzn[0][:,:-1]-rzn[0][:,1:])**2)**0.5,
          ((dl)**2 + (rzn[1][:-1,:]-rzn[1][1:,:])**2)**0.5]
    drc = [((dl)**2 + (rznc[0][:,:-1]-rznc[0][:,1:])**2)**0.5,
          ((dl)**2 + (rznc[1][:-1,:]-rznc[1][1:,:])**2)**0.5]          
    
    # nz - z component of unit normal vector from mid point of plates
    nz = [dl*rzp[0]/dr[0],
          dl*rzp[1]/dr[1]]
    nzc = [dl/drc[0],
           dl/drc[1]]
    
    # beta - correction factor when applying harmonic mean
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

    

    # initialise a corrected array for current
    bc = np.ones((3,ny-1,nx-1))
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
#    elif evaluation_point == 'midpoint':
#        # initialise a corrected array for current
#        bc = np.ones((2,ny-1,nx-1))
#        # x values
#        bc[0] = stats.hmean([bchv[0,1],bchv[0,0]],axis=0)
#        # y values
#        bc[1] = stats.hmean([bchv[1,1],bchv[1,0]],axis=0)

    # assign connectors perpendicular to fault
    bc[2] = np.mean([aperture[:-1,:-1],aperture[:-1,1:],aperture[1:,:-1],aperture[1:,1:]],axis=0)    

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
    bf3 = np.ones((3,ny-1,nx-1))
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
#    elif evaluation_point == 'midpoint':
#        bf3 = np.ones((2,ny-1,nx-1))
#        # x values
#        bf3[0] = stats.hmean([bf3beta[0,1],bf3beta[0,0]],axis=0)
#        # y values
#        bf3[1] = stats.hmean([bf3beta[1,1],bf3beta[1,0]],axis=0)

    bf = bf3**(1./3.)
    
    # assign connectors perpendicular to fault
    bf[2] = np.mean([aperture[:-1,:-1],aperture[:-1,1:],aperture[1:,:-1],aperture[1:,1:]],axis=0)    

    return bf, bc
    

def get_value_edges(arr, hv = 1):
    """
    
    Get an array value on the "edges" between nodes by taking a mean along the
    x and y axis.

    Parameters
    ----------
    arr : numpy array, shape (n, m)
        Array to calculate edge values for.
    hv : int, optional
        Half volume to calculate for. 1 (First half volume) will include the first
        value across the axis of averaging but not the last.  2 (second half
        volume) will exclude the first value across the axis of averaging but
        include the last. The default is 1.

    Returns
    -------
    numpy array, shape (2, n-1, m-1)
        Averaged values on edges defined for the x and y directions

    """
    if hv == 1:
        # first half volume
        return np.array([(arr[1:,:-1] + arr[:-1,:-1])/2., (arr[:-1,1:] + arr[:-1,:-1])/2.])
    elif hv == 2:
        # second half volume
        return np.array([(arr[1:,1:] + arr[:-1,1:])/2., (arr[1:,1:] + arr[1:,:-1])/2.])


def get_value_plane_centre(arr):
    """
    
    Get an array value on the centre of the plane between 4 nodes by taking a 
    mean

    Parameters
    ----------
    arr : numpy array, shape (n, m)
        Array to calculate plane values for.

    Returns
    -------
    numpy array, shape (n-1, m-1)
        Averaged values on planes

    """
    value_plane_centre = (arr[:-1,1:] + arr[:-1,:-1] + arr[1:,1:] + arr[1:,:-1])/4.
    return np.array([value_plane_centre, value_plane_centre])




def get_half_volume_aperture(bN, bNsm, rN, dl, prop='hydraulic', hv=1):
    """
    
    Get the mean hydraulic/electric aperture along each successive half-
    aperture along a fault plane.
    
    For hydraulic aperture the Local Cubic Law correction of Brush & Thomson 
    (2003) equation 33, originally from Nichol et al 1999, is used with
    modification that the fault surfaces are smoothed for calculation of the
    theta angle between plates and the midpoint, with smoothing dependent on
    fault separation, taken as an X-point median around the point where
    X is 1/2 the number of horizontal cells fault separation. E.g. if cellsize
    is 0.1mm and fault separation is 1mm then it would be a 5-point median
    around each point
    
    For electric aperture the correction of Kirkby et al. (2016) equation 23 is
    applied.

    Parameters
    ----------
    bN : array, shape (m, n)
        DESCRIPTION.
    bNsm : TYPE
        DESCRIPTION.
    rN : TYPE
        DESCRIPTION.
    dl : float
        cell size in metres.
    prop : TYPE, optional
        DESCRIPTION. The default is 'hydraulic'.
    hv : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    b_hv : TYPE
        DESCRIPTION.
    beta_hv : TYPE
        DESCRIPTION.

    """
    
    # dl for the half volume
    dlhv = dl/2.
    
    # aperture at edges between node points. Dimensions 2, nx, ny as different for 
    # flow in x and y directions
    bf_hv = get_value_edges(bN, hv=hv)
    bfsm_hv = get_value_edges(bNsm, hv=hv)

    # aperture at centre of plates
    bP = get_value_plane_centre(bN)
    bPsm = get_value_plane_centre(bNsm)
    
    # z component of position vector at edges (perpendicular to flow) between faces
    rf_hv = get_value_edges(rN, hv=hv)
    rP = get_value_plane_centre(rN)
    
    # z component of the unit normal vector from the midpoint (by definition, nz = deltal/(|rf - rP|))
    nz_hv = dlhv/(((rf_hv - rP)**2 + dlhv**2)**0.5)
    

    if prop == 'hydraulic':
        # define correction terms kappa and beta
        kappa_hv = nz_hv**2
        beta_hv = nz_hv**4
    
        # define angle theta (using smoothed plates)
        theta_hv = np.arctan(kappa_hv*np.abs(bfsm_hv - bPsm)/dlhv)
        
        # theta correction factor
        thetacorr_hv = 3*(np.tan(theta_hv) - theta_hv)/(np.tan(theta_hv))**3.
        
        # at very low theta angles the correction becomes unstable
        thetacorr_hv[np.abs(theta_hv) < 1e-3] = 1.0
        
        # corrected aperture for the half volume
        b_hv = ((2.*(bf_hv**2.)*(bP**2.)/(bf_hv + bP))*thetacorr_hv)**(1./3.)
        
        
    elif prop == 'electric':
        # beta = nz**2 for current
        beta_hv = nz_hv**2

        b_hv = (bf_hv - bP)/(np.log(bf_hv) - np.log(bP))
        b_hv[np.abs(bf_hv - bP) < 1e-6] = bf_hv[np.abs(bf_hv - bP) < 1e-6]
    
    return b_hv, beta_hv, bP


def smooth_fault_surfaces(h1,h2,fs,dl):
    ks = int(round(fs/dl/2.))
    size = max(h1.shape) - 1
    
    if ks > size*5:
        h1sm = np.ones_like(h1)*0.
        h2sm = h1sm + fs
    if ks > 1:
        h1sm = median_filter(h1,size=ks)
        h2sm = median_filter(h2,size=ks)
        h2sm[h2sm-h1sm < 1e-10] = h1sm[h2sm-h1sm < 1e-10] + 1e-10
    else:
        h1sm = h1.copy()
        h2sm = h2.copy()
        
    return h1sm, h2sm
    



def correct_aperture_for_geometry(h1,b,fs,dl,km=1e-18):
    """
    
    Get mean hydraulic and electric aperture along fault surfaces.
    
    For hydraulic aperture the Local Cubic Law correction of Brush & Thomson 
    (2003) equation 33, originally from Nichol et al 1999, is used with the
    following modifications:
        - the fault surfaces are smoothed for calculation of the
          theta angle between plates and the midpoint, with smoothing dependent 
          on fault separation, taken as an X-point median around the point where
          X is 1/2 the number of horizontal cells fault separation. E.g. if 
          cellsize is 0.1mm and fault separation is 1mm then it would be a 
          5-point median around each point
        - average values are centred on the planes not on the edges between 
          planes as in B & T.
    
    For electric aperture the correction of Kirkby et al. (2016) equation 23 is
    applied.

    Parameters
    ----------
    h1 : array, shape (m, n)
        surface elevations for fault surface 1.
    h2 : array, shape (m, n)
        surface elevations for fault surface 2.
    fs : float
        separation between the two fault planes.
    dl : float
        cell size in metres.

    Returns
    -------
    bmean_hydraulic : array, shape (m, n)
        DESCRIPTION.
    bmean_electric : TYPE
        DESCRIPTION.

    """
    h2 = h1 + b
    
    # higher threshold for zero aperture to ensure calculations are stable
    zero_ap = np.where(h2-h1 < 1e-10)
    h2[zero_ap] = h1[zero_ap] + 1e-10
    
    h1sm,h2sm = smooth_fault_surfaces(h1, h2, fs, dl)

    # aperture at nodes
    bN = h2 - h1
    bNsm = h2sm - h1sm
    
    # horizontal length of half volumes
    dlhv = dl/2.
    
    # midpoint between plates, at nodes (smoothed)
    rN = (h1sm + h2sm)/2.
    
    b_hv1, beta_hv1, bP = get_half_volume_aperture(bN, bNsm, rN, dl, prop='hydraulic',hv=1)
    b_hv2, beta_hv2, bP = get_half_volume_aperture(bN, bNsm, rN, dl, prop='hydraulic',hv=2)
    
    b3 = dl/(dlhv/(beta_hv1*b_hv1**3.) + dlhv/(beta_hv2*b_hv2**3.))
    
    bmean_hydraulic = b3**(1./3.)
    
    # add apertures for flow in direction perpendicular to fault plane
    bmean_hydraulic = np.array([bmean_hydraulic[0], bmean_hydraulic[1], bP[0]])
    
    be_hv1, betae_hv1, bP = get_half_volume_aperture(bN, bNsm, rN, dl, prop='electric',hv=1)
    be_hv2, betae_hv2, bP = get_half_volume_aperture(bN, bNsm, rN, dl, prop='electric',hv=2)
    
    bmean_electric = dl/(dlhv/(betae_hv1*be_hv1) + dlhv/(betae_hv2*be_hv2))
    
    # add apertures for flow in direction perpendicular to fault plane
    bmean_electric = np.array([bmean_electric[0], bmean_electric[1], bP[0]])
    # print(bmean_electric[1])
    
    return bmean_hydraulic, bmean_electric
    


def subsample_fault_edges(fault_edges,subsample_factor):
    # update fault edges for a subsampled rock volume
    
    return ((fault_edges - 1)/subsample_factor+1).astype(int)



def get_plane(fault_edges):
    # get plane (e.g. xy, xz, yz plane). Returns an integer representing
    # index of orthogonal direction to plane (0=yz, 1=xz, 2=xy)
    return np.where(np.ptp(fault_edges,axis=(0,1))==0)[0][0]

def subsample_aperture(aperture_list, fault_edges, factor):
    if factor % 2 != 0:
        factor = max(factor-1, 2)
    
    hw = int(factor/2)
    
    mean_aperture_c = np.zeros_like(aperture_list)
    new_aperture_list = []
    
    # loop through physical aperture, hydraulic aperture (1) and electric aperture (2)
    for i in range(3):
        new_aperture_list.append([])
        # loop through fault surfaces in list:
        for iii in range(len(aperture_list[i])):
            # if hydraulic aperture, hydraulic resistance proportional to aperture**3
            # need to account for this in averaging
            if i==1:
                pp = 3
            else:
                pp = 1
            # work out which plane the fault is in (0=yz, 1=xz, 2=xy)
            plane = get_plane(fault_edges[iii])
            # if faults are in the yz plane
            if plane == 0:
                # average for apertures perpendicular to fault plane (average of 4 surrounding cells)
                mean_aperture_c[i][iii][:,:,:,0] = aperture_list[i][iii][:,:,:,0]
                mean_aperture_c[i][iii][:-hw,:-hw,:,0] = np.mean([aperture_list[i][iii][:-hw,:-hw,:,0],
                                                          aperture_list[i][iii][:-hw,hw:,:,0],
                                                          aperture_list[i][iii][:-hw,:-hw,:,0],
                                                          aperture_list[i][iii][hw:,:-hw,:,0]],
                                                        axis=0)
                # average for apertures parallel to plane (harmonic means)
                mean_aperture_c[i][iii][:,:-hw,:,1] = stats.hmean([aperture_list[i][iii][:,:-hw,:,1]**pp,
                                                          aperture_list[i][iii][:,hw:,:,1]**pp],
                                                        axis=0)**(1./pp)
                mean_aperture_c[i][iii][:-hw,:,:,2] = stats.hmean([aperture_list[i][iii][:-hw,:,:,2]**pp,
                                                          aperture_list[i][iii][hw:,:,:,2]**pp],
                                                        axis=0)**(1./pp)
                
                new_aperture = mean_aperture_c[i][iii][::2,::2]

            
            elif plane == 1:
                # if faults are in the xz plane
                # average for apertures perpendicular to fault plane (average of 4 surrounding cells)
                mean_aperture_c[i][iii][:,:,:,1] = aperture_list[i][iii][:,:,:,1]
                mean_aperture_c[i][iii][:-hw,:,:-hw,1] = np.mean([aperture_list[i][iii][:-hw,:,:-hw,1],
                                                          aperture_list[i][iii][:-hw,:,hw:,1],
                                                          aperture_list[i][iii][:-hw,:,:-hw,1],
                                                          aperture_list[i][iii][hw:,:,:-hw,1]],
                                                        axis=0)
                
                # average for apertures parallel to plane (harmonic means)
                mean_aperture_c[i][iii][:,:,:-hw,0] = stats.hmean([aperture_list[i][iii][:,:,:-hw,0]**pp,
                                                          aperture_list[i][iii][:,:,hw:,0]**pp],
                                                        axis=0)**(1./pp)
                mean_aperture_c[i][iii][:-hw,:,:,2] = stats.hmean([aperture_list[i][iii][:-hw,:,:,2]**pp,
                                                          aperture_list[i][iii][hw:,:,:,2]**pp],
                                                        axis=0)**(1./pp)
                new_aperture = mean_aperture_c[i][iii][::2,:,::2]

            elif plane == 2:
                # if faults are in the xz plane
                # average for apertures perpendicular to fault plane (average of 4 surrounding cells)
                mean_aperture_c[i][iii][:,:,:,2] = aperture_list[i][iii][:,:,:,2]
                mean_aperture_c[i][iii][:,:-hw,:-hw,2] = np.mean([aperture_list[i][iii][:,:-hw,:-hw,2],
                                                          aperture_list[i][iii][:,:-hw,hw:,2],
                                                          aperture_list[i][iii][:,:-hw,:-hw,2],
                                                          aperture_list[i][iii][:,hw:,:-hw,2]],
                                                        axis=0)
                
                # average for apertures parallel to plane (harmonic means)
                mean_aperture_c[i][iii][:,:,:-hw,0] = stats.hmean([aperture_list[i][iii][:,:,:-hw,0]**pp,
                                                          aperture_list[i][iii][:,:,hw:,0]**pp],
                                                        axis=0)**(1./pp)
                mean_aperture_c[i][iii][:,:-hw,:,1] = stats.hmean([aperture_list[i][iii][:,:-hw,:,1]**pp,
                                                          aperture_list[i][iii][:,hw:,:,1]**pp],
                                                        axis=0)**(1./pp)
                new_aperture = mean_aperture_c[i][iii][:,::2,::2]

            new_aperture_list[i].append(new_aperture)
    
    return new_aperture_list
