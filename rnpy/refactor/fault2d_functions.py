# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:39:21 2025

@author: alisonk
"""
import numpy as np
import os
from scipy.ndimage import median_filter
from utils import medfilt_subsampled


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



def build_fault_pair(size,size_noclip,fractal_dimension=2.4,cellsize=2.5e-4,elevation_scalefactor=1e-3,
                     mismatch_wavelength_cutoff=None,beta_glover=None,
                     random_numbers_dir=None):
    """
    Build a fault pair by the method of Ishibashi et al 2015 JGR (and previous
    authors). Uses numpy n dimensional inverse fourier transform. Returns two
    fault surfaces
    =================================inputs====================================
    size, integer = dimensions (number of cells across) for fault (fault will be square)
    size_noclip = size of fault prior to clipping to size of volume (used
                  to calculate scaling of elevation)
    fractal_dimension, float = fractal dimension of returned fault, recommended values in range 
               [2.,2.5]
    std, float = scaling factor for heights, heights are adjusted so their 
                 standard deviation equals elevation_scalefactor * (size * cs)**0.5
                 multiplied by size (=ncells in one direction, the surface is
                 square). Surface 2 will be scaled by the same factor as surface
                 1 
    cellsize, float = cellsize
    mismatch_wavelength_cutoff, float = cutoff wavelength in metres for matching of faults, the two 
                fault surfaces will match at wavelengths greater than the 
                cutoff frequency, default is 1mm (1e-3)
    beta_glover = beta factor to use if matching the fault planes using the method of Glover. 
                if not provided, uses the default method.
    random_numbers_dir = directory containing random numbers to use to generate
                fault surfaces if desired to fix random seed

    ===========================================================================    
    """
    
    if size % 2 != 0:
        size += 1
    
    fc = 1./mismatch_wavelength_cutoff

    
    
    # get frequency components
    pl = np.fft.fftfreq(size+1,d=cellsize)#*1e-3/cellsize
    pl[0] = 1.
    # define frequencies in 2d
    p,q = np.meshgrid(pl[:int(size/2)+1],pl)
    # define f
    
    f = 1./(1./p**2+1./q**2)**0.5#*(2**.5)

    # take an average of the x and y frequency magnitudes, as matching parameters measured on profiles
    f2 = 2./(1./np.abs(p)+1./np.abs(q))

    gamma = f2/fc
    gamma[f2 > fc] = 1.


    if random_numbers_dir:
        R1 = np.loadtxt(os.path.join(random_numbers_dir,'R1.dat'))
        R2 = np.loadtxt(os.path.join(random_numbers_dir,'R2.dat'))
    # define 2 sets of uniform random numbers
    else:
        R1 = np.random.random(size=np.shape(f))
        R2 = np.random.random(size=np.shape(f))
    np.savetxt(os.path.join(r'C:\tmp\R1.dat'),R1,fmt='%.4f')
    np.savetxt(os.path.join(r'C:\tmp\R2.dat'),R2,fmt='%.4f')
    # define fourier components
    y1 = prepare_ifft_inputs((p**2+q**2)**(-(4.-fractal_dimension)/2.)*np.exp(1j*2.*np.pi*R1))
    
    if beta_glover:
        gamma[f2>2.*fc] = 0.
        gamma[f2<2.*fc] = beta_glover*(1.-(f2[f2<2.*fc]/(2.*fc)))
        y2 = prepare_ifft_inputs((p**2+q**2)**(-(4.-fractal_dimension)/2.)*np.exp(1j*2.*np.pi*(R1*gamma+R2*(1.-gamma))))
    else:
        gamma = f2/fc
        gamma[f2 > fc] = 1.
        y2 = prepare_ifft_inputs((p**2+q**2)**(-(4.-fractal_dimension)/2.)*np.exp(1j*2.*np.pi*(R1+gamma*R2)))
   
    
    # use inverse discrete fast fourier transform to get surface heights
    h1 = np.fft.irfftn(y1,y1.shape)
    h2 = np.fft.irfftn(y2,y2.shape)
    
    # scale so that standard deviation is as specified
    if elevation_scalefactor is not None:
        std = elevation_scalefactor*(cellsize*size_noclip)**(3.-fractal_dimension)
        ic = int(h1.shape[1]/2)
        i0 = int(ic-size_noclip/2)
        i1 = int(ic+size_noclip/2)
        meanstd = np.average([np.std(line[i0:i1]) for line in h1])
        scaling_factor = std/meanstd
        h1 = h1*scaling_factor
        h2 = h2*scaling_factor
        
    return h1, h2


def smooth_fault_surfaces(h1,h2,fs,dl):
    ks = int(round(fs/dl/2.))
    ssrate = int(ks/20)
    size = max(h1.shape) - 1
    
    if ks > size:
        h1sm = np.ones_like(h1)*0.
        h2sm = h1sm + fs
    elif ssrate > 1:
        h1sm = medfilt_subsampled(h1,ks,ssrate)
        h2sm = medfilt_subsampled(h2,ks,ssrate)
        h2sm[h2sm-h1sm < 1e-10] = h1sm[h2sm-h1sm < 1e-10] + 1e-10
    elif ks > 1:
        h1sm = median_filter(h1,size=ks,mode='nearest')
        h2sm = median_filter(h2,size=ks,mode='nearest')
        h2sm[h2sm-h1sm < 1e-10] = h1sm[h2sm-h1sm < 1e-10] + 1e-10
    else:
        h1sm = h1.copy()
        h2sm = h2.copy()
        
    return h1sm, h2sm


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


def correct_aperture_for_geometry(h1,b,fs,dl,smooth_midpoint=True,
                                  min_ap=1e-20):
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
    h2 : array, shape (m, n)
        surface elevations for bottom fault surface
    b : array, shape (m, n)
        apertures
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
    zero_ap = np.where(h2-h1 < min_ap)
    h2[zero_ap] = h1[zero_ap] + min_ap
    
    if smooth_midpoint:
        h1sm,h2sm = smooth_fault_surfaces(h1, h2, fs, dl)
    else:
        h1sm,h2sm = h1.copy(), h2.copy()

    # aperture at nodes
    bN = h2 - h1
    bNsm = h2sm - h1sm
    
    # horizontal length of half volumes
    dlhv = dl/2.
    
    # midpoint between plates, at nodes (smoothed)
    rN = (h1sm + h2sm)/2.
    
    b_hv1, beta_hv1, bP = get_half_volume_aperture(bN, bNsm, rN, dl, prop='hydraulic',hv=1)
    b_hv2, beta_hv2, bP = get_half_volume_aperture(bN, bNsm, rN, dl, prop='hydraulic',hv=2)
    
    for arr in [b_hv1,beta_hv1,b_hv2,beta_hv2]:
        arr[np.isnan(arr)] = min_ap
    
    b3 = dl/(dlhv/(beta_hv1*b_hv1**3.) + dlhv/(beta_hv2*b_hv2**3.))
    
    
    bmean_hydraulic = b3**(1./3.)
    
    # add apertures for flow in direction perpendicular to fault plane
    bmean_hydraulic = np.array([bmean_hydraulic[0], bmean_hydraulic[1], bP[0]])
    bmean_hydraulic[bmean_hydraulic < min_ap] = min_ap
    
    be_hv1, betae_hv1, bP = get_half_volume_aperture(bN, bNsm, rN, dl, prop='electric',hv=1)
    be_hv2, betae_hv2, bP = get_half_volume_aperture(bN, bNsm, rN, dl, prop='electric',hv=2)
    
    bmean_electric = dl/(dlhv/(betae_hv1*be_hv1) + dlhv/(betae_hv2*be_hv2))
    
    # add apertures for flow in direction perpendicular to fault plane
    bmean_electric = np.array([bmean_electric[0], bmean_electric[1], bP[0]])
    bmean_electric[bmean_electric < min_ap] = min_ap
    
    return bmean_hydraulic, bmean_electric



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


def build_aperture(fault_uvw,
                          ncells,
                          cellsize,
                          fault_separation=1e-4, 
                          fault_surfaces = None,
                          offset=0, 
                          deform_fault_surface=False,
                          fractal_dimension = 2.5, 
                          mismatch_wavelength_cutoff = None, 
                          elevation_scalefactor = None,
                          correct_aperture_for_geometry = True,
                          aperture_type = 'random',
                          aperture_list=None,
                          aperture_electric = None,
                          aperture_hydraulic = None,
                          preserve_negative_apertures = False,
                          random_numbers_dir=None):

    size_noclip = ncells*cellsize
    size = (ncells + 2)*cellsize
    
    build = False
    if fault_surfaces is None:
        
        build = True
    else:
        try:
            h1,h2 = fault_surfaces
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
            h1,h2 = build_fault_pair(size, size_noclip, 
                                          fractal_dimension=fractal_dimension,
                                          cellsize=cellsize,
                                          elevation_scalefactor=elevation_scalefactor,
                                          mismatch_wavelength_cutoff=mismatch_wavelength_cutoff,
                                          beta_glover=None,
                                          random_numbers_dir=random_numbers_dir)
        else:
            h1,h2 = [np.zeros((size,size))]*2
            
            
    h1d = h1.copy()
    h2d = h2.copy()            
    if offset > 0:
        
        if deform_fault_surface:
            print("deforming fault surface")
            b, h1dd, h2dd, overlap_avg = offset_faults_with_deformation(h1, h2, 
                                                           fault_separation, 
                                                           offset)
            h1d[offset:,offset:] = h1dd
            h2d[offset:,:-offset] = h2dd
        else:
            print("not deforming fault surface")
            b = h1[offset:,offset:] - h2[offset:,:-offset] + fault_separation
    else:
        b = h1 - h2 + fault_separation
        
    # set zero values to really low value to allow averaging
    if not preserve_negative_apertures:
        b[b <= 1e-50] = 1e-50
        
    if aperture_type in ['random', 'list']:
        # opportunity to provide corrected apertures
        if aperture_electric is not None:
            bc = aperture_electric
            bf = aperture_hydraulic
        else:
            if correct_aperture_for_geometry:
                print("correcting for geometry")
                bf, bc = correct_aperture_for_geometry(h1d[offset:,offset:],b,fault_separation,cellsize)
            else:
                print("not correcting apertures for geometry")
                # bf, bc = [np.array([b[:-1,:-1]]*3)]*2
                bf, bc = [[np.mean([b[1:,1:],b[1:,:-1],
                                    b[:-1,1:],b[:-1,:-1]],axis=0)]*3 for _ in range(2)]
    else:
        print("not correcting apertures for geometry 2")
        bf, bc = [[np.mean([b[1:,1:],b[1:,:-1],
                            b[:-1,1:],b[:-1,:-1]],axis=0)]*3 for _ in range(2)]
    tmp_aplist = []
    # physical aperture
    bphy = [np.mean([b[1:,1:],b[1:,:-1],
                        b[:-1,1:],b[:-1,:-1]],axis=0)]*3