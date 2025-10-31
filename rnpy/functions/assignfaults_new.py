# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:55:34 2016

@author: a1655681
"""

import numpy as np
import rnpy.functions.array as rna
import rnpy.functions.faultaperture as rnfa
from scipy.interpolate import interp1d
from rnpy.functions.utils import clip_iterable_parameters
import copy


def get_faultlength_distribution(lvals, volume, alpha=10, a=3.5):
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

    Nf = np.zeros(len(lvals) - 1)

    for i in range(len(lvals) - 1):
        lmin, lmax = lvals[i : i + 2]
        Nf[i] = (alpha / (1.0 - a)) * lmax ** (1.0 - a) * volume - (
            alpha / (1.0 - a)
        ) * lmin ** (1.0 - a) * volume

    return Nf


def get_Nf(gamma, alpha, R, lvals_range, ndim=2):
    """
    Get Number of faults within area R within bin ranges provided by lvals_range

    Parameters
    ----------
    gamma : TYPE
        density exponent.
    alpha : TYPE
        density constant.
    R : float
        Dimension in metres of the area
    lvals_range : TYPE
        Bin ranges of fault lengths.
    ndim : integer
        Number of dimensions, usually 2 or 3. Note. User must adjust a so
        it is relevant to ndim

    Returns
    -------
    Number of faults in each bin range (len(lvals_range)-1).

    """
    Nf = []
    for i in range(len(lvals_range) - 1):
        lmin, lmax = lvals_range[i : i + 2]

        Nf_greater_than_lmin = (alpha / (gamma - 1.0)) * lmin ** (1.0 - gamma) * R**ndim
        Nf_greater_than_lmax = (alpha / (gamma - 1.0)) * lmax ** (1.0 - gamma) * R**ndim
        # faults_less_lmax =
        Nf = np.append(Nf, (Nf_greater_than_lmin - Nf_greater_than_lmax))
        # Nf = np.append(Nf, alpha/(a-1.)*lmin**(1.-a)*R2 - alpha/(a-1.)*lmax**(1.-a)*R2)

    return np.around(Nf).astype(np.int64)


def get_alpha(
    gamma,
    R,
    lvals_center,
    lvals_range,
    fw,
    porosity_target,
    alpha_start=0.0,
    ndim=2,
):
    """


    Parameters
    ----------
    gamma : TYPE
        Density exponent.
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
    ndim : integer
        Number of dimensions, usually 2 or 3. Note. User must adjust a so
        it is relevant to ndim

    Returns
    -------
    Nf : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    lvals_range : TYPE
        DESCRIPTION.

    """

    alpha = alpha_start * 1.0
    Nf = get_Nf(gamma, alpha, R, lvals_range, ndim=ndim)

    while np.sum(fw * Nf * lvals_center ** (ndim - 1)) / (R**ndim) < porosity_target:
        Nf = get_Nf(gamma, alpha, R, lvals_range, ndim=ndim)
        alpha += 0.01

    return alpha


def create_random_fault(network_size, faultsizerange, plane):
    """
    create a fault in random location along specified plane. Fault will be
    truncated if it extends out of the network.


    network_size = list,tuple or array containing size in x, y and z directions
    faultsizerange = list,tuple or array containing minimum and maximum size
    plane = yz, xz, or xy plane

    """
    network_size = np.array(network_size)
    lmin, lmax = faultsizerange
    # random position and random size within range
    x, y, z = np.random.random(3) * network_size
    size = np.random.random() * (lmax - lmin) + lmin

    if "x" not in plane:
        fracturecoords = [
            [
                [x, y - size / 2.0, z - size / 2.0],
                [x, y + size / 2.0, z - size / 2.0],
            ],
            [
                [x, y - size / 2.0, z + size / 2.0],
                [x, y + size / 2.0, z + size / 2.0],
            ],
        ]
    elif "y" not in plane:
        fracturecoords = [
            [
                [x - size / 2.0, y, z - size / 2.0],
                [x + size / 2.0, y, z - size / 2.0],
            ],
            [
                [x - size / 2.0, y, z + size / 2.0],
                [x + size / 2.0, y, z + size / 2.0],
            ],
        ]
    elif "z" not in plane:
        fracturecoords = [
            [
                [x - size / 2.0, y - size / 2.0, z],
                [x + size / 2.0, y - size / 2.0, z],
            ],
            [
                [x - size / 2.0, y + size / 2.0, z],
                [x + size / 2.0, y + size / 2.0, z],
            ],
        ]
    fracturecoords = np.array(fracturecoords)
    fracturecoords[fracturecoords < 0.0] = 0.0
    #    for i in range(3):
    #        fracturecoords[fracturecoords[:,:,i] > network_size[i]] = network_size[i]

    return fracturecoords


def get_random_plane(pxyz):
    """
    select a plane (yz,xz, or xy) according to relative probability pxyz

    """
    planeind = int(np.random.choice(3, p=pxyz / sum(pxyz)))
    plane = "xyz".replace("xyz"[planeind], "")

    return plane


def getplane(fracturecoord):
    """
    get the plane that the fracture lies in

    """
    planeind = [len(np.unique(fracturecoord[:, :, i])) for i in range(3)].index(1)

    return "xyz".replace("xyz"[planeind], "")


def checkintersection(fracturecoord, plane, elevation, return_2dcoords=False):
    """ """
    fractureplane = getplane(fracturecoord)
    length = 0.0

    # ensure plane is written in a consistent order
    if plane in ["yx", "zy", "zx"]:
        plane = plane[::-1]

    # if fracture is in the same plane as the test plane then there is no
    # intersection (ignore coinciding planes)
    if plane == fractureplane:
        if return_2dcoords:
            coords2d = np.zeros((2, 2))
            return length, coords2d
        else:
            return length

    if plane == "xy":
        fmin, fmax = np.amin(fracturecoord[:, :, 2]), np.amax(fracturecoord[:, :, 2])
        # i, index with along fracture information
        i = ["xz", "yz"].index(fractureplane)
        coords2d = np.array(
            [
                [fracturecoord[:, :, 0].min(), fracturecoord[:, :, 0].max()],
                [fracturecoord[:, :, 1].min(), fracturecoord[:, :, 1].max()],
            ]
        )
    elif plane == "yz":
        fmin, fmax = np.amin(fracturecoord[:, :, 0]), np.amax(fracturecoord[:, :, 0])
        i = ["xy", "xz"].index(fractureplane) + 1
        coords2d = np.array(
            [
                [fracturecoord[:, :, 1].min(), fracturecoord[:, :, 1].max()],
                [fracturecoord[:, :, 2].min(), fracturecoord[:, :, 2].max()],
            ]
        )
    elif plane == "xz":
        fmin, fmax = np.amin(fracturecoord[:, :, 1]), np.amax(fracturecoord[:, :, 1])
        i = ["xy", "yz"].index(fractureplane) * 2
        coords2d = np.array(
            [
                [fracturecoord[:, :, 0].min(), fracturecoord[:, :, 0].max()],
                [fracturecoord[:, :, 2].min(), fracturecoord[:, :, 2].max()],
            ]
        )

    if (elevation >= fmin) and (elevation <= fmax):
        lmin, lmax = np.amin(fracturecoord[:, :, i]), np.amax(fracturecoord[:, :, i])
        length = lmax - lmin
    else:
        coords2d = np.zeros((2, 2))

    if return_2dcoords:
        return length, coords2d
    else:
        return length


def get_fracture_coords(lvals, networksize, pxyz, return_Nf=False, a=3.5, alpha=10.0):
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
    Nf = get_faultlength_distribution(lvals, volume, alpha=alpha, a=a)

    fracturecoords = []

    for ii in range(len(Nf)):
        # loop through number of whole samples
        ni = 0
        while ni < Nf[ii]:
            # choose plane:
            plane = get_random_plane(pxyz)
            # create a fracture along the plane
            fracturecoords.append(
                create_random_fault(networksize, lvals[ii : ii + 2], plane)
            )

            ni += 1
        # deal with decimal point (fractional number of points) by selecting with probability given by leftover fraction
        randval = np.random.random()
        if randval < Nf[ii] % 1:
            plane = get_random_plane(pxyz)
            fracturecoords.append(
                create_random_fault(networksize, lvals[ii : ii + 2], plane)
            )
            ni += 1

    fracturecoords = np.array(fracturecoords)

    if return_Nf:
        return fracturecoords, Nf
    else:
        return fracturecoords


def coords2indices(fracturecoords, networksize, ncells):
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
    fractureind[fractureind < 0] = 0.0

    nz, ny, nx = ncells

    # normalise to the size of the array
    fractureind[:, :, :, 0] *= nx / networksize[0]
    fractureind[:, :, :, 1] *= ny / networksize[1]
    fractureind[:, :, :, 2] *= nz / networksize[2]

    # change to int for indexing and add 1 to account for nulls on the edges
    fractureind = fractureind.astype(int) + 1

    return fractureind


def add_faults_to_array(faultarray, fractureind):
    """
    add faults to an array where 1 is a faulted cell and 0 is a non-faulted
    cell.


    """

    faultarray *= 0.0

    for fi in fractureind:
        u0, v0, w0 = np.amin(fi, axis=(0, 1))
        u1, v1, w1 = fi.max(axis=(0, 1))
        size = fi.max(axis=(0, 1)) - fi.min(axis=(0, 1))
        perp = list(size).index(min(size))

        if perp == 0:
            faultarray[w0:w1, v0 : v1 + 1, u0, 2, 0] = 1.0
            faultarray[w0 : w1 + 1, v0:v1, u0, 1, 0] = 1.0
        elif perp == 1:
            faultarray[w0:w1, v0, u0 : u1 + 1, 2, 1] = 1.0
            faultarray[w0 : w1 + 1, v0, u0:u1, 0, 1] = 1.0
        elif perp == 2:
            faultarray[w0, v0:v1, u0 : u1 + 1, 1, 2] = 1.0
            faultarray[w0, v0 : v1 + 1, u0:u1, 0, 2] = 1.0

    faultarray = rna.add_nulls(faultarray)

    return faultarray


def get_faultsize(duvw, offset):
    """
    get fault size based on the u,v,w extents of the fault

    """

    size = int(np.amax(duvw) + 6 + offset)  # 2*(max(0.2*np.amax(duvw),6))
    size += size % 2

    return size


def get_faultpair_inputs(
    fractal_dimension,
    elevation_scalefactor,
    mismatch_wavelength_cutoff,
    cellsize,
):
    faultpair_inputs = dict(D=fractal_dimension, scalefactor=elevation_scalefactor)
    if mismatch_wavelength_cutoff is not None:
        faultpair_inputs["lc"] = mismatch_wavelength_cutoff
    if cellsize is not None:
        faultpair_inputs["cs"] = cellsize

    return faultpair_inputs


def offset_faults_with_deformation(h1, h2, fs, offset):
    h1n = h1.copy()
    h2n = h2.copy()

    overlap_avg = 0

    # progressively move along fault plane
    for oo in range(offset):
        # offset fault surfaces by one cell
        if oo == 0:
            # apply fault separation on first step
            h1n = h1n[:-1, 1:] + fs
        else:
            h1n = h1n[:-1, 1:]
        h2n = h2n[:-1, :-1]
        # compute aperture
        ap = h1n - h2n

        # sum overlapping heights (i.e. height of gouge created)
        # don't include edges which will eventually be gone
        apcalc = ap[
            : ap.shape[0] - (offset - oo - 1),
            : ap.shape[1] - (offset - oo - 1),
        ]

        # apcalc = ap
        # print(np.sum(apcalc[apcalc < 0]))
        overlap_avg -= np.sum(apcalc[apcalc < 0]) / apcalc.size

        # remove negative apertures
        # first, compute new fault surface height (=average of height 1 and height
        # 2, i.e. both fault surfaces have been "eroded" by an equal amount)
        newheight = np.mean([h1n, h2n], axis=0)
        # newheight = h2n[:]
        h1n[ap < 0] = newheight[ap < 0]
        h2n[ap < 0] = newheight[ap < 0]

    return ap, h1n, h2n, overlap_avg


def assign_fault_aperture(
    fault_uvw,
    ncells,
    cs=0.25e-3,
    fault_separation=1e-4,
    fault_surfaces=None,
    offset=0,
    deform_fault_surface=False,
    fractal_dimension=2.5,
    mismatch_wavelength_cutoff=None,
    elevation_scalefactor=None,
    elevation_prefactor=1.0,
    correct_aperture_for_geometry=True,
    aperture_type="random",
    fill_array=True,
    aperture_list=None,
    aperture_list_electric=None,
    aperture_list_hydraulic=None,
    preserve_negative_apertures=False,
    random_numbers_dir=None,
    minimum_aperture=None,
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

    nx, ny, nz = ncells

    if aperture_type == "list":
        fill_array = True

    if fill_array:
        ap_array = np.array(
            [np.ones((nz + 2, ny + 2, nx + 2, 3, 3)) * 1e-50] * 3
        )  # yz, xz and xy directions

    if (aperture_type != "list") or (aperture_list is None):
        aperture_list = []

    # if either electric or hydraulic aperture is None, set them both to None
    if aperture_list_electric is None:
        aperture_list_hydraulic = None
    if aperture_list_hydraulic is None:
        aperture_list_electric = None
    aperture_list_c = []
    aperture_list_f = []

    if not np.iterable(fault_separation):
        fault_separation = np.ones(len(fault_uvw)) * fault_separation

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
        u0, v0, w0 = np.amin(nn, axis=(0, 1))
        u1, v1, w1 = np.amax(nn, axis=(0, 1))

        # get size of original fault so we get correct scale factor
        size_noclip = max(u1 - u0, v1 - v0, w1 - w0)

        # make sure extents are within the array
        u1 = min(u1, nx + 1)
        v1 = min(v1, ny + 1)
        w1 = min(w1, nz + 1)
        u0 = min(u0, nx + 1)
        v0 = min(v0, ny + 1)
        w0 = min(w0, nz + 1)
        # size in the x, y and z directions
        duvw = np.array([u1 - u0, v1 - v0, w1 - w0])
        du, dv, dw = (duvw * 0.5).astype(int)

        # define size, add some padding to account for edge effects and make
        # the fault square as I am not sure if fft works properly for non-
        # square geometries

        # if list of apertures provided, assign these to the array. Because these
        # apertures have been pre-trimmed to fit the array, we assign them
        # differently to if they were calculated from scratch.
        print("aperture type", aperture_type)
        if (aperture_type == "list") and (aperture_list is not None):
            du1, dv1, dw1 = u1 - u0, v1 - v0, w1 - w0

            # loop through true aperture, hydraulic aperture, electric aperture
            for iii, ap in enumerate(aperture_list):
                print("ap.shape", ap.shape)
                dperp = list(duvw).index(0)

                if dperp == 0:
                    try:
                        ap_array[iii, w0 : w1 + 1, v0 : v1 + 1, u0 - 1 : u1 + 1] += ap[
                            i
                        ][
                            : dw1 + 1, : dv1 + 1
                        ]  # np.amax([ap_array[iii,w0:w1+1,v0:v1+1,u0-1:u1+1],ap[i][:dw1+1,:dv1+1]],axis=0)
                    except:
                        #                            print "aperture wrong shape, resetting fault extents"
                        u1, v1, w1 = (
                            np.array([u0, v0, w0])
                            + ap[i][: dw1 + 1, : dv1 + 1].shape[2::-1]
                            - np.array([2, 1, 1])
                        )
                        #                            print dperp,u0,v0,w0,u1,v1,w1
                        ap_array[iii, w0 : w1 + 1, v0 : v1 + 1, u0 - 1 : u1 + 1] += ap[
                            i
                        ][: dw1 + 1, : dv1 + 1]
                elif dperp == 1:
                    try:
                        ap_array[iii, w0 : w1 + 1, v0 - 1 : v0 + 1, u0 : u1 + 1] += ap[
                            i
                        ][
                            : dw1 + 1, :, : du1 + 1
                        ]  # np.amax([ap_array[iii,w0:w1+1,v0-1:v0+1,u0:u1+1],ap[i][:dw1+1,:,:du1+1]],axis=0)
                    except:
                        #                            print "aperture wrong shape, resetting fault extents"
                        u1, v1, w1 = (
                            np.array([u0, v0, w0])
                            + ap[i][: dw1 + 1, :, : du1 + 1].shape[2::-1]
                            - np.array([1, 2, 1])
                        )
                        #                            print dperp,u0,v0,w0,u1,v1,w1
                        ap_array[iii, w0 : w1 + 1, v0 - 1 : v0 + 1, u0 : u1 + 1] += ap[
                            i
                        ][: dw1 + 1, :, : du1 + 1]
                elif dperp == 2:
                    try:
                        ap_array[iii, w0 - 1 : w0 + 1, v0 : v1 + 1, u0 : u1 + 1] += ap[
                            i
                        ][
                            :, : dv1 + 1, : du1 + 1
                        ]  # np.amax([ap_array[iii,w0-1:w0+1,v0:v1+1,u0:u1+1],ap[i][:,:dv1+1,:du1+1]],axis=0)
                    except:
                        #                            print "aperture wrong shape, resetting fault extents"
                        u1, v1, w1 = (
                            np.array([u0, v0, w0])
                            + ap[i][:, : dv1 + 1, : du1 + 1].shape[2::-1]
                            - np.array([1, 1, 2])
                        )
                        #                            print dperp,u0,v0,w0,u1,v1,w1
                        ap_array[iii, w0 - 1 : w0 + 1, v0 : v1 + 1, u0 : u1 + 1] += ap[
                            i
                        ][:, : dv1 + 1, : du1 + 1]
        elif aperture_type not in ["random", "constant"]:
            aperture_type = "random"

        if aperture_type in ["random", "constant"]:
            # if offset between 0 and 1, assume it is a fraction of fault size
            if 0 < offset < 1:
                offset = int(np.round(offset * size_noclip))
            else:
                # ensure it is an integer
                offset = int(offset)

            # get size of fault including padding
            size = get_faultsize(duvw, offset)

            # define direction normal to fault
            direction = list(duvw).index(0)
            # get fault pair inputs as a dictionary
            faultpair_inputs = get_faultpair_inputs(
                fractal_dimension,
                elevation_scalefactor,
                mismatch_wavelength_cutoff,
                cs,
            )

            faultpair_inputs["random_numbers_dir"] = random_numbers_dir
            faultpair_inputs["prefactor"] = elevation_prefactor

            build = False
            if fault_surfaces is None:
                build = True
            else:
                try:
                    h1, h2 = fault_surfaces[i]
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
                (print("building new faults"),)
                if aperture_type == "random":
                    h1, h2 = rnfa.build_fault_pair(
                        size, size_noclip, **faultpair_inputs
                    )
                else:
                    h1, h2 = [np.zeros((size, size))] * 2

            h1d = h1.copy()
            h2d = h2.copy()
            if offset > 0:
                if deform_fault_surface:
                    print("deforming fault surface")
                    b, h1dd, h2dd, overlap_avg = offset_faults_with_deformation(
                        h1, h2, fault_separation[i], offset
                    )
                    h1d[offset:, offset:] = h1dd
                    h2d[offset:, :-offset] = h2dd
                else:
                    print("not deforming fault surface")
                    b = (
                        h1[offset:, offset:]
                        - h2[offset:, :-offset]
                        + fault_separation[i]
                    )
            else:
                b = h1 - h2 + fault_separation[i]

            # # set zero values to really low value to allow averaging
            # if not preserve_negative_apertures:
            #     b[b <= 1e-50] = 1e-50

            # centre indices of array b
            cb = (np.array(np.shape(b)) * 0.5).astype(int)

            if aperture_type in ["random", "list"]:
                # opportunity to provide corrected apertures
                if aperture_list_electric is not None:
                    bc = aperture_list_electric[i]
                    bf = aperture_list_hydraulic[i]
                else:
                    if correct_aperture_for_geometry:
                        print("correcting for geometry")
                        bf, bc = rnfa.correct_aperture_for_geometry(
                            h1d[offset:, offset:], b, fault_separation[i], cs
                        )
                    else:
                        print("not correcting apertures for geometry")
                        # bf, bc = [np.array([b[:-1,:-1]]*3)]*2
                        bf, bc = [
                            np.array(
                                [
                                    np.mean(
                                        [
                                            b[1:, 1:],
                                            b[1:, :-1],
                                            b[:-1, 1:],
                                            b[:-1, :-1],
                                        ],
                                        axis=0,
                                    )
                                ]
                                * 3
                            )
                            for _ in range(2)
                        ]
            else:
                print("not correcting apertures for geometry 2")
                # bf, bc = [np.array([b[:-1,:-1]]*3)]*2
                bf, bc = [
                    np.array(
                        [
                            np.mean(
                                [
                                    b[1:, 1:],
                                    b[1:, :-1],
                                    b[:-1, 1:],
                                    b[:-1, :-1],
                                ],
                                axis=0,
                            )
                        ]
                        * 3
                    )
                    for _ in range(2)
                ]
            tmp_aplist = []
            # physical aperture
            bphy = [
                np.mean([b[1:, 1:], b[1:, :-1], b[:-1, 1:], b[:-1, :-1]], axis=0)
            ] * 3
            # bphy = [b[1:,1:]]*3

            # set minimum aperture
            if not preserve_negative_apertures:
                b[b <= minimum_aperture] = minimum_aperture
                bf[bf <= minimum_aperture] = minimum_aperture
                bc[bc <= minimum_aperture] = minimum_aperture

                for i in range(2):
                    bphy[i][bphy[i] <= minimum_aperture] = minimum_aperture

            # assign the corrected apertures to aperture array

            for ii, bb in enumerate([bphy, bf, bc]):
                b0, b1, b2 = bb
                if direction == 0:
                    b0vals, b1vals, b2vals = (
                        b0[
                            cb[0] - dw : cb[0] + dw + duvw[2] % 2 + 1,
                            cb[1] - dv : cb[1] + dv + duvw[1] % 2,
                        ],
                        b1[
                            cb[0] - dw : cb[0] + dw + duvw[2] % 2,
                            cb[1] - dv : cb[1] + dv + duvw[1] % 2 + 1,
                        ],
                        b2[
                            cb[0] - dw : cb[0] + dw + duvw[2] % 2 + 1,
                            cb[1] - dv : cb[1] + dv + duvw[1] % 2 + 1,
                        ]
                        / 2.0,
                    )
                    if fill_array:
                        if w1 - w0 + 1 > int(np.shape(b2vals)[0]):
                            print(
                                "indices don't match up, w0 {}, w1 {}, b2vals shape[0] {}".format(
                                    w0, w1, b2vals.shape[0]
                                )
                            )
                            w1 = int(np.shape(b2vals)[0]) + w0 - 1
                        elif w1 - w0 + 1 < int(np.shape(b2vals)[0]):
                            print(
                                "indices don't match up, w0 {}, w1 {}, b2vals shape[0] {}".format(
                                    w0, w1, b2vals.shape[0]
                                )
                            )
                            b2vals = b2vals[: w1 - w0 + 1]
                            b1vals = b1vals[: w1 - w0]
                            b0vals = b0vals[: w1 - w0 + 1]
                        if v1 - v0 + 1 > int(np.shape(b2vals)[1]):
                            print(
                                "indices don't match up, v0 {}, v1 {}, b2vals shape[1] {}".format(
                                    v0, v1, b2vals.shape[1]
                                )
                            )
                            v1 = int(np.shape(b2vals)[1]) + v0 - 1
                        elif v1 - v0 + 1 < int(np.shape(b2vals)[1]):
                            print(
                                "indices don't match up, v0 {}, v1 {}, b2vals shape[1] {}".format(
                                    v0, v1, b2vals.shape[1]
                                )
                            )
                            b2vals = b2vals[:, : v1 - v0 + 1]
                            b1vals = b1vals[:, : v1 - v0 + 1]
                            b0vals = b0vals[:, : v1 - v0]
                        # faults perpendicular to x direction, i.e. yz plane
                        ap_array[ii, w0 : w1 + 1, v0 : v1 + 1, u0 - 1, 0, 0] += b2vals
                        ap_array[ii, w0 : w1 + 1, v0 : v1 + 1, u0, 0, 0] += b2vals
                        # y direction opening in x direction
                        ap_array[ii, w0 : w1 + 1, v0:v1, u0, 1, 0] += b0vals
                        # z direction opening in x direction
                        ap_array[ii, w0:w1, v0 : v1 + 1, u0, 2, 0] += b1vals
                    # print "assigning aperture to list"
                    aperture = np.zeros((w1 - w0 + 1, v1 - v0 + 1, 2, 3, 3))
                    aperture[:, :, 0, 0, 0] = b2vals
                    aperture[:, :, 1, 0, 0] = b2vals
                    aperture[:, :-1, 1, 1, 0] = b0vals
                    aperture[:-1, :, 1, 2, 0] = b1vals

                elif direction == 1:
                    b0vals, b1vals, b2vals = (
                        b0[
                            cb[0] - dw : cb[0] + dw + duvw[2] % 2 + 1,
                            cb[1] - du : cb[1] + du + duvw[0] % 2,
                        ],
                        b1[
                            cb[0] - dw : cb[0] + dw + duvw[2] % 2,
                            cb[1] - du : cb[1] + du + duvw[0] % 2 + 1,
                        ],
                        b2[
                            cb[0] - dw : cb[0] + dw + duvw[2] % 2 + 1,
                            cb[1] - du : cb[1] + du + duvw[0] % 2 + 1,
                        ]
                        / 2.0,
                    )
                    if fill_array:
                        # correct for slight discrepancies in array shape
                        if w1 - w0 + 1 > int(np.shape(b2vals)[0]):
                            print(
                                "indices don't match up, w0 {}, w1 {}, b2vals shape[0] {}".format(
                                    w0, w1, b2vals.shape[0]
                                )
                            )
                            w1 = int(np.shape(b2vals)[0]) + w0 - 1
                        elif w1 - w0 + 1 < int(np.shape(b2vals)[0]):
                            print(
                                "indices don't match up, w0 {}, w1 {}, b2vals shape[0] {}".format(
                                    w0, w1, b2vals.shape[0]
                                )
                            )
                            b2vals = b2vals[: w1 - w0 + 1]
                            b1vals = b1vals[: w1 - w0]
                            b0vals = b0vals[: w1 - w0 + 1]
                        if u1 - u0 + 1 > int(np.shape(b2vals)[1]):
                            print(
                                "indices don't match up, u0 {}, u1 {}, b2vals shape[1] {}".format(
                                    u0, u1, b2vals.shape[1]
                                )
                            )
                            u1 = int(np.shape(b2vals)[1]) + u0 - 1
                        elif u1 - u0 + 1 < int(np.shape(b2vals)[1]):
                            print(
                                "indices don't match up, u0 {}, u1 {}, b2vals shape[1] {}".format(
                                    u0, u1, b2vals.shape[1]
                                )
                            )
                            b2vals = b2vals[:, : u1 - u0 + 1]
                            b1vals = b1vals[:, : u1 - u0 + 1]
                            b0vals = b0vals[:, : u1 - u0]
                        # faults perpendicular to y direction, i.e. xz plane
                        ap_array[ii, w0 : w1 + 1, v0 - 1, u0 : u1 + 1, 1, 1] += b2vals
                        ap_array[ii, w0 : w1 + 1, v0, u0 : u1 + 1, 1, 1] += b2vals
                        # x direction opening in y direction
                        ap_array[ii, w0 : w1 + 1, v0, u0:u1, 0, 1] += b0vals
                        # z direction opening in y direction
                        ap_array[ii, w0:w1, v0, u0 : u1 + 1, 2, 1] += b1vals
                    # print "assigning aperture to list"
                    aperture = np.zeros((w1 + 1 - w0, 2, u1 + 1 - u0, 3, 3))
                    aperture[:, 0, :, 1, 1] = b2vals
                    aperture[:, 1, :, 1, 1] = b2vals
                    aperture[:, 1, :-1, 0, 1] = b0vals
                    aperture[:-1, 1, :, 2, 1] = b1vals

                elif direction == 2:
                    b0vals, b1vals, b2vals = (
                        b0[
                            cb[0] - dv : cb[0] + dv + duvw[1] % 2 + 1,
                            cb[1] - du : cb[1] + du + duvw[0] % 2,
                        ],
                        b1[
                            cb[0] - dv : cb[0] + dv + duvw[1] % 2,
                            cb[1] - du : cb[1] + du + duvw[0] % 2 + 1,
                        ],
                        b2[
                            cb[0] - dv : cb[0] + dv + duvw[1] % 2 + 1,
                            cb[1] - du : cb[1] + du + duvw[0] % 2 + 1,
                        ]
                        / 2.0,
                    )
                    if fill_array:
                        # correct for slight discrepancies in array shape
                        if v1 - v0 + 1 > int(np.shape(b2vals)[0]):
                            print(
                                "indices don't match up, v0 {}, v1 {}, b2vals shape[0] {}".format(
                                    v0, v1, b2vals.shape[0]
                                )
                            )
                            v1 = int(np.shape(b2vals)[0]) + v0 - 1
                        elif v1 - v0 + 1 < int(np.shape(b2vals)[0]):
                            print(
                                "indices don't match up, v0 {}, v1 {}, b2vals shape[0] {}".format(
                                    v0, v1, b2vals.shape[0]
                                )
                            )
                            b2vals = b2vals[: v1 - v0 + 1]
                            b1vals = b1vals[: v1 - v0]
                            b0vals = b0vals[: v1 - v0 + 1]
                        if u1 - u0 + 1 > int(np.shape(b2vals)[1]):
                            print(
                                "indices don't match up, u0 {}, u1 {}, b2vals shape[1] {}".format(
                                    u0, u1, b2vals.shape[1]
                                )
                            )
                            u1 = int(np.shape(b2vals)[1]) + u0 - 1
                        elif u1 - u0 + 1 < int(np.shape(b2vals)[1]):
                            print(
                                "indices don't match up, u0 {}, u1 {}, b2vals shape[1] {}".format(
                                    u0, u1, b2vals.shape[1]
                                )
                            )
                            b2vals = b2vals[:, : u1 - u0 + 1]
                            b1vals = b1vals[:, : u1 - u0 + 1]
                            b0vals = b0vals[:, : u1 - u0]
                        # faults perpendicular to z direction, i.e. xy plane
                        ap_array[ii, w0 - 1, v0 : v1 + 1, u0 : u1 + 1, 2, 2] += b2vals
                        ap_array[ii, w0, v0 : v1 + 1, u0 : u1 + 1, 2, 2] += b2vals
                        # x direction opening in z direction
                        ap_array[ii, w0, v0 : v1 + 1, u0:u1, 0, 2] += b0vals
                        # y direction opening in z direction
                        ap_array[ii, w0, v0:v1, u0 : u1 + 1, 1, 2] += b1vals
                    # print "assigning aperture to list"
                    aperture = np.zeros((2, v1 + 1 - v0, u1 + 1 - u0, 3, 3))
                    aperture[0, :, :, 2, 2] = b2vals
                    aperture[1, :, :, 2, 2] = b2vals
                    aperture[1, :, :-1, 0, 2] = b0vals
                    aperture[1, :-1, :, 1, 2] = b1vals

                tmp_aplist.append(aperture)

                bvals[-1].append([bb, b0, b1])

            faultheights.append([h1, h2])
            # average overlap height per cell * cellsize ** 2 * number of cells
            overlap_vol.append(
                overlap_avg * cs**2 * np.prod(np.array(duvw)[np.array(duvw) > 0])
            )
            aperture_list.append(tmp_aplist[0])
            aperture_list_f.append(tmp_aplist[1])
            aperture_list_c.append(tmp_aplist[2])

    #        ap_array[i] *= fault_array

    if fill_array:
        for ii in range(3):
            rna.add_nulls(ap_array[ii])
        if aperture_type == "list":
            #            print len(aperture_list)
            aperture_list_f = aperture_list[1]
            aperture_list_c = aperture_list[2]
            aperture_list = aperture_list[0]
        if not preserve_negative_apertures:
            ap_array[(np.isfinite(ap_array)) & (ap_array < 2e-50)] = 2e-50
        aperture_c = ap_array[2]
        aperture_f = ap_array[1]
        aperture_array = ap_array[0]
        return (
            aperture_list,
            aperture_list_f,
            aperture_list_c,
            aperture_array,
            aperture_f,
            aperture_c,
            faultheights,
            overlap_vol,
        )
    else:
        return (
            aperture_list,
            aperture_list_f,
            aperture_list_c,
            faultheights,
            overlap_vol,
        )


def update_from_precalculated(rv, effective_apertures_fn, permeability_matrix=1e-18):
    effective_apertures = np.loadtxt(effective_apertures_fn)

    # add extreme values so that we cover the entire interpolation range
    # < min; hydraulic aperture = sqrt(matrix permeability * 12); electric aperture = 1e-50
    # > max; fault sep = hydraulic aperture = electric aperture
    first_row = [-1, np.sqrt(12 * permeability_matrix), 1e-50]
    last_row = [1, 1, 1]
    effective_apertures = np.concatenate(
        [[first_row], effective_apertures, [last_row]], axis=0
    )

    # create interpolation functions for effective apertures
    feah = interp1d(effective_apertures[:, 0], effective_apertures[:, 1])
    feae = interp1d(effective_apertures[:, 0], effective_apertures[:, 2])

    # update aperture values
    for i in range(len(rv.aperture)):
        for j in range(len(rv.aperture[0])):
            for k in range(len(rv.aperture[0, 0])):
                for ii in range(3):
                    jjlist = [0, 1, 2]
                    jjlist.remove(ii)
                    for jj in jjlist:
                        if rv.fault_array[i, j, k, jj, ii] and np.isfinite(
                            rv.aperture[i, j, k, jj, ii]
                        ):
                            rv.aperture_hydraulic[i, j, k, jj, ii] = feah(
                                rv.aperture[i, j, k, jj, ii]
                            )
                            rv.aperture_electric[i, j, k, jj, ii] = feae(
                                rv.aperture[i, j, k, jj, ii]
                            )

    # initialise electrical and hydraulic resistance
    rv.initialise_electrical_resistance()
    rv.initialise_permeability()

    return rv


def add_random_fault_sticks_to_arrays(
    Rv,
    Nfval,
    fault_length_m,
    fault_width,
    hydraulic_aperture,
    resistivity,
    pz,
    fault_lengths_assigned=None,
):
    ncells = Rv.ncells[1]
    cellsize = Rv.cellsize[1]
    Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = cellsize

    # array to record what fault lengths were assigned where
    if fault_lengths_assigned is None:
        fault_lengths_assigned = np.zeros_like(Rv.resistivity)

    if Nfval > 0:
        orientationj = np.random.choice([0, 1], p=[1.0 - pz, pz], size=Nfval)
        faultsj = orientationj * int(fault_length_m / cellsize)
        orientationi = (1 - orientationj).astype(int)
        faultsi = orientationi * int(fault_length_m / cellsize)

        # j axis (vertical faults) open in the y direction
        idxo = np.ones_like(faultsj, dtype=int)
        # i axis (horizontal faults) open in the z direction
        idxo[faultsi > 0] = 2

        # i axis (horizontal faults) are associated with y connectors
        idxc = np.ones_like(faultsj, dtype=int)
        # j axis (vertical faults) are associated with z connectors
        idxc[faultsj > 0] = 2

        # can only initialise a fault where there isn't one already of that size or bigger
        if np.any(Rv.aperture >= fault_width):
            i = np.zeros(Nfval, dtype=int)
            j = np.zeros(Nfval, dtype=int)
            # faults in i direction (horizontal) are y connectors opening in z direction
            available_ij_i = np.column_stack(
                np.where(Rv.aperture[:, :, 1, 1, 2] < fault_width)
            )
            # make a list of random indices to pull from available_ij_i
            random_indices_i = np.random.choice(
                np.arange(len(available_ij_i)),
                replace=False,
                size=sum(orientationi),
            )
            i[orientationi == 1] = available_ij_i[random_indices_i][:, 1]
            j[orientationi == 1] = available_ij_i[random_indices_i][:, 0]
            # faults in j direction (horizontal) are y connectors opening in z direction
            available_ij_j = np.column_stack(
                np.where(Rv.aperture[:, :, 1, 2, 1] < fault_width)
            )
            # make a list of random indices to pull from available_ij_i
            random_indices_j = np.random.choice(
                np.arange(len(available_ij_j)),
                replace=False,
                size=sum(orientationj),
            )
            j[orientationj == 1] = available_ij_j[random_indices_j][:, 0]
            i[orientationj == 1] = available_ij_j[random_indices_j][:, 1]
        else:
            # pick a random location for the fault centre
            i = np.random.randint(1, ncells + 2, size=Nfval)
            j = np.random.randint(1, ncells + 2, size=Nfval)

        j0 = (j - faultsj / 2).astype(int)
        j1 = (j + faultsj / 2).astype(int)

        # add extra length if we have an odd-cellsize length fault as cutting
        # in half truncates the fault
        if int(fault_length_m / cellsize) % 2 == 1:
            extra_bit = np.random.choice([0, 1], size=Nfval)
            j0 -= (extra_bit * orientationj).astype(int)
            j1 += ((1 - extra_bit) * orientationj).astype(int)

        # truncate so we don't go out of bounds
        j0[j0 < 1] = 1
        j1[j1 > ncells + 1] = ncells + 1

        i0 = (i - faultsi / 2).astype(int)
        i1 = (i + faultsi / 2).astype(int)

        # add extra length if we have an odd-cellsize length fault as cutting
        # in half truncates the fault
        if int(fault_length_m / cellsize) % 2 == 1:
            i0 -= (extra_bit * orientationi).astype(int)
            i1 += ((1 - extra_bit) * orientationi).astype(int)

        # truncate so we don't go out of bounds
        i0[i0 < 1] = 1
        i1[i1 > ncells + 1] = ncells + 1

        # initialise indices to update
        idx_i = i0 * 1
        idx_j = j0 * 1

        # randomly pull list of resistivity and hydraulic aperture values from
        # the array, if they are arrays.
        assign_dict = {}
        idxs = None
        for val, name in [[hydraulic_aperture, "aph"], [resistivity, "resf"]]:
            if np.iterable(val):
                # choose some random indices
                # only define idxs once (use same indices for all properties)
                if idxs is None:
                    idxs = np.random.choice(np.arange(len(val)), size=idx_j.shape)

                if len(val.shape) == 2:
                    assign_dict[name] = np.zeros(len(idx_j))
                    assign_dict[name][np.where(orientationi)] = val[:, 0][idxs][
                        np.where(orientationi)
                    ]
                    assign_dict[name][np.where(orientationj)] = val[:, 1][idxs][
                        np.where(orientationj)
                    ]

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

        for add_idx in range(int(fault_length_m / cellsize)):
            # filter to take out indices where aperture is already larger than proposed update
            # or null in the resistivity array
            filt = np.array(
                [
                    k
                    for k in range(len(idx_j))
                    if (
                        Rv.aperture[idx_j[k], idx_i[k], 1, idxc[k], idxo[k]]
                        < fault_width
                        and not np.isnan(Rv.resistivity[idx_j[k], idx_i[k], 1, idxc[k]])
                    )
                ]
            )
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

                Rv.aperture[idx_jj, idx_ii, 1, idxc_i, idxo_i] = fault_width  #
                # print(Rv.aperture[idx_j, idx_i,1,1,idxo])
                Rv.aperture_electric[idx_jj, idx_ii, 1, idxc_i, idxo_i] = fault_width
                Rv.aperture_hydraulic[idx_jj, idx_ii, 1, idxc_i, idxo_i] = (
                    values_to_assign["aph"]
                )
                Rv.resistance[idx_jj, idx_ii, 1, idxc_i] = (
                    values_to_assign["resf"] / cellsize
                )
                Rv.resistivity[idx_jj, idx_ii, 1, idxc_i] = values_to_assign["resf"]
                fault_lengths_assigned[idx_jj, idx_ii, 1, idxc_i] = fault_length_m
                idx_i[np.all([faultsi > 0], axis=0)] += 1
                idx_j[np.all([faultsj > 0], axis=0)] += 1

                idx_i[idx_i > ncells + 1] = ncells + 1
                idx_j[idx_j > ncells + 1] = ncells + 1

    return Rv, fault_lengths_assigned


def add_random_fault_planes_to_arrays(
    Rv,
    Nfval,
    fault_length_m,  # primary in-plane extent (meters)
    fault_widths,  # scalar or 1D array length Nfval (per-fault widths)
    hydraulic_aperture_pairs,  # scalar, (Nfval,), or (Nfval,2) per plane, per direction
    resistivity_pairs,  # scalar, (Nfval,), or (Nfval,2) per plane, per direction
    pxyz=(
        1 / 3,
        1 / 3,
        1 / 3,
    ),  # NORMAL probabilities: (Px, Py, Pz) i.e., x, y, z
    fault_span_m=None,  # secondary in-plane extent; None -> same as fault_length_m
    fault_lengths_assigned=None,
    normals=None,  # optional: 1D array length Nfval with {0,1,2} for normals (x,y,z)
    max_tries=1000,  # attempts per plane (strict_mode=True)
    raise_on_shortfall=False,  # if strict_mode and can't place Nfval planes, raise
    rng=None,  # optional: numpy Generator for reproducibility
    full_rectangle=False,
):
    """
    Insert 3D fault planes into Rv connector arrays (in place), updating BOTH off-diagonal
    pairs per plane, with per-direction properties. Spatial arrays have ghost padding.

    -------------------------------------------------------------------------------
    Inputs & layout
    -------------------------------------------------------------------------------
    Rv.ncells   : [Nx, Ny, Nz] (cartesian order)
    Rv.cellsize : [dx, dy, dz] (cartesian meters)

    Array spatial order (axes 0,1,2) is [z, y, x] with **ghost padding** (interior indices [1..N+1]):
      Rv.aperture.shape            == (nz+2, ny+2, nx+2, 3, 3)
      Rv.aperture_electric.shape   == (nz+2, ny+2, nx+2, 3, 3)
      Rv.aperture_hydraulic.shape  == (nz+2, ny+2, nx+2, 3, 3)
      Rv.resistivity.shape         == (nz+2, ny+2, nx+2, 3)
      Rv.resistance.shape          == (nz+2, ny+2, nx+2, 3)

    Connector/opening axes (last dims): 0=x, 1=y, 2=z; diagonal entries are unused.

    -------------------------------------------------------------------------------
    Orientation probabilities
    -------------------------------------------------------------------------------
    pxyz is given as NORMAL probabilities: (Px, Py, Pz) meaning normals along x, y, z.

    -------------------------------------------------------------------------------
    Plane normal → TWO off-diagonal (connector, opening) pairs (opening along normal)
    -------------------------------------------------------------------------------
      normal x (yz-plane): pairs [(y,x), (z,x)] → (1,0), (2,0)
      normal y (xz-plane): pairs [(x,y), (z,y)] → (0,1), (2,1)
      normal z (xy-plane): pairs [(x,z), (y,z)] → (0,2), (1,2)

    The **two columns** of hydraulic_aperture_pairs/resistivity_pairs map to the
    pair order above (column 0 → first pair, column 1 → second pair).

    -------------------------------------------------------------------------------
    strict_mode (optional)
    -------------------------------------------------------------------------------
    If strict_mode=True, the function retries random centers (up to max_tries) and
    **pre-checks** the entire rectangle (both pairs) before writing:
      - If require_full_rectangle=True: ALL cells in the rectangle must be updatable.
      - Else: at least SOME cells in the rectangle must be updatable.
    If raise_on_shortfall=True and it's impossible to place Nfval planes, raise.

    Returns
    -------------------------------------------------------------------------------
    Rv (modified), fault_lengths_assigned (same shape as Rv.resistivity).
    """
    # print("starting")
    # --- Meta (cartesian orders for counts and sizes) ---
    ncells = np.asarray(Rv.ncells, dtype=int)  # [Nx, Ny, Nz]
    cellsize = np.asarray(Rv.cellsize, dtype=float)  # [dx, dy, dz]
    assert ncells.shape == (3,), "Rv.ncells must be [Nx, Ny, Nz]"
    assert cellsize.shape == (3,), "Rv.cellsize must be [dx, dy, dz]"

    Nx, Ny, Nz = map(int, ncells)

    # Array spatial order is [z, y, x] with ghost padding (+2 each)
    arr_shape = Rv.aperture.shape[:3]
    expected = (Nz + 2, Ny + 2, Nx + 2)
    if arr_shape != expected:
        raise ValueError(
            f"Unexpected spatial shape {arr_shape}. Expected (nz+2, ny+2, nx+2)={expected}."
        )

    # Map cartesian axis -> array axis
    # cart: 0=x, 1=y, 2=z  -> array: 2=x, 1=y, 0=z
    CART2ARR = {0: 2, 1: 1, 2: 0}

    # Interior index bounds per ARRAY axis (inclusive), because of ghost padding:
    #   interior indices are [1 .. N+1] on each axis
    ARR_MIN_INCL = {0: 1, 1: 1, 2: 1}  # array axes: 0=z, 1=y, 2=x
    ARR_MAX_INCL = {0: Nz + 1, 1: Ny + 1, 2: Nx + 1}  # inclusive upper bounds

    # Plane normal → (connector, opening) pairs with opening along the normal
    PAIRS = {
        0: [(1, 0), (2, 0)],  # normal x → opening x; connectors y, z
        1: [(0, 1), (2, 1)],  # normal y → opening y; connectors x, z
        2: [(0, 2), (1, 2)],  # normal z → opening z; connectors x, y
    }

    # Baseline electric aperture per connector axis (finite only)
    for idxc in range(3):  # connector axis (x=0, y=1, z=2)
        for idxo in range(3):  # opening axis (x=0, y=1, z=2)
            mask = np.isfinite(Rv.aperture_electric[..., idxc, idxo])
            Rv.aperture_electric[..., idxc, idxo][mask] = cellsize[idxc]

    # Record-keeping array
    if fault_lengths_assigned is None:
        fault_lengths_assigned = np.zeros_like(Rv.resistivity, dtype=float)

    if Nfval <= 0:
        return Rv, fault_lengths_assigned

    rng = np.random.default_rng() if rng is None else rng

    # NORMAL probabilities: (Px, Py, Pz)
    if normals is None:
        Px, Py, Pz = pxyz
        p_normals = np.asarray([Px, Py, Pz], dtype=float)
        p_normals = p_normals / p_normals.sum()
        normals = rng.choice(
            [0, 1, 2], size=Nfval, p=p_normals
        )  # 0=x, 1=y, 2=z normals
    else:
        normals = np.asarray(normals, dtype=int)
        if normals.shape != (Nfval,) or not np.isin(normals, [0, 1, 2]).all():
            raise ValueError(
                "`normals` must be a 1D array of length Nfval with values in {0,1,2}."
            )

    if fault_span_m is None:
        fault_span_m = fault_length_m

    # Helpers
    def cells_from_meters(m, cart_axis):
        return max(1, int(round(float(m) / float(cellsize[cart_axis]))))

    def per_fault_values(val, name):
        if np.iterable(val) and not np.isscalar(val):
            arr = np.asarray(val, dtype=float)
            if arr.shape != (Nfval,):
                raise ValueError(
                    f"`{name}` must be scalar or shape (Nfval,). Got {arr.shape}."
                )
            return arr
        else:
            return np.full(Nfval, float(val), dtype=float)

    def per_fault_pair_values(val, name):
        """
        Accept scalar, (Nfval,), or (Nfval,2); return (Nfval,2).
        Column 0 → first pair for the normal; Column 1 → second pair.
        """
        if np.iterable(val) and not np.isscalar(val):
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 1:
                if arr.shape != (Nfval,):
                    raise ValueError(
                        f"`{name}` must be scalar, (Nfval,), or (Nfval,2). Got {arr.shape}."
                    )
                return np.column_stack([arr, arr])
            elif arr.ndim == 2 and arr.shape == (Nfval, 2):
                return arr
            else:
                raise ValueError(
                    f"`{name}` must be scalar, (Nfval,), or (Nfval,2). Got {arr.shape}."
                )
        else:
            s = float(val)
            return np.tile([[s, s]], (Nfval, 1))

    widths_per_plane = per_fault_values(fault_widths, "fault_widths")
    span_per_plane = per_fault_values(fault_span_m, "fault_span_m")
    aph_pairs = per_fault_pair_values(
        hydraulic_aperture_pairs, "hydraulic_aperture_pairs"
    )
    res_pairs = per_fault_pair_values(resistivity_pairs, "resistivity_pairs")

    # --- Center selection (array order [z,y,x], 1-based interior), prefer cells where either pair is available ---
    cz = np.zeros(Nfval, dtype=int)
    cy = np.zeros(Nfval, dtype=int)
    cx = np.zeros(Nfval, dtype=int)
    print("cx, cy, cz", cx, cy, cz)
    # print("initialised centers")

    for p in range(Nfval):
        n = int(normals[p])
        width_val = float(widths_per_plane[p])

        if full_rectangle:
            # determine in-plane cartesian axes and corresponding array axes
            inplane_cart = [ax for ax in (0, 1, 2) if ax != n]
            cartA, cartB = inplane_cart[0], inplane_cart[1]
            arrA, arrB = CART2ARR[cartA], CART2ARR[cartB]

            # required length in cells along each in-plane cartesian axis
            LA = cells_from_meters(fault_length_m, cartA)
            LB = cells_from_meters(span_per_plane[p], cartB)
            halfA = LA // 2
            halfB = LB // 2

            pairs = PAIRS[n]
            apA = Rv.aperture[..., pairs[0][0], pairs[0][1]]
            apB = Rv.aperture[..., pairs[1][0], pairs[1][1]]
            available_zyx = np.column_stack(
                np.where((apA < width_val) | (apB < width_val))
            )

            # Filter to interior [1 .. N+1] along each ARRAY axis
            if available_zyx.size > 0:
                a = available_zyx
                mask_interior = (
                    (a[:, 0] >= ARR_MIN_INCL[0])
                    & (a[:, 0] <= ARR_MAX_INCL[0])
                    & (a[:, 1] >= ARR_MIN_INCL[1])
                    & (a[:, 1] <= ARR_MAX_INCL[1])
                    & (a[:, 2] >= ARR_MIN_INCL[2])
                    & (a[:, 2] <= ARR_MAX_INCL[2])
                )
                a = a[mask_interior]

                # Further require enough margin so the full LA x LB rectangle fits
                if a.size > 0:
                    mask_fit = (
                        (a[:, arrA] >= ARR_MIN_INCL[arrA] + halfA)
                        & (a[:, arrA] <= ARR_MAX_INCL[arrA] - halfA)
                        & (a[:, arrB] >= ARR_MIN_INCL[arrB] + halfB)
                        & (a[:, arrB] <= ARR_MAX_INCL[arrB] - halfB)
                    )
                    available_zyx = a[mask_fit]
                else:
                    available_zyx = a

            if available_zyx.size == 0:
                # fallback: random centre inside the SAFE interior ranges so full plane fits
                min_vals = [ARR_MIN_INCL[0], ARR_MIN_INCL[1], ARR_MIN_INCL[2]]
                max_vals = [ARR_MAX_INCL[0], ARR_MAX_INCL[1], ARR_MAX_INCL[2]]

                # enforce margins for in-plane axes
                min_vals[arrA] = ARR_MIN_INCL[arrA] + halfA
                max_vals[arrA] = ARR_MAX_INCL[arrA] - halfA
                min_vals[arrB] = ARR_MIN_INCL[arrB] + halfB
                max_vals[arrB] = ARR_MAX_INCL[arrB] - halfB

                # if margin impossible (min > max) fall back to full interior for that axis
                for ai in range(3):
                    if min_vals[ai] > max_vals[ai]:
                        min_vals[ai] = ARR_MIN_INCL[ai]
                        max_vals[ai] = ARR_MAX_INCL[ai]

                cz[p] = rng.integers(min_vals[0], max_vals[0] + 1)
                cy[p] = rng.integers(min_vals[1], max_vals[1] + 1)
                cx[p] = rng.integers(min_vals[2], max_vals[2] + 1)
            else:
                ridx = rng.integers(0, len(available_zyx))
                cz[p], cy[p], cx[p] = available_zyx[ridx]  # [z, y, x]
        else:
            n = int(normals[p])
            width_val = float(widths_per_plane[p])

            pairs = PAIRS[n]
            apA = Rv.aperture[..., pairs[0][0], pairs[0][1]]
            apB = Rv.aperture[..., pairs[1][0], pairs[1][1]]
            available_zyx = np.column_stack(
                np.where((apA < width_val) | (apB < width_val))
            )

            # Filter to interior [1 .. N+1] along each ARRAY axis
            if available_zyx.size > 0:
                a = available_zyx
                mask_interior = (
                    (a[:, 0] >= ARR_MIN_INCL[0])
                    & (a[:, 0] <= ARR_MAX_INCL[0])
                    & (a[:, 1] >= ARR_MIN_INCL[1])
                    & (a[:, 1] <= ARR_MAX_INCL[1])
                    & (a[:, 2] >= ARR_MIN_INCL[2])
                    & (a[:, 2] <= ARR_MAX_INCL[2])
                )
                available_zyx = a[mask_interior]

            if available_zyx.size == 0:
                # fallback: random centre anywhere in interior (1-based)
                cz[p] = rng.integers(ARR_MIN_INCL[0], ARR_MAX_INCL[0] + 1)
                cy[p] = rng.integers(ARR_MIN_INCL[1], ARR_MAX_INCL[1] + 1)
                cx[p] = rng.integers(ARR_MIN_INCL[2], ARR_MAX_INCL[2] + 1)
            else:
                ridx = rng.integers(0, len(available_zyx))
                cz[p], cy[p], cx[p] = available_zyx[ridx]  # [z, y, x]

        # print(f"got centers {p}")

    # --- Assignment ---
    placed = 0
    for p in range(Nfval):
        n = int(normals[p])  # 0=x, 1=y, 2=z
        width_val = float(widths_per_plane[p])

        # In-plane axes (cartesian ids)
        inplane_cart = [ax for ax in (0, 1, 2) if ax != n]
        cartA, cartB = inplane_cart[0], inplane_cart[1]
        arrA, arrB = CART2ARR[cartA], CART2ARR[cartB]

        LA = cells_from_meters(fault_length_m, cartA)
        LB = cells_from_meters(span_per_plane[p], cartB)

        # Centre in ARRAY order (1-based interior index)
        c_arr = [int(cz[p]), int(cy[p]), int(cx[p])]  # [z, y, x]

        # Odd-size handling (random extra cell)
        extraA = rng.integers(0, 2) if (LA % 2 == 1) else 0
        extraB = rng.integers(0, 2) if (LB % 2 == 1) else 0

        # Inclusive bounds on A/B axes (interior [1 .. N+1]) → convert to exclusive slice ends (+1)
        A0 = max(ARR_MIN_INCL[arrA], c_arr[arrA] - LA // 2 - (extraA == 1))
        A1_inc = min(
            ARR_MAX_INCL[arrA], c_arr[arrA] + LA // 2 + (extraA == 0)
        )  # inclusive
        B0 = max(ARR_MIN_INCL[arrB], c_arr[arrB] - LB // 2 - (extraB == 1))
        B1_inc = min(
            ARR_MAX_INCL[arrB], c_arr[arrB] + LB // 2 + (extraB == 0)
        )  # inclusive

        # Fix the normal axis at centre; march along arrA; assign strips across arrB
        sl = [
            slice(c_arr[0], c_arr[0] + 1),  # z
            slice(c_arr[1], c_arr[1] + 1),  # y
            slice(c_arr[2], c_arr[2] + 1),
        ]  # x

        aph_vals_pair = aph_pairs[p]  # (2,)
        res_vals_pair = res_pairs[p]  # (2,)
        pairs = PAIRS[n]  # [(idxc, idxo), (idxc, idxo)]

        # Iterate A with inclusive range; slice B to include last cell (+1 at end)
        for posA in range(A0, A1_inc + 1):  # include A1_inc
            sl[arrA] = slice(posA, posA + 1)
            sl[arrB] = slice(B0, B1_inc + 1)  # include B1_inc

            # Update BOTH off-diagonal pairs
            for pair_idx, (idxc, idxo) in enumerate(pairs):
                aph_val = float(aph_vals_pair[pair_idx])
                res_val = float(res_vals_pair[pair_idx])

                ap_region = Rv.aperture[sl[0], sl[1], sl[2], idxc, idxo]
                res_region = Rv.resistivity[sl[0], sl[1], sl[2], idxc]

                mask = (ap_region < width_val) & np.isfinite(res_region)
                if not np.any(mask):
                    continue

                # Apertures
                ap_region[mask] = width_val
                Rv.aperture[sl[0], sl[1], sl[2], idxc, idxo] = ap_region

                ae_region = Rv.aperture_electric[sl[0], sl[1], sl[2], idxc, idxo]
                ae_region[mask] = width_val
                Rv.aperture_electric[sl[0], sl[1], sl[2], idxc, idxo] = ae_region

                ah_region = Rv.aperture_hydraulic[sl[0], sl[1], sl[2], idxc, idxo]
                ah_region[mask] = aph_val
                Rv.aperture_hydraulic[sl[0], sl[1], sl[2], idxc, idxo] = ah_region

                # Electrical properties (on connector channel)
                res_vals_region = Rv.resistivity[sl[0], sl[1], sl[2], idxc]
                res_vals_region[mask] = res_val
                Rv.resistivity[sl[0], sl[1], sl[2], idxc] = res_vals_region

                R_region = Rv.resistance[sl[0], sl[1], sl[2], idxc]
                R_region[mask] = (
                    res_val / cellsize[idxc]
                )  # spacing along connector axis
                Rv.resistance[sl[0], sl[1], sl[2], idxc] = R_region

                # Record length for THIS connector channel
                rec_region = fault_lengths_assigned[sl[0], sl[1], sl[2], idxc]
                rec_region[mask] = float(fault_length_m)
                fault_lengths_assigned[sl[0], sl[1], sl[2], idxc] = rec_region

        placed += 1
        # print(f"placed {placed}")

    return Rv, fault_lengths_assigned


def build_normal_faulting_update_dict(
    Rv,
    Nfval,
    fault_length_m,
    fault_span_m,
    fw,
    aph0,
    aph1,
    resistivity0,
    resistivity1,
    pxyz,
    fault_lengths_assigned=None,
):
    """
    Builds input dictionaries for add_random_fault_planes_to_arrays function.
    Assumes a normal faulting regime with faults in x, y, and z directions.
    Assigns correct orientation and dimensions of faults based on normal regime
    (i.e. - vertical (along-strike) faults are square with length=fault_length_m,
            with aph0 and resistivity0 vertical and aph1 and resistivity1 horizontal
          - horizontal faults are rectangular with along-strike length=fault_length_m and
            across-strike length=fault_span_m, with aph0 and resistivity0 across-strike and
            aph1 and resistivity1 along-strike
          - vertical (across-strike), with across-strike length=fault_span_m and vertical
            extent = fault_length_m, with a aph0/ap1 and resistivity0/resistivity1
            randomly assigned as either horizontal/vertical or vertical/horizontal


    Args:
        i (_type_): _description_
        Nfval (_type_): _description_
        fault_length_m (_type_): Maximum fault length (along-strike and vertical length)
        fault_span_center (_type_): Minimum fault length (across-strike length)
        fw (_type_): fault widths (1 per fault)
        aph0 (_type_): hydraulic aperture perpendicular to slip
        aph1 (_type_): hydraulic aperture parallel to slip
        resistivity0 (_type_): resistivity perpendicular to slip
        resistivity1 (_type_): resistivity parallel to slip
        pxyz (_type_): probability of faults in x, y, z directions (should sum to 1.0)
    """
    fault_update_dict = {"x": {}, "y": {}, "z": {}}

    # normal x-direction faults
    # aph0 = perpendicular to slip = z direction (second axis), aph1 = parallel to slip = y direction (first axis)
    # along-strike length = length, down-dip = length
    px = pxyz[0]
    Nx = int(round(Nfval * px))
    fault_update_dict["x"]["Nfval"] = Nx
    fault_update_dict["x"]["fault_length_m"] = fault_length_m
    fault_update_dict["x"]["fault_span_m"] = fault_length_m
    idxs = np.random.choice(len(aph0), size=Nx, replace=True)
    fault_update_dict["x"]["fault_widths"] = fw[idxs]
    fault_update_dict["x"]["hydraulic_aperture_pairs"] = np.column_stack(
        [aph1[idxs], aph0[idxs]]
    )
    fault_update_dict["x"]["resistivity_pairs"] = np.column_stack(
        [resistivity1[idxs], resistivity0[idxs]]
    )
    fault_update_dict["x"]["pxyz"] = (1.0, 0.0, 0.0)

    # normal y-direction faults
    # aph0 and aph1 are a mix of perp and parallel to slip
    # across-strike length = span, down-dip = length
    py = pxyz[1]
    Ny = int(round(Nfval * py))
    fault_update_dict["y"]["Nfval"] = Ny
    fault_update_dict["y"]["fault_length_m"] = fault_span_m
    fault_update_dict["y"]["fault_span_m"] = fault_length_m
    idxs = np.random.choice(len(aph0), size=Ny, replace=True)
    swap_idxs = np.random.choice(Ny, size=Ny // 2, replace=False)
    fault_update_dict["y"]["fault_widths"] = fw[idxs]
    fault_update_dict["y"]["hydraulic_aperture_pairs"] = np.column_stack(
        [aph0[idxs], aph1[idxs]]
    )
    fault_update_dict["y"]["hydraulic_aperture_pairs"][swap_idxs] = fault_update_dict[
        "y"
    ]["hydraulic_aperture_pairs"][swap_idxs][:, ::-1]
    fault_update_dict["y"]["resistivity_pairs"] = np.column_stack(
        [resistivity1[idxs], resistivity0[idxs]]
    )
    fault_update_dict["y"]["resistivity_pairs"][swap_idxs] = fault_update_dict["y"][
        "resistivity_pairs"
    ][swap_idxs][:, ::-1]
    fault_update_dict["y"]["pxyz"] = (0.0, 1.0, 0.0)

    # normal z-direction faults
    # aph0 = perpendicular to slip = x direction (first axis), aph1 = parallel to slip = y direction (second axis)
    # along-strike length = length, across-strike = span
    pz = pxyz[2]
    Nz = int(round(Nfval * pz))
    fault_update_dict["z"]["Nfval"] = Nz
    fault_update_dict["z"]["fault_length_m"] = fault_span_m
    fault_update_dict["z"]["fault_span_m"] = fault_length_m
    idxs = np.random.choice(len(aph0), size=Nz, replace=True)
    fault_update_dict["z"]["fault_widths"] = fw[idxs]
    fault_update_dict["z"]["hydraulic_aperture_pairs"] = np.column_stack(
        [aph0[idxs], aph1[idxs]]
    )
    fault_update_dict["z"]["resistivity_pairs"] = np.column_stack(
        [resistivity0[idxs], resistivity1[idxs]]
    )
    fault_update_dict["z"]["pxyz"] = (0.0, 0.0, 1.0)

    if fault_lengths_assigned is not None:
        for direction in "xyz":
            fault_update_dict[direction]["fault_lengths_assigned"] = (
                fault_lengths_assigned[direction]
            )

    return fault_update_dict


def populate_rock_volume_with_all_faults(
    Rv,
    Nf,
    fault_length_center,
    fault_span_center,
    fw,
    aph0,
    aph1,
    resistivity0,
    resistivity1,
    pxyz,
):
    # make a copy of Rv to be used for additional faults
    Rv_start = copy.deepcopy(Rv)

    # for each direction: start with biggest faults and assign. Then check available cells.
    # if not enough available cells, then reduce Nf and try again.
    fault_lengths_assigned = {"x": None, "y": None, "z": None}

    # initialise dictionary to contain additional faults
    added_Rv = {}

    # start with biggest faults
    for i in np.arange(len(Nf))[::-1]:
        # get inputs to add Nf[i] faults of length fault_length_center[i] to Rv
        fault_update_dict = build_normal_faulting_update_dict(
            Rv,
            Nf[i],
            fault_length_center[i],
            fault_span_center[i],
            fw,
            aph0,
            aph1,
            resistivity0,
            resistivity1,
            pxyz=pxyz,
            fault_lengths_assigned=fault_lengths_assigned,
        )

        for direction in "xyz":
            # determine how many cells need to be assigned
            N, span, length = [
                fault_update_dict[direction][param]
                for param in ["Nfval", "fault_span_m", "fault_length_m"]
            ]
            print("N, span, length=", N, span, length)
            if direction == "x":
                nc0, nc1 = (
                    min(span / Rv.cellsize[2], Rv.ncells[2]),
                    min(length / Rv.cellsize[1], Rv.ncells[1]),
                )
            elif direction == "y":
                nc0, nc1 = (
                    min(span / Rv.cellsize[2], Rv.ncells[2]),
                    min(length / Rv.cellsize[0], Rv.ncells[0]),
                )
            elif direction == "x":
                nc0, nc1 = (
                    min(span / Rv.cellsize[0], Rv.ncells[0]),
                    min(length / Rv.cellsize[1], Rv.ncells[1]),
                )
            cells_to_be_assigned = N * nc0 * nc1
            # determine number of available cells for assignment
            if fault_lengths_assigned[direction] is None:
                nz, ny, nx = np.array(Rv.resistivity.shape)[:-1] - 2
                n_available = nx * ny * nz
            else:
                n_available_array = np.sum(
                    fault_lengths_assigned[direction][1:-1, 1:-1, 1:-1] == 0,
                    axis=(0, 1, 2),
                )
                idxs = np.array(
                    [idx for idx in [0, 1, 2] if idx != "xyz".index(direction)]
                )
                n_available = int(n_available_array[idxs].sum() / 2)
            print(
                "i",
                i,
                "direction",
                direction,
                "n_available",
                n_available,
                "to be assigned",
                cells_to_be_assigned,
            )
            # if there are not enough spaces, need to make additional rock volumes, the resistances and
            # permeabilities of which will be added in parallel to the Rv later.
            if n_available < cells_to_be_assigned:
                print(f"making updated Rv for direction {direction}, i {i}")
                # initialise dictionary
                if i not in added_Rv.keys():
                    added_Rv[i] = {}
                    if direction not in added_Rv[i].keys():
                        added_Rv[i][direction] = []
                # determine how many extra Rv need to be made
                n_rounds = int(np.floor(cells_to_be_assigned / n_available))
                print("n_rounds", n_rounds)
                # determine how many faults in the extra Rv
                n_per_round = np.floor(n_available / (nc0 * nc1))
                # create the extra Rv's
                for n in range(n_rounds):
                    fault_update_dict_new = clip_iterable_parameters(
                        fault_update_dict[direction],
                        int(n * n_per_round),
                        int((n + 1) * n_per_round),
                    )

                    # make a dummy Rv with same inputs
                    Rv_update = copy.deepcopy(Rv_start)

                    # add faults to the Rv and update permeability
                    Rvn, _ = add_random_fault_planes_to_arrays(
                        Rv_update, **fault_update_dict_new
                    )
                    Rvn.initialise_permeability()
                    added_Rv[i][direction].append(Rvn)
                print("Nfval", fault_update_dict[direction]["Nfval"])

                # update the input dict to just contain the additional faults not added to the extra Rvs
                fault_update_dict[direction] = clip_iterable_parameters(
                    fault_update_dict[direction],
                    int((n + 1) * n_per_round),
                    fault_update_dict[direction]["Nfval"],
                )
                print("Nfval", fault_update_dict[direction]["Nfval"])

            # add faults to the main Rv
            Rv, fault_lengths_assigned[direction] = add_random_fault_planes_to_arrays(
                Rv,
                **fault_update_dict[direction],
            )

    Rv.initialise_permeability()

    Rv_list = [Rv]
    for key in added_Rv.keys():
        for dir in added_Rv[key].keys():
            Rv_list += added_Rv[key][dir]

    # compute all the apertures and resistivities in all the rock volumes including extras
    ap_list = np.array([Rvi.aperture for Rvi in Rv_list])
    aph_list = np.array([Rvi.aperture_hydraulic for Rvi in Rv_list])
    res_list = np.array([Rvi.resistivity for Rvi in Rv_list])

    # add up total using in-parallel calculation for hydraulic aperture and resistivity
    ap_total = np.nansum(ap_list, axis=0)
    Rv.aperture = ap_total
    Rv.aperture_hydraulic = (
        12 * np.nansum(ap_list * aph_list**2 / 12, axis=0) / ap_total
    ) ** 0.5
    Rv.resistivity = len(res_list) / np.nansum(1.0 / res_list, axis=0)
    Rv.initialise_permeability()
    Rv.compute_conductive_fraction()
