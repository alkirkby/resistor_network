import numpy as np
import sys
import os
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")
plt.ion()

if os.name == "nt":
    sys.path.append(r"C:\git\resistor_network")
    print("appended to path")

from rnpy.core.resistornetwork import Rock_volume
from rnpy.functions.assignfaults_new import (
    get_Nf,
    get_alpha,
    add_random_fault_planes_to_arrays,
)
from rnpy.functions.readoutputs import read_fault_params


def get_unique_properties(sort_array, property_array_list=None, compute_means=True):
    if len(np.unique(sort_array)) < len(sort_array):
        sort_array_unique = np.unique(sort_array)
        # get mean of other properties by sort_array
        if property_array_list is not None:
            prop_arrays_sorted = []
            for property_array in property_array_list:
                prop_array_i_sorted = [
                    np.array(property_array[sort_array == ll])
                    for ll in sort_array_unique
                ]

                if compute_means:
                    prop_array_i_sorted = np.array(
                        [np.mean(pa) for pa in prop_array_i_sorted]
                    )

                prop_arrays_sorted.append(prop_array_i_sorted)

    else:
        sort_array_unique, property_array_list = (
            sort_array,
            property_array_list,
        )
    if property_array_list is None:
        return sort_array_unique
    else:
        return sort_array_unique, prop_arrays_sorted


def filter_by_min_max(
    min_val,
    max_val,
    lvals_center_unique,
    array_list,
    use_indices=False,
):
    filtered_array_list = []
    for arr in array_list:
        if max_val is None:
            max_val, idx1 = (
                np.amax(lvals_center_unique) + 1,
                len(lvals_center_unique) + 1,
            )
        else:
            idx1 = (
                np.where(lvals_center_unique <= max_val)[-1][-1]
                - len(lvals_center_unique)
                + 1
            )  # relative to the end
        if min_val is None:
            min_val, idx0 = 0, 0
        else:
            idx0 = np.where(lvals_center_unique > min_val)[0][0]
        if use_indices:
            filtered_array_list.append(arr[idx0:idx1])
        else:
            filt = np.all([lvals_center > min_val, lvals_center <= max_val], axis=0)
            filtered_array_list.append(arr[filt])

    return filtered_array_list


rfluid = 0.5
direction = "z"
nc = 20
fault_aspect_ratio = 0.2
cellsize = 0.001
R = 0.02
lmin, lmax = None, 0.03
gamma = 4.1
pxyz = (0.8, 0.1, 0.1)

wd = r"C:\Users\alisonk.GNS\OneDrive - GNS Science\Energy_Futures_Project_2_Geophysics\Rock_property_modelling\summary_data_from_models"

# read fault lengths (center of bin) + pre-computed properties from json file
lvals_center, lvals_range, fw, aph0, resistivity0 = read_fault_params(
    os.path.join(wd, "fault_k_aperture_rf%s_y.npy" % (rfluid)),
    None,
)
lvals_center, lvals_range, fw, aph1, resistivity1 = read_fault_params(
    os.path.join(wd, "fault_k_aperture_rf%s_z.npy" % (rfluid)),
    None,
)
lvals_center_unique, [fw_mean] = get_unique_properties(
    lvals_center, property_array_list=[fw], compute_means=True
)
lvals_range_unique = get_unique_properties(lvals_range)
# fault span (center of bin)
fault_span_center = fault_aspect_ratio * lvals_center

R_for_alpha_calc = 10 * lvals_center_unique.max()


alpha = get_alpha(
    gamma=gamma,
    R=R_for_alpha_calc,
    lvals_center=lvals_center_unique,
    lvals_range=lvals_range_unique,
    fw=fw_mean,
    porosity_target=0.05,
    alpha_start=0.0,
    ndim=3,
)
Nf = get_Nf(gamma, alpha, R, lvals_range_unique, ndim=3)

aph0, aph1, resistivity0, resistivity1, fw, lvals_center = filter_by_min_max(
    lmin,
    lmax,
    lvals_center_unique,
    [aph0, aph1, resistivity0, resistivity1, fw, lvals_center],
    use_indices=False,
)
Nf, lvals_range_unique, lvals_center_unique = filter_by_min_max(
    lmin,
    lmax,
    lvals_center_unique,
    [Nf, lvals_range_unique, lvals_center_unique],
    use_indices=True,
)
fw[:] = 2e-5
aph0[:] = 1e-6  # np.median(aph0)
aph1[:] = 2e-6  # np.median(aph1)
resistivity0[:] = 4  # np.median(resistivity0)
resistivity1[:] = 3  # np.median(resistivity1)


i = 0

# normal x-direction faults
# aph0 = perpendicular to slip = z direction (second axis), aph1 = parallel to slip = y direction (first axis)
# along-strike length = length, down-dip = length
px = pxyz[0]
Nx = int(round(Nf[i] * px))
length_x = lvals_center_unique[i]
span_x = lvals_center_unique[i]
idxs = np.random.choice(len(aph0), size=Nx, replace=True)
fwidths_x = fw[idxs]
hyd_ap_pairs_x = np.column_stack([aph1[idxs], aph0[idxs]])
res_pairs_x = np.column_stack([resistivity1[idxs], resistivity0[idxs]])

# normal y-direction faults
# aph0 and aph1 are a mix of perp and parallel to slip
# across-strike length = span, down-dip = length
py = pxyz[1]
Ny = int(round(Nf[i] * py))
length_y = fault_span_center[i]
span_y = lvals_center_unique[i]
# indices to randomly choose apertures
idxs = np.random.choice(len(aph0), size=Ny, replace=True)
swap_idxs = np.random.choice(Ny, size=Ny // 2, replace=False)

fwidths_y = fw[idxs]
hyd_ap_pairs_y = np.column_stack([aph0[idxs], aph1[idxs]])
hyd_ap_pairs_y[swap_idxs] = hyd_ap_pairs_y[swap_idxs][:, ::-1]
res_pairs_y = np.column_stack([resistivity0[idxs], resistivity1[idxs]])
res_pairs_y[swap_idxs] = res_pairs_y[swap_idxs][:, ::-1]


# normal z-direction faults
# aph0 = perpendicular to slip = x direction (first axis), aph1 = parallel to slip = y direction (second axis)
# along-strike length = length, across-strike = span
pz = pxyz[0]
Nz = Nf[i] - Nx - Ny
length_z = lvals_center_unique[i]
span_z = fault_span_center[i]
idxs = np.random.choice(len(aph0), size=Nz, replace=True)
fwidths_z = fw[idxs]
hyd_ap_pairs_z = np.column_stack([aph0[idxs], aph1[idxs]])
res_pairs_z = np.column_stack([resistivity0[idxs], resistivity1[idxs]])


t0 = time.time()
Rv = Rock_volume(ncells=(nc, nc, nc), cellsize=cellsize)
Rv.aperture[np.isfinite(Rv.aperture)] = 2e-50
Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = 2e-50
Rv.initialise_electrical_resistance()

# initialise hydraulic aperture to reflect matrix permeability
for i in range(3):
    Rv.aperture_hydraulic[:, :, :, i][
        np.isfinite(Rv.aperture_hydraulic[:, :, :, i])
    ] = (12 * Rv.permeability_matrix[i]) ** 0.5


Rv, fault_lengths_assigned_x = add_random_fault_planes_to_arrays(
    Rv,
    Nx,
    fault_length_m=length_x,
    fault_span_m=span_x,
    fault_widths=fwidths_x,
    hydraulic_aperture_pairs=hyd_ap_pairs_x,  # (N,2)
    resistivity_pairs=res_pairs_x,  # (N,2)
    pxyz=(1.0, 0.0, 0.0),
)
Rv, fault_lengths_assigned_y = add_random_fault_planes_to_arrays(
    Rv,
    Ny,
    fault_length_m=length_y,
    fault_span_m=span_y,
    fault_widths=fwidths_y,
    hydraulic_aperture_pairs=hyd_ap_pairs_y,  # (N,2)
    resistivity_pairs=res_pairs_y,  # (N,2)
    pxyz=(0.0, 1.0, 0.0),
)
Rv, fault_lengths_assigned_z = add_random_fault_planes_to_arrays(
    Rv,
    Nz,
    fault_length_m=length_z,
    fault_span_m=span_z,
    fault_widths=fwidths_z,
    hydraulic_aperture_pairs=hyd_ap_pairs_z,  # (N,2)
    resistivity_pairs=res_pairs_z,  # (N,2)
    pxyz=(0.0, 0.0, 1.0),
)

Rv.initialise_permeability()
# fault_lengths_assigned = (
#     fault_lengths_assigned_x + fault_lengths_assigned_y
# + fault_lengths_assigned_z)
# Rv.solve_resistor_network2()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d", aspect="equal")
# mask = np.any(fault_lengths_assigned > 0, axis=3).transpose(2, 1, 0)
mask = np.any(Rv.permeability > 1.1e-18, axis=3).transpose(2, 1, 0)
# ax.view_init(elev=30, azim=-90)
ax.voxels(mask)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show(block=False)
