import numpy as np
import sys
import os
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib
import copy

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
    build_normal_faulting_update_dict,
)
from rnpy.functions.readoutputs import read_fault_params, get_equivalent_rho
from rnpy.functions.utils import (
    get_unique_properties,
    filter_by_min_max,
)
from dask.distributed import Client
from dask import delayed, compute

rfluid = 0.5
direction = "z"
nc = 24
fault_aspect_ratio = 0.2
gamma = 4.1
porosity_target = 0.02

# micro scale
res_matrix = np.array([1000.0, 1000.0, 1000.0])
k_matrix = np.array([1e-18, 1e-18, 1e-18])
cellsize, lmin, lmax = 0.002, None, 0.05
pxyz = (1 / 3, 1 / 3, 1 / 3)

# # meso scale
res_matrix = np.array([676.98144846, 476.26921402, 470.81550976])
# res_matrix = np.array([1000.0, 1000.0, 1000.0])
k_matrix = np.array([8.96531544e-17, 4.05607225e-16, 4.54526428e-16])
cellsize, lmin, lmax = 0.02, 0.051, 0.5
pxyz = (0.6, 0.2, 0.2)

# # macro scale
res_matrix = np.array([646.8899511, 332.59344532, 395.94582483])
k_matrix = np.array([1.61760051e-16, 1.08027302e-14, 1.71143332e-15])
cellsize, lmin, lmax = 0.2, 0.51, 5.0
pxyz = (0.7, 0.15, 0.15)

# # reservoir scale
res_matrix = np.array([632.99173118, 276.66530104, 359.5144438])
k_matrix = np.array([1.29098467e-15, 2.34268624e-13, 1.91038064e-13])
cellsize, lmin, lmax = 2.0, 5.1, 50.0

R = cellsize * nc

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
# # convert resistivities to equivalent values for the cell
# resistivity0 = get_equivalent_rho(resistivity0, fw, cellsize)
# resistivity1 = get_equivalent_rho(resistivity1, fw, cellsize)


lvals_center_unique, [fw_mean] = get_unique_properties(
    lvals_center, property_array_list=[fw], compute_means=True
)
lvals_range_unique = get_unique_properties(lvals_range)
# fault span (center of bin)
fault_span_center = fault_aspect_ratio * lvals_center_unique
fault_span_range = fault_aspect_ratio * lvals_range_unique


# fw[:] = 2e-5
# fw_mean[:] = 2e-5
# aph0[:] = 1e-6  # np.median(aph0)
# aph1[:] = 2e-6  # np.median(aph1)
# resistivity0[:] = 4  # np.median(resistivity0)
# resistivity1[:] = 3  # np.median(resistivity1)


# mean fault radius given span is shorter than length
def mean_diameter(pxyz, length, span):
    return (
        pxyz[0] * length**2 + pxyz[1] * length * span + pxyz[2] * length * span
    ) ** 0.5


R_for_alpha_calc = 10 * lvals_center_unique.max()
mean_diameter_range = mean_diameter(pxyz, lvals_range_unique, fault_span_range)
mean_diameter_center = mean_diameter(pxyz, lvals_center_unique, fault_span_center)

alpha = get_alpha(
    gamma=gamma,
    R=R_for_alpha_calc,
    lvals_center=mean_diameter_center,
    lvals_range=mean_diameter_range,
    fw=fw_mean,
    porosity_target=porosity_target,
    alpha_start=0.0,
    ndim=3,
)
Nf = get_Nf(gamma, alpha, R, lvals_range_unique, ndim=3)
print("Nf", Nf)

NfLarge = get_Nf(gamma, alpha, R_for_alpha_calc, mean_diameter_range, ndim=3)
porosity_theory = NfLarge * mean_diameter_center**2 * fw_mean / R_for_alpha_calc**3

aph0, aph1, resistivity0, resistivity1, fw, lvals_center = filter_by_min_max(
    lmin,
    lmax,
    lvals_center,
    [aph0, aph1, resistivity0, resistivity1, fw, lvals_center],
    use_indices=False,
)
print(Nf, lvals_center_unique)
Nf, lvals_range_unique, lvals_center_unique = filter_by_min_max(
    lmin,
    lmax,
    lvals_center_unique,
    [Nf, lvals_range_unique, lvals_center_unique],
    use_indices=True,
)
print(Nf, lvals_center_unique)

t0 = time.time()
Rv = Rock_volume(ncells=(nc, nc, nc), cellsize=cellsize)
Rv.aperture[np.isfinite(Rv.aperture)] = 2e-50
Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = 2e-50
Rv.initialise_electrical_resistance()
Rv.resistivity_matrix = res_matrix
Rv.permeability_matrix = k_matrix

# initialise hydraulic aperture to reflect matrix permeability
for i in range(3):
    Rv.aperture_hydraulic[:, :, :, i][
        np.isfinite(Rv.aperture_hydraulic[:, :, :, i])
    ] = (12 * Rv.permeability_matrix[i]) ** 0.5

fault_numbers_assigned = {"x": None, "y": None, "z": None}
fault_numbers_assigned_dict = {}
for direction in "xyz":
    fault_numbers_assigned_dict[direction] = {}
    for i in np.arange(len(Nf))[::-1]:  # largest to smallest
        fault_update_dict = build_normal_faulting_update_dict(
            Nf[i],
            lvals_center_unique[i],
            fault_span_center[i],
            fw,
            aph0,
            aph1,
            resistivity0,
            resistivity1,
            pxyz=pxyz,
            fault_numbers_assigned=fault_numbers_assigned,
        )

        Rv, fault_numbers_assigned[direction] = add_random_fault_planes_to_arrays(
            Rv, **fault_update_dict[direction]
        )
        fault_numbers_assigned_dict[direction][lvals_center_unique[i]] = copy.deepcopy(
            fault_numbers_assigned[direction]
        )
    Rv.initialise_permeability()
    Rv.initialise_electrical_resistance()

# porosity_actual = [
#     (fault_lengths_assigned["x"] == lvals_center_unique[ii]).sum()
#     * cellsize**2
#     * fw_mean[ii]
#     / (2 * R**3)
#     + (fault_lengths_assigned["y"].sum() == lvals_center_unique[ii])
#     * cellsize**2
#     * fw_mean[ii]
#     / (2 * R**3)
#     + (fault_lengths_assigned["z"].sum() == lvals_center_unique[ii])
#     * cellsize**2
#     * fw_mean[ii]
#     / (2 * R**3)
#     for ii in range(len(Nf))
# ]

x = np.arange(-cellsize, R + cellsize, cellsize)
plt.figure()
# plt.imshow(np.log10(Rv.permeability[:, 10, :, 2]), vmin=-15, vmax=-11)
plt.pcolormesh(x, x, np.log10(Rv.permeability[:, 10, :, 2]), vmin=-15, vmax=-8)
plt.colorbar()

plt.figure()
# plt.imshow(Rv.aperture[:, 10, :, 2, 0], vmin=1e-6, vmax=1e-3)
plt.pcolormesh(x, x, Rv.aperture[:, 10, :, 2, 0], vmin=1e-6, vmax=1e-2)
plt.colorbar()

for lval in lvals_center_unique:
    plt.figure()
    # plt.imshow(Rv.aperture[:, 10, :, 2, 0], vmin=1e-6, vmax=1e-3)
    plt.pcolormesh(x, x, fault_numbers_assigned_dict["x"][lval][:, 10, :, 2])
    plt.colorbar()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# mask = np.any(Rv.permeability > 1.1e-18, axis=3).transpose(2, 1, 0)
# ax.view_init(elev=30, azim=-90)
# ax.voxels(mask)

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show(block=False)
Rv.solve_resistor_network2()
print(Rv.permeability_bulk)
print(Rv.resistivity_bulk)
Rv.compute_conductive_fraction()
print(Rv.conductive_fraction)
print(np.nanmax(Rv.aperture))
