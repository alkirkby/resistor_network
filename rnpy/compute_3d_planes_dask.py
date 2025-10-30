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
from rnpy.functions.readoutputs import read_fault_params
from rnpy.functions.utils import get_unique_properties, filter_by_min_max


rfluid = 0.5
direction = "z"
nc = 60
fault_aspect_ratio = 0.2
cellsize = 0.001
R = cellsize * nc
lmin, lmax = None, 0.05
gamma = 4.1
porosity_target = 0.05
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
    porosity_target=porosity_target,
    alpha_start=0.0,
    ndim=3,
)
Nf = get_Nf(gamma, alpha, R, lvals_range_unique, ndim=3)

aph0, aph1, resistivity0, resistivity1, fw, lvals_center = filter_by_min_max(
    lmin,
    lmax,
    lvals_center,
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

Rv_start = copy.deepcopy(Rv)

# for each direction: start with biggest faults and assign. Then check available cells.
# if not enough available cells, then reduce Nf and try again.
fault_lengths_assigned = {"x": None, "y": None, "z": None}


def clip_iterable_parameters(fault_dict, i0, i1):
    update_dict = copy.deepcopy(fault_dict)
    for key in update_dict.keys():
        if key != "pxyz":
            if type(update_dict[key]) is np.ndarray:
                if len(update_dict[key]) == update_dict["Nfval"]:
                    update_dict[key] = update_dict[key][i0:i1]
    update_dict["Nfval"] = i1 - i0
    return update_dict


added_Rv = {}

for i in np.arange(len(Nf))[::-1]:
    # start with biggest faults
    fault_update_dict = build_normal_faulting_update_dict(
        Rv,
        Nf[i],
        lvals_center_unique[i],
        fault_span_center[i],
        fw,
        aph0,
        aph1,
        resistivity0,
        resistivity1,
        pxyz=pxyz,
        fault_lengths_assigned=fault_lengths_assigned,
    )

    for direction in "x":
        N, span, length = [
            fault_update_dict[direction][param]
            for param in ["Nfval", "fault_span_m", "fault_length_m"]
        ]
        print(N, span, length)
        cells_to_be_assigned = N * span * length / cellsize**2
        if fault_lengths_assigned[direction] is None:
            n_available = nc**3
        else:
            n_available_array = np.sum(
                fault_lengths_assigned[direction][1:-1, 1:-1, 1:-1] == 0,
                axis=(0, 1, 2),
            )
            idxs = np.array([idx for idx in [0, 1, 2] if idx != "xyz".index(direction)])
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
        if n_available < cells_to_be_assigned:
            print(f"making updated Rv for direction {direction}, i {i}")
            if i not in added_Rv.keys():
                added_Rv[i] = {}
                if direction not in added_Rv[i].keys():
                    added_Rv[i][direction] = []
            n_rounds = int(np.floor(cells_to_be_assigned / n_available))
            print("n_rounds", n_rounds)
            n_per_round = np.floor(n_available * cellsize**2 / (span * length))

            for n in range(n_rounds):
                fault_update_dict_new = clip_iterable_parameters(
                    fault_update_dict[direction],
                    int(n * n_per_round),
                    int((n + 1) * n_per_round),
                )
                # make a dummy Rv with same inputs
                Rv_update = copy.deepcopy(Rv_start)

                Rvn, fault_lengths_assigned_n = add_random_fault_planes_to_arrays(
                    Rv_update, **fault_update_dict_new
                )
                Rvn.initialise_permeability()
                added_Rv[i][direction].append(Rvn)
            print("Nfval", fault_update_dict[direction]["Nfval"])
            fault_update_dict[direction] = clip_iterable_parameters(
                fault_update_dict[direction],
                int((n + 1) * n_per_round),
                fault_update_dict[direction]["Nfval"],
            )
            print("Nfval", fault_update_dict[direction]["Nfval"])

        Rv, fault_lengths_assigned[direction] = add_random_fault_planes_to_arrays(
            Rv,
            **fault_update_dict[direction],
        )


Rv.initialise_permeability()


# fault_lengths_assigned = (
#     fault_lengths_assigned_x + fault_lengths_assigned_y + fault_lengths_assigned_z
# )
# Rv.solve_resistor_network2()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d", aspect="equal")
# # mask = np.any(fault_lengths_assigned[direction] > 0, axis=3).transpose(2, 1, 0)
# mask = np.any(Rv.permeability > 1.1e-18, axis=3).transpose(2, 1, 0)
# # ax.view_init(elev=30, azim=-90)
# ax.voxels(mask)

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show(block=False)
# Rv.aperture_hydraulic
fig, axs = plt.subplots(1, 2)
vmin, vmax = 0, 5e-15
for i in range(2):
    m = axs[i].imshow(Rv.permeability[:, 10, :, 2], vmin=vmin, vmax=vmax)
Rv.compute_conductive_fraction()
print(Rv.conductive_fraction)
# Rv.solve_resistor_network2()
print(Rv.permeability_bulk)

Rv_list = [Rv]
for key in added_Rv.keys():
    for dir in added_Rv[key].keys():
        Rv_list += added_Rv[key][dir]

ap_list = np.array([Rvi.aperture for Rvi in Rv_list])
ap_total = np.nansum(ap_list, axis=0)
aph_list = np.array([Rvi.aperture_hydraulic for Rvi in Rv_list])
Rv.aperture = ap_total
Rv.aperture_hydraulic = (
    12 * np.nansum(ap_list * aph_list**2 / 12, axis=0) / ap_total
) ** 0.5
Rv.initialise_permeability()
Rv.compute_conductive_fraction()
print(Rv.conductive_fraction)
# Rv.solve_resistor_network2()
print(Rv.permeability_bulk)

fig, axs = plt.subplots(1, 2)
for i in range(2):
    m = axs[i].imshow(Rv.permeability[:, 10, :, 2], vmin=vmin, vmax=vmax)


# fig, axs = plt.subplots(1, 2)
# vmin, vmax = 0, 4e-5
# for i in range(2):
#     m = axs[i].imshow(ap_list[0][:, 10, :, i + 1, 0], vmin=vmin, vmax=vmax)

# fig, axs = plt.subplots(1, 2)
# vmin, vmax = 0, 4e-5
# for i in range(2):
#     m = axs[i].imshow(ap_list[1][:, 10, :, i + 1, 0], vmin=vmin, vmax=vmax)

# fig, axs = plt.subplots(1, 2)
# vmin, vmax = 0, 4e-5
# for i in range(2):
#     m = axs[i].imshow(ap_total[:, 10, :, i + 1, 0], vmin=vmin, vmax=vmax)

# fig, axs = plt.subplots(1, 2)
# vmin, vmax = 0, 2e-15
# for i in range(2):
#     m = axs[i].imshow(
#         added_Rv[0]["x"][0].permeability[:, 10, :, i + 1], vmin=vmin, vmax=vmax
#     )
# # plt.colorbar(m)

# fig, axs = plt.subplots(1, 2)
# vmin, vmax = 0, 4e-5
# for i in range(2):
#     m = axs[i].imshow(
#         added_Rv[0]["x"][0].aperture[:, 10, :, i + 1, 0], vmin=vmin, vmax=vmax
#     )
# plt.colorbar(m)

# fig, axs = plt.subplots(1, 2)
# vmin, vmax = None, None
# for i in range(2):
#     m = axs[i].imshow(
#         added_Rv[0]["x"][0].resistivity[:, 10, :, i + 1], vmin=vmin, vmax=vmax
#     )
# # plt.colorbar(m)

# fig, axs = plt.subplots(1, 2)
# vmin, vmax = 0, 2e-6
# for i in range(2):
#     axs[i].imshow(
#         added_Rv[0]["x"][0].aperture_hydraulic[:, 10, :, i + 1, 0], vmin=vmin, vmax=vmax
#     )
# # plt.colorbar(m)
