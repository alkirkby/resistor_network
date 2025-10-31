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
    populate_rock_volume_with_all_faults,
)
from rnpy.functions.readoutputs import read_fault_params, get_equivalent_rho
from rnpy.functions.utils import (
    get_unique_properties,
    filter_by_min_max,
    clip_iterable_parameters,
)


rfluid = 0.5
direction = "z"
nc = 20
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
# convert resistivities to equivalent values for the cell
resistivity0 = get_equivalent_rho(resistivity0, fw, cellsize)
resistivity1 = get_equivalent_rho(resistivity1, fw, cellsize)

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
# fw[:] = 2e-5
# aph0[:] = 1e-6  # np.median(aph0)
# aph1[:] = 2e-6  # np.median(aph1)
# resistivity0[:] = 4  # np.median(resistivity0)
# resistivity1[:] = 3  # np.median(resistivity1)


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


populate_rock_volume_with_all_faults(
    Rv,
    Nf,
    lvals_center_unique,
    fault_span_center,
    fw,
    aph0,
    aph1,
    resistivity0,
    resistivity1,
    pxyz,
)
