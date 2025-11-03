import numpy as np
import sys
import os
from tests import TEST_DATA_ROOT
from unittest import TestCase

if os.name == "nt":
    sys.path.append(r"C:\git\resistor_network")
    print("appended to path")

from rnpy.core.resistornetwork import Rock_volume
from rnpy.functions.assignfaults_new import (
    add_random_fault_planes_to_arrays,
    build_normal_faulting_update_dict,
)
from rnpy.functions.readoutputs import get_equivalent_rho


nc = 20
cellsize = 0.001
R = cellsize * (nc + 1)


# fix some values
fw = np.ones(100) * 2e-5
aph0 = np.ones(100) * 1e-6  # np.median(aph0)
aph1 = np.ones(100) * 2e-6  # np.median(aph1)
resistivity0 = np.ones(100) * 4  # np.median(resistivity0)
resistivity1 = np.ones(100) * 3  # np.median(resistivity1)
# convert resistivities to equivalent values for the cell


# test case 1: ten faults with normal x, normal y, normal z
#
class testFaultPlanes3d(TestCase):
    def test_many_faults(self):
        for direction in "xz":
            ii = 0

            Nf = [10]
            n_assigned = 10
            lvals_center_unique = fault_span_center = [0.05]  # fill the whole plane

            pxyz = np.zeros(3)  # + 1.0 / 3
            pxyz[list("xyz").index(direction)] = 1.0

            Rv = Rock_volume(ncells=(nc, nc, nc), cellsize=cellsize)
            Rv.aperture[np.isfinite(Rv.aperture)] = 2e-50
            Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = 2e-50
            Rv.initialise_electrical_resistance()

            # initialise hydraulic aperture to reflect matrix permeability
            for i in range(3):
                Rv.aperture_hydraulic[:, :, :, i][
                    np.isfinite(Rv.aperture_hydraulic[:, :, :, i])
                ] = (12 * Rv.permeability_matrix[i]) ** 0.5

            fault_update_dict = build_normal_faulting_update_dict(
                Nf[ii],
                lvals_center_unique[ii],
                fault_span_center[ii],
                fw,
                aph0,
                aph1,
                resistivity0,
                resistivity1,
                pxyz=pxyz,
                fault_numbers_assigned=None,
            )

            fault_lengths_assigned = {}
            Rv, fault_lengths_assigned[direction] = add_random_fault_planes_to_arrays(
                Rv, **fault_update_dict[direction]
            )
            Rv.initialise_permeability()
            Rv.initialise_electrical_resistance()
            Rv.solve_resistor_network2()

            if direction == "x":
                # vertical, along-strike
                (
                    i0,
                    i1,
                ) = 2, 1
            elif direction == "z":
                # across, along-strike
                i0, i1 = 0, 1

            # number of faults assigned (due to random process sometimes get overlap)

            rm, km = Rv.resistivity_matrix, Rv.permeability_matrix
            k_theory = np.ones(3) * km
            r_theory = np.ones(3) * rm

            # i0 k uses aph0
            k_theory[i0] = (
                n_assigned * fw[0] * aph0[0] ** 2 / 12
                + (R - n_assigned * fw[0]) * km[i0]
            ) / R
            # i1 k uses aph1
            k_theory[i1] = (
                n_assigned * fw[0] * aph1[0] ** 2 / 12
                + (R - n_assigned * fw[0]) * km[i1]
            ) / R
            # i0 res uses resistivity0
            r_theory[i0] = R / (
                n_assigned * fw[0] / resistivity0[0] + (R - n_assigned * fw[0]) / rm[i0]
            )
            # i1 res uses resistivity0
            r_theory[i1] = R / (
                n_assigned * fw[0] / resistivity1[0] + (R - n_assigned * fw[0]) / rm[i1]
            )

            assert np.all(np.abs(k_theory - Rv.permeability_bulk) / k_theory < 1e-3)
            assert np.all(np.abs(r_theory - Rv.resistivity_bulk) / r_theory < 1e-3)
