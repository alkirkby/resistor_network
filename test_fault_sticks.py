# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:05:59 2021

@author: alisonk
"""

from unittest import TestCase
from tests import TEST_DATA_ROOT
from rnpy.core.resistornetwork import Rock_volume
from rnpy.functions.readoutputs import get_equivalent_rho
from rnpy.functions.assignfaults_new import add_random_fault_sticks_to_arrays
import numpy as np
import os


# set some parameters
nc = 20  # number of cells
cellsize = 0.001  # cellsize
R = cellsize * (nc + 1)  # size of area
# fix some values
fw = 2e-5
aph = np.ones(20) * 2e-6  # np.median(aph1)
resistivity = np.ones(20) * 4  # np.median(resistivity0)


# convert resistivities to equivalent values for the cell
resistivityeq = get_equivalent_rho(resistivity, fw, cellsize)

# target faults to assign
Nf = 10

fault_length = 0.05


class testFaultSticks(TestCase):

    def test_build_run_random_aperture(self):

        for pz in [1.0, 0.0]:
            if pz == 1.0:
                idx = 2
            elif pz == 0.0:
                idx = 1

            Rv = Rock_volume(
                ncells=(0, nc, nc), cellsize=cellsize, resistivity_fluid=0.5
            )
            Rv.aperture[np.isfinite(Rv.aperture)] = 2e-50
            Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = 2e-50
            Rv.initialise_electrical_resistance()
            km = Rv.permeability_matrix[idx]
            rm = Rv.resistivity_matrix[idx]

            # initialise hydraulic aperture to reflect matrix permeability
            for i in range(3):
                Rv.aperture_hydraulic[:, :, :, i][
                    np.isfinite(Rv.aperture_hydraulic[:, :, :, i])
                ] = (12 * Rv.permeability_matrix[i]) ** 0.5

            np.random.seed(1)
            fault_lengths_assigned = None

            # add fault stick to array
            Rv, fault_lengths_assigned = add_random_fault_sticks_to_arrays(
                Rv,
                Nf,
                fault_length,
                fw,
                aph,
                resistivityeq,
                pz,
                fault_lengths_assigned=fault_lengths_assigned,
            )
            # actual faults added
            if pz == 1.0:
                N_actual = (Rv.aperture[1, :, 1, idx, 3 - idx] > 1e-49).sum()
            else:
                N_actual = (Rv.aperture[:, 1, 1, idx, 3 - idx] > 1e-49).sum()

            Rv.initialise_permeability()

            Rv.solve_resistor_network2()

            # i1 k uses aph1
            k_theory = (
                N_actual * fw * aph[0] ** 2 / 12 + (R - N_actual * fw) * km
            ) / R

            # i0 res uses resistivity0
            r_theory = R / (
                N_actual * fw / resistivity[0] + (R - N_actual * fw) / rm
            )

            print(Rv.permeability_bulk[idx])
            print(k_theory)

            assert (Rv.permeability_bulk[idx] - k_theory) / k_theory < 1e-3
            assert (Rv.resistivity_bulk[idx] - r_theory) / r_theory < 1e-3
