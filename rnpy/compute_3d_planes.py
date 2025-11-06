import numpy as np
import sys
import os
import argparse
import time
import matplotlib.pyplot as plt

import copy


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
from rnpy.functions.command_line import parse_arguments_3d
from dask.distributed import Client
from dask import delayed, compute


if __name__ == "__main__":
    # check if it is run by command line or not
    is_cmdline = True if (sys.stdin.isatty() and sys.stdout.isatty()) else False

    if is_cmdline:
        input_parameters = parse_arguments_3d(sys.argv)
        for key in input_parameters.keys():
            print(key, input_parameters[key])
        R = input_parameters["width_m"]
        nc = int(R / input_parameters["cellsize_mm"])
    else:
        input_parameters = {}
        # set defaults
        input_parameters["repeats"] = 1
        input_parameters["resistivity_fluid"] = 0.5
        nc = 20
        input_parameters["fault_aspect_ratio"] = 0.2
        input_parameters["gamma"] = 4.1
        input_parameters["target_porosity"] = 0.02
        input_parameters["resistivity_matrix"] = np.array([1000.0, 1000.0, 1000.0])
        input_parameters["permeability_matrix"] = np.array([1e-18, 1e-18, 1e-18])
        (
            input_parameters["cellsize_mm"],
            input_parameters["lmin"],
            input_parameters["lmax"],
        ) = 0.005, None, 0.05
        input_parameters["width_m"] = R = input_parameters["cellsize_mm"] * nc
        input_parameters["pxyz"] = (0.8, 0.1, 0.1)
        input_parameters["working_directory"] = (
            r"C:\Users\alisonk.GNS\OneDrive - GNS Science\Energy_Futures_Project_2_Geophysics\Rock_property_modelling\summary_data_from_models"
        )
        input_parameters["n_workers"] = 1

    client = Client(
        n_workers=input_parameters["n_workers"],
        threads_per_worker=2,
        # local_directory=local_dir,
        processes=True,
        dashboard_address=None,
        memory_limit="auto",
    )

    # read fault lengths (center of bin) + pre-computed properties from json file
    lvals_center, lvals_range, fw, aph0, resistivity0 = read_fault_params(
        os.path.join(
            input_parameters["working_directory"],
            "fault_k_aperture_rf%s_y.npy" % (input_parameters["resistivity_fluid"]),
        ),
        None,
    )
    lvals_center, lvals_range, fw, aph1, resistivity1 = read_fault_params(
        os.path.join(
            input_parameters["working_directory"],
            "fault_k_aperture_rf%s_z.npy" % (input_parameters["resistivity_fluid"]),
        ),
        None,
    )

    lvals_center_unique, [fw_mean] = get_unique_properties(
        lvals_center, property_array_list=[fw], compute_means=True
    )
    lvals_range_unique = get_unique_properties(lvals_range)
    # fault span (center of bin)
    fault_span_center = input_parameters["fault_aspect_ratio"] * lvals_center_unique
    fault_span_range = input_parameters["fault_aspect_ratio"] * lvals_range_unique

    # mean fault radius given span is shorter than length
    def mean_diameter(pxyz, length, span):
        return (
            pxyz[0] * length**2 + pxyz[1] * length * span + pxyz[2] * length * span
        ) ** 0.5

    R_for_alpha_calc = 10 * lvals_center_unique.max()
    mean_diameter_range = mean_diameter(
        input_parameters["pxyz"], lvals_range_unique, fault_span_range
    )
    mean_diameter_center = mean_diameter(
        input_parameters["pxyz"], lvals_center_unique, fault_span_center
    )

    input_parameters["alpha"] = get_alpha(
        gamma=input_parameters["gamma"],
        R=R_for_alpha_calc,
        lvals_center=mean_diameter_center,
        lvals_range=mean_diameter_range,
        fw=fw_mean,
        porosity_target=input_parameters["target_porosity"],
        alpha_start=0.0,
        ndim=3,
    )

    Nf = get_Nf(
        input_parameters["gamma"],
        input_parameters["alpha"],
        R,
        lvals_range_unique,
        ndim=3,
    )
    print("Nf", Nf)

    NfLarge = get_Nf(
        input_parameters["gamma"],
        input_parameters["alpha"],
        R_for_alpha_calc,
        mean_diameter_range,
        ndim=3,
    )
    porosity_theory = NfLarge * mean_diameter_center**2 * fw_mean / R_for_alpha_calc**3

    aph0, aph1, resistivity0, resistivity1, fw, lvals_center = filter_by_min_max(
        input_parameters["lmin"],
        input_parameters["lmax"],
        lvals_center,
        [aph0, aph1, resistivity0, resistivity1, fw, lvals_center],
        use_indices=False,
    )

    Nf, lvals_range_unique, lvals_center_unique = filter_by_min_max(
        input_parameters["lmin"],
        input_parameters["lmax"],
        lvals_center_unique,
        [Nf, lvals_range_unique, lvals_center_unique],
        use_indices=True,
    )

    @delayed
    def setup_and_solve(input_parameters):
        # initialise a rock volume
        Rv = Rock_volume(
            ncells=(nc, nc, nc),
            cellsize=input_parameters["cellsize_mm"],
            resistivity_matrix=input_parameters["resistivity_matrix"],
            resistivity_fluid=input_parameters["resistivity_fluid"],
            permeability_matrix=input_parameters["permeability_matrix"],
        )

        # set background apertures, etc to matrix values
        Rv.aperture[np.isfinite(Rv.aperture)] = 2e-50
        Rv.aperture_electric[np.isfinite(Rv.aperture_electric)] = 2e-50
        Rv.initialise_electrical_resistance()
        Rv.resistivity_matrix = input_parameters["resistivity_matrix"]
        Rv.permeability_matrix = input_parameters["permeability_matrix"]

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
                    pxyz=input_parameters["pxyz"],
                    fault_numbers_assigned=fault_numbers_assigned,
                )

                Rv, fault_numbers_assigned[direction] = (
                    add_random_fault_planes_to_arrays(
                        Rv, **fault_update_dict[direction]
                    )
                )
                fault_numbers_assigned_dict[direction][lvals_center_unique[i]] = (
                    copy.deepcopy(fault_numbers_assigned[direction])
                )
            Rv.initialise_permeability()
            Rv.initialise_electrical_resistance()

        Rv.solve_resistor_network2()
        print(Rv.permeability_bulk)
        print(Rv.resistivity_bulk)
        Rv.compute_conductive_fraction()
        print(Rv.conductive_fraction)
        print(np.nanmax(Rv.aperture))
        return Rv, fault_numbers_assigned_dict

    t0 = time.time()
    simulations = [
        setup_and_solve(input_parameters) for _ in range(input_parameters["repeats"])
    ]
    results = compute(*simulations)
    t1 = time.time()
    print(f"{input_parameters['repeats']} simulations took {t1 - t0}s")
    Rv_list = [results[i][0] for i in range(len(results))]
    assigned_dict_list = [results[i][1] for i in range(len(results))]

    if not is_cmdline:
        import matplotlib

        matplotlib.use("Qt5Agg")
        plt.ion()
        Rv = Rv_list[0]
        fault_numbers_assigned_dict = assigned_dict_list[0]
        x = np.arange(
            -input_parameters["cellsize_mm"],
            R + input_parameters["cellsize_mm"],
            input_parameters["cellsize_mm"],
        )
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

    def get_header(param_dict):
        header = ""
        for key in param_dict.keys():
            header += f"{key} {param_dict[key]:}\n"
        return header

    prop_suffix = (
        f"gamma{input_parameters['gamma']:.1f}_R{R}_rf{input_parameters['resistivity_fluid']}_"
        + f"por{input_parameters['target_porosity']:.2f}pc_pxyz{'_'.join([str(f) for f in input_parameters['pxyz']])}"
        + f"_cs{input_parameters['cellsize_mm']}mm"
    )
    output_fn = f"FaultVolume_{prop_suffix}.dat"
    output_fn = os.path.join(input_parameters["working_directory"], output_fn)
    header = get_header(input_parameters)
    header += "conductive_fraction "
    for param in ["resistivity", "permeability"]:
        for direction in "xyz":
            header += f"{param}_{direction}"

    out_array = np.vstack(
        [
            np.concatenate(
                [[Rv.conductive_fraction], Rv.resistivity_bulk, Rv.permeability_bulk]
            )
            for Rv in Rv_list
        ]
    )

    np.savetxt(output_fn, out_array, header=header, fmt=["%.4f"] + ["%.3e"] * 6)
