import argparse
import numpy as np


def parse_arguments_3d(arguments):
    """
    parse arguments from command line
    """

    argument_names = [
        ["target_porosity", "p", "target porosity", 1, float],
        ["width_m", "R", "model width", 1, float],
        ["cellsize_mm", "c", "cellsize for fault stick model", 1, float],
        ["gamma", None, "density exponent gamma", 1, float],
        [
            "pxyz",
            None,
            "probability of a fault in x, y, z direction",
            3,
            float,
        ],
        ["resistivity_fluid", "rf", "resistivity of fluid", 1, float],
        ["resistivity_matrix", "rm", "resistivity of matrix", "*", float],
        ["permeability_matrix", "km", "permeability of matrix", "*", float],
        ["lmin", None, "minimum fault length", 1, float],
        ["lmax", None, "maximum fault length", 1, float],
        [
            "fault_aspect_ratio",
            None,
            "maximum fault length for cross-faults",
            1,
            float,
        ],
        [
            "working_directory",
            "wd",
            "working directory for inputs and outputs",
            1,
            str,
        ],
        ["repeats", "r", "number of repeats", 1, int],
        ["n_workers", "nw", "number of workers to parallelise by", 1, int],
        # ["threads_per_worker", None, "number of threads per worker", 1, int],
    ]

    parser = argparse.ArgumentParser()

    for i in range(len(argument_names)):
        longname, shortname, helpmsg, nargs, vtype = argument_names[i]
        action = "store"
        longname = "--" + longname

        if shortname is None:
            parser.add_argument(
                longname, help=helpmsg, nargs=nargs, type=vtype, action=action
            )
        else:
            shortname = "-" + shortname
            parser.add_argument(
                shortname,
                longname,
                help=helpmsg,
                nargs=nargs,
                type=vtype,
                action=action,
            )

    args = parser.parse_args(arguments[1:])

    input_parameters = {}
    # assign parameters to correct dictionaries. Only allowing fault separation
    # as a loop parameter at this point.
    for at in args._get_kwargs():
        if at[1] is not None:
            value = at[1]
            if len(value) == 1:
                input_parameters[at[0]] = value[0]
            else:
                input_parameters[at[0]] = np.array(value)

    return input_parameters
