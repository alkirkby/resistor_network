# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:15:03 2021

@author: alisonk
"""

import numpy as np


def roundsf(number, sf):
    """
    round a number to a specified number of significant figures (sf)
    """
    # can't have < 1 s.f.
    sf = max(sf, 1.0)

    if np.iterable(number):
        print("iterable")
        rounding = (np.ceil(-np.log10(np.abs(number)) + sf - 1.0)).astype(int)
        return np.array(
            [np.round(number[ii], rounding[ii]) for ii in range(len(rounding))]
        )
    else:
        if number == 0:
            rounding = 1
        else:
            rounding = int(np.ceil(-np.log10(np.abs(number)) + sf - 1.0))

        return np.round(number, rounding)


def get_logspace_array(val_min, val_max, vals_per_decade, include_outside_range=True):
    """
    get a list of values, evenly spaced in log space and making sure it is
    including values on multiples of 10

    :returns:
        numpy array containing list of values

    :inputs:
        min_val = minimum value
        max_val = maximum value
        vals_per_decade = number of values per decade
        include_outside_range = option whether to start and finish the value
                                list just inside or just outside the bounds
                                specified by val_min and val_max
                                default True

    """

    log_val_min = np.log10(val_min)
    log_val_max = np.log10(val_max)

    # check if log_val_min is a whole number
    if log_val_min % 1 > 0:
        # list of vals, around the minimum val, that will be present in specified
        # vals per decade
        aligned_logvals_min = np.linspace(
            np.floor(log_val_min), np.ceil(log_val_min), vals_per_decade + 1
        )
        lpmin_diff = log_val_min - aligned_logvals_min
        # index of starting val, smallest value > 0
        if include_outside_range:
            spimin = np.where(lpmin_diff > 0)[0][-1]
        else:
            spimin = np.where(lpmin_diff < 0)[0][0]
        start_val = aligned_logvals_min[spimin]
    else:
        start_val = log_val_min

    if log_val_max % 1 > 0:
        # list of vals, around the maximum val, that will be present in specified
        # vals per decade
        aligned_logvals_max = np.linspace(
            np.floor(log_val_max), np.ceil(log_val_max), vals_per_decade + 1
        )
        lpmax_diff = log_val_max - aligned_logvals_max
        # index of starting val, smallest value > 0
        if include_outside_range:
            spimax = np.where(lpmax_diff < 0)[0][0]
        else:
            spimax = np.where(lpmax_diff > 0)[0][-1]
        stop_val = aligned_logvals_max[spimax]
    else:
        stop_val = log_val_max

    return np.logspace(
        start_val, stop_val, int(round((stop_val - start_val) * vals_per_decade + 1))
    )


def get_bin_ranges_from_centers(centers):
    """
    get bin ranges from centers, assuming logarithmic variation

    """
    # get evenly spaced bins in log space
    internal_edges = 10 ** np.mean(
        [np.log10(centers[1:]), np.log10(centers[:-1])], axis=0
    )
    # round to appropriate number of decimal places
    internal_edges = roundsf(internal_edges, 2)
    # add outer edges
    bins = np.insert(internal_edges, 0, roundsf(2 * centers[0] - internal_edges[0], 1))
    bins = np.append(bins, 2 * centers[-1] - internal_edges[-1])

    return bins


def get_unique_properties(sort_array, property_array_list=None, compute_means=True):
    """_summary_

    Args:
        sort_array (_type_): _description_
        property_array_list (_type_, optional): _description_. Defaults to None.
        compute_means (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
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
    lvals_center,
    array_list,
    use_indices=False,
):
    """_summary_

    Args:
        min_val (_type_): _description_
        max_val (_type_): _description_
        lvals_center_unique (_type_): _description_
        array_list (_type_): _description_
        use_indices (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    filtered_array_list = []
    for arr in array_list:
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = np.amax(lvals_center) + 1

        if use_indices:
            idx0 = np.where(lvals_center > min_val)[0][0]
            idx1 = np.where(lvals_center <= max_val)[-1][-1] - len(lvals_center) + 1
            filtered_array_list.append(arr[idx0:idx1])
        else:
            filt = np.all([lvals_center > min_val, lvals_center <= max_val], axis=0)
            filtered_array_list.append(arr[filt])

    return filtered_array_list
