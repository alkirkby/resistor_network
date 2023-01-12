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
    sf = max(sf,1.)
    
    if np.iterable(number):
        print("iterable")
        rounding = (np.ceil(-np.log10(np.abs(number)) + sf - 1.)).astype(int)
        return np.array([np.round(number[ii],rounding[ii]) for ii in range(len(rounding))])
    else:
        if number == 0:
            rounding = 1
        else:
            rounding = int(np.ceil(-np.log10(np.abs(number)) + sf - 1.))
    
        return np.round(number, rounding)
    
    
def get_logspace_array(val_min,val_max,vals_per_decade,include_outside_range=True):
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
        aligned_logvals_min = np.linspace(np.floor(log_val_min),np.ceil(log_val_min),vals_per_decade + 1)
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
        aligned_logvals_max = np.linspace(np.floor(log_val_max),np.ceil(log_val_max),vals_per_decade + 1)
        lpmax_diff = log_val_max - aligned_logvals_max
        # index of starting val, smallest value > 0
        if include_outside_range:
            spimax = np.where(lpmax_diff < 0)[0][0]
        else:
            spimax = np.where(lpmax_diff > 0)[0][-1]
        stop_val = aligned_logvals_max[spimax]
    else:
        stop_val = log_val_max        
        
    return np.logspace(start_val,stop_val,int(round((stop_val-start_val)*vals_per_decade + 1)))
