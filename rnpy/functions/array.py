# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681

functions dealing with assigning properties to arrays, including faults, 
fault apertures, permeability, resistivity, etc

- adding nulls to correct edges of a fault, resistivity, permeability or aperture array 
- adding fault to an array

"""
from __future__ import division, print_function
import numpy as np
import rnpy.functions.faultaperture as rnfa



def add_nulls(in_array):
    """
    initialise a fault, resistivity, permeability, aperture etc array by
    putting nulls at edges in correct spots
    
    """

    in_array[:,:,-1,0] = np.nan
    in_array[:,-1,:,1] = np.nan
    in_array[-1,:,:,2] = np.nan
    in_array[:,:,0] = np.nan
    in_array[:,0,:] = np.nan
    in_array[0,:,:] = np.nan
    
    return in_array
