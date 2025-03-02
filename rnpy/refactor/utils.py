# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:44:46 2025

@author: alisonk
"""

from scipy.interpolate import RegularGridInterpolator
import numpy as np
from scipy.ndimage import median_filter

def medfilt_subsampled(h1,ks,ssrate):
    """
    For high kernel sizes it's faster to subsample, take a median filter then
    interpolate to the original nodes, and gives almost the same result

    Parameters
    ----------
    h1 : TYPE
        DESCRIPTION.
    ks : TYPE
        DESCRIPTION.
    ssrate : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    xi, yi = np.arange(h1.shape[1]),np.arange(h1.shape[0])
    xlr,ylr = xi[::ssrate],yi[::ssrate]
    h1_inp = h1[::ssrate,::ssrate]
    
    if xi[-1] not in xlr:
        xlr = np.append(xlr,xi[-1])
        h1_inp = np.vstack([h1_inp.T,h1_inp[:,-2:-1].T]).T
    if yi[-1] not in ylr:
        ylr = np.append(ylr,yi[-1])
        h1_inp = np.vstack([h1_inp,h1_inp[-2:-1]])
        
        
    xi,yi = np.meshgrid(xi,yi)
    
    h1sm_lr = median_filter(h1_inp,
                            size=int(ks/ssrate),
                            mode='nearest')
    
    func = RegularGridInterpolator((xlr, ylr), h1sm_lr.T)
    
    return func((xi,yi))