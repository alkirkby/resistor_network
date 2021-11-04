# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 17:20:37 2021

@author: alisonk
"""
import numpy as np
import matplotlib.pyplot as plt
from rnpy.functions.readoutputs import bulk_permeability,bulk_cfraction
fn = r'C:\Rock_property_modelling\model_runs\aperture_correction\0x400x400_tst4\outputs.dat'

data = np.loadtxt(fn)
csmax = np.amax(data[:,8])


def bulk_permeability(permeability,x_cellsize,cellsize_max,permeability_matrix=1e-18):
        return (permeability*x_cellsize + \
             permeability_matrix*(cellsize_max - x_cellsize))/cellsize_max

k = bulk_permeability(data[:,7],data[:,8],csmax)
cf = bulk_cfraction(data[:,1],data[:,8],csmax)
ap = cf*csmax
kpar =cf*ap**2./12. + (1.-cf)*1e-18
plt.figure()
plt.loglog(cf,k,'.-')
plt.loglog(cf,kpar)
# plt.loglog(data[:,1],kpar)