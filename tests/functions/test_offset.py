# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:25:00 2021

@author: alisonk
"""

import unittest
import numpy as np
from rnpy.functions.assignfaults_new import assign_fault_aperture
from rnpy.core.resistornetwork import Rock_volume


fault_uvw = np.array([[[[ 1,  1,  1],[ 1, 11,  1]],
                                    [[ 1,  1, 11],[ 1, 11, 11]]]])

ncells = [0,10,10]

aperture_list,aperture_list_f,aperture_list_c,\
aperture_array,aperture_f,aperture_c,faultheights = \
    assign_fault_aperture(fault_uvw,ncells)

np.save(r'C:\git\resistor_network\data\fault_aperture\aperture_list_fs0.1mm_0x10x10')

# class TestAssignFaultAperture(unittest.TestCase()):
#     def __init__(self):
#         # initialise params
#         self.fault_uvw = np.array([[[[ 1,  1,  1],[ 1, 11,  1]],
#                                     [[ 1,  1, 11],[ 1, 11, 11]]]])
#         self.ncells = 10
#         self.cs = 0.25e-3
#         self.fault_surfaces = None
#         self.offset=0
#         self.fractal_dimension = 2.5 
#         self.mismatch_wavelength_cutoff = None 
#         self.elevation_scalefactor = None
#         self.aperture_type = 'random'
#         self.aperture_list=None
#         self.preserve_negative_apertures = False
        
#         self.input_dict = {}
#         for param in dir(self):
#             if not (param.startswith('_') or param == 'input_dict'):
#                 self.input_dict[param] = getattr(self,param)
        
#     def testBase(self):
        
    
    
    
