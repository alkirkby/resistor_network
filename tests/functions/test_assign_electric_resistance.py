# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:48:17 2022

@author: alisonk
"""

from rnpy.core.resistornetwork import Rock_volume
from unittest import TestCase
from tests import TEST_DATA_ROOT
import os
import numpy as np


class testAssignElectricResistance(TestCase):
    
    def test_no_matrix_current(self):
        
        rv = Rock_volume(ncells=(0,4,4), matrix_current=False)
        rv.solve_resistor_network2()
        
        retest = rv.cellsize[2]*rv.resistivity_fluid/\
            (rv.cellsize[1]*rv.aperture_electric[1:-1,1:,1,2,0])
        
        retest2 = rv.cellsize[1]*rv.resistivity_fluid/\
            (rv.cellsize[2]*rv.aperture_electric[1:,1:-1,1,1,0])
        
        
        assert np.all(np.abs(retest-rv.resistance[1:-1,1:,1,2])/retest < 1e-8)
        assert np.all(np.abs(retest2-rv.resistance[1:,1:-1,1,1])/retest2 < 1e-8)
        
        
    def test_matrix_current(self):
        
        rv = Rock_volume(ncells=(0,4,4),matrix_current=True)
        
        rv.solve_resistor_network2()
        
        # electric resistance of fracture
        ref = rv.cellsize[2]*rv.resistivity_fluid/\
            (rv.cellsize[1]*rv.aperture_electric[1:-1,1:,1,2,0])
            
        # hydraulic resistance of fluid
        rem = rv.cellsize[2]*rv.resistivity_matrix/\
            (rv.cellsize[1]*(rv.cellsize[0] - rv.aperture_electric[1:-1,1:,1,2,0]))
                
        retest = 1./(1./ref + 1./rem)
        
        
        # electric resistance of fracture
        ref2 = rv.cellsize[1]*rv.resistivity_fluid/\
            (rv.cellsize[2]*rv.aperture_electric[1:,1:-1,1,1,0])
            
        # hydraulic resistance of fluid
        rem2 = rv.cellsize[1]*rv.resistivity_matrix/\
            (rv.cellsize[2]*(rv.cellsize[0] - rv.aperture_electric[1:,1:-1,1,1,0]))
                
        retest2 = 1./(1./ref2 + 1./rem2)
        
        
        assert np.all(np.abs(retest-rv.resistance[1:-1,1:,1,2])/retest < 1e-8)
        assert np.all(np.abs(retest2-rv.resistance[1:,1:-1,1,1])/retest2 < 1e-8)