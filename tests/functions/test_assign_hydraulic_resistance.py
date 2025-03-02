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


class testAssignHydraulicResistance(TestCase):
    def test_no_matrix_flow(self):
        rv = Rock_volume(ncells=(0,4,4),matrix_flow=False)
        rv.solve_resistor_network2()
        
        rhtest = rv.fluid_viscosity*rv.cellsize[2]/\
            (rv.cellsize[1]*(rv.aperture_hydraulic[1:-1,1:,1,2,0]**3/12))
        
        rhtest2 = rv.fluid_viscosity*rv.cellsize[1]/\
            (rv.cellsize[2]*(rv.aperture_hydraulic[1:,1:-1,1,1,0]**3/12))
        
        
        assert np.all(np.abs(rhtest-rv.hydraulic_resistance[1:-1,1:,1,2])/rhtest < 1e-8)
        assert np.all(np.abs(rhtest2-rv.hydraulic_resistance[1:,1:-1,1,1])/rhtest2 < 1e-8)
    def test_matrix_flow(self):
        rv = Rock_volume(ncells=(0,4,4),matrix_flow=True)
        rv.solve_resistor_network2()
        
        # hydraulic resistance of fracture
        rhf = rv.fluid_viscosity*rv.cellsize[2]/\
            (rv.cellsize[1]*(rv.aperture_hydraulic[1:-1,1:,1,2,0]**3/12))
            
        # hydraulic resistance of fluid
        rhm = rv.fluid_viscosity*rv.cellsize[2]/\
            (rv.cellsize[1]*(rv.cellsize[0]-rv.aperture_hydraulic[1:-1,1:,1,2,0])*\
             rv.permeability_matrix)
                
        rhtest = 1./(1./rhf + 1./rhm)
        
        
        # hydraulic resistance of fracture
        rhf2 = rv.fluid_viscosity*rv.cellsize[1]/\
            (rv.cellsize[2]*(rv.aperture_hydraulic[1:,1:-1,1,1,0]**3/12))
            
        # hydraulic resistance of fluid
        rhm2 = rv.fluid_viscosity*rv.cellsize[1]/\
            (rv.cellsize[2]*(rv.cellsize[0]-rv.aperture_hydraulic[1:,1:-1,1,1,0])*\
             rv.permeability_matrix)
                
        rhtest2 = 1./(1./rhf2 + 1./rhm2)
        # rhtest2 = rv.fluid_viscosity*rv.cellsize[1]/\
        #     (rv.cellsize[2]*(rv.aperture_hydraulic[1:,1:-1,1,1,0]**3/12))
        
        
        assert np.all(np.abs(rhtest-rv.hydraulic_resistance[1:-1,1:,1,2])/rhtest < 1e-8)
        assert np.all(np.abs(rhtest2-rv.hydraulic_resistance[1:,1:-1,1,1])/rhtest2 < 1e-8)