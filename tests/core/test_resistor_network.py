# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:05:59 2021

@author: alisonk
"""

from unittest import TestCase
from rnpy.core.resistornetwork import Rock_volume
import numpy as np

fslist = np.array([-5e-5,0,1e-4,1e-3])
answers_random = np.array([[2.24332230e-18, 2.20290378e-18, 4.49777941e+02, 4.57030723e+02,
                            4.33686852e-02],
                           [1.27452537e-17, 1.15037930e-17, 8.76109797e+01, 1.05124100e+02,
                            9.36148739e-02],
                           [8.93747276e-11, 9.12669546e-11, 6.90802212e-01, 6.65184938e-01,
                            2.47462321e-01],
                           [6.25600568e-08, 6.25765542e-08, 1.30755015e-01, 1.30770742e-01,
                            7.67130875e-01]])


class testResistorNetwork2d(TestCase):
    def SetUp(self):
        self.inputs = dict(ncells = [0,50,50],
                      cellsize = [1e-6,0.00025,0.00025],
                      aperture_type = 'random',
                      fault_dict = {'random_numbers_dir':r'C:\tmp',
                                    'correct_aperture_for_geometry':False})
        self.fault_separations = [-5e-4,0,1e-4,1e-3]
        
    def test_random(self):
        
        
        for i, fs in enumerate(fslist):
            rv = Rock_volume(ncells=[0,50,50],cellsize=[1e-6,0.00025,0.00025],
                             aperture_type='random',
                            
                             fault_dict={'fault_separation':fs,
                                         'random_numbers_dir':r'C:\tmp',
                                         'correct_aperture_for_geometry':False})
            rv.solve_resistor_network2()
            rv.compute_conductive_fraction()
            cf = rv.conductive_fraction
            
            assert np.all(np.abs(np.log10(rv.permeability_bulk[1:]) - \
                                 np.log10(answers_random[i,:2])) < 1e-6)
            assert np.all(np.abs(np.log10(rv.resistivity_bulk[1:]) - \
                                 np.log10(answers_random[i,2:4])) < 1e-6)
            assert np.abs(rv.conductive_fraction - \
                          answers_random[i,4]) < 1e-6
                
    def test_constant_aperture(self):
        for i, fs in enumerate(fslist[1:]):
            rv = Rock_volume(ncells=[0,50,50],cellsize=[1.5e-3,0.00025,0.00025],
                             aperture_type='constant',
                            
                             fault_dict={'fault_separation':fs,
                                         'random_numbers_dir':r'C:\tmp',
                                         'correct_aperture_for_geometry':False})
            rv.solve_resistor_network2()
            rv.compute_conductive_fraction()
            cf = rv.conductive_fraction
            
            kpar = (fs**3/12 + (rv.cellsize[0]-fs)*1e-18)/rv.cellsize[0]
            rpar = rv.cellsize[0]/(fs/rv.resistivity_fluid + \
                                   (rv.cellsize[0]-fs)/rv.resistivity_matrix)
            
            print(kpar,rv.permeability_bulk[1:])
            assert np.all(np.abs(np.log10(rv.permeability_bulk[1:]) - \
                                 np.log10(kpar)) < 1e-6)
            assert np.all(np.abs(np.log10(rv.resistivity_bulk[1:]) - \
                                 np.log10(rpar)) < 1e-6)
            assert np.abs(rv.conductive_fraction - fs/rv.cellsize[0]) < 1e-6        
        
# print(kpar/rv.permeability_bulk)

