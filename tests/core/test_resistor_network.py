# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:05:59 2021

@author: alisonk
"""

from unittest import TestCase
from tests import TEST_DATA_ROOT
from rnpy.core.resistornetwork import Rock_volume
import numpy as np
import os


fslist = np.array([-5e-5,0,1e-4,1e-3])
answers_random = np.array([[1.73284218e-17, 1.06714967e-17, 8.17900190e+01, 1.05105761e+02,
        6.53146709e-02, 9.69876112e-06],
       [5.35171176e-13, 5.92874402e-13, 3.57549634e+00, 3.00208897e+00,
        1.39673338e-01, 2.76313696e-05],
       [1.86834298e-10, 1.94015496e-10, 3.51813459e-01, 3.55504947e-01,
        3.42590799e-01, 1.01918208e-04],
       [6.91658272e-08, 6.91874067e-08, 1.19971879e-01, 1.19970509e-01,
        8.34879287e-01, 9.99670993e-04]])

answers_correct_for_geometry = np.array([[2.10174310e-18, 2.19934103e-18, 1.19056679e+02, 1.30716506e+02,
        6.70436480e-02, 9.69876112e-06],
       [1.32956139e-14, 1.99286489e-17, 7.66790999e+00, 7.38556092e+00,
        1.42949772e-01, 2.76313696e-05],
       [1.39927905e-10, 1.46553639e-10, 3.90499450e-01, 3.99839019e-01,
        3.49842422e-01, 1.01918208e-04],
       [6.28227106e-08, 6.29537145e-08, 1.25039311e-01, 1.24930121e-01,
        8.33013975e-01, 9.99670993e-04]])



class testResistorNetwork2d(TestCase):
        
    def test_build_run_random_aperture(self):
        
        for i, fs in enumerate(fslist):
            rv = Rock_volume(aperture_type='random',
                             fault_separation=fs,
                             ncells = [0,50,50],
                             cellsize = [1e-6,0.00025,0.00025],
                             fault_dict = {'random_numbers_dir':os.path.join(TEST_DATA_ROOT,'random_seeds'),
                                    'correct_aperture_for_geometry':False})
                             # **self.inputs)
            rv.solve_resistor_network2()
            rv.compute_conductive_fraction()
            
            assert np.all(np.abs(np.log10(rv.permeability_bulk[1:]) - \
                                 np.log10(answers_random[i,:2])) < 1e-6)
            assert np.all(np.abs(np.log10(rv.resistivity_bulk[1:]) - \
                                 np.log10(answers_random[i,2:4])) < 1e-6)
            assert np.abs(rv.conductive_fraction - \
                          answers_random[i,4]) < 1e-6
            assert np.abs(rv.aperture_mean[0] - \
                          answers_random[i,5]) < 1e-6

    def test_build_run_random_aperture_correct_for_geometry(self):
        
        for i, fs in enumerate(fslist):
            rv = Rock_volume(aperture_type='random',
                             fault_separation=fs,
                             ncells = [0,50,50],
                             cellsize = [1e-6,0.00025,0.00025],
                             fault_dict = {'random_numbers_dir':os.path.join(TEST_DATA_ROOT,'random_seeds'),
                                    'correct_aperture_for_geometry':True})
                             # **self.inputs)
            rv.solve_resistor_network2()
            rv.compute_conductive_fraction()
            
            assert np.all(np.abs(np.log10(rv.permeability_bulk[1:]) - \
                                 np.log10(answers_correct_for_geometry[i,:2])) < 1e-6)
            assert np.all(np.abs(np.log10(rv.resistivity_bulk[1:]) - \
                                 np.log10(answers_correct_for_geometry[i,2:4])) < 1e-6)
            assert np.abs(rv.conductive_fraction - \
                          answers_correct_for_geometry[i,4]) < 1e-6
            assert np.abs(rv.aperture_mean[0] - \
                          answers_correct_for_geometry[i,5]) < 1e-6

                
    def test_build_run_constant_aperture(self):
        
        for i, fs in enumerate(fslist[1:]):
            rv = Rock_volume(aperture_type='constant',
                             fault_separation=fs,
                             ncells = [0,50,50],
                             cellsize = [1e-6,0.00025,0.00025],
                             fault_dict = {'random_numbers_dir':os.path.join(TEST_DATA_ROOT,'random_seeds'),
                                    'correct_aperture_for_geometry':False})
            rv.solve_resistor_network2()
            rv.compute_conductive_fraction()
            
            kpar = (fs**3/12 + (rv.cellsize[0]-fs)*1e-18)/rv.cellsize[0]
            rpar = rv.cellsize[0]/(fs/rv.resistivity_fluid + \
                                   (rv.cellsize[0]-fs)/rv.resistivity_matrix)
            
            assert np.all(np.abs(np.log10(rv.permeability_bulk[1:]) - \
                                 np.log10(kpar)) < 1e-6)
            assert np.all(np.abs(np.log10(rv.resistivity_bulk[1:]) - \
                                 np.log10(rpar)) < 1e-6)
            assert np.abs(rv.conductive_fraction - fs/rv.cellsize[0]) < 1e-6  
            assert np.abs(rv.aperture_mean[0] - max(fs,0)) < 1e-6            

        
# print(kpar/rv.permeability_bulk)

