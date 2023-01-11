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
        0.06601688, 9.69876112e-06],
       [5.35171176e-13, 5.92874402e-13, 3.57549634e+00, 3.00208897e+00,
        0.14023532, 2.76313696e-05],
       [1.86834298e-10, 1.94015496e-10, 3.51813459e-01, 3.55504947e-01,
        0.3426331, 1.01918208e-04],
       [6.91658272e-08, 6.91874067e-08, 1.19971879e-01, 1.19970509e-01,
        0.83486371, 9.99670993e-04]])

answers_correct_for_geometry = np.array([[2.10161483e-18, 2.19919172e-18, 1.98574878e+02, 2.13298141e+02,
        0.06785351, 9.69876112e-06],
       [1.32950664e-14, 1.99272152e-17, 1.15965765e+01, 2.16141248e+01,
        0.14360331, 2.76313696e-05],
       [1.39927856e-10, 1.46553593e-10, 3.93255590e-01, 4.04320986e-01,
        0.34995828, 1.01918208e-04],
       [6.28227106e-08, 6.29537145e-08, 1.25039311e-01, 1.24930121e-01,
        0.83299222, 9.99670993e-04]])


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

