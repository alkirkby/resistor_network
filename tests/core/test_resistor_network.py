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
answers_random = np.array([[7.50048317e-18, 7.51587040e-18, 1.46387203e+02, 1.44007499e+02,
        6.90992817e-02, 1.06361192e-05],
       [1.02380298e-13, 2.37852562e-13, 5.52321147e+00, 5.66698014e+00,
        1.36055920e-01, 2.76592197e-05],
       [1.41729347e-10, 1.45402777e-10, 4.00600242e-01, 3.92771709e-01,
        3.23166052e-01, 9.79159375e-05],
       [6.76350417e-08, 6.76683082e-08, 1.21228648e-01, 1.21234753e-01,
        8.26395872e-01, 9.94063828e-04]])

answers_correct_for_geometry = np.array([[2.35662990e-18, 2.15568980e-18, 1.80088256e+02, 1.73688314e+02,
        7.03263676e-02, 1.06361192e-05],
       [1.31408966e-17, 1.20461147e-17, 1.31121216e+01, 1.32449537e+01,
        1.31518895e-01, 2.76592197e-05],
       [9.98931008e-11, 1.03066585e-10, 4.78246317e-01, 4.63979230e-01,
        3.15764907e-01, 9.79159375e-05],
       [6.13476960e-08, 6.17287924e-08, 1.26077644e-01, 1.25742522e-01,
        8.28335215e-01, 9.94063828e-04]])



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

