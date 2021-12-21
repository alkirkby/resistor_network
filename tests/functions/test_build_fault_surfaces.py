# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:18:51 2021

@author: alisonk
"""

from unittest import TestCase
from tests import TEST_DATA_ROOT
import os
import numpy as np
from rnpy.functions.faultaperture import build_fault_pair

class testCorrectForGeometry(TestCase):
    def setUp(self):
        self.h1, self.h2 = build_fault_pair(70,70,D=2.4,
                                            cs=2.5e-4,
                                            scalefactor=0.003,
                                            random_numbers_dir=os.path.join(TEST_DATA_ROOT,'random_seeds'))
        
    def test_build_faultpair(self):
        h1test = np.load(os.path.join(TEST_DATA_ROOT,'fault_surfaces','h1.npy'))
        h2test = np.load(os.path.join(TEST_DATA_ROOT,'fault_surfaces','h2.npy'))
        
        assert np.all(np.abs(h1test-self.h1) < 1e-8)
        assert np.all(np.abs(h2test-self.h2) < 1e-8)
        