# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:30:13 2021

@author: alisonk
"""

import os

TEST_ROOT = os.path.normpath(
    os.path.abspath(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)

TEST_DATA_ROOT = os.path.join(TEST_ROOT,'data')