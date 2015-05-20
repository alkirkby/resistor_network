# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681
"""

import scipy.sparse.linalg as linalg
    
def solve_matrix(A,b):
    """
    solve the matrix equation Ab = C
    
    """
   
    return linalg.spsolve(A,b)
