# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:37:59 2021

@author: alisonk
"""

a = np.arange(6).reshape(2,3)

b= np.ones_like(a)


with np.nditer(a) as it:
   for x in it:
       x[...] = 2 * x