# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681

functions dealing with assigning properties to arrays, including faults, 
fault apertures, permeability, resistivity, etc

- adding nulls to correct edges of a fault, resistivity, permeability or aperture array 
- adding fault to an array

"""
from __future__ import division, print_function
import numpy as np




def embed_network(prop,embedfiles,pembed_m,pembed_f,vmatrix,vfluid):
    """
    embed a network with resistivity values from smaller networks
    
    prop = array containing property values, 2 unique values vmatrix and vfluid
    embedfiles = list containing full path to text file containing x, y and z 
                 property values for the embedment
    pembed_m = numpy array or list containing x,y,z probability of embedment 
               for matrix (closed) cells in prop
    pembed_f = numpy array or list containing x,y,z probability of embedment 
               for open cells in prop
    vmatrix,vfluid = property values for matrix and fluid in prop
    
    """
    
    # get values to embed
    embedvals = [np.loadtxt(f) for f in embedfiles]
    
    # get number of fluid and matrix cells in prop array
    nf = len(prop[prop==vmatrix])
    nm = len(prop[prop==vfluid])
    
    # get number of of fluid and matrix cells to embed
    pf = float(nf)*pembed_f
    pm = float(nm)*pembed_m
    
    # embed fluid cells with probability pembed_f
    count = 0
    n = 0
    ii = 0
    
    for p,val in [[pf,vfluid],[pm,vmatrix]]:
        while count < p:
            if n >= len(embedvals[ii]):
                n = 0
            k,j,i = [np.random.randint(1,int(nn)-1) for nn in np.shape(prop)]
            if np.all(prop[np.isfinite(prop[k,j,i])] == val):
                prop[k,j,i] = embedvals[ii][n]
                n += 1
                count += 1
        ii += 1