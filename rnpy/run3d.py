# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:07:46 2016

@author: a1655681
"""

import numpy as np
import sys
import rnpy.core.resistornetwork as rn
import rnpy.functions.faultaperture as rnfa
import rnpy.functions.assignfaults as rnaf
import os
import os.path as op
import time
import itertools

argument_names = [['ncells','n','number of cells x,y and z direction',3,int],
                  ['cellsize','c','cellsize in x,y and z direction',3,float],
                  ['pconnectionx','px','probability of connection in x direction','*',float],
                  ['pconnectiony','py','probability of connection in y direction','*',float],
                  ['pconnectionz','pz','probability of connection in z direction','*',float],
                  ['resistivity_matrix','rm','','*',float],
                  ['resistivity_fluid','rf','','*',float],
                  ['permeability_matrix','km','','*',float],
                  ['fluid_viscosity','mu','','*',float],
                  ['fault_assignment',None,'how to assign faults, random or list, '\
                                           'if list need to provide fault edges',1,str],
                  ['offset',None,'number of cells offset between fault surfaces','*',float],
                  ['length_max',None,'maximum fault length, if specifying random faults','*',float],
                  ['length_decay',None,'decay in fault length, if specifying random '\
                                       'fault locations, for each fault: '\
                                       'faultlength = length_max*exp(-length_decay*R)'\
                                       'where R is a random number in [0,1]','*',float],
                  ['mismatch_wavelength_cutoff',None,
                  'wavelength cutoff for matching between faults','*',float],
                  ['elevation_scalefactor',None,
                  'scale factor for standard deviation in elevation of fault surfaces','*',float],
                  ['fractal_dimension',None,
                  'fractal dimension of fault surfaces, recommended values in range (2.0,2.5)',
                  '*',float],
                  ['fault_separation',None,'amount to separate faults by, in metres','*',float],
                  ['fault_edges','fe','indices of fault edges in x,y,z directions '\
                                      'xmin xmax ymin ymax zmin zmax',6,int],
                  ['aperture_type',None,'type of aperture, random or constant',1,str],
                  ['workdir','wd','working directory',1,str],
                  ['outfile','o','output file name',1,str],
                  ['solve_properties','sp','which property to solve, current, fluid or currentfluid (default)',1,str],
                  ['solve_direction','sd','which direction to solve, x, y, z or a combination, e.g. xyz (default), xy, xz, y, etc',1,str],
                  ['repeats','r','how many times to repeat each permutation',1,int]]


def read_arguments(arguments, argument_names):
    """
    takes list of command line arguments obtained by passing in sys.argv
    reads these and updates attributes accordingly
    """
    
    import argparse
    
            
    parser = argparse.ArgumentParser()

    for longname,shortname,helpmsg,nargs,vtype in argument_names:
        if longname == 'fault_edges':
            action = 'append'
        else:
            action = 'store'
        longname = '--'+longname
        
        if shortname is None:
            parser.add_argument(longname,help=helpmsg,
                                nargs=nargs,type=vtype,action=action)
        else:
            shortname = '-'+shortname
            parser.add_argument(shortname,longname,help=helpmsg,
                                nargs=nargs,type=vtype,action=action)
    

    args = parser.parse_args(arguments[1:])

    loop_parameters = {}
    fixed_parameters = {}   
    faultsurface_keys = ['fractal_dimension',
                         'elevation_scalefactor',
                         'mismatch_wavelength_cutoff']
    faultsurface_parameters = {}
    
    for at in args._get_kwargs():
        if at[1] is not None:
            if at[0] == 'fault_edges':
                nf = len(at[1])
                value = np.reshape(at[1],(nf,3,2))
            else:
                value = at[1]
            if at[0] in ['permeability_matrix','resistivity_matrix','resistivity_fluid']:
                fixed_parameters[at[0]] = at[1]
            elif at[0] in faultsurface_keys:
                if type(value) != list:
                    value = [value]
                if len(value) > 0:
                    faultsurface_parameters[at[0]] = value
            elif at[0] == 'repeats':
                faultsurface_parameters['repeat'] = range(at[1][0])
            elif type(value) == list:
                if len(value) > 0:
                    if len(value) == 1:
                        fixed_parameters[at[0]] = value[0]
                    elif at[0] in ['ncells','cellsize']:
                        fixed_parameters[at[0]] = value
                    else:
                        loop_parameters[at[0]] = value
            else:
                fixed_parameters[at[0]] = value
    

    return fixed_parameters, loop_parameters, faultsurface_parameters