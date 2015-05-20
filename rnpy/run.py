# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:52:26 2015

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


arguments = sys.argv[1:]

def read_arguments(arguments):
    """
    takes list of command line arguments obtained by passing in sys.argv
    reads these and updates attributes accordingly
    """
    
    import argparse
    
            
    parser = argparse.ArgumentParser()
    
    # arguments to put into parser:
    # [longname,shortname,help,nargs,type]
    argument_names = [['ncells','n','number of cells x,y and z direction',3,int],
                      ['cellsize','c','cellsize in x,y and z direction',3,float],
                      ['pconnectionx','px','probability of connection in x direction','*',float],
                      ['pconnectiony','py','probability of connection in y direction','*',float],
                      ['pconnectionz','pz','probability of connection in z direction','*',float],
                      ['resistivity_matrix','rm','','*',float],
                      ['resistivity_fluid','rf','','*',float],
                      ['permeability_matrix','km','','*',float],
                      ['fluid_viscosity','mu','','*',float]
                      ['fault_assignment',None,'how to assign faults, random or list, '\
                                               'if list need to provide fault edges',1,str],
                      ['offset',None,'number of cells offset between fault surfaces','*',float],
                      ['length_max',None,'maximum fault length, if specifying random faults','*',float],
                      ['length_decay',None,'decay in fault length, if specifying random '\
                                           'fault locations, for each fault: '\
                                           'faultlength = length_max*exp(-length_decay*R)'\
                                           'where R is a random number in [0,1]','*',float],
                      ['mismatch_frequency_cutoff',None,
                      'frequency cutoff for matching between faults','*',float],
                      ['elevation_standard_deviation',None,
                      'standard deviation in elevation of fault surfaces','*',float],
                      ['fractal_dimension',None,
                      'fractal dimension of fault surfaces, recommended values in range (2.0,2.5)',
                      '*',float],
                      ['fault_separation',None,'amount to separate faults by, in metres','*',float],
                      ['fault_edges','fe','indices of fault edges in x,y,z directions '\
                                          'xmin xmax ymin ymax zmin zmax',6,int],
                      ['aperture_assignment',None,'type of aperture assignment, random or constant',1,str],
                      ['workdir','wd','working directory',1,str],
                      ['outfile','o','output file name',1,str],
                      ['solve_properties','sp','which property to solve, current, fluid or currentfluid (default)',1,str],
                      ['solve_direction','sd','which direction to solve, x, y, z or a combination, e.g. xyz (default), xy, xz, y, etc',1,str],
                      ['repeats','r','how many times to repeat each permutation',1,int]]
                      
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
    

    args = parser.parse_args(arguments)

    loop_parameters = {}
    fixed_parameters = {}   
    faultsurface_keys = ['fractal_dimension',
                         'elevation_standard_deviation',
                         'mismatch_wavelength_cutoff']
    faultsurface_parameters = {}
    
    for at in args._get_kwargs():
        if at[1] is not None:
            if at[0] == 'fault_edges':
                nf = len(at[1])
                value = np.reshape(at[1],(nf,3,2))
            else:
                value = [at[1]]
            
            if at[0] in faultsurface_keys:
                if type(value) != list:
                    value = [value]
                faultsurface_parameters[at[0]] = value
            elif at[0] == 'repeats':
                faultsurface_parameters[at[0]] = range(at[1])
            elif type(at[1]) != list:
                fixed_parameters[at[0]] = [value]
            elif len(value) == 1:
                fixed_parameters[at[0]] = value
            else:
                loop_parameters[at[0]] = value

    
    
    return fixed_parameters, loop_parameters, faultsurface_parameters


def initialise_inputs(fixed_parameters, loop_parameters, faultsurface_parameters):
    """
    make a list of run parameters
    """
    import itertools

    list_of_inputs = []

    # create list of all the different variables, need to ensure that fault surface
    # inputs are on the outermost loops
    loop_inputs = [val for val in itertools.product(*loop_parameters.values())]
    faultsurface_inputs = [val for val in itertools.product(*faultsurface_parameters.values())]
    if len(faultsurface_inputs) > 0:
        variablelist = [val for val in itertools.product(faultsurface_inputs,loop_inputs)]
    else:
        variablelist = loop_inputs
    
    # create a list of keys for all loop inputs including faultsurface, faultsurface
    # keywords first
    keys = faultsurface_parameters.keys()
    keys += loop_parameters.keys()

    # number of different fault surface variations, including repeats
    nfv = len(faultsurface_inputs)
    
    # get ncells, if it's not in commandline arguments need to get the default
    if 'ncells' in fixed_parameters.keys():
        ncells = fixed_parameters['ncells']
    else:
        ro = rn.Rock_volume()
        ncells = ro.ncells

    # get cellsize
    if 'cellsize' in fixed_parameters.keys():
        cellsize = fixed_parameters['cellsize']
    else:
        ro = rn.Rock_volume()
        cellsize = ro.cellsize    
    
    for iv,variable in enumerate(variablelist):
        # initialise a dictionary
        input_dict = fixed_parameters.copy()
        # add loop parameters including fault surface variables
        for k, key in enumerate(keys):
            input_dict[key] = variable[k]
        # check if we need to create a new fault surface pair
        if iv % (len(variablelist)/nfv) == 0:
            size = rnaf.get_faultsize(np.array(ncells))
            D = faultsurface_inputs['fractal_dimension']
            std = faultsurface_inputs['elevation_standard_deviation']
            lc = faultsurface_inputs['mismatch_wavelength_cutoff']
            heights = np.array(rnfa.build_fault_pair(size,lc=lc,D=D,
                                                     std=std, cs=cellsize))
        # in every case until we create a new pair, the fault surface pair is the same
        input_dict['fault_surfaces'] = heights
        
        list_of_inputs.append(input_dict)
        
    return list_of_inputs



def divide_inputs(work_to_do,size):
    """
    divide list of inputs into chunks to send to each processor
    
    """
    chunks = [[] for _ in range(size)]
    for i,d in enumerate(work_to_do):
        chunks[i%size].append(d)

    return chunks
    
    
def run(list_of_inputs,rank,wd,save_array=True):
    """
    generate and run a random resistor network
    takes a dictionary of inputs to be used to create a resistivity object
    """
    
    r_objects = []

    r = 0
    for input_dict in list_of_inputs:
        # initialise random resistor network
        ro = rn.Rock_volume(**input_dict)
        # solve the network
        ro.solve_resistor_network()
        # append result to list of r objects
        r_objects.append(ro)
        print "run {} completed".format(r)
        if save_array:
            for prop in ['resistivity','permeability',
                         'current','flowrate','aperture_array']:
                np.save(os.path.join(wd,'{}{}_{}'.format(prop,rank,r)),
                        getattr(ro,prop)
                        )
        r += 1
    return r_objects
    
    
def setup_and_run_suite(arguments):
    """
    set up and run a suite of runs in parallel using mpi4py
    """
    
    from mpi4py import MPI
    
    # sort out rank and size info
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    print 'Hello! My name is {}. I am process {} of {}'.format(name,rank,size)
   
    # get inputs from the command line
    fixed_parameters, loop_parameters, faultsurface_parameters = read_arguments(arguments)
    if 'workdir' in fixed_parameters.keys():
        wd = fixed_parameters['workdir']
    else:
        wd = './model_runs'
    
    if rank == 0:
        list_of_inputs = initialise_inputs(fixed_parameters, 
                                           loop_parameters, 
                                           faultsurface_parameters)
        inputs = divide_inputs(list_of_inputs,size)
    else:
        list_of_inputs = None
        inputs = None

    if rank == 0:
        if not os.path.exists(wd):
            os.mkdir(wd)
        wd = os.path.abspath(wd)

    wd2 = os.path.join(wd,'arrays')

    if rank == 0:
        if not os.path.exists(wd2):
            os.mkdir(wd2)
    else:
        # wait for wd2 to be created
        while not os.path.exists(wd2):
            time.sleep(1)
            print '.',

    inputs_sent = comm.scatter(inputs,root=0)
    r_objects = run(inputs_sent,rank,wd2)
    outputs_gathered = comm.gather(r_objects,root=0)
     
    if rank == 0:
        print "gathering outputs...",
        # flatten list, outputs currently a list of lists for each rank, we
        # don't need the outputs sorted by rank
        ogflat = []
        count = 1
        for group in outputs_gathered:
            for ro in group:
                ogflat.append(ro)
                count += 1
        print "gathered outputs into list, now sorting"
        # get list of fixed parameters
        ro0 = ogflat[0]
        fixed_paramkeys = []
        for fpm in fixed_parameters.keys():
            if fpm not in ['ncells','cellsize']:
                # check it is a r object parameter
                if hasattr(ro0,fpm):
                    fixed_paramkeys.append(fpm)
        
        # get list of variable parameters
        variable_paramkeys = []
        for vpm in faultsurface_parameters.keys() + loop_parameters.keys():
            if hasattr(ogflat[0],fpm):
                # check it is a r object parameter
                variable_paramkeys.append(fpm)            
        
        header = '# suite of resistor network simulations\n'
        for pm in ['ncells','cellsize']:
            header += pm + '# {} {} {}\n'.format(*(getattr(ro,pm)))
        header += '# fixed parameters\n'
        header += ' '.join(fpm)+'\n'
        header += ' '.join([str(getattr(ro0,pm)) for pm in fpm])+'\n'
        header += '# variable parameters\n'
        header += ' '.join(vpm)
        
        output_array = np.array([[getattr(ro,pm) for pm in variable_paramkeys] \
                                  for ro in ogflat]).T
                                      
        np.savetxt(op.join(wd,fixed_parameters['outfile']),
                   output_array,
                   header = header, fmt='%.3e')
                  
                   
if __name__ == "__main__":
    setup_and_run_suite(sys.argv[1:])

