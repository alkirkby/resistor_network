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
                      ['fluid_viscosity','mu','','*',float],
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
                value = at[1]
            if at[0] in faultsurface_keys:
                if type(value) != list:
                    value = [value]
                faultsurface_parameters[at[0]] = value
            elif at[0] == 'repeats':
                faultsurface_parameters[at[0]] = range(at[1][0])
            elif type(value) == list:
                if len(value) == 1:
                    print "len 1",at[0],value
                    fixed_parameters[at[0]] = value[0]
                elif at[0] in ['ncells','cellsize']:
                    fixed_parameters[at[0]] = value
                else:
                    loop_parameters[at[0]] = value
            else:
                print "not list",at[0],value
                fixed_parameters[at[0]] = value
    
    print "fixed_parameters read in from command line",fixed_parameters
    print "loop parameters read in from command line",loop_parameters
    print "fault surface parameters read in from command line",faultsurface_parameters


    return fixed_parameters, loop_parameters, faultsurface_parameters


def initialise_inputs(fixed_parameters, loop_parameters, faultsurface_parameters):
    """
    make a list of run parameters
    """
    import itertools

    list_of_inputs = []

    # create list of all the different variables, need to ensure that fault surface
    # inputs are on the outermost loops
    if len(loop_parameters) > 1:
        loop_inputs = [list(val) for val in itertools.product(*loop_parameters.values())]
    else:
        loop_inputs = [[val] for val in loop_parameters.values()[0]]
    if len(faultsurface_parameters) > 1:
        faultsurface_inputs = [list(val) for val in itertools.product(*faultsurface_parameters.values())]
    else:
        faultsurface_inputs = [[val] for val in faultsurface_parameters.values()[0]]  

    print "faultsurface_parameters",faultsurface_parameters
    print "loop_parameters",loop_parameters

    print "faultsurface_values",faultsurface_parameters.values()
    print "loop_parameters_values",loop_parameters.values()

    print "loop inputs",loop_inputs
    print "faultsurface_inputs",faultsurface_inputs
    if len(faultsurface_inputs) > 0:
        print "len(faultsurface_inputs > 0 so creating new variablelist"
        variablelist = [list(val) for val in itertools.product(faultsurface_inputs,loop_inputs)]
        print variablelist
        variablelist2 = []
        for vline in variablelist:
            vlist_temp = []
            for vv in vline:
                for v in vv:
                    print "v",v
                    vlist_temp.append(v)
                variablelist2.append(vlist_temp)
        variablelist = variablelist2
    else:
        print "variablelist = loop_inputs"
        variablelist = loop_inputs
    print "got variable list",variablelist    
    # create a list of keys for all loop inputs including faultsurface, faultsurface
    # keywords first
    keys = faultsurface_parameters.keys()
    keys += loop_parameters.keys()
    print "got keys", keys
    # number of different fault surface variations, including repeats
    nfv = min(len(faultsurface_inputs),1)
    
    # intialise a rock volume to get the defaults from
    print "initialising a rock volume to get defaults"
    ro = rn.Rock_volume(build=False)

    for fparam in ['ncells','cellsize']:
        if fparam not in fixed_parameters.keys():
            fixed_parameters[fparam] = getattr(ro,fparam)
    print fixed_parameters
    print "initialising variables"
    for iv,variable in enumerate(variablelist):
        offset = 0
        # initialise a dictionary
        input_dict = fixed_parameters.copy()
        # add loop parameters including fault surface variables
        for k, key in enumerate(keys):
            input_dict[key] = variable[k]
            if key == 'offset':
                offset = variable[k]
        # check if we need to create a new fault surface pair
        if iv % (len(variablelist)/nfv) == 0:
            size = rnaf.get_faultsize(np.array(fixed_parameters['ncells']),offset)
            print "size",size
            hinput = {}
            for inputname,param in [['D','fractal_dimension'],
                                    ['std','elevation_standard_deviation'],
                                    ['lc','mismatch_wavelength_cutoff']]:
                hinput[inputname] = ro.fault_dict[param]
            hinput['cs'] = np.average(fixed_parameters['cellsize'])
            print "creating height array"
            heights = np.array(rnfa.build_fault_pair(size, **hinput))
        # in every case until we create a new pair, the fault surface pair is the same
        input_dict['fault_surfaces'] = heights
        print input_dict
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
    print "got command line arguments, now divide inputs" 
    if rank == 0:
        print "rank is zero, getting list of inputs"
        list_of_inputs = initialise_inputs(fixed_parameters, 
                                           loop_parameters, 
                                           faultsurface_parameters)
        print "rank is zero, dividing inputs"
        inputs = divide_inputs(list_of_inputs,size)
        print "divided inputs"
    else:
        list_of_inputs = None
        inputs = None
    if rank == 0:
        if not os.path.exists(wd):
            os.mkdir(wd)
    else:
        while not os.path.exists(wd):
            time.sleep(1)
            print rank,"waiting for wd......"
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
    print "sending jobs out"
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
        fixedp_out = {}
        for fpm in fixed_parameters.keys():
            if fpm not in ['ncells','cellsize','workdir']:
                # check it is a r object parameter
                if (fpm == 'fault_edges') and (ro.fault_edges is not None):
                    fixedp_out[fpm] = ''.join(['['+','.join([str(fp) for fp in np.array(fe).flatten()])+']' for fe in ro.fault_edges])
                elif hasattr(ro0,fpm):
                    fixedp_out[fpm] = getattr(ro0,fpm)
                elif fpm in ro0.fault_dict.keys():
                    fixedp_out[fpm] = ro0.fault_dict[fpm]
        
        # get list of variable parameters
        varp_out = {}
        varpkeys = []
        print ro0.fault_dict
        print "faultsurface_parameters.keys() + loop_parameters.keys()",faultsurface_parameters.keys() + loop_parameters.keys()
        for ro in ogflat:    
            for vpm in faultsurface_parameters.keys() + loop_parameters.keys():
                if hasattr(ro,vpm):
                    if vpm not in varp_out.keys():
                        varp_out[vpm] = []
                        varpkeys.append(vpm)
                    varp_out[vpm].append(getattr(ro,vpm))
                elif vpm in ro.fault_dict.keys():
                    if vpm not in varp_out.keys():
                        varp_out[vpm] = []
                        varpkeys.append(vpm)
                    varp_out[vpm].append(ro.fault_dict[vpm])
            for vpo in ['resistivity_bulk','permeability_bulk']:
                for direction in range(3):
                    vpokey = vpo + 'xyz'[direction]
                    if vpokey not in varp_out.keys():
                        varp_out[vpokey] = []
                        varpkeys.append(vpokey)
                    varp_out[vpokey].append(getattr(ro,vpo)[direction]) 

        header = 'suite of resistor network simulations\n'
        for pm in ['ncells','cellsize']:
            header += pm + '{} {} {}\n'.format(*(getattr(ro0,pm)))
        header += 'fixed parameters\n'
        header += ' '.join(fixedp_out.keys())+'\n'
        header += ' '.join([str(varf) for varf in fixedp_out.values()])+'\n'
        header += 'variable parameters\n'
        header += ' '.join(varpkeys)

        output_array = np.array([varp_out[vkey] for vkey in varpkeys]).T
        print header
        print output_array                                      
        if 'outfile' in fixed_parameters.keys():
            outfile = fixed_parameters['outfile']
        else:
            outfile = 'outputs.dat'
        np.savetxt(op.join(wd,outfile),
                   output_array,
                   header = header, fmt='%.3e')
                  
                   
if __name__ == "__main__":
    setup_and_run_suite(sys.argv[1:])

