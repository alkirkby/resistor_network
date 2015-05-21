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
                    fixed_parameters[at[0]] = value[0]
                elif at[0] in ['ncells','cellsize']:
                    fixed_parameters[at[0]] = value
                else:
                    loop_parameters[at[0]] = value
            else:
                fixed_parameters[at[0]] = value
    

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

    if len(faultsurface_inputs) > 0:
        variablelist = [list(val) for val in itertools.product(faultsurface_inputs,loop_inputs)]
        variablelist2 = []
        for vline in variablelist:
            vlist_temp = []
            for vv in vline:
                for v in vv:
                    vlist_temp.append(v)
                variablelist2.append(vlist_temp)
        variablelist = variablelist2
    else:
        variablelist = loop_inputs
    # create a list of keys for all loop inputs including faultsurface, faultsurface
    # keywords first
    keys = faultsurface_parameters.keys()
    keys += loop_parameters.keys()
    # number of different fault surface variations, including repeats
    nfv = min(len(faultsurface_inputs),1)
    
    # intialise a rock volume to get the defaults from
    ro = rn.Rock_volume(build=False)

    for fparam in ['ncells','cellsize']:
        if fparam not in fixed_parameters.keys():
            fixed_parameters[fparam] = getattr(ro,fparam)
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
            hinput = {}
            for inputname,param in [['D','fractal_dimension'],
                                    ['std','elevation_standard_deviation'],
                                    ['lc','mismatch_wavelength_cutoff']]:
                hinput[inputname] = ro.fault_dict[param]
            hinput['cs'] = np.average(fixed_parameters['cellsize'])
            heights = np.array(rnfa.build_fault_pair(size, **hinput))
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
    
    

def write_output(ro, loop_variables, outfilename, newfile):
    """
    prepare an output file for writing to
    """
    variables = vars(ro)
    output_variables = ['contact_area','aperture_mean']
    output_variables += ['resistivity_bulk'+ri for ri in 'xyz']
    output_variables += ['permeability_bulk'+ri for ri in 'xyz']
    loop_input_output = loop_variables + output_variables
    
    variablekeys = []
    fixeddict = {}
    variabledict = {}

    for vkey in variables.keys():        
        append = False
        if type(variables[vkey]) in [np.float64,float,str]:
            keys, values, append = [vkey], [variables[vkey]], True
        elif type(variables[vkey]) == dict:
            keys = [kk for kk in variables[vkey].keys() if type(variables[vkey][kk]) in [np.float64,float,str]]
            values = [variables[vkey][kk] for kk in keys]
            append = True
        elif vkey in ['resistivity_bulk','permeability_bulk']:
            keys, values, append = [vkey+'xyz'[i] for i in range(3)], [variables[vkey][i] for i in range(3)], True
        if append:
            for k,key in enumerate(keys):
                if key in loop_input_output:
                    variablekeys.append(key)
                    variabledict[key] = values[k]
                else:
                    fixeddict[key] = values[k]

    output_line = [variabledict[vkey] for vkey in variablekeys]

    if newfile:
        with open(outfilename, 'wb') as outfile:
            header = '# suite of resistor network simulations\n'
            for pm in ['ncells','cellsize']:
                header += '# ' + pm + ' {} {} {}\n'.format(*(getattr(ro,pm)))
            header += '### fixed parameters ###\n'
            header += '# '+'\n# '.join([' '.join([key,str(fixeddict[key])]) for key in fixeddict.keys()])+'\n'
            header += '### variable parameters ###\n'
            header += '# '+' '.join(loop_input_output)
            outfile.write(header)
            outfile.write('\n'+' '.join(['%.3e'%oo for oo in output_line]))
    else:
        print "appending to old file"
        with open(outfilename, 'ab') as outfile:
            outfile.write('\n'+' '.join(['%.3e'%oo for oo in output_line]))



def run(list_of_inputs,rank,wd,outfilename,loop_variables,save_array=True):
    """
    generate and run a random resistor network
    takes a dictionary of inputs to be used to create a resistivity object
    """
    
    r_objects = []
    
    ofb = op.basename(outfilename)
    ofp = op.dirname(outfilename)
    if '.' in ofb:
        di = ofb.index('.')
        outfilename = op.join(ofp, ofb[:di] + str(rank) + ofb[di:])
    else:
        outfilename = outfilename + str(rank)

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
        if r == 0:
            newfile = True
        else:
            newfile = False
        write_output(ro,loop_variables,outfilename,newfile)
        r += 1
        
    return outfilename
  
    
def setup_and_run_suite(arguments, argument_names):
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
    fixed_parameters, loop_parameters, faultsurface_parameters = read_arguments(arguments, argument_names)
    if 'workdir' in fixed_parameters.keys():
        wd = fixed_parameters['workdir']
    else:
        wd = './model_runs'
    wd2 = os.path.join(wd,'arrays')

    if rank == 0:
        # get inputs
        list_of_inputs = initialise_inputs(fixed_parameters, 
                                           loop_parameters, 
                                           faultsurface_parameters)
        # divide inputs
        inputs = divide_inputs(list_of_inputs,size)
        
        # make working directories
        if not os.path.exists(wd):
            os.mkdir(wd)
        wd = os.path.abspath(wd)
        if not os.path.exists(wd2):
            os.mkdir(wd2)
         
    else:
        list_of_inputs = None
        inputs = None
        while not os.path.exists(wd2):
            time.sleep(1)
            print 'process {} waiting for wd'.format(rank)

    # initialise outfile
    if 'outfile' in fixed_parameters.keys():
        outfile = fixed_parameters['outfile']
    else:
        outfile = 'outputs.dat'

    print "sending jobs out"
    inputs_sent = comm.scatter(inputs,root=0)
    outfilenames = run(inputs_sent,
                       rank,
                       wd2,
                       op.join(wd,outfile),
                       loop_parameters.keys()+faultsurface_parameters.keys())
    outputs_gathered = comm.gather(outfilenames,root=0)
    
    if rank == 0:
        outfn = outputs_gathered[0]
        outarray = np.loadtxt(outfn)
        outfile0 = open(outfn)
        line = outfile0.readline()
        header = ''

        while line[0] == '#':
            header += line
            line = outfile0.readline()
        header = header.strip()
        count = 0
        for outfn in outputs_gathered:
            if count > 0:
                outarray = np.vstack([outarray,np.loadtxt(outfn)])
            count += 1

        np.savetxt(op.join(wd,outfile),outarray,header=header,fmt='%.3e',comments='')
        
                   
if __name__ == "__main__":
    setup_and_run_suite(sys.argv[1:],argument_names)

