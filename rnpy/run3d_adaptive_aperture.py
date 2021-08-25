# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:06:10 2019

@author: alktrt
"""

from rnpy.core.resistornetwork import Rock_volume
from rnpy.functions.assignproperties import update_all_apertures
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import time

trace_mem = True
if trace_mem:
    import tracemalloc


def parse_arguments(arguments):
    """
    parse arguments from command line
    """
    
    
    argument_names = [['ncells','n','number of cells x,y and z direction',3,int],
                      ['cellsize','c','cellsize in x,y and z direction',1,float],
                      ['resistivity_matrix','rm','resistivity of matrix rock','*',float],
                      ['resistivity_fluid','rf','resistivity of fluid filling fractures','*',float],
                      ['fault_assignment',None,'how to assign faults, random or list, '\
                                               'if list need to provide fault edges',1,str],
                      ['faultlength_max',None,'maximum fault length, if specifying random faults',1,float],
                      ['faultlength_min',None,'minimum fault length, if specifying random faults',1,float],
                      ['alpha',None,'alpha value (scaling constant) for fault network',1,float],
                      ['num_fault_separation','nfs','number of fault separation '+\
                       'values, code will auto choose these values',1,float],
                      ['aperture_type',None,'type of aperture, random or constant',1,str],
                      ['workdir','wd','working directory',1,str],
                      ['outfile','o','output file name',1,str],
                      ['solver_type','st','type of solver, bicg or direct',1,str],
                    #  ['solve_properties','sp','which property to solve, current, fluid or currentfluid (default)',1,str],
                    #  ['solve_direction','sd','which direction to solve, x, y, z or a combination, e.g. xyz (default), xy, xz, y, etc',1,str],
                      ['repeats','r','how many times to repeat each permutation',1,int]]
    
    parser = argparse.ArgumentParser()
    
    
    for i in range(len(argument_names)):
        longname,shortname,helpmsg,nargs,vtype = argument_names[i]
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
    
    # initialise defaults (other defaults defined internally in rock_volume object)
    input_parameters = {'workdir':os.getcwd()}  
    suite_parameters = {'outfile':'outputs','num_fault_separation':10}
    repeats = 1
    
    # assign parameters to correct dictionaries. Only allowing fault separation
    # as a loop parameter at this point.
    for at in args._get_kwargs():
        if at[1] is not None:
            value = at[1]
            if at[0] == 'repeats':
                repeats = value[0]
            elif at[0] in suite_parameters.keys():
                suite_parameters[at[0]] = value[0]
            else:
                input_parameters[at[0]] = value[0]
            
    return input_parameters, suite_parameters, repeats
    
def initialise_inputs(input_parameters):
    """
    initialise a rock volume to generate inputs for a suite of runs
    
    """
    RockVol = Rock_volume(**input_parameters)
    
    # set defaults
    inputs = {}
    inputs.update(input_parameters)
    inputs['mismatch_wavelength_cutoff'] = 1e-3 # cutoff wavelength for mismatching of opposing surfaces in faults
    inputs['elevation_scalefactor'] = 1.9e-3 # scaling factor to scale fault elevations by
    inputs['fractal_dimension'] = 2.4 # fractal dimension used to calculate surfaces
    inputs['a'] = 3.5
    inputs['solve_properties'] = 'current_fluid'
    inputs['solve_direction'] = 'z'
    
    # store all parameters in input dict
    for att in ['ncells','resistivity_matrix','resistivity_fluid','fault_assignment',
                'fractal_dimension','faultlength_max','faultlength_min','alpha',
                'a','elevation_scalefactor','aperture_type','fault_edges',
                'fault_surfaces','cellsize']:
        # check if it is an attribute in RockVol
        if hasattr(RockVol,att):
            attval = getattr(RockVol,att)
            if type(attval) == np.ndarray:
                inputs[att] = attval.copy()
            else:
                inputs[att] = attval
        # if not, check fault dictionary
        elif att in RockVol.fault_dict.keys():
            inputs[att] = RockVol.fault_dict[att]
        
    
            
        
    return inputs

def save_arrays(RockVol,property_names,suffix):
    """
    save arrays to numpy files
    """
    
    for att in property_names:
        if hasattr(RockVol,att):
            data = getattr(RockVol,att)
        else:
            data = RockVol.fault_dict[att]
        suffix = '_fs%.2e_mm'%(np.mean(RockVol.fault_dict['fault_separation'])*1e3)
        dirpath = os.path.join(RockVol.workdir,'arrays')
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        np.save(os.path.join(dirpath,att+suffix),data)
        
        

def run_adaptive(repeats, input_parameters, numfs, outfilename, rank):
    """
    run a set of rock volumes with adaptive fault separation (auto calculated
    based on position of percolation threshold)
    
    """
    # initial fault separation values
    
    first = True
    
    
    wd = input_parameters['workdir']
    iruntimefn = os.path.join(wd,'iruntime.dat')
    nx,ny,nz = [input_parameters['ncells']]*3
    
    
    
    for r in repeats:
        input_parameters_new = {'fault_assignment':'random'}
        # offset start time so we don't load all the memory
        time.sleep(rank*10)
        input_parameters_new.update(initialise_inputs(input_parameters))
        input_parameters_new['fault_assignment'] = 'list'
        fault_separations = np.array([-1.,0.,10.])*input_parameters_new['cellsize'][2]
        # initialise arrays to contain bulk resistivity and conductive fractions
        cfractions = np.ones_like(fault_separations)*np.nan
        resbulk = np.ones_like(fault_separations)*np.nan
        kbulk = np.ones_like(fault_separations)*np.nan
        props_to_save = ['aperture','current','fault_surfaces']
        
        # run initial set of runs
        for i, fs in enumerate(fault_separations):
            if trace_mem:
                tracemalloc.start()
            input_parameters_new['fault_separation'] = fs
            RockVolI = Rock_volume(**input_parameters_new)
    
            t0 = time.time()
            if trace_mem:
                current, peaksetup = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                tracemalloc.start()
            RockVolI.solve_resistor_network2(method=input_parameters['solver_type'])
            
            if trace_mem:
                current, peaksolve = tracemalloc.get_traced_memory()
                tracemalloc.stop()

            iruntime=time.time()-t0
            print('time taken',iruntime)
            if os.path.isfile(iruntimefn):
                stat = 'a'
            else:
                stat = 'w'
            with open(iruntimefn,stat) as iruntimefile:
                iruntimefile.write('%1i %1i %1i %.4f %.4f %.4f %.4f %s\n'%(nx,ny,nx,iruntime, fs, peaksetup, peaksolve,input_parameters['solver_type']))
         
            RockVolI.compute_conductive_fraction()
            cfractions[i] = RockVolI.conductive_fraction
            resbulk[i] = RockVolI.resistivity_bulk[2]
            kbulk[i] = RockVolI.permeability_bulk[2]
            if r == 0:
                save_arrays(RockVolI,props_to_save,'r%1i'%r)
    
        # run infilling runs
        count = len(fault_separations)
        
        while count < numfs:
                
            # compute differences between adjacent resistivity points on curve
            resjump = np.log10(resbulk[:-1])-np.log10(resbulk[1:])
            kjump = np.log10(kbulk[1:])-np.log10(kbulk[:-1])
            
            # index after which we have the maximum jump
            #i = int(np.where(resjump==max(resjump))[0])
            i = int(np.where(kjump == max(kjump))[0])
            
            # new fault separation to insert (halfway across max jump)
            newfs = np.mean(fault_separations[i:i+2])
            input_parameters_new['fault_separation'] = newfs
            
            # create a rock volume
            if trace_mem:
                tracemalloc.start()
                
            RockVol = Rock_volume(**input_parameters_new)
            t0 = time.time()
#            print(input_parameters['solver_type'])
            if trace_mem:
                current, peaksetup = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                tracemalloc.start()

            RockVol.solve_resistor_network2(method=input_parameters['solver_type'])

            if trace_mem:
                current, peaksolve = tracemalloc.get_traced_memory()
                tracemalloc.stop()            
            
            iruntime=time.time()-t0
            print('time taken',iruntime)
            with open(iruntimefn,stat) as iruntimefile:
                iruntimefile.write('%1i %1i %1i %.4f %.4f %.4f %.4f %s\n'%(nx,ny,nx,iruntime,newfs, peaksetup, peaksolve,input_parameters['solver_type']))
         
            # insert resistivity bulk, conductive fraction & new fault separation to arrays
            resbulk = np.insert(resbulk,i+1,RockVol.resistivity_bulk[2])
            kbulk = np.insert(kbulk, i+1, RockVol.permeability_bulk[2])
            #print(resbulk)
            #print(kbulk)
            RockVol.compute_conductive_fraction()
            cfractions = np.insert(cfractions,i+1,RockVol.conductive_fraction)
            fault_separations = np.insert(fault_separations,i+1,newfs)
            
            if r == 0:
                save_arrays(RockVol,props_to_save,'r%1i'%r)
            
            count += 1
            
        if first:
            
            fs_master = fault_separations.copy()
            rb_master = resbulk.copy()
            kb_master = kbulk.copy()
            cf_master = cfractions.copy()
            rp_master = np.ones(len(fs_master))*r
            first = False
        else:
            fs_master = np.hstack([fs_master,fault_separations])
            rb_master = np.hstack([rb_master,resbulk])
            kb_master = np.hstack([kb_master,kbulk])
            cf_master = np.hstack([cf_master,cfractions])
            rp_master = np.hstack([rp_master,np.ones(len(fault_separations))*r])
            

        
        
        write_outputs(input_parameters,fs_master,cf_master,rb_master,kb_master,rp_master, rank, outfilename)
        
    return outfilename


def write_outputs(input_parameters,fault_separations,cfractions,resbulk,kbulk,repeatno, rank, outfilename):
    """
    write outputs to a file
    
    """
    # variable names
    variablekeys = ['fault_separation','conductive_fraction']
    variablekeys += ['resistivity_bulk_z']
    variablekeys += ['permeability_bulk_z']
    variablekeys += ['repeat']

    # values for above headings
    output_lines = np.vstack([fault_separations,
                              cfractions,
                              resbulk,
                              kbulk,
                              repeatno]).T

    # create a dictionary containing fixed variables
    fixeddict = {}
    for param in ['cellsize','ncells']:
        value = input_parameters[param]
        if np.iterable(value):
            fixeddict[param] = ' '.join([str(val) for val in input_parameters[param]])
        else:
            fixeddict[param] = str(value)
    for param in ['resistivity_matrix','resistivity_fluid','fault_assignment',
                'fractal_dimension','faultlength_max','faultlength_min','alpha',
                'a','elevation_scalefactor','aperture_type']:
        if param in input_parameters.keys():
            fixeddict[param] = input_parameters[param]


    header = '# suite of resistor network simulations\n'
    
    header += '### fixed parameters ###\n'
    header += '# '+'\n# '.join([' '.join([key,str(fixeddict[key])]) for key in fixeddict.keys()])+'\n'
    header += '### variable parameters ###\n'
    header += '# '+' '.join(variablekeys)
    np.savetxt(outfilename,
               output_lines,fmt='%.3e',header=header,comments='')

    return outfilename
      
    
def combine_outputs(outputs_gathered):
    
    first = True
    
    for op in outputs_gathered:
        
        if first:
            data = np.loadtxt(op)
            # read header
            header = ""
            with open(op) as ff:
                for line in ff:
                    if line.startswith('#'):
                        header += line
                        
            first = False
        else:
            data = np.vstack([data,np.loadtxt(op)])
            
    bn = os.path.basename(outputs_gathered[0])
    newbn = bn[:-6]+bn[-4:]
    np.savetxt(os.path.join(os.path.dirname(op),newbn),data,fmt='%.3e',
               header=header,comments='')


def get_mpi_specs():
    """
    determine if using mpi and get processor specs
    """
    
    try:
        from mpi4py import MPI
        
        # sort out rank and size info
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        name = MPI.Get_processor_name()
    except:
        comm=None
        size=1
        rank=0
        name='Fred'
    
    print('Hello! My name is {}. I am process {} of {}'.format(name,rank,size))
    
    return comm, name,rank,size


def divide_inputs(repeats,procsize,rank):
    
    chunks = [[] for _ in range(procsize)]
    for r in range(repeats):
        chunks[r%procsize].append(r)
        
    return chunks


def setup_and_run_suite(arguments):
    """
    master function to set up and run a set of rock volumes
    
    """
    t0 = time.time()
    
    # get variables from command line
    input_parameters, suite_parameters, repeats = parse_arguments(arguments)
    
    # fill input parameters with defaults and initialise faults
#    input_parameters = initialise_inputs(input_parameters)
    
    comm, name, rank, size= get_mpi_specs()
    
    # get workdir
    wd = input_parameters['workdir']
    
    # define output filename
    outfilename = os.path.join(wd,'outputs%02i.dat'%rank)
    

    if rank == 0:
        chunks = divide_inputs(repeats,size,rank)
        
        # make working directories
        if not os.path.exists(wd):
            os.mkdir(wd)
        wd = os.path.abspath(wd)
    
    else:       
        chunks = None


    if comm is not None:
        inputs_sent = comm.scatter(chunks,root=0)
    else:
        inputs_sent = range(repeats)
        
    outfilenames = run_adaptive(inputs_sent,
                                input_parameters,
                                suite_parameters['num_fault_separation'],
                                outfilename,
                                rank)

    if comm is not None:
        outputs_gathered = comm.gather(outfilenames,root=0)
    else:
        outputs_gathered = [outfilenames]
        
#    print outputs_gathered
    if rank == 0:
        combine_outputs(outputs_gathered)
        
        runtime=time.time()-t0
        
        print("simulations took %1i s to run"%(runtime))
        runtimefn = os.path.join(wd,'runtime.dat')
        
        if os.path.isfile(runtimefn):
            stat = 'a'
        else:
            stat = 'w'
        
        nx,ny,nz = [input_parameters['ncells']]*3
        with open(runtimefn,stat) as runtimefile:
            runtimefile.write('%1i %1i %1i %.4f %s\n'%(nx,ny,nz,runtime,input_parameters['solver_type']))
    

if __name__ == "__main__":

    setup_and_run_suite(sys.argv)          
    
