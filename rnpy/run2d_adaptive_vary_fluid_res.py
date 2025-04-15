# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:39:22 2025

@author: alisonk
"""

from rnpy.core.resistornetwork import Rock_volume
from rnpy.functions.assignproperties import update_all_apertures
from rnpy.functions.assignfaults_new import update_from_precalculated
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
                      ['cellsize','c','cellsize in x,y and z direction',3,float],
                      ['resistivity_matrix','rm','resistivity of matrix rock','*',float],
                      ['resistivity_fluid','rf','resistivity of fluid filling fractures','*',float],
                      ['fault_assignment',None,'how to assign faults, random or list, '\
                                               'if list need to provide fault edges',1,str],
                      ['faultlength_max',None,'maximum fault length, if specifying random faults',1,float],
                      ['faultlength_min',None,'minimum fault length, if specifying random faults',1,float],
                      ['offset',None,'offset on fault, (distance in m)',1,float],
                      ['deform_fault_surface',None,'whether or not to deform fault surfaces when applying offset',1,str],
                      ['fault_gouge',None,'whether or not to add fault gouge to the fault cavity',1,str],
                      ['alpha',None,'alpha value (scaling constant) for fault network',1,float],
                      ['num_fault_separation','nfs','number of fault separation '+\
                       'values, code will auto choose these values',1,float],
                      ['aperture_type',None,'type of aperture, random or constant',1,str],
                      ['mismatch_wavelength_cutoff',None,'cutoff wavelength for '+\
                       'mismatching of opposing surfaces in faults',1,float],
                      ['matrix_flow',None,'whether or not to include fluid flow in matrix surrounding fracture',1,str],
                      ['matrix_current',None,'whether or not to include current in matrix surrounding fracture',1,str],
                      ['correct_aperture_for_geometry',None,'whether or not to apply correction of aperture for local slopes in plate walls',1,str],
                      ['elevation_scalefactor',None,'scaling factor to scale fault elevations by',1,float],
                      ['elevation_prefactor',None,'prefactor to scale elevations by',1,float],
                      ['fractal_dimension',None,'fractal dimension used to calculate surfaces',1,float],
                      ['fault_surfaces_fn',None,'option to provide filename containing fault surfaces',1,str],
                      ['workdir','wd','working directory',1,str],
                      ['outfile','o','output file name',1,str],
                      ['solver_type','st','type of solver, bicg or direct',1,str],
                      
                      ['solve_properties','sp','which property to solve, current, fluid or currentfluid (default)',1,str],
                      ['solve_direction','sd','which direction to solve, x, y, z or a combination, e.g. xyz (default), xy, xz, y, etc',1,str],
                      ['repeats','r','how many times to repeat each permutation',1,int],
                      ['random_numbers_dir',None,'option to provide random seeds for creating fault planes as a file',1,str]]
    
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
    input_parameters = {'workdir':os.getcwd(), 'solver_type':'direct'}  
    suite_parameters = {'outfile':'outputs','num_fault_separation':10}
    repeats = 1
    
    # assign parameters to correct dictionaries. Only allowing fault separation
    # as a loop parameter at this point.
    for at in args._get_kwargs():
        if at[1] is not None:
            value = at[1]
            if at[0] == 'repeats':
                repeats = value[0]
            elif at[0] in ['ncells','cellsize']:
                if len(value) == 1:
                    input_parameters[at[0]] = [value[0]]*3
                else:
                    input_parameters[at[0]] = value
            elif at[0] in suite_parameters.keys():
                suite_parameters[at[0]] = value[0]
            elif at[0] == 'resistivity_fluid':
                input_parameters[at[0]] = value
            else:
                input_parameters[at[0]] = value[0]

    for key in input_parameters.keys():
        if type(input_parameters[key]) == 'str':
            input_parameters[key] = input_parameters[key].strip()

    if 'elevation_scalefactor' in input_parameters.keys():
        if input_parameters['elevation_scalefactor'] == 0.:
            input_parameters['elevation_scalefactor'] = None
            
    for att in ['deform_fault_surface','correct_aperture_for_geometry',
                'matrix_flow','matrix_current','fault_gouge']:
        if att in input_parameters.keys():
            if str.lower(input_parameters[att]) == 'true':
                input_parameters[att] = True
            else:
                input_parameters[att] = False
                

    return input_parameters, suite_parameters, repeats

    
def initialise_inputs(input_parameters):
    """
    initialise a rock volume to generate inputs for a suite of runs.
    This ensures that the fault surfaces are the same for all the different
    fault separation values
    
    """

    # set defaults
    inputs = {}
    inputs.update(input_parameters)

    # we are allowing multiple resistivity values
    # if it is resistivity, set the initial one as the first in the list.
    if (('resistivity_fluid' in input_parameters.keys()) and \
        np.iterable(input_parameters['resistivity_fluid'])):
        inputs['resistivity_fluid'] = input_parameters['resistivity_fluid'][0]
        print('updating resistivity fluid to first of list')
    
    
    RockVol = Rock_volume(**inputs)
    

    # store all parameters in input dict
    for att in ['ncells','resistivity_matrix','resistivity_fluid','fault_assignment',
                'fractal_dimension','faultlength_max','faultlength_min','alpha',
                'a','elevation_scalefactor','aperture_type','fault_edges',
                'fault_surfaces','cellsize','offset','deform_fault_surface',
                'random_numbers_dir']:

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
            
    if 'fault_surfaces_fn' in input_parameters.keys():
        inputs['fault_surfaces'] = np.load(input_parameters['fault_surfaces_fn'])
            
    print(inputs)
        
    return inputs


def validate_offset(offset, ncells):
    if offset < 1:
        return int(ncells * offset)
    else:
        return(int(offset))

def ap(h1,h2,offset,fs, remove_negative=False):
    if offset > 0:
        aperture = h1[offset:,offset:] - h2[offset:,:-offset] + fs
    else:
        aperture = h1 - h2 + fs
        
    if remove_negative:
        aperture[aperture < 0] = 0
    return aperture

def contact_area(ap):
    return (ap < 0).sum()/ap.size

def get_start_fault_separation(RockVol, fs0=0.0, fs1=1e-3):

    offset = validate_offset(RockVol.fault_dict['offset'], RockVol.ncells[1])
    h1,h2 = RockVol.fault_dict['fault_surfaces'][0]
    
    # get first fault separation
    while contact_area(ap(h1,h2,offset,fs0)) < 1.0:
        fs0 -= 1e-5
    
    # get second fault separation
    aperture = ap(h1,h2,offset,fs1, remove_negative=True)
    while np.mean(aperture)/np.amax(aperture) < 0.9:
        fs1 += 1e-4
        aperture = ap(h1,h2,offset,fs1, remove_negative=True)
        
    return np.array([fs0, 0.0, fs1])
    

def save_arrays(RockVol,property_names,suffix):
    """
    save arrays to numpy files
    """
    
    for att in property_names:
        if hasattr(RockVol,att):
            data = getattr(RockVol,att)
        else:
            data = RockVol.fault_dict[att]
        suffix = '_fs%.6e_mm'%(np.mean(RockVol.fault_dict['fault_separation'])*1e3)
        dirpath = os.path.join(RockVol.workdir,'arrays')
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        np.save(os.path.join(dirpath,att+suffix),data)
        
    # remove fault surfaces, as we only need to save them once
    for propname in ['fault_edges','fault_surfaces']:
        if propname in property_names:
            property_names.remove(propname)

    return property_names

        
def setup_run_single_model(input_parameters_new, iruntimefn):
    
    if trace_mem:
        tracemalloc.start()
    
    t0a = time.time()
            
    
    RockVolI = Rock_volume(**input_parameters_new)
    
    t0 = time.time()
    if trace_mem:
        current, peaksetup = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        
    solver_type = 'pardiso'
    

    RockVolI.solve_resistor_network2(method=solver_type)
    
    if trace_mem:
        current, peaksolve = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    iruntime=time.time()-t0
    setuptime = t0 - t0a
    print('time taken',iruntime)
    if os.path.isfile(iruntimefn):
        stat = 'a'
    else:
        stat = 'w'
        
    nx,ny,nz = input_parameters_new['ncells']
    if trace_mem:
        with open(iruntimefn,stat) as iruntimefile:
            iruntimefile.write('%1i %1i %1i %.4f %.4f %.4e %.4f %.4f %s\n'%(nx,ny,nz,iruntime, setuptime,
                                                                       input_parameters_new['fault_separation'], 
                                                                       peaksetup, peaksolve,
                                                                       solver_type))
    else:
        with open(iruntimefn,stat) as iruntimefile:
            iruntimefile.write('%1i %1i %1i %.4f %.4f %.4e %s\n'%(nx,ny,nz,
                                                                  iruntime, setuptime,
                                                                       input_parameters_new['fault_separation'], 
                                                                       solver_type))
            
    RockVolI.compute_conductive_fraction()
    
    return RockVolI


def run_adaptive(repeats, input_parameters, numfs, outfilename, rank):
    """
    run a set of rock volumes with adaptive fault separation (auto calculated
    based on position of percolation threshold)
    
    """
    # initial fault separation values
    
    first = True
    
    
    wd = input_parameters['workdir']
    iruntimefn = os.path.join(wd,'iruntime%1i.dat'%rank)
    
    nx,ny,nz = input_parameters['ncells']
    
    
    
    for r in repeats:
        
        input_parameters_new = {'fault_assignment':'random'}
        # offset start time so we don't load all the memory
        time.sleep(rank*10)
        input_parameters_new.update(initialise_inputs(input_parameters))
        input_parameters_new['fault_assignment'] = 'list'
        
        RockVolI = Rock_volume(**input_parameters_new)
        fault_separations = get_start_fault_separation(RockVolI, fs0=0.0, fs1=1e-3)
        
        
        cfractions = np.ones(3)*np.nan
        contactarea = np.ones(3)*np.nan
        resbulk = np.ones((3,1,3))*np.nan
        kbulk = np.ones((3,3))*np.nan
        cellsizes = np.ones((3,3))*np.nan

        props_to_save = ['aperture','current','flowrate','fault_surfaces','fault_edges']
        
        # run initial set of runs
        # 
        for i in range(3):
            # set x cell size to a small number if it's a 2d array
            for idx in np.where(np.array(input_parameters['ncells'])==0)[0]:
                input_parameters['cellsize'][idx] = 1e-8

            
            input_parameters_new['fault_separation'] = fault_separations[i]

            RockVolI = setup_run_single_model(input_parameters_new, iruntimefn)
            
            
            
            cfractions[i] = RockVolI.conductive_fraction
            contactarea[i] = RockVolI.contact_area[0]
            resbulk[i,0] = RockVolI.resistivity_bulk
            kbulk[i] = RockVolI.permeability_bulk
            cellsizes[i] = RockVolI.cellsize
            
            if r == 0:
                props_to_save = save_arrays(RockVolI,props_to_save,'r%1i'%r)

                
        # run infilling runs
        count = len(fault_separations)
        
        while count < numfs:
            # set x cell size to a small number if it's a 2d array
            for idx in np.where(np.array(input_parameters['ncells'])==0)[0]:
                input_parameters['cellsize'][idx] = 1e-8
                
            # compute differences between adjacent permeability points on curve
            kjump = np.log10(kbulk[1:])-np.log10(kbulk[:-1])
            kjump = np.amax(kjump,axis=1)
            i = int(np.where(kjump == max(kjump))[0])
            
            # new fault separation to insert (halfway across max jump)
            newfs = np.mean(fault_separations[i:i+2])
            input_parameters_new['fault_separation'] = newfs
                
            RockVol = setup_run_single_model(input_parameters_new, iruntimefn)

            # insert resistivity bulk, conductive fraction & new fault separation to arrays
            resbulk = np.insert(resbulk,i+1,np.zeros_like(resbulk[0]),axis=0)
            resbulk[i+1, 0] = RockVol.resistivity_bulk
            kbulk = np.insert(kbulk, i+1, RockVol.permeability_bulk,axis=0)
            cellsizes = np.insert(cellsizes, i+1, RockVol.cellsize,axis=0)
            cfractions = np.insert(cfractions,i+1,RockVol.conductive_fraction)
            contactarea = np.insert(contactarea,i+1,RockVol.contact_area[0])

            fault_separations = np.insert(fault_separations,i+1,newfs)
            
            if r == 0:
                props_to_save = save_arrays(RockVol,props_to_save,'r%1i'%r)

            
            count += 1
            

        # repeat the modelling at different fluid resistivities
        for j in range(1, len(input_parameters['resistivity_fluid'])):
            # insert resistivity bulk, conductive fraction & new fault separation to arrays
            resbulk = np.insert(resbulk,j,np.zeros(3),axis=1)
            for i in range(len(fault_separations)):
                input_parameters_new['fault_separation'] = fault_separations[i]
                input_parameters_new['resistivity_fluid'] = input_parameters['resistivity_fluid'][j]
                input_parameters_new['solve_properties'] = 'current'
                RockVol = setup_run_single_model(input_parameters_new,iruntimefn)
                resbulk[i,j] = RockVol.resistivity_bulk
                
                print(resbulk.shape)

        print(resbulk)

                
        print(input_parameters['resistivity_fluid'])
        outputs_dict = {'fault_separation':fault_separations,
                        'porosity':cfractions,
                        'contact_area':contactarea,
                        'cellsize':cellsizes,
                        'permeability_bulk':kbulk,
                        'resistivity_bulk':resbulk
                        }
        
        dtypes = []
        array_dict = {}        
        for key in outputs_dict.keys():
            value = outputs_dict[key]
            # 1d array, just make a column
            if len(value.shape) == 1:
                dtypes.append((key,float))
                array_dict[key] = value
                # print('appended array, shape',key, array_dict[key].shape)
            elif len(value.shape) == 2:
                for i, val in enumerate('xyz'):
                    name = key+'_'+val
                    # print(key,val)
                    dtypes.append((name, float))
                    array_dict[name] = value[:,i]
                    # print('appended array, shape',name, array_dict[name].shape)
            elif len(value.shape) == 3:
                if key == 'resistivity_bulk':
                    for j, resval in enumerate(input_parameters['resistivity_fluid']):
                        for i, val in enumerate('xyz'):
                            name = key+'_%s_rf%s'%(val,resval)
                            dtypes.append((name, float))
                            array_dict[name] = value[:,j,i]
                            # print('appended array, shape',name,array_dict[name].shape)
            output_array = np.zeros(len(value),
                                    dtype=dtypes)
            for name in array_dict.keys():
                output_array[name] = array_dict[name]

                
        np.save(r'C:\tmp\output_array',output_array)
        
        # update input_parameters_new
        input_parameters_new.update(input_parameters)
        
        write_outputs(input_parameters_new, output_array, outfilename)
        

    return outfilename


def write_outputs(input_parameters, output_array, outfilename):
    """
    write outputs to a file
    
    """

    # create header with fixed parameters
    header = '# suite of resistor network simulations\n'
    header += '### fixed parameters ###\n'
    
    # add to header with all the inputs
    for param in input_parameters.keys():
        if param not in output_array.dtype.names:
            if np.iterable(input_parameters[param]):
                if not np.iterable(input_parameters[param][0]):
                    header += '# ' + param + ' ' + ' '.join([str(val) for val in input_parameters[param]]) + '\n'
            elif type(input_parameters[param]) == dict:
                
                for key in input_parameters[param].keys():
                    # don't want any arrays with dim > 1 in the header
                    print(key,input_parameters[param][key])
                    print(np.iterable(input_parameters[param][key]))
                    if not np.iterable(input_parameters[param][key]):
                        header += '# ' + key + ' ' + str(input_parameters[param][key]) + '\n'
            else:
                header += '# ' + param + ' ' + str(input_parameters[param]) + '\n'
    header += '### variable parameters ###\n'
    header += '# ' + ' '.join(output_array.dtype.names)

    np.savetxt(outfilename,
               output_array,fmt='%.6e',header=header,comments='')

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
    np.savetxt(os.path.join(os.path.dirname(op),newbn),data,fmt='%.6e',
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
        
        nx,ny,nz = input_parameters['ncells']
        print(runtimefn)
        with open(runtimefn,stat) as runtimefile:
            runtimefile.write('%1i %1i %1i %.4f %s\n'%(nx,ny,nz,runtime,input_parameters['solver_type']))
    

if __name__ == "__main__":

    setup_and_run_suite(sys.argv)
    