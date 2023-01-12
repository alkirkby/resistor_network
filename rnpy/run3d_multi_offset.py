#-*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:06:10 2019

@author: alktrt
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
                      ['permeability_gouge',None,'permeability of gouge material','*',float],
                      ['fault_assignment',None,'how to assign faults, random or list, '\
                                               'if list need to provide fault edges',1,str],
                      ['faultlength_max',None,'maximum fault length, if specifying random faults',1,float],
                      ['faultlength_min',None,'minimum fault length, if specifying random faults',1,float],
                      ['fault_separation',None,'fault separation, (distance in m)',1,float],
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
                      ['effective_apertures_fn',None,'file containing precalculated effective apertures',1,str],
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
    initialise a rock volume to generate inputs for a suite of runs
    
    """
    print(input_parameters)
    RockVol = Rock_volume(**input_parameters)
    
    # set defaults
    inputs = {}
    inputs.update(input_parameters)
    inputs['solve_properties'] = 'current_fluid'
    
    # store all parameters in input dict
    for att in ['ncells','resistivity_matrix','resistivity_fluid','fault_assignment',
                'fractal_dimension','faultlength_max','faultlength_min','alpha',
                'a','elevation_scalefactor','aperture_type','fault_edges',
                'fault_surfaces','cellsize','fault_separation','deform_fault_surface',
                'random_numbers_dir','permeability_gouge']:
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
        suffix = '_oo%.6e_mm'%(RockVol.fault_dict['offset'])
        dirpath = os.path.join(RockVol.workdir,'arrays')
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        np.save(os.path.join(dirpath,att+suffix),data)
        
        
def get_solver_type(solver_type,fs,ncells):
    if solver_type == 'adapt':
        if ((np.abs(fs) < 5e-5) or (np.product(np.array(ncells)+1) < 10000)):
            solver_type = 'direct'
        else:
            solver_type = 'bicg'
    return solver_type


def get_meanstd(h,size_noclip):
    ic = int(h.shape[1]/2)
    i0,i1 = ic - int(size_noclip/2), ic + int(size_noclip/2)
    return np.mean([np.std(line[i0:i1]) for line in h])

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
        offsets = np.arange(0,0.42,0.01)[::-1]
        input_parameters['offset'] = offsets[0]
        input_parameters_new.update(initialise_inputs(input_parameters))
        input_parameters_new['fault_assignment'] = 'list'
        
        # if we are updating apertures then need to preserve negative apertures
        if 'effective_apertures_fn' in input_parameters.keys():
            if 'fault_dict' not in input_parameters_new.keys():
                input_parameters_new['fault_dict'] = {}
                
            input_parameters_new['fault_dict']['preserve_negative_apertures'] = True
            

            
        fs = 1e-4


        cfractions = np.ones_like(offsets)*np.nan
        contactarea = np.ones_like(offsets)*np.nan
        resbulk = np.ones((offsets.shape[0],3))*np.nan
        kbulk = np.ones((offsets.shape[0],3))*np.nan
        cellsizes = np.ones((offsets.shape[0],3))*np.nan
        gouge_areas = np.zeros_like(offsets)*np.nan
        gouge_fractions = np.zeros_like(offsets)*np.nan

        props_to_save = ['aperture','current','flowrate','fault_surfaces','fault_edges']
        
        # run initial set of runs
        for i in np.arange(len(offsets)):
            
            # set x cell size to a small number if it's a 2d array
            for idx in np.where(np.array(input_parameters['ncells'])==0)[0]:
                input_parameters['cellsize'][idx] = 1e-8
            oo = offsets[i]
            print("offset",oo)
            if trace_mem:
                tracemalloc.start()
            input_parameters_new['offset'] = oo
            input_parameters_new['fault_separation'] = fs
            t0a = time.time()
            
            
            
            RockVolI = Rock_volume(**input_parameters_new)
            
            if 'effective_apertures_fn' in input_parameters.keys():
            # if input_parameters['effective_apertures_fn'] in :
                RockVolI = update_from_precalculated(RockVolI,input_parameters['effective_apertures_fn'])
    
            t0 = time.time()
            if trace_mem:
                current, peaksetup = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                tracemalloc.start()
                
            solver_type = get_solver_type(input_parameters['solver_type'],fs, RockVolI.ncells)
            
            
            if 'fault_gouge' in input_parameters.keys():
                if input_parameters['fault_gouge']:
                    
                    RockVolI.add_fault_gouge()
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
            if trace_mem:
                with open(iruntimefn,stat) as iruntimefile:
                    iruntimefile.write('%1i %1i %1i %.4f %.4f %.4e %.4f %.4f %s\n'%(nx,ny,nz,iruntime, setuptime,
                                                                                fs, peaksetup, peaksolve,
                                                                                input_parameters['solver_type']))
            else:
                with open(iruntimefn,stat) as iruntimefile:
                    iruntimefile.write('%1i %1i %1i %.4f %.4f %.4e %s\n'%(nx,ny,nz,iruntime, setuptime,
                                                                                fs, 
                                                                                input_parameters['solver_type']))                
         
            RockVolI.compute_conductive_fraction()
            cfractions[i] = RockVolI.conductive_fraction
            contactarea[i] = RockVolI.contact_area[0]
            resbulk[i] = RockVolI.resistivity_bulk
            kbulk[i] = RockVolI.permeability_bulk
            cellsizes[i] = RockVolI.cellsize
            gouge_areas[i] = RockVolI.gouge_area_fraction
            gouge_fractions[i] = RockVolI.gouge_fraction
            
            # if r == 0:
            save_arrays(RockVolI,props_to_save,'r%1i'%r)
            # only save 1 copy of fault surfaces
            for propname in ['fault_edges','fault_surfaces']:
                if propname in props_to_save:
                    props_to_save.remove(propname)
                

            
        if first:
            
            oo_master = offsets.copy()
            rb_master = resbulk.copy()
            kb_master = kbulk.copy()
            cf_master = cfractions.copy()
            rp_master = np.ones(len(oo_master))*r
            cs_master = cellsizes.copy()
            ca_master = contactarea.copy()
            ga_master = gouge_areas.copy()
            gf_master = gouge_fractions.copy()
            first = False
        else:
            oo_master = np.hstack([oo_master,offsets])
            rb_master = np.concatenate([rb_master,resbulk])
            kb_master = np.concatenate([kb_master,kbulk])
            cf_master = np.hstack([cf_master,cfractions])
            rp_master = np.hstack([rp_master,np.ones(len(offsets))*r])
            cs_master = np.concatenate([cs_master,cellsizes])
            ca_master = np.hstack([ca_master,contactarea])
            ga_master = np.hstack([ga_master,gouge_areas])
            gf_master = np.hstack([gf_master,gouge_fractions])

        
        write_outputs(input_parameters_new,oo_master,cf_master,rb_master,
                      kb_master,rp_master,cs_master, ca_master, ga_master,
                      gf_master,
                      rank, outfilename)
        
    return outfilename


def write_outputs(input_parameters,offsets,cfractions,resbulk,
                  kbulk,repeatno,cellsizex,contactarea, gouge_areas,
                  gouge_fractions,
                  rank, outfilename):
    """
    write outputs to a file
    
    """
    # variable names
    variablekeys = ['offset','conductive_fraction']
    variablekeys += ['resistivity_bulk_%s'%val for val in 'xyz']
    variablekeys += ['permeability_bulk_%s'%val for val in 'xyz']
    variablekeys += ['cellsize_%s'%val for val in 'xyz']
    variablekeys += ['contact_area']
    variablekeys += ['repeat']
    variablekeys += ['gouge_fraction','gouge_area_fraction']
    

    # values for above headings
    output_lines = np.vstack([offsets,
                              cfractions,
                              resbulk.T,
                              kbulk.T,
                              cellsizex.T,
                              contactarea,
                              repeatno,
                              gouge_areas,
                              gouge_fractions]).T

    # create a dictionary containing fixed variables
    fixeddict = {}
    for param in ['cellsize','ncells']:
        value = input_parameters[param]
        if np.iterable(value):
            fixeddict[param] = ' '.join([str(val) for val in input_parameters[param]])
        else:
            fixeddict[param] = str(value)
    for param in ['resistivity_matrix','resistivity_fluid','permeability_matrix',
                  'fault_assignment','fractal_dimension','faultlength_max',
                  'faultlength_min','alpha','a','elevation_scalefactor',
                  'aperture_type','fault_separation','mismatch_wavelength_cutoff',
                  'matrix_flow','matrix_current','correct_aperture_for_geometry',
                  'deform_fault_surface','permeability_gouge','porosity_gouge']:
        if param in input_parameters.keys():
            fixeddict[param] = input_parameters[param]


    header = '# suite of resistor network simulations\n'
    
    header += '### fixed parameters ###\n'
    header += '# '+'\n# '.join([' '.join([key,str(fixeddict[key])]) for key in fixeddict.keys()])+'\n'
    header += '### variable parameters ###\n'
    header += '# '+' '.join(variablekeys)
    np.savetxt(outfilename,
               output_lines,fmt='%.6e',header=header,comments='')

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
    
