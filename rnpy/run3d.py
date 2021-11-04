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
                  ['pconnection','p','probability of a fault in x, y and z direction',3,float],
                  ['resistivity_matrix','rm','','*',float],
                  ['resistivity_fluid','rf','','*',float],
                  ['permeability_matrix','km','','*',float],
                  ['fluid_viscosity','mu','',1,float],
                  ['fault_assignment',None,'how to assign faults, random or list, '\
                                           'if list need to provide fault edges',1,str],
                  ['offset',None,'number of cells offset between fault surfaces',1,float],
                  ['faultlength_max',None,'maximum fault length, if specifying random faults',1,float],
                  ['faultlength_min',None,'minimum fault length, if specifying random faults',1,float],
                  ['alpha',None,'alpha value (scaling constant) for fault network',1,float],
                  ['mismatch_wavelength_cutoff',None,
                  'wavelength cutoff for matching between faults',1,float],
                  ['elevation_scalefactor',None,
                  'scale factor for standard deviation in elevation of fault surfaces',1,float],
                  ['fractal_dimension',None,
                  'fractal dimension of fault surfaces, recommended values in range (2.0,2.5)',
                  1,float],
                  ['fault_separation',None,'amount to separate faults by, in metres','*',float],
                  ['fault_edges','fe','indices of fault edges in x,y,z directions '\
                                      'xmin xmax ymin ymax zmin zmax',6,int],
                  ['aperture_type',None,'type of aperture, random or constant',1,str],
                  ['workdir','wd','working directory',1,str],
                  ['outfile','o','output file name',1,str],
                  ['solve_properties','sp','which property to solve, current, fluid or currentfluid (default)',1,str],
                  ['solve_direction','sd','which direction to solve, x, y, z or a combination, e.g. xyz (default), xy, xz, y, etc',1,str],
                  ['solve_method','sm','solver method, direct or iterative (relaxation)',1,str],
                  ['vsurf','vs','voltage at top of volume for modelling',1,float],
                  ['vbase','vb','voltage at top of volume for modelling',1,float],
                  ['psurf','ps','pressure at top of volume for modelling',1,float],
                  ['pbase','pb','pressure at top of volume for modelling',1,float],
                  ['tolerance','tol','tolerance for the iterative solver',1,float],
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

    loop_parameters = {'fault_separation':[0.0]}
    # initialise fixed parameters, giving a default for workdir
    fixed_parameters = {'workdir':os.getcwd()}   
    repeats = [0]
    
    for at in args._get_kwargs():
        if at[1] is not None:
            if at[0] == 'fault_edges':
                nf = len(at[1])
                value = np.reshape(at[1],(nf,3,2))
            else:
                value = at[1]
            if at[0] == 'repeats':
                repeats = at[1][0]
            elif at[0] in ['permeability_matrix','resistivity_matrix','resistivity_fluid']:
                loop_parameters[at[0]] = value
            elif type(value) == list:
                if len(value) > 0:
                    if ((len(value) == 1) and (at[0]!='fault_separation')):
                        fixed_parameters[at[0]] = value[0]
                    elif at[0] in ['ncells','cellsize']:
                        fixed_parameters[at[0]] = value
                    else:
                        loop_parameters[at[0]] = value

            else:
                fixed_parameters[at[0]] = value
                

    print "loop_parameters",loop_parameters
    return fixed_parameters, loop_parameters, repeats


def initialise_inputs(fixed_parameters, loop_parameters, repeats, rank, size):
    """
    make a list of run parameters
    the list is 2 dimensional, with length of first dimension given by number 
    of repeats and resistivity and permeability permutations, length of second
    dimension given by number of fault separation values.
    
    """

    for param,default in [['solve_properties','currentfluid'],
                          ['solve_direction','xyz'],
                          ['solve_method','direct'],
                          ['tolerance',1e-6],
                          ['permeability_matrix',1e-18],
                          ['resistivity_matrix',1e3],
                          ['resistivity_fluid',1e-1]]:
                              if param not in fixed_parameters.keys():
                                  fixed_parameters[param] = default
    
    input_list = []
    solveproperties = []
    for prop in ['current','fluid']:
        if prop in fixed_parameters['solve_properties']:
            solveproperties.append(prop)
    
    propvals = []
    for key in ['permeability_matrix','resistivity_matrix','resistivity_fluid']:
        if key in loop_parameters.keys():
            propvals.append(loop_parameters[key])
        else:
            propvals.append([fixed_parameters[key]])
            
    kmvals,rmvals,rfvals = propvals
    
    if not np.iterable(repeats):
        repeats = range(repeats)

    for r in repeats:
        # define names for fault edge and fault surface arrays to be saved
        fename,fsname = 'fault_edges%02i.npy'%r, 'fault_surfaces%02i.npy'%r
        input_dict = {}
        input_dict.update(fixed_parameters)
        if 'solve_direction' in input_dict.keys():
            solvedirections = input_dict['solve_direction']
        else:
            solvedirections = 'xyz'
        input_dict['repeat'] = r
        # create a rock volume and get the fault surfaces and fault edges
        ro = rn.Rock_volume(**input_dict)
        # save to a file so they can be accessed by all processors
        np.save(op.join(ro.workdir,fename),ro.fault_edges)
        np.save(op.join(ro.workdir,fsname),ro.fault_dict['fault_surfaces'])
        # set the variables to the file name
        input_dict['fault_edgesname'] = fename
        input_dict['fault_surfacename'] = fsname

        # loop through solve directions  
        print rmvals,rfvals
        for i,sd in enumerate(solvedirections):
                # update input dict so we deal with one solve direction at a time
                input_dict['solve_direction'] = sd
                # loop through current and then fluid
                # solving each separately, to allow each to be done on a separate
                # cpu.
                for prop in solveproperties:
                    input_dict['solve_properties'] = prop
                    if prop == 'fluid':
                        for km in kmvals:
                            input_dict['permeability_matrix'] = km
                            tmp_list = []
                            for fs in loop_parameters['fault_separation']:
                                input_dict['fault_separation'] = fs
                                tmp_list.append(input_dict.copy())
                            input_list.append(tmp_list)
                    elif prop == 'current':
                        
                        for rm,rf in itertools.product(rmvals,rfvals):
                            input_dict['resistivity_matrix'] = rm
                            input_dict['resistivity_fluid'] = rf
                            tmp_list = []
                            for fs in loop_parameters['fault_separation']:
                                input_dict['fault_separation'] = fs
                                #print "input_dict",input_dict
                                tmp_list.append(input_dict.copy())
                                #print fs
                            input_list.append(tmp_list)
    #print "input_list",input_list                    
    return input_list
    

def divide_inputs(work_to_do,size):
    """
    divide list of inputs into chunks to send to each processor. Need to split
    them up so that (a) get same repeat on same processor if possible,
    and (b) relatively evenly distribute runs so that we get a few at the 
    percolation threshold and a few away from the percolation threshold on each
    processor
    
    """
    repeats = len(work_to_do)
    procgroups = [[] for _ in range(repeats)]
        
    chunks = [[] for _ in range(size)]
    if size >= repeats:
        print "size > repeats"
        for i in range(size):
            procgroups[i%repeats].append(i)
        for i,pg in enumerate(procgroups):
            # number of processors in the group
            npg = len(pg)
            # split work up amongst the processors we have in the group
            for wi,ww in enumerate(work_to_do[i]):
                chunks[(wi%npg)*repeats+i].append(ww)
    else:
        wi = 0
        skip = len(work_to_do)*len(work_to_do[0])/size + 1
        for wlst in work_to_do:
            for ww in wlst:
                chunks[wi/skip].append(ww)
                
                wi += 1
    
    # reassign some parts of the chunks to even up lengths
    lengths = [len(_) for _ in chunks]
    while max(lengths) - min(lengths) > 1:
        minl,maxl = min(lengths),max(lengths)
        for i in range(size):
            # find any max lengths
            if lengths[i] == maxl:
                for ii in range(size):
                    # find the next min length and append the max length
                    if lengths[ii] == minl:
                        chunks[ii].append(chunks[i][-1])
                        chunks[i] = chunks[i][:-1]
                        lengths = [len(_) for _ in chunks]
                        break

    return chunks

def get_boundary_conditions(ro,input_dict,sdno,solveproperty):
    # create default boundary conditions
    bcs = [0.,ro.ncells[sdno]*ro.cellsize[sdno]*50.]

    if solveproperty ==  'current':
        prefix = 'v'
    else:
        prefix = 'p'


    for i,pm in enumerate([prefix+'surf',prefix+'base']):
        if pm in input_dict.keys():    
            bcs[i] = input_dict[pm]

    return bcs
    
def write_output(ro, outfilename, newfile, repeatno, rank, runno, direction):
    
    # make a list containing the variable names to store
    # start with variables with three directions (x,y and z)
    variablekeys = ['aperture_mean' + dd for dd in 'xyz']
    variablekeys += ['contact_area' + dd for dd in ['yz','xz','xy']]
    variablekeys += ['conductive_fraction']
    variablekeys += [param+dd for param in ['permeability','resistivity'] for dd in 'xyz']

    # add single-valued variables
    variablekeys += ['fault_separation','repeat','rank','run_no']
    if hasattr(ro,'permeability_bulk'):
        ro.permeability_bulk[ro.permeability_bulk==0] = np.nan
    # output line
#    print ro.aperture_mean[dno],ro.contact_area[dno],\
#                            ro.permeability[dno],ro.resistivity[dno],\
#                            ro.resistivity_matrix,ro.resistivity_fluid,\
#                            ro.permeability_matrix,ro.fault_dict['fault_separation'],\
#                            repeatno,rank,runno
    output_line = np.hstack([ro.aperture_mean,ro.contact_area,
                             [ro.conductive_fraction],
                            ro.permeability_bulk,ro.resistivity_bulk,
                            np.median(ro.fault_dict['fault_separation']),
                            repeatno,rank,runno])
        
    # create a dictionary containing fixed variables
    fixeddict = {}
    for param in ['cellsize','ncells','pconnection']:
        fixeddict[param] = ' '.join([str(val) for val in getattr(ro,param)])
    for param in ['workdir','fluid_viscosity','fault_assignment','offset',
                  'fractal_dimension','faultlength_max','faultlength_min',
                  'alpha','a','mismatch_wavelength_cutoff','aperture_type',
                  'resistivity_matrix','resistivity_fluid','permeability_matrix']:
        if param in ro.fault_dict.keys():
            fixeddict[param] = ro.fault_dict[param]
        else:
            fixeddict[param] = getattr(ro,param)
        
    # write to file. If newfile flag is True, then create header and make a new file
    # otherwise append to existing file
    if newfile:
        with open(outfilename, 'wb') as outfile:
            header = '# suite of resistor network simulations\n'
            
            header += '### fixed parameters ###\n'
            header += '# '+'\n# '.join([' '.join([key,str(fixeddict[key])]) for key in fixeddict.keys()])+'\n'
            header += '### variable parameters ###\n'
            header += '# '+' '.join(variablekeys)
            outfile.write(header)
            outfile.write('\n'+' '.join(['%.3e'%oo for oo in output_line]))
    else:
        with open(outfilename, 'ab') as outfile:
            outfile.write('\n'+' '.join(['%.3e'%oo for oo in output_line]))


def gather_outputs(outputs_gathered, wd, outfile, delete_originals=True) :
    """
    gathers all the outputs written to individual files for each rank, to a 
    master file.
    
    
    """
    outfn = outputs_gathered[0]
    outarray = np.loadtxt(outfn)
    with open(outfn) as outfile0:
        line = outfile0.readline()
        header = ''
    
        while line[0] == '#':
            header += line
            line = outfile0.readline()
    header = header.strip()
    count = 0
    for outfn in outputs_gathered:
        if count > 0:
            try:
                outarray = np.vstack([outarray,np.loadtxt(outfn)])
            except IOError:
                print "Failed to find file {}, skipping and moving to the next file".format(outfn)
        count += 1

    np.savetxt(op.join(wd,outfile),outarray,header=header,fmt='%.3e',comments='')

    outarray[np.isinf(outarray)] = np.nan

    # get rid of nans
    count = 0
    for r in np.unique(outarray[:,-3]):
        for fs in np.unique(outarray[:,-4]):
            ind = np.where(np.all([outarray[:,-3]==r,outarray[:,-4]==fs],axis=0))[0]
            if len(ind) > 0:
                line = np.nanmax(outarray[ind],axis=0)
                if count == 0:
                    outarray2 = line.copy()
                else:
                    outarray2 = np.vstack([outarray2,line])
                count += 1
        if count == 1:
            outarray2 = np.array([outarray2])

    np.savetxt(op.join(wd,outfile[:-4]+'f.dat'),outarray2,header=header,fmt='%.3e',comments='')

    if delete_originals:
        time.sleep(5)
        for outfn in outputs_gathered:
            if outfile not in outfn:
                os.remove(outfn)    

def run(list_of_inputs,rank,arraydir,outfilename,save_array=True,array_savepath=None):
    #print "list_of_inputs",list_of_inputs
    
    ofb = op.basename(outfilename)
    ofp = op.dirname(outfilename)
    if '.' in ofb:
        di = ofb.index('.')
        outfilename = op.join(ofp, ofb[:di] + str(rank) + ofb[di:])
    else:
        outfilename = outfilename + str(rank)   
       
    newfile = True 
    runno=1
    print "running rock volumes on rank {}".format(rank)
    # stagger the starts to hopefully prevent memory crash
    if 'ncells' in list_of_inputs[0].keys():
        if np.amax(list_of_inputs[0]['ncells']) > 50:
            time.sleep(rank%16*30)
    for ii,input_dict in enumerate(list_of_inputs):
        
        if array_savepath == None:
            array_savepath = input_dict['workdir']

        input_dict['fault_surfaces'] = np.load(op.join(input_dict['workdir'],input_dict['fault_surfacename']))
        input_dict['fault_edges'] = np.load(op.join(input_dict['workdir'],input_dict['fault_edgesname']))
        input_dict['fault_assignment'] = 'list'

        rno = input_dict['repeat']
        
        # determine whether this is a new volume
        newvol = True
        if input_dict['solve_method'] == 'relaxation':
            if ii > 0:
                # check if the same rock volume (as determined by repeat) as previous
                if rno == input_dict['repeat']:
                    newvol = False
        # define rno, for next loop
        rno = input_dict['repeat']
        # determine solve direction (integer)
        sd= input_dict['solve_direction']
        sdno = list('xyz').index(sd)
        
        ro = rn.Rock_volume(**input_dict)
        
        if save_array:
            if ((rno == 0) and (sd=='z')):
                np.save(op.join(array_savepath,'aperture0_fs%.1e.npy'%input_dict['fault_separation']),ro.aperture)
                np.save(op.join(array_savepath,'faultarray0.npy'),ro.fault_array)
       
 
        Vstart = None
        
        if input_dict['solve_method'] == 'relaxation':
            if not newvol:
                if input_dict['solve_properties'] == 'current':
                    Vstart = ro.voltage[:,:,:,sd].copy().transpose(1,0,2)
                elif input_dict['solve_properties'] == 'fluid':
                    Vstart = ro.pressure[:,:,:,sd].copy().transpose(1,0,2)
                
        Vsurf,Vbase = get_boundary_conditions(ro,input_dict,sdno,input_dict['solve_properties'])
                
        t0 = time.time() 
        if input_dict['solve_method'] == 'direct':
            ro.solve_resistor_network2(Vstart=Vstart,Vsurf=Vsurf,Vbase=Vbase,
                                      method='direct')
        elif input_dict['solve_method'] == 'relaxation':
            ro.solve_resistor_network2(Vstart=Vstart,Vsurf=Vsurf,Vbase=Vbase,
                                      method='relaxation',itstep=100,
                                      tol=input_dict['tolerance'])
        ro.compute_conductive_fraction()
        print 'time to solve {} using {} method on rank {}, {} s'.format(input_dict['solve_properties'],input_dict['solve_method'],rank,time.time()-t0)

        if save_array:
            if ((rno == 0) and (sd == 'z')):
                if 'current' in input_dict['solve_properties']:
                    np.save(op.join(array_savepath,'current0z_fs%.1e.npy'%input_dict['fault_separation']),ro.current)
                elif 'fluid' in input_dict['solve_properties']:
                    np.save(op.join(array_savepath,'flowrate0z_fs%.1e.npy'%input_dict['fault_separation']),ro.flowrate)


        write_output(ro, outfilename, newfile, rno, rank, runno, input_dict['solve_direction'])
        runno += 1
        newfile = False
    
    return outfilename
 

        
def setup_and_run_suite(arguments, argument_names):
    """
    set up and run a suite of runs in parallel using mpi4py
    """
    try:
        from mpi4py import MPI
        mpi_import = True
        # sort out rank and size info
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        name = MPI.Get_processor_name()
    except:
        mpi_import = False
        size=1
        rank=0
        name='Fred'
    
    print 'Hello! My name is {}. I am process {} of {}'.format(name,rank,size)
   
    # get inputs from the command line
    fixed_parameters, loop_parameters, repeats = read_arguments(arguments, argument_names)
    
    # get workdir
    if 'workdir' in fixed_parameters.keys():
        wd = fixed_parameters['workdir']
    else:
        wd = './model_runs'
    # define a subdirectory to store arrays
    wd2 = os.path.join(wd,'arrays')

    if rank == 0:
        list_of_inputs = initialise_inputs(fixed_parameters, loop_parameters, repeats, rank, size)
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
        # wait for rank 1 to generate folder
        while not os.path.exists(wd2):
            time.sleep(1)
            
            
    # initialise outfile name
    if 'outfile' in fixed_parameters.keys():
        outfile = fixed_parameters['outfile']
    else:
        outfile = 'outputs.dat'
        
    #print "sending jobs out, rank {}".format(rank)
    if mpi_import:
        inputs_sent = comm.scatter(inputs,root=0)
        #print "inputs have been sent", len(inputs_sent), "rank", rank
    else:
        inputs_sent = inputs[0]
    outfilenames = run(inputs_sent,
                       rank,
                       wd2,
                       op.join(wd,outfile),
                       array_savepath=wd2)

    if mpi_import:
        outputs_gathered = comm.gather(outfilenames,root=0)
    else:
        outputs_gathered = [outfilenames]
    
    if rank == 0:
        gather_outputs(outputs_gathered, wd, outfile)

                   
if __name__ == "__main__":
    setup_and_run_suite(sys.argv,argument_names)
