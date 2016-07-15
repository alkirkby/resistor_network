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
import rnpy.functions.array as rna
import os
import os.path as op
import time
import itertools

argument_names = [['splitn','n','number of subvolumes in x,y and z direction',3,int],
                  ['subvolume_size','sn','number of cells in each subvolume (3 integers for size in x, y and z directions)',3,int]
                  ['cellsize','c','cellsize in x,y and z direction',3,float],
                  ['pconnection','p','probability of a fault in x, y and z direction',3,float],
                  ['resistivity_matrix','rm','',1,float],
                  ['resistivity_fluid','rf','',1,float],
                  ['permeability_matrix','km','',1,float],
                  ['fluid_viscosity','mu','',1,float],
                  ['fault_assignment',None,'how to assign faults, random or list, '\
                                           'if list need to provide fault edges',1,str],
                  ['offset',None,'number of cells offset between fault surfaces',1,float],
                  ['faultlength_max',None,'maximum fault length, if specifying random faults',1,float],
                  ['faultlength_min',None,'minimum fault length, if specifying random faults',1,float],
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
                  ['repeats','r','how many times to repeat each permutation',1,int]
                  ['comparison_arrays','','what sort of comparison arrays to build, bulk, array, bulk_array (both), or none',1,str]]


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
    # initialise fixed parameters, giving some defaults
    fixed_parameters = {'workdir':os.getcwd(),'comparison_arrays':'none',
                        'splitn':np.array([3,3,3]),'subvolume_size':np.array([4,4,4])}
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
            elif type(value) == list:
                if len(value) > 0:
                    if len(value) == 1:
                        fixed_parameters[at[0]] = value[0]
                    elif at[0] in ['ncells','cellsize','subvolume_size']:
                        fixed_parameters[at[0]] = value
                    else:
                        loop_parameters[at[0]] = value
            else:
                fixed_parameters[at[0]] = value
                
    fixed_parameters['ncells'] = (fixed_parameters['subvolume_size'] + 1) * fixed_parameters['splitn']

    print "loop_parameters",loop_parameters
    return fixed_parameters, loop_parameters, repeats


def initialise_inputs_master(fixed_parameters,loop_parameters,repeats):
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
                          ['resistivity_fluid',1e-1],
                          ['update_cellsize_tf',False]]:
                              if param not in fixed_parameters.keys():
                                  fixed_parameters[param] = default
    
    input_list = []

    if fixed_parameters['comparison_arrays'] != 'none':
        fixed_parameters['build_arrays'] = True
    else:
        fixed_parameters['build_arrays'] = False

    for r in range(repeats):
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


        for fs in loop_parameters['fault_separation']:
            input_dict['fault_separation'] = fs
            input_list.append(input_dict.copy())
                  
    return input_list

def object2dict(Object):
    """
    get attributes of an object as a dict
    """    
    
    import inspect    
    
    odict = {}
    
    for key in dir(Object):
        if not key.startswith('__'):
            if not inspect.ismethod(getattr(Object,key)):
                odict[key] = getattr(Object,key)
    
    return odict


def calculate_comparison_volumes(Rock_volume_list,subvolume_size,properties=None,
                             boundary_res = 1e40,boundary_hydres = 1e60,directions=None):
    """
    calculate the resistivity in x y and z direction of a subsetted volume
    (different dimensions for solving in x y and z directions)
    
    """
    kbulk,rbulk= [],[]    
    
    for rom in Rock_volume_list:
        properties = rom.solve_properties
        directions = rom.solve_direction
            
        nc = np.array(rom.ncells)
        n = subvolume_size + 1
        
        inputs = dict(update_cellsize_tf=False, fault_assignment='none',
                      cellsize=rom.cellsize,solve_properties=properties)    
        
        romx = rn.Rock_volume(ncells=[nc-n,nc,nc],solve_direction='x',**inputs)
        romy = rn.Rock_volume(ncells=[nc,nc-n,nc],solve_direction='y',**inputs)
        romz = rn.Rock_volume(ncells=[nc,nc,nc-n],solve_direction='z',**inputs)
        
        for att,br in [['resistivity',boundary_res],['hydraulic_resistance',boundary_hydres]]:
            arrtoset = getattr(rom,att)        
            
            # x direction array
            if 'x' in directions:
                arr = getattr(romx,att)
                arr[1:,1:,1:-1,0] = arrtoset[1:,1:,1:-n-1,0]
                arr[1:,1:-1,1:,1] = arrtoset[1:,1:-1,1:-n,1]
                arr[1:-1,1:,1:,2] = arrtoset[1:-1,1:,1:-n,2]
                arr[:,-1,:,0] = boundary_res
                arr[-1,:,:,0] = boundary_res
                setattr(romx,att,rna.add_nulls(arr))
                romx.solve_resistor_network2()
    
            # y direction array
            if 'y' in directions:
                arr = getattr(romy,att)
                arr[1:,1:,1:-1,0] = arrtoset[1:,1:-n,1:-1,0]
                arr[1:,1:-1,1:,1] = arrtoset[1:,1:-n-1,1:,1]
                arr[1:-1,1:,1:,2] = arrtoset[1:-1,1:-n,1:,2]
                arr[-1,:,:,1] = boundary_res
                arr[:,:,-1,1] = boundary_res
                setattr(romy,att,rna.add_nulls(arr))
                romy.solve_resistor_network2()
            
            # z direction array
            if 'z' in directions:
                arr = getattr(romz,att)
                arr[1:,1:,1:-1,0] = arrtoset[1:-n,1:,1:-1,0]
                arr[1:,1:-1,1:,1] = arrtoset[1:-n,1:-1,1:,1]
                arr[1:-1,1:,1:,2] = arrtoset[1:-n-1,1:,1:,2]
                arr[:,-1,:,2] = boundary_res
                arr[:,:,-1,2] = boundary_res
                setattr(romz,att,rna.add_nulls(arr))
                romz.solve_resistor_network2()
    
            
            rbulk.append(np.array([romx.resistivity_bulk[0],romy.resistivity_bulk[1],romz.resistivity_bulk[2]])*(nc/(nc+1.))**2)
            kbulk.append(np.array([romx.permeability_bulk[0],romy.permeability_bulk[1],romz.permeability_bulk[2]])*((nc+1.)/nc)**2)

        return rbulk, kbulk
        
        
        
def segment_faults(faultedges,aperturelist,indices,subvolume_size,buf=4):
    """
    get fault edges and apertures for a given subvolume within a network and
    trim fault where necessary. Returns local fault edges and apertures

    faultedges = list containing edges from the larger fault volume (shape
                 (nfaults,2,2,3))
    aperturelist = list, apertures from the larger fault volume, length 3. 
                   The first item is the aperture, the second is the fluid-flow
                   geometry corrected aperture and the third is the electrical
                   current geometry corrected aperture.
                   each aperture list contains array/list of shape 
                   (nfaults,dz+2,dv+2,du+2,3,3) where du, dv and dw are the 
                   extents of the fault in x,y and z directions
    indices = list, x,y,z indices of the subvolume of interest, with respect to the
              larger fault volume. For example, the subvolume in the first row,
              second row back, third column would have indices of [0,1,2]
    subvolume_size = int, size of the subvolume, given as nn + 1 where nn is the
                     actual size of the rock volume, as the final dimensions of
                     the rock volume increase by 1. Only cubic subvolumes
                     accepted
    
    """
    aperture_list,faultedge_list = [],[]
    # rename inputs to make them shorter and easier to deal with
    n = subvolume_size + 1
    sx,sy,sz = indices
    
    # minimum and maximum extents of faults in x, y and z directions
    uvw0 = np.amin(faultedges,axis=(1,2))
    uvw1 = np.amax(faultedges,axis=(1,2))
                
    # initialise lists to contain the local fault edges in the subvolume, the
    # aperture, geometry corrected aperture (fluids) and geometry corrected 
    # aperture (current)
    local_fault_edges = []
    ap,apc,apf = [],[],[]
    
    # get fracture indices that fall within the volume of interest (+buffer)
    for fi in np.where(np.all([np.any([np.all([(sx+1)*n + buf >= uvw0[:,0],uvw0[:,0] > max(sx*n - buf,0)],axis=0), 
                                      np.all([max(sx*n - buf,0) < uvw1[:,0],uvw1[:,0] <= (sx+1)*n + buf],axis=0), 
                                      np.all([uvw0[:,0] <= max(sx*n - buf,0),uvw1[:,0] > (sx+1)*n + buf],axis=0)],axis=0),
                               np.any([np.all([(sy+1)*n + buf >= uvw0[:,1],uvw0[:,1] > max(sy*n - buf,0)],axis=0),
                                      np.all([max(sy*n - buf,0) < uvw1[:,1],uvw1[:,1] <= (sy+1)*n + buf],axis=0),
                                      np.all([uvw0[:,1] <= max(sy*n - buf,0),uvw1[:,1] > (sy+1)*n + buf],axis=0)],axis=0),
                               np.any([np.all([(sz+1)*n + buf >= uvw0[:,2],uvw0[:,2] > max(sz*n - buf,0)],axis=0),
                                       np.all([max(sz*n - buf,0) < uvw1[:,2],uvw1[:,2] <= (sz+1)*n + buf],axis=0),
                                       np.all([uvw0[:,2] <= max(sz*n - buf,0),uvw1[:,2] > (sz+1)*n + buf],axis=0)],axis=0)],axis=0))[0]:

        # local fault extents
        lfe = faultedges[fi] - np.array([sx*n,sy*n,sz*n])
        lfe[lfe < 1 - buf] = 1 - buf
        lfe[lfe > n + buf] = n + buf
        local_fault_edges.append(lfe)
        
        # direction perpendicular to fault
        dperp = list(np.amax(faultedges[fi],axis=(0,1)) - np.amin(faultedges[fi],axis=(0,1))).index(0)
        
        # minimum and maximum indices for aperture to cut it if it extends over multiple sub volumes
        ap0 = np.array([sx*n,sy*n,sz*n]) - buf + 1 - np.amin(faultedges[fi],axis=(0,1))
        ap0[ap0 < 0] = 0.
        ap1 = ap0 + np.amax(lfe,axis=(0,1)) - np.amin(lfe,axis=(0,1))
        
        ap0[dperp] = 0.
        ap1[dperp] = 2.

        # append aperture, aperture corrected for fluid flow, aperture corrected for current
        ap.append(aperturelist[0][fi][ap0[2]:ap1[2]+1,ap0[1]:ap1[1]+1,ap0[0]:ap1[0]+1])
        apf.append(aperturelist[1][fi][ap0[2]:ap1[2]+1,ap0[1]:ap1[1]+1,ap0[0]:ap1[0]+1])
        apc.append(aperturelist[2][fi][ap0[2]:ap1[2]+1,ap0[1]:ap1[1]+1,ap0[0]:ap1[0]+1])

    aperture_list.append([ap,apf,apc])
    faultedge_list.append(local_fault_edges)

    return faultedge_list,aperture_list


def initialise_inputs_subvolumes(faultedge_list,aperture_list,subvolume_size,splitn,inputdict={},buf=4):
    """
    divide a large rock volume into subvolumes and prepare a list containing 
    input dictionaries for all the subvolumes.

    """
    input_list = []
    
    inputdict['fault_assignment'] = 'list'
    inputdict['aperture_type'] = 'list'
    inputdict['update_cellsize_tf'] = False
    inputdict['build_arrays'] = True
    inputdict['array_buffer'] = buf

    for sz in range(splitn):
        for sy in range(splitn):
            for sx in range(splitn):
                # initialise dict with default parameters
                localidict = {}
                localidict.update(inputdict)
                # get aperture and faults
                localfaults,localap = segment_faults(faultedge_list,aperture_list,subvolume_size,[sx,sy,sz],buf=buf)
                localidict['aperture_list'] = localap
                localidict['fault_edges'] = np.array(localfaults)
                # store indices to make it easier to put back together
                localidict['indices'] = [sx,sy,sz]
                
                input_list.append(localidict)
                
    return input_list





def divide_inputs(work_to_do,size):
    """
    divide list of inputs into chunks to send to each processor
    
    """
    chunks = [[] for _ in range(size)]
    for i,d in enumerate(work_to_do):
        chunks[i%size].append(d)

    return chunks


def write_outputs_subvolumes(outputs_gathered, outfile):
    """
    """
    
    
    count = 0
    for line in outputs_gathered:
        rbulk,kbulk,indices = outputs_gathered[:3]
        outline = np.hstack([rbulk,kbulk,indices])
        if count == 0:
            outarray = np.array([outline])
        else:
            outarray = np.vstack([outarray,outline])
        count += 1
        
    np.savetxt(outfile,outarray,fmt='%.3e',comments='')


def run_subvolumes(input_list,return_objects=False):
    """
    
    """
    ro_list = []
    rlist,klist,indices = [],[],[]
    
    
    for input_dict in input_list:
        rbulk, kbulk = np.ones(3)*np.nan, np.ones(3)*np.nan
        di = 'xyz'.index(input_dict['solve_direction'])
        ros = rn.Rock_volume(**input_dict)
        ros.solve_resistor_network2()
        rbulk[di] = ros.resistivity_bulk[di]
        kbulk[di] = ros.permeability_bulk[di]
        if return_objects:
            ro_list.append(ros)
        
        rlist.append(rbulk)
        klist.append(kbulk)
        indices.append(ros.indices)
    
    if return_objects:
        return rbulk,kbulk,indices,ro_list
    else:
        return rbulk,kbulk,indices      



def scatter_run_subvolumes(input_list,size,rank,comm,outfile,return_objects=False):
    """
    initialise and run subvolumes
    
    """
    nn = input_list[0]['subvolume_size'] 
    directions = input_list[0]['solve_direction']
    properties = input_list[0]['solve_properties']

    input_list_sep = []
    for idict in input_list:
        directions = idict['solve_direction']
        properties = idict['solve_properties']
        for sd in directions:
            di = 'xyz'.index(sd)
            ncells = np.array([nn,nn,nn])
            ncells[di] += 1
            idict['ncells'] = ncells
            for sp in properties:
                idict['solve_direction'] = sd
                idict['solve_properties'] = sp
                input_list_sep.append(idict.copy())
    
    input_list_divided = divide_inputs(input_list_sep,size)
    inputs_sent = comm.scatter(input_list_divided,root=0)
    bulk_props = run_subvolumes(inputs_sent,return_objects=return_objects)
    outputs_gathered = comm.gather(bulk_props,root=0)
    
    if rank == 0:
        write_outputs_subvolumes(outputs_gathered, outfile)   

    return outputs_gathered


def construct_array_from_subvolumes():



def build_master(list_of_inputs):
    """
    initialise master rock volumes
    
    """
    ro_list_sep, ro_list = [],[]
    repeat = None
    solve_properties = []
    for pp in ['current','fluid']:
        if pp in list_of_inputs[0]['solve_properties']:
            solve_properties.append(pp)
        
    
    for input_dict in list_of_inputs:
        if repeat != input_dict['repeat']:
            input_dict['fault_edges'] = None
            input_dict['fault_surfaces'] = None
            repeat = input_dict['repeat']
        else:
            input_dict['fault_edges'] = ro.fault_edges
            input_dict['fault_surfaces'] = ro.fault_dict['fault_surfaces']

        ro = rn.Rock_volume(**input_dict)
        ro_list.append(ro)
        
        solve_direction = ro.solve_direction
        for sp in solve_properties:
            for sd in solve_direction:
                ro.solve_direction = sd
                ro.solve_properties = sp
                ro_list_sep.append(ro)
    
    return ro_list,ro_list_sep


def write_outputs_comparison(outputs_gathered, outfile) :
    """
    gathers all the outputs written to individual files for each rank, to a 
    master file.
    
    
    """
    
    print "len(outputs_gathered)",len(outputs_gathered)
    print "len(outputs_gathered[0])",len(outputs_gathered[0])    
    
    count = 0
    for kbulk,rbulk in outputs_gathered:
        line = np.hstack([kbulk,rbulk])
        if count == 0:
            outarray = np.array([line])
        else:
            outarray = np.vstack([outarray,line])
        count += 1

    np.savetxt(outfile,outarray,fmt='%.3e',comments='')
   



def run_comparison(ro_list,subvolume_size,rank,size,comm,outfile):
    
    
    
    if comm is not None:
        rolist_divided = divide_inputs(ro_list,size)
        inputs_sent = comm.scatter(rolist_divided,root=0)
        bulk_props = calculate_comparison_volumes(inputs_sent,subvolume_size)
        outputs_gathered = comm.gather(bulk_props,root=0)
        
        if rank == 0:
            write_outputs_comparison(outputs_gathered, outfile)

    else:
        for ro in ro_list:
            rbulk,kbulk = calculate_comparison_volumes(ro,subvolume_size)


def setup_and_run_segmented_volume(arguments, argument_names):
    """
    set up and run a suite of runs in parallel using mpi4py
    """
    use_mpi = True
    try:
        from mpi4py import MPI
    except ImportError:
        print "Cannot import mpi4py, running in serial"
        use_mpi = False
        
        
    if use_mpi:
        # sort out rank and size info
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        name = MPI.Get_processor_name()
        print 'Hello! My name is {}. I am process {} of {}'.format(name,rank,size)
    else:
        size,rank,comm = 1,0,None

    
    # get inputs from the command line
    fixed_parameters, loop_parameters, repeats = read_arguments(arguments, argument_names)
    
    # get workdir
    if 'workdir' in fixed_parameters.keys():
        wd = fixed_parameters['workdir']
    else:
        wd = './model_runs'
    # define a subdirectory to store arrays
    wd2 = os.path.join(wd,'arrays')
    
    # create inputs for master rock volumes
    list_of_inputs_master = initialise_inputs_master(fixed_parameters, loop_parameters, repeats)

    time.sleep(10)
    
    if rank == 0:        
        # make working directories
        if not os.path.exists(wd):
            os.mkdir(wd)
        wd = os.path.abspath(wd)
        if not os.path.exists(wd2):
            os.mkdir(wd2)
    else:
        
        list_of_inputs_master = None
        # wait for rank 1 to generate folder
        while not os.path.exists(wd2):
            time.sleep(1)

    # initialise outfile name
    if 'outfile' in fixed_parameters.keys():
        outfile = fixed_parameters['outfile']
    else:
        outfile = 'outputs.dat'

    # get list of master rock volumes. Two lists. The first has all the rock volumes
    # the second has all the solve properties and directions separated out for
    # parallel processing
    ro_list, ro_list_sep = build_master(list_of_inputs_master)
    # run comparison
    if 'bulk' in fixed_parameters['comparison_arrays']:
        run_comparison(ro_list_sep,rank,size,comm,op.join(wd,'comparison_'+outfile))

    
    # create subvolumes
    subvolume_input_list = []
    for rr in range(len(ro_list)):
        ro = ro_list[rr]
        input_dict = list_of_inputs_master[rr]
        faultedge_list = ro.fault_edges
        aperture_list = ro.fault_dict['aperture_list']
        subvolume_input_list += initialise_inputs_subvolumes(faultedge_list,
                                                             aperture_list,
                                                             input_dict['subvolume_size'],
                                                             input_dict['splitn'],
                                                             inputdict=input_dict,
                                                             buf=4)
    outputs_gathered = scatter_run_subvolumes(subvolume_input_list,
                                              size,rank,comm,
                                              outfile,
                                              return_objects=False)
    
    
    
    
                   
if __name__ == "__main__":
    setup_and_run_segmented_volume(sys.argv,argument_names)

