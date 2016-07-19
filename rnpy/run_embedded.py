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
import copy

argument_names = [['splitn','n','number of subvolumes in x,y and z direction',3,int],
                  ['subvolume_size','sn','number of cells in each subvolume (3 integers for size in x, y and z directions)',3,int],
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
                  ['repeats','r','how many times to repeat each permutation',1,int],
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
    
    # loop through command line arguments and sort according to type
    for at in args._get_kwargs():
        if at[1] is not None:
            # reshape fault edges to correct shape they are provided as an argument
            if at[0] == 'fault_edges':
                nf = len(at[1])
                value = np.reshape(at[1],(nf,3,2))
            else:
                value = at[1]
            # repeats doesn't go into input dict
            if at[0] == 'repeats':
                repeats = at[1][0]
            # except for ncells, cellsize and subvolume size, all list arguments
            # are loop parameters
            elif type(value) == list:
                if len(value) > 0:
                    if len(value) == 1:
                        fixed_parameters[at[0]] = value[0]
                    elif at[0] in ['ncells','cellsize','subvolume_size']:
                        fixed_parameters[at[0]] = np.array(value)
                    else:
                        loop_parameters[at[0]] = value
            else:
                fixed_parameters[at[0]] = value
                
    fixed_parameters['ncells'] = (fixed_parameters['subvolume_size'] + 1) * fixed_parameters['splitn']

    return fixed_parameters, loop_parameters, repeats


def initialise_inputs_master(fixed_parameters,loop_parameters,repeats):
    """
    make a list of run parameters
    the list is 2 dimensional, with length of first dimension given by number 
    of repeats and resistivity and permeability permutations, length of second
    dimension given by number of fault separation values.
    
    """
    # set some defaults
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
    
    # only build the array in the master volume if we need it for comparions
    # (large arrays will result in a memory error)
    if fixed_parameters['comparison_arrays'] != 'none':
        fixed_parameters['build_arrays'] = True
    else:
        fixed_parameters['build_arrays'] = False

    for r in range(repeats):
        # define names for fault edge and fault surface arrays to be saved
        fename,fsname = 'fault_edges%02i.npy'%r, 'fault_surfaces%02i.npy'%r
        input_dict = {}
        input_dict.update(fixed_parameters)
        # set id - will become an attribute of the rock volume
        input_dict['id'] = r
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
        #print "inputs to comparison volume",inputs 
        #print "directions"      
        kbulk1,rbulk1 = np.ones(4)*np.nan,np.ones(4)*np.nan

        for att,br,sp,bulk,prop in [['resistivity',boundary_res,'current',rbulk1,'resistivity'],['hydraulic_resistance',boundary_hydres,'fluid',kbulk1,'permeability']]:
            if sp in properties:
                arrtoset = getattr(rom,att)        
                inputs['solve_properties'] = sp
                # x direction array
                if 'x' in directions:
                    inputs['solve_direction'] = 'x'
                    ncells = nc.copy()
                    ncells[0] -= n[0]
                    romx = rn.Rock_volume(ncells=ncells,**inputs)
                    arr = getattr(romx,att)
                    arr[1:,1:,1:-1,0] = arrtoset[1:,1:,1:-n[0]-1,0]
                    arr[1:,1:-1,1:,1] = arrtoset[1:,1:-1,1:-n[0],1]
                    arr[1:-1,1:,1:,2] = arrtoset[1:-1,1:,1:-n[0],2]
                    arr[:,-1,:,0] = boundary_res
                    arr[-1,:,:,0] = boundary_res
                    setattr(romx,att,rna.add_nulls(arr))
                    romx.solve_resistor_network2()
                    factor = (nc[1]*nc[2])/((nc[1]+1.)*(nc[2]+1.))
                    if att == 'resistivity':
                        bulk[0] = romx.resistivity_bulk[0]*factor
                    else:
                        bulk[0] = romx.permeability_bulk[0]/factor
                    #print "x direction",sp,bulk,prop
        
                # y direction array
                if 'y' in directions:
                    inputs['solve_direction'] = 'y'
                    ncells = nc.copy()
                    ncells[1] -= n[1]
                    romy = rn.Rock_volume(ncells=ncells,**inputs)
                    arr = getattr(romy,att)
                    arr[1:,1:,1:-1,0] = arrtoset[1:,1:-n[1],1:-1,0]
                    arr[1:,1:-1,1:,1] = arrtoset[1:,1:-n[1]-1,1:,1]
                    arr[1:-1,1:,1:,2] = arrtoset[1:-1,1:-n[1],1:,2]
                    arr[-1,:,:,1] = boundary_res
                    arr[:,:,-1,1] = boundary_res
                    setattr(romy,att,rna.add_nulls(arr))
                    romy.solve_resistor_network2()
                    factor = (nc[0]*nc[2])/((nc[0]+1.)*(nc[2]+1.))
                    if att == 'resistivity':
                        bulk[1] = romy.resistivity_bulk[1]*factor
                    else:
                        bulk[1] = romy.permeability_bulk[1]/factor
                    #print "y direction",sp,bulk,prop
                
                # z direction array
                if 'z' in directions:
                    inputs['solve_direction'] = 'z'
                    ncells = nc.copy()
                    ncells[2] -= n[2]
                    romz = rn.Rock_volume(ncells=ncells,**inputs)
                    arr = getattr(romz,att)
                    arr[1:,1:,1:-1,0] = arrtoset[1:-n[1],1:,1:-1,0]
                    arr[1:,1:-1,1:,1] = arrtoset[1:-n[1],1:-1,1:,1]
                    arr[1:-1,1:,1:,2] = arrtoset[1:-n[1]-1,1:,1:,2]
                    arr[:,-1,:,2] = boundary_res
                    arr[:,:,-1,2] = boundary_res
                    setattr(romz,att,rna.add_nulls(arr))
                    romz.solve_resistor_network2()
                    factor = (nc[0]*nc[1])/((nc[0]+1.)*(nc[1]+1.))
                    if att == 'resistivity':
                        bulk[2] = romz.resistivity_bulk[2]*factor
                    else:
                        bulk[2] = romz.permeability_bulk[2]/factor
                    #print "z direction",sp,bulk,prop
        
        rbulk1[-1] = rom.id
        kbulk1[-1] = rom.id

    	rbulk.append(rbulk1)
        kbulk.append(kbulk1)

    rbulk = np.array(rbulk)
    kbulk = np.array(kbulk)
    #print "rbulk,kbulk",rbulk,kbulk

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
    # rename inputs to make them shorter and easier to deal with
    n = np.array(subvolume_size) + 1
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
    for fi in np.where(np.all([np.any([np.all([(sx+1)*n[0] + buf >= uvw0[:,0],uvw0[:,0] > max(sx*n[0] - buf,0)],axis=0), 
                                      np.all([max(sx*n[0] - buf,0) < uvw1[:,0],uvw1[:,0] <= (sx+1)*n[0] + buf],axis=0), 
                                      np.all([uvw0[:,0] <= max(sx*n[0] - buf,0),uvw1[:,0] > (sx+1)*n[0] + buf],axis=0)],axis=0),
                               np.any([np.all([(sy+1)*n[1] + buf >= uvw0[:,1],uvw0[:,1] > max(sy*n[1] - buf,0)],axis=0),
                                      np.all([max(sy*n[1] - buf,0) < uvw1[:,1],uvw1[:,1] <= (sy+1)*n[1] + buf],axis=0),
                                      np.all([uvw0[:,1] <= max(sy*n[1] - buf,0),uvw1[:,1] > (sy+1)*n[1] + buf],axis=0)],axis=0),
                               np.any([np.all([(sz+1)*n[2] + buf >= uvw0[:,2],uvw0[:,2] > max(sz*n[2] - buf,0)],axis=0),
                                       np.all([max(sz*n[2] - buf,0) < uvw1[:,2],uvw1[:,2] <= (sz+1)*n[2] + buf],axis=0),
                                       np.all([uvw0[:,2] <= max(sz*n[2] - buf,0),uvw1[:,2] > (sz+1)*n[2] + buf],axis=0)],axis=0)],axis=0))[0]:

        # local fault extents
        lfe = faultedges[fi] - np.array([sx*n[0],sy*n[1],sz*n[2]])
        lfe[lfe < 1 - buf] = 1 - buf
        for i in range(3):
            lfe[:,:,i][lfe[:,:,i] > n[i] + buf] = n[i] + buf
        #print "fi,lfe",fi,lfe
        local_fault_edges.append(lfe)
        
        # direction perpendicular to fault
        dperp = list(np.amax(faultedges[fi],axis=(0,1)) - np.amin(faultedges[fi],axis=(0,1))).index(0)
        
        # minimum and maximum indices for aperture to cut it if it extends over multiple sub volumes
        ap0 = np.array([sx*n[0],sy*n[1],sz*n[2]]) - buf + 1 - np.amin(faultedges[fi],axis=(0,1))
        ap0[ap0 < 0] = 0.
        ap1 = ap0 + np.amax(lfe,axis=(0,1)) - np.amin(lfe,axis=(0,1))
        
        ap0[dperp] = 0.
        ap1[dperp] = 2.

        # append aperture, aperture corrected for fluid flow, aperture corrected for current
        ap.append(aperturelist[0][fi][ap0[2]:ap1[2]+1,ap0[1]:ap1[1]+1,ap0[0]:ap1[0]+1])
        apf.append(aperturelist[1][fi][ap0[2]:ap1[2]+1,ap0[1]:ap1[1]+1,ap0[0]:ap1[0]+1])
        apc.append(aperturelist[2][fi][ap0[2]:ap1[2]+1,ap0[1]:ap1[1]+1,ap0[0]:ap1[0]+1])

    aperture_list = [ap,apf,apc]

    return local_fault_edges,aperture_list


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

    for sz in range(splitn[2]):
        for sy in range(splitn[1]):
            for sx in range(splitn[0]):
                # initialise dict with default parameters
                localidict = {}
                localidict.update(inputdict)
                # get aperture and faults
                localfaults,localap = segment_faults(faultedge_list,aperture_list,subvolume_size,[sx,sy,sz],buf=buf)
                localidict['aperture_list'] = localap
                localidict['fault_edges'] = np.array(localfaults)
                print "localfaults",localidict['fault_edges']
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
    ro_masterlist = []
    ro_masterlist_sorted = []
    #print "outputs_gathered",outputs_gathered
    count = 0
    for line in outputs_gathered:
        if len(line) == 5:
            rbulk,kbulk,indices,ridlist,rolist = line
            ro_masterlist += rolist
        else:
            rbulk,kbulk,indices,ridlist = line[:4]
            ro_masterlist = None
        # reshape ridlist so it can be stacked with the other arrays
        ridlist = np.array(ridlist).reshape(len(ridlist),1)
        print "ridlist",ridlist
        outline = np.hstack([rbulk,kbulk,indices,ridlist])
        if count == 0:
            outarray = outline.copy()
        else:
            outarray = np.vstack([outarray,outline])
        count += 1
    
    # now go through and put all entries for each rock volume on one line
    # also sort the rock objects by the same order
    for rid in np.unique(outarray[:,-1]):
        for iz in np.unique(outarray[:,-2]):
            for iy in np.unique(outarray[:,-3]):
                for ix in np.unique(outarray[:,-4]):
                    ind = np.where(np.all(outarray[:,-4:]==np.array([ix,iy,iz,rid]),axis=1))[0]
                    line = np.nanmax(outarray[ind],axis=0)
                    roxyz = np.zeros(3,dtype=object)
                    for ii in range(3):
                        for iii in ind:
                            if np.any(np.isfinite(outarray[iii][np.array([ii,ii+3])])):
                                roxyz[ii] = ro_masterlist_sorted[iii]
                                break
                        
                    # only append the first instance to the sorted ro list
                    ro_masterlist_sorted.append(roxyz)
                    # add the line to the output array
                    if count == 0:
                        outarray2 = line.copy()
                    else:
                        outarray2 = np.vstack([outarray2,line])
                    count += 1

    if count == 1:
        outarray2 = np.array([outarray2])

    np.savetxt(outfile,outarray2,fmt=['%.3e']*6+['%3i']*4,comments='')
    
    if rolist is None:
        return outarray2
    else:
        return outarray2,ro_masterlist_sorted


def run_subvolumes(input_list,return_objects=False):
    """
    
    """
    ro_list = []
    rlist,klist,indices = [],[],[]
    ridlist = [] 
    
    for input_dict in input_list:
        #print "subvolume input dict",input_dict
        rbulk, kbulk = np.ones(3)*np.nan, np.ones(3)*np.nan
        di = 'xyz'.index(input_dict['solve_direction'])
        ros = rn.Rock_volume(**input_dict)
        ros.solve_resistor_network2()
        rbulk[di] = ros.resistivity_bulk[di]
        kbulk[di] = ros.permeability_bulk[di]
        
        if return_objects:
            ro_list.append(copy.copy(ros))
        
        rlist.append(rbulk.copy())
        klist.append(kbulk.copy())
        ridlist.append(ros.id)
        indices.append(ros.indices)
    
    if return_objects:
        return rlist,klist,indices,ridlist,ro_list
    else:
        return rlist,klist,indices,ridlist



def scatter_run_subvolumes(input_list,size,rank,comm,outfile,return_objects=False):
    """
    initialise and run subvolumes
    
    """
    if rank == 0:
        nn = input_list[0]['subvolume_size']
        directions = input_list[0]['solve_direction']
        #properties = input_list[0]['solve_properties']

        input_list_sep = []
        for idict in input_list:
            directions = idict['solve_direction']
            properties = []
            for attr in ['current','fluid']:
                if attr in idict['solve_properties']:
                    properties.append(attr)
            for sd in directions:
                di = 'xyz'.index(sd)
                ncells = np.array(nn)
                ncells[di] += 1
                idict['ncells'] = ncells
                for sp in properties:
                    idict['solve_direction'] = sd
                    idict['solve_properties'] = sp
                    input_list_sep.append(idict.copy())
    
        input_list_divided = divide_inputs(input_list_sep,size)
    else:
        input_list_divided = None

    inputs_sent = comm.scatter(input_list_divided,root=0)
    bulk_props = run_subvolumes(inputs_sent,return_objects=return_objects)
    outputs_gathered = comm.gather(bulk_props,root=0)
    
    if rank == 0:
        outarray = write_outputs_subvolumes(outputs_gathered, outfile)
    else:
        outarray = None

    return outarray


#def construct_array_from_subvolumes():
def compare_arrays(ro_list,ro_list_seg,indices,subvolume_size):
    """
    """
    
    indicesz = np.unique(outarray[:,-2])
    indicesy = np.unique(outarray[:,-3])
    indicesx = np.unique(outarray[:,-4])
    splitn = [np.amax(ind)+1 for ind in [indicesx,indicesy,indicesz]]
    n = np.array(subvolume_size) + 1
    nn = subvolume_size
    ncellsx,ncellsy,ncellsz = [[nn[2] + 1,nn[1],nn[0]],
                               [nn[2],nn[1] + 1,nn[0]],
                               [nn[2],nn[1],nn[0] + 1]]
    
    compiled_faults = np.zeros((splitn*n[2]+2,splitn*n[1]+2,splitn*n[0]+2,3,3))
    compiled_ap = np.zeros((splitn*n[2]+2,splitn*n[1]+2,splitn*n[0]+2,3,3))
    compiled_res = np.zeros((splitn*n[2]+2,splitn*n[1]+2,splitn*n[0]+2,3))
    compiled_hr = np.zeros((splitn*n[2]+2,splitn*n[1]+2,splitn*n[0]+2,3))    
        
    testfaults,testap,testres,testhr = [],[],[],[]        
        
    count = 0
    for rom in ro_list:
        for rid in np.unique(indices[-1]):
            if rom.id == rid:
                for sz in indicesz:
                    for sy in indicesy:
                        for sx in indicesx:
                            ind = np.where(np.all(indices[:,-4:]==np.array([sx,sy,sz,rid]),axis=1))[0][0]
                            rox,roy,roz = ro_list_seg[ind]
                            compiled_ap[1+sz*n[2]:1+sz*n[2]+ncellsz[2],1+sy*n[1]:1+(sy+1)*n,1+sx*n[0]:1+(sx+1)*n[0],2] = roz.aperture[1:ncellsz[2]+1,1:,1:,2]
                            compiled_ap[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+sy*n[1]+ncellsy[1],1+sx*n[0]:1+(sx+1)*n[0],1] = roy.aperture[1:,1:ncellsy[1]+1,1:,1]
                            compiled_ap[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+(sy+1)*n,1+sx*n[0]:1+sx*n[0]+ncellsx[0],0] = rox.aperture[1:,1:,1:ncellsx[0]+1,0]
                            compiled_faults[1+sz*n[2]:1+sz*n[2]+ncellsz[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+(sx+1)*n[0],2] = roz.fault_array[1:ncellsz[2]+1,1:,1:,2]
                            compiled_faults[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+sy*n[1]+ncellsy[1],1+sx*n[0]:1+(sx+1)*n[0],1] = roy.fault_array[1:,1:ncellsy[1]+1,1:,1]
                            compiled_faults[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+sx*n[0]+ncellsx[0],0] = rox.fault_array[1:,1:,1:ncellsx[0]+1,0]
                            compiled_res[1+sz*n[2]:1+sz*n[2]+ncellsz[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+(sx+1)*n[0],2] = roz.resistivity[1:ncellsz[2]+1,1:,1:,2]
                            compiled_res[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+sy*n[1]+ncellsy[1],1+sx*n[0]:1+(sx+1)*n[0],1] = roy.resistivity[1:,1:ncellsy[1]+1,1:,1]
                            compiled_res[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+sx*n[0]+ncellsx[0],0] = rox.resistivity[1:,1:,1:ncellsx[0]+1,0]
                            compiled_hr[1+sz*n[2]:1+sz*n[2]+ncellsz[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+(sx+1)*n[0],2] = roz.hydraulic_resistance[1:ncellsz[2]+1,1:,1:,2]
                            compiled_hr[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+sy*n[1]+ncellsy[1],1+sx*n[0]:1+(sx+1)*n[0],1] = roy.hydraulic_resistance[1:,1:ncellsy[1]+1,1:,1]
                            compiled_hr[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+sx*n[0]+ncellsx[0],0] = rox.hydraulic_resistance[1:,1:,1:ncellsx[0]+1,0]    

                diff_faults = np.unique(compiled_faults-rom.fault_array)
                diff_faults[np.isnan(diff_faults)] = 0
                diff_ap = np.unique(compiled_ap-rom.aperture)
                diff_ap[np.isnan(diff_ap)] = 0                
                diff_res = np.unique(compiled_res-rom.resistivity)
                diff_res[np.isnan(diff_res)] = 0                
                diff_hr = np.unique(compiled_hr-rom.hydraulic_resistance)
                diff_hr[np.isnan(diff_faults)] = 0                


def build_master(list_of_inputs):
    """
    initialise master rock volumes
    
    """
    # two lists, one with all rock volumes, the other split up by solve direction
    # and solve properties
    ro_list_sep, ro_list = [],[]
    repeat = None
    solve_properties = []
    
    for pp in ['current','fluid']:
        if pp in list_of_inputs[0]['solve_properties']:
            solve_properties.append(pp)
        
    
    for input_dict in list_of_inputs:
        # only initialise new faults if we are moving to a new volume
        if repeat != input_dict['id']:
            input_dict['fault_edges'] = None
            input_dict['fault_surfaces'] = None
            repeat = input_dict['id']
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
                ro_list_sep.append(copy.copy(ro))
                
    return ro_list,ro_list_sep


def write_outputs_comparison(outputs_gathered, outfile) :
    """
    gathers all the outputs written to individual files for each rank, to a 
    master file.
    
    
    """
  
    
    # first gather all outputs into a master array
    count = 0
    for kbulk,rbulk in outputs_gathered:
        line = np.hstack([kbulk[:,:3],rbulk])
        if count == 0:
            outarray = line.copy()
        else:
            outarray = np.vstack([outarray,line])
        count += 1
 
    # now go through and put all entries for each rock volume on one line
    count = 0
    for r in np.unique(outarray[:,-1]):
        line = np.nanmax(outarray[outarray[:,-1]==r],axis=0)
        if count == 0:
            outarray2 = line.copy()
        else:
            outarray2 = np.vstack([outarray2,line])
        count += 1
    if count == 1:
        outarray2 = np.array([outarray2])

    print "saving outputs to file {}".format(outfile)
    np.savetxt(outfile,outarray2,fmt='%.3e',comments='')
   



def run_comparison(ro_list,subvolume_size,rank,size,comm,outfile):
    """
    run comparison arrays to compare to segmented volume (different volumes for
    the x, y and z directions)
    
    """
    
    if comm is not None:
        print "setting up comparison volumes in parallel"
        if rank == 0:
            rolist_divided = divide_inputs(ro_list,size)
        else:
            rolist_divided = None
        inputs_sent = comm.scatter(rolist_divided,root=0)
        bulk_props = calculate_comparison_volumes(inputs_sent,subvolume_size)
        outputs_gathered = comm.gather(bulk_props,root=0)
        
        if rank == 0:
            print "writing comparison outputs"
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
    subvolume_size = list_of_inputs_master[0]['subvolume_size']
    if rank == 0:
        ro_list, ro_list_sep = build_master(list_of_inputs_master)
    else:
        ro_list, ro_list_sep = None,None
    # run comparison
    if 'bulk' in fixed_parameters['comparison_arrays']:
        run_comparison(ro_list_sep,subvolume_size,rank,size,comm,op.join(wd,'comparison_'+outfile))

    
    # create subvolumes
    if rank == 0:
        subvolume_input_list = []
        for rr in range(len(ro_list)):
            ro = ro_list[rr]
            input_dict = list_of_inputs_master[rr].copy()
            # have to solve 3 directions in subvolumes regardless of directions being solved in master
            input_dict['solve_direction'] = 'xyz'
            faultedge_list = ro.fault_edges
            aperture_list = ro.fault_dict['aperture_list']
            subvolume_input_list += initialise_inputs_subvolumes(faultedge_list,
                                                             aperture_list,
                                                             input_dict['subvolume_size'],
                                                             input_dict['splitn'],
                                                             inputdict=input_dict,
                                                             buf=4)
    else:
        subvolume_input_list = None
        
    # determine whether we need the segmented rock volumes for future analysis/comparison
    if 'array' in list_of_inputs_master[0]['comparison_arrays']:
        return_objects = True
    else:
        return_objects = False
    
    # run the subvolumes and return an array, containing results + indices +
    # rock volume ids + (optionally) rock volume objects
    if return_objects:
        outarray,ro_list_seg = scatter_run_subvolumes(subvolume_input_list,
                                                      size,rank,comm,
                                                      op.join(wd,'subvolumes_'+outfile),
                                                      return_objects=True)
        # assemble the individual pieces into master arrays (aperture, 
        # resistivity and hydraulic resistance and compare these to the
        # original ones.
        if rank == 0:
            compare_arrays(ro_list,ro_list_seg,outarray[:,-4:])
    else:
        outarray = scatter_run_subvolumes(subvolume_input_list,
                                          size,rank,comm,
                                          op.join(wd,'subvolumes_'+outfile),
                                          return_objects=False)

    if rank == 0:
        print "outarray",outarray
    
    
    
    
                   
if __name__ == "__main__":
    setup_and_run_segmented_volume(sys.argv,argument_names)

