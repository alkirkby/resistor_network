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
import rnpy.functions.assignproperties as rnap
import rnpy.functions.array as rna
import os
import os.path as op
import time
import itertools
import copy




def read_arguments(arguments):
    """
    takes list of command line arguments obtained by passing in sys.argv
    reads these and updates attributes accordingly
    """
    
    import argparse
    

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
                    elif at[0] in ['ncells','cellsize','subvolume_size','splitn']:
                        fixed_parameters[at[0]] = np.array(value)
                    else:
                        loop_parameters[at[0]] = value
            else:
                fixed_parameters[at[0]] = value
                
    fixed_parameters['ncells'] = (fixed_parameters['subvolume_size'] + 1) * fixed_parameters['splitn']

    return fixed_parameters, loop_parameters, repeats


def initialise_inputs_master(fixed_parameters,loop_parameters,repeats,savepath,return_objects=False):
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
    if return_objects:
        ro_list = []
    
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
        input_dict['fault_assignment'] = 'random'
        input_dict['aperture_type'] = 'random'

        count = 0
        for fs in loop_parameters['fault_separation']:
            if count > 0:
                input_dict['fault_assignment'] = 'list'
                input_dict['fault_surfaces'] = ro.fault_dict['fault_surfaces']
                input_dict['fault_edges'] = ro.fault_edges
            count += 1
            input_dict['fault_separation'] = fs
            aplistname = op.join(savepath,'aperture_list_%1i_fs%.1e.npy'%(r,np.median(input_dict['fault_separation'])))
            ro = rn.Rock_volume(**input_dict)
            # save aperture list to a file
            np.save(aplistname,ro.fault_dict['aperture_list'])
            # reset to None to save memory
            input_dict['fault_surfaces'] = None
            input_dict['fault_edges'] = None
            input_list.append(input_dict.copy())
            if return_objects:
                ro_list.append(copy.copy(ro))
        # save to a file so they can be accessed by all processors
        np.save(op.join(savepath,fename),ro.fault_edges)
        np.save(op.join(savepath,fsname),ro.fault_dict['fault_surfaces'])

        

        # set the variables to the file name
        input_dict['fault_edgesname'] = fename
        input_dict['fault_surfacename'] = fsname
        input_dict['fault_assignment'] = 'list'


    time.sleep(10)
 
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


def calculate_comparison_volumes(input_list,subvolume_size,rank,tmp_outfile=None,properties=None,
                                 boundary_res = 1e40,boundary_hydres = 1e60,directions=None):
    """
    calculate the resistivity in x y and z direction of a subsetted volume
    (different dimensions for solving in x y and z directions)
    
    """
    
    count = 0
    if np.amax(input_list[0]['ncells']) > 40:
        time.sleep(60*rank)
    for input_dict in input_list:
        input_dict = input_dict.copy()
        input_dict['fault_edges'] = np.load(input_dict['faultedge_file'])
        input_dict['fault_assignment'] = 'list'
        input_dict['aperture_list'] = np.load(input_dict['aperturelist_file'])
        input_dict['aperture_assignment'] = 'list'
        
        rom = rn.Rock_volume(**input_dict)
        
        
        properties = rom.solve_properties
        directions = rom.solve_direction
            
        nc = np.array(rom.ncells)
        n = subvolume_size + 1
        
        inputs = dict(update_cellsize_tf=False, fault_assignment='none',
                      cellsize=rom.cellsize,solve_properties=properties)    
        kbulk1,rbulk1 = np.ones(3)*np.nan,np.ones(3)*np.nan

        for att,br,sp,bulk,prop in [['resistivity',boundary_res,'current',rbulk1,'resistivity'],['hydraulic_resistance',boundary_hydres,'fluid',kbulk1,'permeability']]:
            if sp in properties:
                arrtoset = getattr(rom,att)        
                inputs['solve_properties'] = sp
                # x direction array
                if 'x' in directions:
                    inputs['solve_direction'] = 'x'
                    ncells = nc.copy()
                    ncells[0] -= n[0]
                    print "setting up comparison resistor network"
                    romx = rn.Rock_volume(ncells=ncells,**inputs)
                    arr = getattr(romx,att)
                    arr[1:,1:,1:-1,0] = arrtoset[1:,1:,1:-n[0]-1,0]
                    arr[1:,1:-1,1:,1] = arrtoset[1:,1:-1,1:-n[0],1]
                    arr[1:-1,1:,1:,2] = arrtoset[1:-1,1:,1:-n[0],2]
                    arr[:,-1,:,0] = boundary_res
                    arr[-1,:,:,0] = boundary_res
                    setattr(romx,att,rna.add_nulls(arr))
                    print "solving comparison resistor network"
                    romx.solve_resistor_network2()
                    print "comparison resistor network solved"
                    factor = (nc[1]*nc[2])/((nc[1]+1.)*(nc[2]+1.))
                    if att == 'resistivity':
                        bulk[0] = romx.resistivity_bulk[0]*factor
                    else:
                        bulk[0] = romx.permeability_bulk[0]/factor
        
                # y direction array
                if 'y' in directions:
                    inputs['solve_direction'] = 'y'
                    ncells = nc.copy()
                    ncells[1] -= n[1]
                    print "setting up comparison resistor network"
                    romy = rn.Rock_volume(ncells=ncells,**inputs)
                    arr = getattr(romy,att)
                    arr[1:,1:,1:-1,0] = arrtoset[1:,1:-n[1],1:-1,0]
                    arr[1:,1:-1,1:,1] = arrtoset[1:,1:-n[1]-1,1:,1]
                    arr[1:-1,1:,1:,2] = arrtoset[1:-1,1:-n[1],1:,2]
                    arr[-1,:,:,1] = boundary_res
                    arr[:,:,-1,1] = boundary_res
                    setattr(romy,att,rna.add_nulls(arr))
                    print "solving comparison resistor network"
                    romy.solve_resistor_network2()
                    print "comparison resistor network solved"
                    factor = (nc[0]*nc[2])/((nc[0]+1.)*(nc[2]+1.))
                    if att == 'resistivity':
                        bulk[1] = romy.resistivity_bulk[1]*factor
                    else:
                        bulk[1] = romy.permeability_bulk[1]/factor
                
                # z direction array
                if 'z' in directions:
                    inputs['solve_direction'] = 'z'
                    ncells = nc.copy()
                    ncells[2] -= n[2]
                    print "setting up comparison resistor network"
                    romz = rn.Rock_volume(ncells=ncells,**inputs)
                    arr = getattr(romz,att)
                    arr[1:,1:,1:-1,0] = arrtoset[1:-n[1],1:,1:-1,0]
                    arr[1:,1:-1,1:,1] = arrtoset[1:-n[1],1:-1,1:,1]
                    arr[1:-1,1:,1:,2] = arrtoset[1:-n[1]-1,1:,1:,2]
                    arr[:,-1,:,2] = boundary_res
                    arr[:,:,-1,2] = boundary_res
                    setattr(romz,att,rna.add_nulls(arr))
                    print "solving comparison resistor network"
                    romz.solve_resistor_network2()
                    print "comparison resistor network solved"
                    factor = (nc[0]*nc[1])/((nc[0]+1.)*(nc[1]+1.))
                    if att == 'resistivity':
                        bulk[2] = romz.resistivity_bulk[2]*factor
                    else:
                        bulk[2] = romz.permeability_bulk[2]/factor
                        
        if hasattr(rom,'aperture_mean'):
            apm = rom.aperture_mean
        else:
            apm = np.ones(3)*np.nan

        if hasattr(rom,'contact_area'):
            ca = rom.contact_area
        else:
            ca = np.ones(3)*np.nan                        
                        
        line = np.hstack([rbulk1,kbulk1,apm,ca,[np.median(rom.fault_dict['fault_separation'])],[rom.id]])
        
        if count == 0:
            outarray = np.array([line.copy()])
            count += 1
        else:
            outarray = np.vstack([outarray,line])

        if tmp_outfile is not None:
            np.savetxt(tmp_outfile,outarray,fmt=['%.3e']*9+['%.3f']*3+['%.3e','%3i'],header='resx resy resz kx ky kz apmx apmy apmz cax cay caz fs rid')
    
    

    return outarray
        
        
        
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
        
        # direction perpendicular to fault
        dperp = list(np.amax(faultedges[fi],axis=(0,1)) - np.amin(faultedges[fi],axis=(0,1))).index(0)

        for i in range(3):
            if i != dperp:
                lfe[:,:,i][lfe[:,:,i] < 1. - buf] = 1. - buf
                lfe[:,:,i][lfe[:,:,i] > n[i] + buf] = n[i] + buf
     
        local_fault_edges.append(lfe)

        
        # minimum and maximum indices for aperture to cut it if it extends over multiple sub volumes
        ap0 = np.array([sx*n[0],sy*n[1],sz*n[2]]) - buf + 1 - np.amin(faultedges[fi],axis=(0,1))#
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


def initialise_inputs_subvolumes(splitn,inputdict,buf=4):
    """
    divide a large rock volume into subvolumes, reset some properties and assign indices

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
                localidict = inputdict.copy()

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
    
    count = 0
    for line in outputs_gathered:
        if len(line) == 2:
            print "gathering ro list and outputs" 
            line1,rolist = line
            ro_masterlist += rolist
        else:
            line1 = line
            ro_masterlist = None

        if count == 0:
            outarray = line1.copy()
        else:
            outarray = np.vstack([outarray,line1])
        count += 1
    
    # now go through and put all entries for each rock volume on one line
    # also sort the rock objects by the same order
    count = 0
    for rid in np.unique(outarray[:,-1]):
        for fs in np.unique(outarray[:,-5]):
            for iz in np.unique(outarray[:,-2]):
                for iy in np.unique(outarray[:,-3]):
                    for ix in np.unique(outarray[:,-4]):
                        ind = np.where(np.all(outarray[:,-5:]==np.array([fs,ix,iy,iz,rid]),axis=1))[0]
                        line = np.nanmax(outarray[ind],axis=0)
                        roxyz = np.zeros(3,dtype=object)
                        if ro_masterlist is not None:
                            for ii in range(3):
                                for iii in ind:
                                    if np.any(np.isfinite(outarray[iii][np.array([ii,ii+3])])):
                                        roxyz[ii] = ro_masterlist[iii]
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

    np.savetxt(outfile,outarray2,fmt=['%.3e']*9+['%.3f']*3+['%.3e']+['%3i']*4,comments='')
    
    if ro_masterlist is None:
        return outarray2
    else:
        return outarray2,ro_masterlist_sorted


def run_subvolumes(input_list,subvolume_size,return_objects=False,tmp_outfile=None):
    """
    
    """
    if return_objects:
        ro_list = []
    faultedge_file = None
    count = 0    
    print "running subvolumes for tmp outfile {}".format(tmp_outfile)
    for input_dict in input_list:
        rbulk, kbulk = np.ones(3)*np.nan, np.ones(3)*np.nan
        di = 'xyz'.index(input_dict['solve_direction'])
        #if faultedge_file is not None:
            # only reload if it's a different file to previous, save reloading the file multiple times
            #if faultedge_file == input_dict['faultedge_file']:
        faultedge_list = np.load(input_dict['faultedge_file'])
        aperture_list = np.load(input_dict['aperturelist_file'])
        #        faultedge_file = input_dict['faultedge_file']
        # get aperture and faults
        localfaults,localap = segment_faults(faultedge_list,aperture_list,input_dict['indices'],
                                             np.array(subvolume_size).copy(),buf=input_dict['array_buffer'])
        input_dict['aperture_list'] = localap
        input_dict['fault_edges'] = np.array(localfaults)


        ros = rn.Rock_volume(**input_dict)
        ros.solve_resistor_network2()

        # reset to None so it doesn't have to store in memory, this may or may not help with speed/memory issues
        input_dict['aperture_list'] = None
        input_dict['fault_edges'] = None
        
        rbulk[di] = ros.resistivity_bulk[di]
        kbulk[di] = ros.permeability_bulk[di]
        
        if return_objects:
            ro_list.append(copy.copy(ros))

        if hasattr(ros,'aperture_mean'):
            apm = ros.aperture_mean
        else:
            apm = np.ones(3)*np.nan

        if hasattr(ros,'contact_area'):
            ca = ros.contact_area
        else:
            ca = np.ones(3)*np.nan   

        
        line = np.hstack([rbulk,kbulk,apm,ca,
                         [np.median(ros.fault_dict['fault_separation'])],
                          ros.indices,[ros.id]])
        if count == 0:
            outarray = np.array([line.copy()])
        else:
            outarray = np.vstack([outarray,[line]])
 
        if tmp_outfile is not None:
            np.savetxt(tmp_outfile,outarray,fmt=['%.3e']*9+['%.3f']*3+['%.3e']+['%3i']*4,
                       header='resx resy resz kx ky kz apmx apmy apmz cax cay caz fs ix iy iz rid')
                         
        count += 1
        
    if return_objects:
        return outarray,ro_list
    else:
        return outarray



def scatter_run_subvolumes(input_list,subvolume_size,size,rank,comm,outfile,
                           return_objects=False,tmp_outfile=None):
    """
    initialise and run subvolumes
    
    """

    if tmp_outfile is not None:
        tmp_outfile += str(rank)
    print "tmp file name for rank {} created".format(rank) 
    if rank == 0:
        nn = input_list[0]['subvolume_size']
        directions = input_list[0]['solve_direction']
        #properties = input_list[0]['solve_properties']

        input_list_sep = []
        for ii,idict in enumerate(input_list):
            directions = idict['solve_direction']
            properties = []
            for attr in ['current','fluid']:
                if attr in idict['solve_properties']:
                    properties.append(attr)
            for sd in directions:
                di = 'xyz'.index(sd)
                ncells = np.array(nn).copy()
                ncells[di] += 1
                idict['ncells'] = ncells
                for sp in properties:
                    idict['solve_direction'] = sd
                    idict['solve_properties'] = sp
                    input_list_sep.append(idict.copy())
    
        input_list_divided = divide_inputs(input_list_sep,size)
    else:
        input_list_divided = None

    if comm is not None:
        inputs_sent = comm.scatter(input_list_divided,root=0)
        print "scattering subvolumes on rank {}".format(rank)
        bulk_props = run_subvolumes(inputs_sent,subvolume_size,return_objects=return_objects,
                                    tmp_outfile=tmp_outfile)
        outputs_gathered = comm.gather(bulk_props,root=0)
    else:
        outputs_gathered = [run_subvolumes(input_list_sep,subvolume_size,
                                           return_objects=return_objects,
                                           tmp_outfile=tmp_outfile)]
    
    if rank == 0:
        return write_outputs_subvolumes(outputs_gathered, outfile)
    else:
        if return_objects:
            return None,None
        else:
            return None



def compare_arrays(ro_list,ro_list_seg,indices,subvolume_size):
    """
    """
    
    indicesz = np.unique(indices[:,-2])
    indicesy = np.unique(indices[:,-3])
    indicesx = np.unique(indices[:,-4])
    splitn = [np.amax(ind)+1 for ind in [indicesx,indicesy,indicesz]]
    n = np.array(subvolume_size) + 1
    nn = subvolume_size
    ncellsx,ncellsy,ncellsz = [[nn[2] + 1,nn[1],nn[0]],
                               [nn[2],nn[1] + 1,nn[0]],
                               [nn[2],nn[1],nn[0] + 1]]
    
    testfaults,testap,testres,testhr = [],[],[],[]        
    
    for rom in ro_list:
        compiled_faults = np.zeros((splitn[2]*n[2]+2,splitn[1]*n[1]+2,splitn[0]*n[0]+2,3,3))
        compiled_ap = np.zeros((splitn[2]*n[2]+2,splitn[1]*n[1]+2,splitn[0]*n[0]+2,3,3))
        compiled_res = np.zeros((splitn[2]*n[2]+2,splitn[1]*n[1]+2,splitn[0]*n[0]+2,3))
        compiled_hr = np.zeros((splitn[2]*n[2]+2,splitn[1]*n[1]+2,splitn[0]*n[0]+2,3))
        


        rid,fs = rom.id,np.median(rom.fault_dict['fault_separation'])

        for sz in indicesz:
            for sy in indicesy:
                for sx in indicesx:
                    ind = np.where(np.all(indices[:,-5:]==np.array([fs,sx,sy,sz,rid]),axis=1))[0][0]
                    rox,roy,roz = ro_list_seg[ind]
                    compiled_ap[1+sz*n[2]:1+sz*n[2]+ncellsz[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+(sx+1)*n[0],2] = roz.aperture[1:ncellsz[2]+1,1:,1:,2].copy()
                    compiled_ap[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+sy*n[1]+ncellsy[1],1+sx*n[0]:1+(sx+1)*n[0],1] = roy.aperture[1:,1:ncellsy[1]+1,1:,1].copy()
                    compiled_ap[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+sx*n[0]+ncellsx[0],0] = rox.aperture[1:,1:,1:ncellsx[0]+1,0].copy()
                    compiled_faults[1+sz*n[2]:1+sz*n[2]+ncellsz[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+(sx+1)*n[0],2] = roz.fault_array[1:ncellsz[2]+1,1:,1:,2]
                    compiled_faults[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+sy*n[1]+ncellsy[1],1+sx*n[0]:1+(sx+1)*n[0],1] = roy.fault_array[1:,1:ncellsy[1]+1,1:,1]
                    compiled_faults[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+sx*n[0]+ncellsx[0],0] = rox.fault_array[1:,1:,1:ncellsx[0]+1,0]
                    compiled_res[1+sz*n[2]:1+sz*n[2]+ncellsz[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+(sx+1)*n[0],2] = roz.resistivity[1:ncellsz[2]+1,1:,1:,2]
                    compiled_res[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+sy*n[1]+ncellsy[1],1+sx*n[0]:1+(sx+1)*n[0],1] = roy.resistivity[1:,1:ncellsy[1]+1,1:,1]
                    compiled_res[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+sx*n[0]+ncellsx[0],0] = rox.resistivity[1:,1:,1:ncellsx[0]+1,0]
                    compiled_hr[1+sz*n[2]:1+sz*n[2]+ncellsz[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+(sx+1)*n[0],2] = roz.hydraulic_resistance[1:ncellsz[2]+1,1:,1:,2]
                    compiled_hr[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+sy*n[1]+ncellsy[1],1+sx*n[0]:1+(sx+1)*n[0],1] = roy.hydraulic_resistance[1:,1:ncellsy[1]+1,1:,1]
                    compiled_hr[1+sz*n[2]:1+(sz+1)*n[2],1+sy*n[1]:1+(sy+1)*n[1],1+sx*n[0]:1+sx*n[0]+ncellsx[0],0] = rox.hydraulic_resistance[1:,1:,1:ncellsx[0]+1,0]    
    
        # set unused edges to zero (these are removed for calculation of resistivity)  
        rom.aperture[:,-1,:,2] = 0.
        rom.aperture[:,:,-1,2] = 0.
        rom.aperture[-1,:,:,1] = 0.
        rom.aperture[:,:,-1,1] = 0.
        rom.aperture[:,-1,:,0] = 0.
        rom.aperture[-1,:,:,0] = 0.
        rom.fault_array[:,-1,:,2] = 0.
        rom.fault_array[:,:,-1,2] = 0.
        rom.fault_array[-1,:,:,1] = 0.
        rom.fault_array[:,:,-1,1] = 0.
        rom.fault_array[:,-1,:,0] = 0.
        rom.fault_array[-1,:,:,0] = 0.                
        		    
		    
        # set unused edges to high res (these are removed for calculation of resistivity)  
        rom.resistivity[:,-1,:,2] = np.nan
        rom.resistivity[:,:,-1,2] = np.nan
        rom.resistivity[-1,:,:,1] = np.nan
        rom.resistivity[:,:,-1,1] = np.nan
        rom.resistivity[:,-1,:,0] = np.nan
        rom.resistivity[-1,:,:,0] = np.nan
        rom.hydraulic_resistance[:,-1,:,2] = np.nan
        rom.hydraulic_resistance[:,:,-1,2] = np.nan
        rom.hydraulic_resistance[-1,:,:,1] = np.nan
        rom.hydraulic_resistance[:,:,-1,1] = np.nan
        rom.hydraulic_resistance[:,-1,:,0] = np.nan
        rom.hydraulic_resistance[-1,:,:,0] = np.nan


        diff_faults = compiled_faults-rom.fault_array
        diff_faults[np.isnan(diff_faults)] = 0
        diff_ap = compiled_ap-rom.aperture
        diff_ap[np.isnan(diff_ap)] = 0                
        diff_res = compiled_res-rom.resistivity
        diff_res[np.isnan(diff_res)] = 0                
        diff_hr = compiled_hr-rom.hydraulic_resistance
        diff_hr[np.isnan(diff_hr)] = 0                

        testfaults.append(np.all(diff_faults==0))
        testap.append(np.all(diff_ap==0))
        testres.append(np.all(diff_res==0))
        testhr.append(np.all(diff_hr==0))

    return testfaults,testap,testres,testhr



def write_outputs_comparison(outputs_gathered, outfile) :
    """
    gathers all the outputs written to individual files for each rank, to a 
    master file.
    
    
    """
  
    
    # first gather all outputs into a master array
    count = 0
    for lines in outputs_gathered:
#        fslist = np.array(fslist).reshape(len(fslist),1)
#        ridlist = np.array(ridlist).reshape(len(ridlist),1)
#        line = np.hstack([rbulk,kbulk,fslist,ridlist])
        if count == 0:
            outarray = np.array(lines)
        else:
            outarray = np.vstack([outarray,lines])
        count += 1
    
    outarray[np.isinf(outarray)] = np.nan    

    
    # now go through and put all entries for each rock volume on one line
    count = 0
    for r in np.unique(outarray[:,-1]):
        for fs in np.unique(outarray[:,-2]):
            ind = np.where(np.all([outarray[:,-1]==r,outarray[:,-2]==fs],axis=0))[0]
            line = np.nanmax(outarray[ind],axis=0)
            if count == 0:
                outarray2 = line.copy()
            else:
                outarray2 = np.vstack([outarray2,line])
            count += 1
        if count == 1:
            outarray2 = np.array([outarray2])

    np.savetxt(outfile,outarray2,fmt=['%.3e']*9+['%.3f']*3+['%.3e','%2i'],comments='')
   

def run_comparison(input_list,subvolume_size,rank,size,comm,outfile,tmp_outfile=None):
    """
    run comparison arrays to compare to segmented volume (different volumes for
    the x, y and z directions)
    
    """
    if tmp_outfile is not None:
        tmp_outfile += str(rank)
    
    if comm is not None:
        print "setting up comparison volumes in parallel"
        if rank == 0:
            inputlist_divided = divide_inputs(input_list,size)
        else:
            inputlist_divided = None
        inputs_sent = comm.scatter(inputlist_divided,root=0)
        bulk_props = calculate_comparison_volumes(inputs_sent,subvolume_size,rank,tmp_outfile=tmp_outfile)
        outputs_gathered = comm.gather(bulk_props,root=0)
        
        if rank == 0:
            print "writing comparison outputs"
            write_outputs_comparison(outputs_gathered, outfile)

    else:
        outputs_gathered = [list(calculate_comparison_volumes(input_list,subvolume_size,tmp_outfile=tmp_outfile))]

        write_outputs_comparison(outputs_gathered, outfile)



def update_from_subvolumes(arr,outputs,propertyname):
    """
    update an array from subvolume outputs
    
    """
    splitn = np.amax(outputs[:,-4:-1],axis=0).astype(int) + 1

    if propertyname == 'permeability':
        c = 3
    else:
        c = 0
    
    for sz in range(splitn[2]):
        for sy in range(splitn[1]):
            for sx in range(splitn[0]):
                ind = np.where(np.all(outputs[:,-4:-1] ==np.array([sx,sy,sz]),axis=1))[0][0]
                arr[sz+1,sy+1,sx+1,0] = outputs[ind][0 + c]
                arr[sz+1,sy+1,sx+1,1] = outputs[ind][1 + c]
                arr[sz+1,sy+1,sx+1,2] = outputs[ind][2 + c]
    
    return arr          


def build_master_segmented(list_of_inputs,subvolume_outputs,subvolume_size):
    """
    set up master volume to contain the segmented inner volumes.
    """
    
    ro_list_sep = []
    solve_properties = []
    n = np.array(subvolume_size) + 1

    splitn = np.amax(subvolume_outputs[:,-4:-1],axis=0) + 1    

    for pp in ['current','fluid']:
        if pp in list_of_inputs[0]['solve_properties']:
            solve_properties.append(pp)
    
    for input_dict in list_of_inputs:
        # not building new faults or apertures so fault assignment is none
        input_dict['fault_assignment'] = 'none'
        input_dict['build_arrays'] = True
        # number of cells
        input_dict['ncells'] = splitn - 1
        # new cellsize, size of subvolume * cellsize of original volume
        input_dict['cellsize'] = n * input_dict['cellsize']
        rid,fs = input_dict['id'],input_dict['fault_separation']

        ind = np.where(np.all([subvolume_outputs[:,-1] == rid,subvolume_outputs[:,-5]==fs],axis=0))[0]
        outputs = subvolume_outputs[ind]
        ro = rn.Rock_volume(**input_dict)
        ro.resistivity = rna.add_nulls(update_from_subvolumes(ro.resistivity,outputs,'resistivity'))
        ro.permeability = rna.add_nulls(update_from_subvolumes(ro.permeability,outputs,'permeability'))
        ro.hydraulic_resistance = rnap.permeability2hydraulic_resistance(ro.permeability,
                                                                         ro.cellsize,
                                                                         ro.fluid_viscosity)
        ro.aperture_mean = np.nanmax(outputs[:,6:9],axis=0)
        ro.contact_area = np.nanmax(outputs[:,9:12],axis=0)

        solve_directions = input_dict['solve_direction']

        for sp in solve_properties:
            ro.solve_properties = sp
            for sd in solve_directions:
                ro.solve_direction = sd
                ro_list_sep.append(copy.copy(ro))
    
    return ro_list_sep


def run_segmented(ro_list_sep,save_array=True,savepath=None,tmp_outfile=None):
    

    
    count = 0
    for ro in ro_list_sep:
        ro.solve_resistor_network2()
        if (save_array and (savepath is not None)):
            for attname in ['permeability','resistivity','hydraulic_resistance']:
                arr = getattr(ro,attname)
                if arr is not None:
                    np.save(op.join(savepath,attname+'%1i_fs%.1e'%(ro.id,np.median(ro.fault_dict['fault_separation']))),
                            arr)
                            
                            
        if hasattr(ro,'aperture_mean'):
            apm = ro.aperture_mean
        else:
            apm = np.ones(3)*np.nan

        if hasattr(ro,'contact_area'):
            ca = ro.contact_area
        else:
            ca = np.ones(3)*np.nan
            
            
        line = np.hstack([ro.resistivity_bulk,ro.permeability_bulk,apm,ca,
                          [np.median(ro.fault_dict['fault_separation'])],[ro.id]])
        if count == 0:
            outarray = np.array([line.copy()])
            count += 1
        else:
            outarray = np.vstack([outarray,[line]])
            
        if tmp_outfile is not None:
            np.savetxt(tmp_outfile,outarray,fmt=['%.3e']*9+['%.3f']*3+['%.3e','%3i'],
                       header='resx resy resz kx ky kz apmx apmy apmz cax cay caz fs rid')        

    return outarray


def distribute_run_segmented(ro_list,subvolume_size,rank,size,comm,outfile,
                             save_array=True,savepath=None,tmp_outfile=None):
    """
    run comparison arrays to compare to segmented volume (different volumes for
    the x, y and z directions)
    
    """
    
    if tmp_outfile is not None:
        tmp_outfile += str(rank)
    
    if comm is not None:
        print "setting up comparison volumes in parallel"
        if rank == 0:
            rolist_divided = divide_inputs(ro_list,size)
        else:
            rolist_divided = None
        inputs_sent = comm.scatter(rolist_divided,root=0)
        bulk_props = run_segmented(inputs_sent,save_array=save_array,
                                   savepath=savepath,tmp_outfile=tmp_outfile)
        outputs_gathered = comm.gather(bulk_props,root=0)
        
        if rank == 0:
            print "writing comparison outputs"
            write_outputs_comparison(outputs_gathered, outfile)

    else:
        # if not using mpi, don't need to split up
        outputs_gathered = [list(run_segmented(ro_list,save_array=save_array,
                                               savepath=savepath,tmp_outfile=tmp_outfile))]
        write_outputs_comparison(outputs_gathered, outfile)

                         
    

def build_master(list_of_inputs,savepath=None):
    """
    initialise master rock volumes
    
    """
    # two lists, one with all rock volumes, the other split up by solve direction
    # and solve properties
    input_list,input_list_sep = [], []
    solve_properties = []
    
    if return_objects:
        ro_list = []
    
    for pp in ['current','fluid']:
        if pp in list_of_inputs[0]['solve_properties']:
            solve_properties.append(pp)


    for input_dict1 in list_of_inputs:
        # make a copy so we don't change the original
        input_dict = input_dict1.copy()
        # only initialise new faults if we are moving to a new volume
#        if 'fault_edges' in input_dict.keys():
#            faultedges = input_dict['fault_edges']
#        else:
#            faultedges = None
#            
#        if 'fault_surfaces' in input_dict.keys():
#            faultsurfaces = input_dict['fault_surfaces']
#        else:
#            faultsurfaces = None
#        
#        input_dict['aperture_type'] = 'random'
#        print "input_dict",input_dict        
#        input_dict['fault_edges'] = np.load(op.join(input_dict['workdir'],input_dict['fault_edgesname']))
#        input_dict['fault_surfaces'] = np.load(op.join(input_dict['workdir'],input_dict['fault_surfacename']))
#        print "read in fault surfaces file",op.join(input_dict['workdir'],input_dict['fault_surfacename']),"id",input_dict['id']
#
#        build_arrays = input_dict['build_arrays']
#        input_dict['aperture_type'] = 'random'
#        #print "input_dict (master, large array)",input_dict
#        
#        if return_objects:
#            input_dict['build_arrays'] = True
#        else:
#            input_dict['build_arrays'] = False
#        
#        # initialise a rock volume to get the fault edges and aperture list, don't need to build arrays at this point
#        ro = rn.Rock_volume(**input_dict)
#        print "id after reading",ro.id
#
##        print "np.shape(ro.fault_dict['aperture_list'])",np.shape(ro.fault_dict['aperture_list'])
# 
#        # change build_array value back to original value
#        input_dict['build_arrays'] = build_arrays
#        input_dict['aperture_type'] = 'list'        
#        
#        # save aperture list and fault edges to a file and record the filenames in the input_dict
#        if savepath is None:
#            savepath = ro.workdir
#        fename = op.join(savepath,'fault_edges_%1i_fs%.1e.npy'%(ro.id,np.median(ro.fault_dict['fault_separation'])))
#        aplistname = op.join(savepath,'aperture_list_%1i_fs%.1e.npy'%(ro.id,np.median(ro.fault_dict['fault_separation'])))
#        #aparrayname = op.join(savepath,'aperture_%1i_fs%.1e.npy'%(ro.id,np.median(ro.fault_dict['fault_separation'])))        
#        #fspostreadname = op.join(savepath,'fault_surfpost_%1i_fs%.1e.npy'%(ro.id,np.median(ro.fault_dict['fault_separation'])))
#
#        np.save(fename,ro.fault_edges)
#        np.save(aplistname,ro.fault_dict['aperture_list'])
#        #np.save(aparrayname,ro.aperture)
#        #np.save(fspostreadname,ro.fault_dict['fault_surfaces'])
#      
#        input_dict['faultedge_file'] = fename
#        input_dict['aperturelist_file'] = aplistname
#
#        # reset these back to original values
#        input_dict['fault_edges'] = None
#        input_dict['fault_surfaces'] = None
        
        input_list.append(input_dict.copy())
    
#        if return_objects:
#            ro_list.append(copy.copy(ro))
    
        solve_direction = ro.solve_direction
        
        # split into different solve properties and solve directions
        for sp in solve_properties:
            for sd in solve_direction:
                input_dict['solve_direction'] = sd
                input_dict['solve_properties'] = sp
                input_list_sep.append(input_dict.copy())
                
#    if return_objects:
#        return input_list, input_list_sep, ro_list
#    else:
    return input_list, input_list_sep



def setup_and_run_segmented_volume(arguments):
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
    fixed_parameters, loop_parameters, repeats = read_arguments(arguments)
    
    # get workdir
    if 'workdir' in fixed_parameters.keys():
        wd = fixed_parameters['workdir']
    else:
        wd = './model_runs'
    # define a subdirectory to store arrays and temporary files
    wd2 = os.path.join(wd,'arrays')
    wd3 = op.join(wd,'tempfiles')    

    time.sleep(2)

    if rank == 0:
        # make working directories
        if not os.path.exists(wd):
            os.mkdir(wd)
        wd = os.path.abspath(wd)
        for wdn in [wd2,wd3]:
            if not os.path.exists(wdn):
                os.mkdir(wdn)

    # determine whether we need the segmented rock volumes for future analysis/comparison
    if 'array' in list_of_inputs_master[0]['comparison_arrays']:
        return_objects = True
    else:
        return_objects = False



    # create inputs for master rock volumes
    if return_objects:
        list_of_inputs_master,ro_list = initialise_inputs_master(fixed_parameters, loop_parameters, repeats,wd2,return_objects=True)
    else:
        list_of_inputs_master = initialise_inputs_master(fixed_parameters, loop_parameters, repeats,wd2,return_objects=False)

    time.sleep(2)
    
    else:
        print "waiting for directories on rank {}"
        # wait for rank 0 to generate folder up to max of 10 minutes
        tt = 0
        while not os.path.exists(wd3):
            time.sleep(1)
            tt += 1
            if tt > 600:
                break

    # initialise outfile name
    if 'outfile' in fixed_parameters.keys():
        outfile = fixed_parameters['outfile']
        if not outfile.endswith('.dat'):
            outfile += '.dat'
    else:
        outfile = 'outputs.dat'


    # get list of master rock volumes. Two lists. The first has all the rock volumes
    # the second has all the solve properties and directions separated out for
    # parallel processing
    subvolume_size = list_of_inputs_master[0]['subvolume_size']
    if rank == 0:
        t0 = time.time()
        input_list, input_list_sep = build_master(list_of_inputs_master,savepath=wd2)
        print "Initialised master rock volumes in {} s".format(time.time()-t0)
    else:
        input_list, input_list_sep, ro_list = None, None, None
        
    # run comparison for bulk properties
    t1 = time.time()
    if 'bulk' in fixed_parameters['comparison_arrays']:
        t1 = time.time()
        run_comparison(input_list_sep,subvolume_size,rank,size,comm,
                       op.join(wd,'comparison_'+outfile),
                       tmp_outfile=op.join(wd3,'comparison_'+outfile[:-4]+'.tmp'))
        if rank == 0:
            print "Ran comparison arrays in {} s".format(time.time()-t1)

    # create subvolume inputs
    if rank == 0:
        subvolume_input_list = []
        t2 = time.time()
        for rr in range(len(input_list)):
            input_dict = input_list[rr].copy()
            # have to solve 3 directions in subvolumes regardless of directions being solved in master
            input_dict['solve_direction'] = 'xyz'
            subvolume_input_list += initialise_inputs_subvolumes(input_dict['splitn'],
                                                                 input_dict,
                                                                 buf=4)
        print "Initialised subvolume input list in {} s".format(time.time()-t2)
    else:
        subvolume_input_list = None
        

    
    # run the subvolumes and return an array, containing results + indices +
    # rock volume ids + (optionally) rock volume objects
    t3 = time.time()
    print "running subvolumes on rank {}"
    if return_objects:
        outarray,ro_list_seg = scatter_run_subvolumes(subvolume_input_list,list_of_inputs_master[0]['subvolume_size'],
                                                      size,rank,comm,
                                                      op.join(wd,'subvolumes_'+outfile),
                                                      return_objects=True,
                                                      tmp_outfile=op.join(wd3,'subvolumes_'+outfile[:-4]+'.tmp'))
        # assemble the individual pieces into master arrays (aperture, 
        # resistivity and hydraulic resistance and compare these to the
        # original ones.
        if rank == 0:
            print "comparing segmented and original arrays"
            testfaults,testap,testres,testhr = compare_arrays(ro_list,ro_list_seg,outarray[:,-5:],list_of_inputs_master[0]['subvolume_size'])
            print testfaults,testap,testres,testhr
    else:
        outarray = scatter_run_subvolumes(subvolume_input_list,list_of_inputs_master[0]['subvolume_size'],
                                          size,rank,comm,
                                          op.join(wd,'subvolumes_'+outfile),
                                          return_objects=False,
                                          tmp_outfile=op.join(wd3,'subvolumes_'+outfile[:-4]+'.tmp'))
    if rank == 0:
        print "Ran subvolumes in {} s".format(time.time()-t3)
    
    # create and run master volume containing subvolume results
    if rank == 0:
        t4 = time.time()
        ro_list_sep = build_master_segmented(list_of_inputs_master,
                                             outarray,
                                             input_dict['subvolume_size'])
        print "Built a master segmented volume in {} s".format(time.time()-t4)
    else:
        ro_list_sep = None
    #print "running master volume, ro_list_sep on rank",rank,ro_list_sep
    t5 = time.time()    
    distribute_run_segmented(ro_list_sep,list_of_inputs_master[0]['subvolume_size'],
                             rank,size,comm,op.join(wd,'master_'+outfile),
                             save_array=True,savepath=wd2,
                             tmp_outfile=op.join(wd3,'master_'+outfile[:-4]+'.tmp'))
    if rank == 0:
        print "Ran segmented volume in {} s".format(time.time()-t5)
        print "Times: setup master: {} s, run comparison: {} s, setup subvolumes: {} s, run subvolumes: {} s, setup segmented (master): {} s, run segmented (master): {} s".format(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,time.time()-t5)

                   
if __name__ == "__main__":
    setup_and_run_segmented_volume(sys.argv)

