# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:35:11 2015

@author: a1655681
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import rnpy.functions.assignfaults_new as rnaf
import rnpy.functions.assignproperties as rnap
import rnpy.functions.matrixbuild as rnmb

import rnpy.functions.matrixsolve as rnms
import rnpy.functions.array as rna
import rnpy.functions.faultaperture as rnfa
import sys
import time

class Rock_volume():
    """
    ***************Documentation last updated 21 June 2016, not finished*******************
    
    Class to contain volumes to be modelled as a random resistor network.
    workdir = working directory
    ncells = list containing number of nodes in the x,y and z direction, 
             default is [10,10,10]
    cellsize = cellsize for network. Generally this needs to be the same in
               each direction. The exception is where faults of only one
               direction exist in the network, in which case cells perpendicular
               to the fault can have a different size (to accommodate wide faults)
    update_cellsize_tf = True or False, determines whether to update the cellsize
                         in the direction perp to fault, only updates if 
                         there is only one orientation of faults in the network
    pconnection = list of relative probability of connection in the yz,xz, and xy plane 
                  if fault_assignment is random, default [0.33,0.33,0.33] (input
                  list is normalised so that the total = 1.)
    resistivity_matrix = resistivity of the low conductivity matrix
    resistivity_fluid = resistivity of the high conductivity fluid. Used with 
                        fracture diameter to calculate the resistance of 
                        connected bonds
    resistivity = option to provide the resistivity array, if it is not provided
                  then it is calculated from the aperture
    permeability_matrix = permeability of low electrical conductivity matrix
    fractal_dimension = fractal dimension of fault surfaces, float
    fault_separation = separation value for faults, float, or array or list, if
                       array or list is provided then needs to be same length 
                       as fault_edges
    
    
    fluid_viscosity = fluid viscosity, default for freshwater at 20 degrees
    faultlength_max = maximum fault length if res_type is "random"
    faultlength_decay = decay factor to describe shape of fault length
                        distribution function, default 5
                 
    """
    
    def __init__(self, **input_parameters):
        self.workdir = '.' # working directory
        self.ncells = [10,10,10] #ncells in x, y and z directions
        self.cellsize = 1e-3
        self.update_cellsize_tf = True # option to update cellsize if fault is wider than cellsize, only works if there are only faults in one direction.
        self.pconnection = [0.5,0.5,0.5]
        self.resistivity_matrix = 1000.
        self.resistivity_fluid = 0.1
        self.matrix_current=True
        self.matrix_flow=True
        self.resistivity = None
        self.permeability_matrix = 1.e-18
        self.fluid_viscosity = 1.e-3 #default is for freshwater at 20 degrees 
        self.fault_dict = dict(fractal_dimension=2.5,
                               fault_separation = 1e-4,
                               offset = 0,
                               deform_fault_surface=False,
                               faultlength_max = np.amax(self.cellsize)*np.amax(self.ncells),
                               faultlength_min = np.amax(self.cellsize),
                               alpha = 3.,
                               a = 3.5,
                               mismatch_wavelength_cutoff = None,
                               elevation_scalefactor = 1e-3,
                               elevation_prefactor=1,
                               aperture_type = 'random',
                               aperture_list = None,
                               fault_surfaces = None,
                               random_numbers_dir=None,
                               correct_aperture_for_geometry = False,
                               preserve_negative_apertures = False,
                               fault_spacing = 2)
        self.fault_array = None      
        self.fault_edges = None
        self.fault_assignment = 'single_yz' # how to assign faults, 'random' or 'list', or 'single_yz'
        self.aperture = None
        self.aperture_electric = None
        self.aperture_hydraulic = None
        self.solve_properties = 'currentfluid'
        self.solve_direction = 'xyz'
        self.build_arrays = True
        self.array_buffer = 0
        
        self.resistivity_bulk = [np.nan]*3
        self.permeability_bulk = [np.nan]*3
        
        self.indices = None # indices of bigger volume, if the rock volume is part of a larger network
        self.id =None # identification number

        
        update_dict = {}
        #correcting dictionary for upper case keys
        input_parameters_nocase = {}
        for key in list(input_parameters.keys()):
            # only assign if it's a valid attribute
            if hasattr(self,key):
                input_parameters_nocase[key.lower()] = input_parameters[key]
            else:
                for dictionary in [self.fault_dict]:
                    if key in list(dictionary.keys()):
                        input_parameters_nocase[key] = input_parameters[key]
                
        
        update_dict.update(input_parameters_nocase)
        for key in update_dict:
            try:
                # original value defined
                value = getattr(self,key)
                if type(value) == str:
                    try:
                        value = float(update_dict[key])
                    except:
                        value = update_dict[key]
                elif type(value) == dict:
                    value.update(update_dict[key])
                else:
                    value = update_dict[key]
                setattr(self,key,value)
            except:
                try:
                    if key in list(self.fault_dict.keys()):
                        try:
                            value = float(update_dict[key])
                        except:
                            value = update_dict[key]
                        self.fault_dict[key] = value
                except:
                    continue 
        
        if type(self.ncells) in [float,int]:
            self.ncells = (np.ones(3)*self.ncells).astype(int)
            
        if type(self.cellsize) in [float,int]:
            self.cellsize = np.ones(3)*self.cellsize
        else:
            if (type(self.fault_assignment) == 'single_yz') and len(self.cellsize == 3):
                if self.cellsize[1] != self.cellsize[2]:
                    print("y cellsize not equal to z cellsize, updating z cellsize")
                    self.cellsize[2] = self.cellsize[1]
                else:
                    self.cellsize = [np.amin(self.cellsize)]*3

        nx,ny,nz = self.ncells
        
        self._verify_solve_direction()

        if self.build_arrays:
#            print "building arrays"
            #if self.fault_array is None:
            self.build_faults()
                
            #if self.aperture is None:
            self.build_aperture()
                
            self.initialise_electrical_resistance()
            self.initialise_permeability()

            # remove array buffer
            if self.array_buffer > 0:
                
                if np.all(self.fault_array.shape[:3] > np.array(self.ncells) + 2):
#                    print "removing buffer"
                    buf = self.array_buffer
                    self.fault_array = rna.add_nulls(self.fault_array[buf:-buf,buf:-buf,buf:-buf])
                    self.fault_edges -= self.array_buffer
                if np.all(self.aperture.shape[:3] > np.array(self.ncells) + 2):
                    buf = self.array_buffer
                    self.aperture = rna.add_nulls(self.aperture[buf:-buf,buf:-buf,buf:-buf])
                    
                self.resistance = rna.add_nulls(self.resistance[buf:-buf,buf:-buf,buf:-buf])
                self.resistivity = rna.add_nulls(self.resistivity[buf:-buf,buf:-buf,buf:-buf])
                self.aperture_electric = rna.add_nulls(self.aperture_electric[buf:-buf,buf:-buf,buf:-buf])
                
                self.hydraulic_resistance = rna.add_nulls(self.hydraulic_resistance[buf:-buf,buf:-buf,buf:-buf])
                self.permeability = rna.add_nulls(self.permeability[buf:-buf,buf:-buf,buf:-buf])
                self.aperture_hydraulic = rna.add_nulls(self.aperture_hydraulic[buf:-buf,buf:-buf,buf:-buf])            

            self.voltage = np.zeros((nz+1,ny+1,nx+1,3))
            self.pressure = np.zeros((nz+1,ny+1,nx+1,3))
        else:
            self.build_faults(create_array=False)
            self.build_aperture()
            
    
    def _verify_solve_direction(self):
        for i, sd in enumerate('xyz'):
            if self.ncells[i] <=2:
                self.solve_direction = self.solve_direction.replace(sd,'')
        

    def build_faults(self,create_array=True):
        """
        initialise a faulted volume. 
        shape is [nz+2,ny+2,nx+2,3,3]
        
        at point x,y,z:
        opening in:
      xdirection  ydirection zdirection
       (yz plane) (xz plane) (xy plane)
               |      |      |
               v      v      v
            [[0,      x(y),  x(z)], <-- x connectors
             [y(x),   0,     y(z)], <-- y connectors
             [z(x),   z(y),    0]]  <-- z connectors
        
        """
        # define number of cells in x, y, z directions
        nx,ny,nz = self.ncells
        
        # first check the shape and see that it conforms to correct dimensions
        # if it doesn't, create a new array with the correct dimensions
        if self.fault_array is not None:
            if self.fault_array.shape != (nz+2 + self.array_buffer*2,
                                          ny+2 + self.array_buffer*2,
                                          nx+2 + self.array_buffer*2,3,3):
                print("Fault array does not conform to dimensions of network, creating a new array!")
                self.fault_array= None
                
        
        if self.fault_array is None:
            if create_array:
#                print "initialising a new array"
                # initialise a fault array
#                print "array_buffer",self.array_buffer,"nx,ny,nz",nx,ny,nz
                self.fault_array = np.zeros([nz+2+self.array_buffer*2,
                                             ny+2+self.array_buffer*2,
                                             nx+2+self.array_buffer*2,3,3])
                if self.fault_edges is not None:
                    self.fault_edges = np.array(self.fault_edges) + self.array_buffer
                # add nulls to the edges
                self.fault_array = rna.add_nulls(self.fault_array)
            
            addfaults = False

            # option to specify fault edges as a list
            if ((self.fault_assignment == 'list') or (self.fault_assignment[0]=='list')):
                if self.fault_edges is not None:
                    # check the dimensions of the fault edges
                    if np.shape(self.fault_edges)[-3:] == (2,2,3):
                        if len(np.shape(self.fault_edges)) == 3:
                            self.fault_edges = np.array([self.fault_edges])
                        addfaults = True
                    else:
                        pass
                        #print "Invalid fault edges"
                    
                        
            # option to specify a single fault in the centre of the yz plane
            elif self.fault_assignment == 'single_yz':
                ix = int(nx/2) + 1
                iy0, iy1 = 1, ny + 1
                iz0, iz1 = 1, nz + 1
                self.fault_edges = np.array([[[[ix,iy0,iz0],[ix,iy1,iz0]],
                                              [[ix,iy0,iz1],[ix,iy1,iz1]]]])
                addfaults = True
                        
            # option to specify a single fault in the centre of the xz plane
            elif self.fault_assignment == 'single_xz':
                iy = int(ny/2) + 1
                ix0, ix1 = 1, nx + 1
                iz0, iz1 = 1, nz + 1
                self.fault_edges = np.array([[[[ix0,iy,iz0],[ix1,iy,iz0]],
                                              [[ix0,iy,iz1],[ix1,iy,iz1]]]])
                addfaults = True            
                        
            # option to specify a single fault in the centre of the xy plane
            elif self.fault_assignment == 'single_xy':
                iz = int(nz/2) + 1
                iy0, iy1 = 1, ny + 1
                ix0, ix1 = 1, nz + 1
                self.fault_edges = np.array([[[[ix0,iy0,iz],[ix0,iy1,iz]],
                                              [[ix1,iy0,iz],[ix1,iy1,iz]]]])
                addfaults = True

            
            elif self.fault_assignment == 'multiple_yz':
                self.fault_dict['fault_spacing'] = int(self.fault_dict['fault_spacing'])
                if nx > 1:
                    start = 2
                else:
                    start = 1
                iy0, iy1 = 1, ny + 1
                iz0, iz1 = 1, nz + 1
                self.fault_edges = np.array([[[[ix,iy0,iz0],[ix,iy1,iz0]],
                                              [[ix,iy0,iz1],[ix,iy1,iz1]]] \
                                              for ix in range(start,nx + 2,self.fault_dict['fault_spacing'])])
                addfaults = True                

            elif self.fault_assignment == 'multiple_xz':
                self.fault_dict['fault_spacing'] = int(self.fault_dict['fault_spacing'])
                if ny > 1:
                    start = 2
                else:
                    start = 1
                ix0, ix1 = 1, nx + 1
                iz0, iz1 = 1, nz + 1
                self.fault_edges = np.array([[[[ix0,iy,iz0],[ix1,iy,iz0]],
                                              [[ix0,iy,iz1],[ix1,iy,iz1]]] \
                                              for iy in range(start,ny + 2,self.fault_dict['fault_spacing'])])
                addfaults = True
                
            elif self.fault_assignment == 'multiple_xy':
                if nz > 1:
                    start = 2
                else:
                    start = 1
                self.fault_dict['fault_spacing'] = int(self.fault_dict['fault_spacing'])
                iy0, iy1 = 1, ny + 1
                ix0, ix1 = 1, nx + 1
                self.fault_edges = np.array([[[[ix0,iy0,iz],[ix0,iy1,iz]],
                                              [[ix1,iy0,iz],[ix1,iy1,iz]]] \
                                              for iz in range(start,nz + 2,self.fault_dict['fault_spacing'])])
                addfaults = True       
                
                
            # single fault in each of the xy, yz, and xz planes
            elif self.fault_assignment == 'single_xyz':
                ix, ix0, ix1 = int(nx/2) + 1, 1, nx + 1
                iy, iy0, iy1 = int(ny/2) + 1, 1, ny + 1
                iz, iz0, iz1 = int(nz/2) + 1, 1, nz + 1
                
                self.fault_edges = np.array([[[[ix,iy0,iz0],[ix,iy1,iz0]],
                                              [[ix,iy0,iz1],[ix,iy1,iz1]]],
                                             [[[ix0,iy,iz0],[ix1,iy,iz0]],
                                              [[ix0,iy,iz1],[ix1,iy,iz1]]],
                                             [[[ix0,iy0,iz],[ix1,iy0,iz]],
                                              [[ix0,iy1,iz],[ix1,iy1,iz]]]])
    
            elif self.fault_assignment == 'random':
                # get log10 of minimum and maximum fault length
                lmin,lmax = [np.log10(self.fault_dict['faultlength_{}'.format(mm)]) for mm in ['min','max']]
                # get number of bins for assigning faults of different length, 20 per order of magnitude
                nbins = int((lmax-lmin)*20.)
                # define bins
                lvals = np.logspace(lmin,lmax,nbins)
                # define network size
                networksize = np.array(self.cellsize) * np.array(self.ncells)
                # define probability of connection and normalise
                pxyz = np.array(self.pconnection)/float(sum(self.pconnection))
                # get fault edges
                fracturecoords = rnaf.get_fracture_coords(lvals,networksize,pxyz,return_Nf = False,
                                                          a=self.fault_dict['a'],alpha=self.fault_dict['alpha'])
                self.fault_edges = rnaf.coords2indices(fracturecoords,networksize,[nx,ny,nz])
                addfaults = True
            else:
                self.fault_assignment = 'none'
                
            
                
            if (addfaults and create_array):
                rnaf.add_faults_to_array(self.fault_array,self.fault_edges)
            else:
#                print "Can't assign faults, invalid fault assignment type or invalid fault edges list provided"
                return
                
            # make fault separation into an array with length same as fault edges
            if type(self.fault_dict['fault_separation']) in [float,int]:
                self.fault_dict['fault_separation'] *= np.ones(len(self.fault_edges))

            
    def build_aperture(self):


            
            
        if self.fault_assignment == 'single_yz':
            # if (and only if) all faults are along yz plane cellsize perpendicular 
            # is allowed to be different from cell size along fault.
            # to build aperture we just need cellsize along the fault plane
            cellsize_faultplane = self.cellsize[1]
        elif self.fault_edges is not None:
            planelist = np.array([rnfa.get_plane(self.fault_edges[i]) for i in \
                                  range(len(self.fault_edges))])
            if (np.all(planelist==0) or np.all(planelist==1)):
                # if all in the yz or xz planes, we can use the z cellsize
                cellsize_faultplane = self.cellsize[2]
            elif np.all(planelist==2):
                # else if all faults in the xy plane use the y cellsize
                cellsize_faultplane = self.cellsize[1]
            else:
                # else cellsize **should** be same in all directions
                cellsize_faultplane = self.cellsize[2]

            
            

        if self.fault_assignment == 'none':
            if self.fault_array is not None:
                self.aperture = rna.add_nulls(np.zeros_like(self.fault_array))
                self.aperture_electric = self.aperture.copy()
                self.aperture_hydraulic = self.aperture.copy()

        else:
            aperture_input = {}
            self.fault_dict['mismatch_wavelength_cutoff'], fc = \
            rnfa.get_faultpair_defaults(cellsize_faultplane,
                                        self.fault_dict['mismatch_wavelength_cutoff'] 
                                        )
            aperture_input['cs'] = cellsize_faultplane
            for key in ['fractal_dimension','fault_separation','offset',
                        'elevation_scalefactor', 'fault_surfaces', 'elevation_prefactor',
                        'mismatch_wavelength_cutoff','aperture_type',
                        'correct_aperture_for_geometry','aperture_list',
                        'preserve_negative_apertures','random_numbers_dir',
                        'deform_fault_surface']:
                            aperture_input[key] = self.fault_dict[key]
            if self.fault_dict['fault_surfaces'] is None:
                print("fault surfaces none!")
                
    
            if self.build_arrays:
                    ap,aph,apc,self.aperture,self.aperture_hydraulic, \
                    self.aperture_electric,self.fault_dict['fault_surfaces'] = \
                    rnaf.assign_fault_aperture(self.fault_edges,
                                               np.array(self.ncells)+self.array_buffer*2,
                                               fill_array=True,**aperture_input)
                    # minimum aperture
                    ap_min = (self.permeability_matrix*12)**0.5
                    self.aperture_hydraulic[self.aperture_hydraulic < ap_min] = ap_min                    
                    
                    self.fault_dict['aperture_list'] = [ap,aph,apc]
            else:
                ap,aph,apc,self.fault_dict['fault_surfaces'] = \
                rnaf.assign_fault_aperture(self.fault_edges,np.array(self.ncells)+self.array_buffer*2,
                                           fill_array=False,**aperture_input)
                self.fault_dict['aperture_list'] = [ap,aph,apc] 
                
                
            if ((self.aperture is not None) and (self.fault_array is not None)):
                self._get_contact_area()
                self._get_mean_aperture()
    
                if self.aperture_hydraulic is None:
                    self.aperture_hydraulic = self.aperture.copy()
                if self.aperture_electric is None:
                    self.aperture_electric = self.aperture.copy()
                
                # update cellsize so it is at least as big as the largest fault aperture
                # but only if it's a 2d network or there are only faults in one direction
                if self.update_cellsize_tf:
                    self.update_cellsize()
    
    def _get_faulted_apertures(self):
        apvals_list, fmask_list = [],[]
        for j in [0,1,2]:
            idxs = [0,1,2]
            idxs.remove(j)
            apvals = np.nanmax([self.aperture[:,:,:,i,j] for i in idxs],axis=0)
            fmask = np.nanmax([self.fault_array[:,:,:,i,j] for i in idxs],axis=0)
            apvals = apvals[np.isfinite(apvals)]
            fmask = fmask[np.isfinite(fmask)]
            apvals_list.append(apvals)
            fmask_list.append(fmask)
        return apvals_list,fmask_list
    
    def _get_contact_area(self):
        self.contact_area = []
        apvals_list,fmask_list = self._get_faulted_apertures()
        for j in [0,1,2]:
            apvals = apvals_list[j]
            fmask = fmask_list[j]
            ca = int(len(apvals[apvals<1e-49]))/fmask.sum()
            if np.isinf(ca):
                ca = 0.
            self.contact_area = np.append(self.contact_area,ca)
    
    def _get_mean_aperture(self):
        self.aperture_mean = []
        apvals_list,fmask_list = self._get_faulted_apertures()
        for j in [0,1,2]:
            apvals = apvals_list[j]
            fmask = fmask_list[j]
            self.aperture_mean.append(np.mean(apvals*fmask))
        
    
    def compute_conductive_fraction(self):
        nx,ny,nz = self.ncells
        csx,csy,csz = self.cellsize
        
        aperture = rnap.update_all_apertures(self.aperture,self.cellsize)
        
        # all apertures opening in x direction (yz plane)
        apx = np.zeros((nz+1,ny+1,nx+1)) 
        # y and z connectors in yz plane are equal except last row and column
        # last row - z connectors missing, last column - y connectors missing
        apx[:-1] = aperture[1:-1,1:,1:,2,0] # z connectors opening in x direction (yz)
        apx[:,:-1] = aperture[1:,1:-1,1:,1,0] # y connectors opening in x direction (yz)
        # fill missing aperture in corner
        try:
            apx[-1,-1] = np.mean([apx[-2,-1],apx[-1,-2]],axis=0)
        # if 2d network
        except IndexError:
            if self.ncells[1] == 0:
                apx[-1,-1] = apx[-2,-1]
            elif self.ncells[0] == 0:
                apx[-1,-1] = apx[-1,-2]
        
        # all apertures opening in y direction (xz plane)
        apy = np.zeros_like(apx) 
        # x and z connectors in xz plane are equal except last row and column
        # last row - z connectors missing, last column - x connectors missing
        apy[:-1] = aperture[1:-1,1:,1:,2,1] # z connectors opening in y direction
        apy[:,:,:-1] = aperture[1:,1:,1:-1,0,1] # x connectors opening in y direction
        # fill missing aperture in corner
        try:
            apy[-1,:,-1] = np.mean([apy[-2,:,-1],apy[-1,:,-2]],axis=0)
        except IndexError:
            if self.ncells[2] == 0:
                apy[-1,:,-1] = apy[-1,:,-2]
            elif self.ncells[0] == 0:
                apy[-1,:,-1] = apy[-2,:,-1]
        
        # all apertures opening in z direction (xy plane)
        apz = np.zeros_like(apx) # all apertures opening in z direction (xy plane)
        # x and y connectors in xz plane are equal except last row and column
        # last row - y connectors missing, last column - x connectors missing
        apz[:,:-1] = aperture[1:,1:-1,1:,1,2]
        apz[:,:,:-1] = aperture[1:,1:,1:-1,0,2]
        # fill missing aperture in corner
        try:
            apz[:,-1,-1] = np.mean([apz[:,-2,-1],apz[:,-1,-2]],axis=0)
        except IndexError:
            if self.ncells[1] == 0:
                apz[:,-1,-1] = apz[:,-1,-2]
            elif self.ncells[0] == 0:
                apz[:,-1,-1] = apz[:,-2,-1]
        
        # conductive volume - add the sum of 3 directions (multiplied by cell area to get volume)
        cv1 = (apx.sum()*csy*csz + apy.sum()*csx*csz + apz.sum()*csx*csy)
        # subtract overlapping areas
        oxy = (apx * apy).sum()*csz
        oxz = (apx * apz).sum()*csy
        oyz = (apy * apz).sum()*csx
        oxyz = (apx * apy * apz).sum()
        
        total_volume = (np.product(apx.shape))*csx*csy*csz
        
        cv = cv1 - oxy - oxz - oyz + oxyz
        
        self.conductive_fraction = cv/total_volume


              
  
    def update_cellsize(self):
        if ((self.fault_assignment in [pre+suf for pre in ['single_','multiple_']\
            for suf in ['xy','yz','xz']]) or (min(self.ncells)==0)):
            for i in range(3):
                apih = self.aperture_hydraulic[:,:,:,:,i][np.isfinite(self.aperture_hydraulic[:,:,:,:,i])]
                apie = self.aperture_electric[:,:,:,:,i][np.isfinite(self.aperture_electric[:,:,:,:,i])]
    
                for api in [apih,apie]:
                    if len(api) > 0:
                        apmax = np.amax(api)
                        if self.cellsize[i] < apmax:
                            rounding = -int(np.ceil(np.log10(self.cellsize[i])))+2
                            # need to use ceil function so it always rounds up
                            self.cellsize[i] = np.ceil(apmax*10.**rounding)*10.**(-rounding)
            

    def initialise_electrical_resistance(self):
        """
        initialise a resistivity array

        """
        
        self.resistance,self.resistivity,self.aperture_electric = \
        rnap.get_electrical_resistance(self.aperture_electric,
                                      self.resistivity_matrix,
                                      self.resistivity_fluid,
                                      self.cellsize,
                                      matrix_current=self.matrix_current)
        rna.add_nulls(self.resistivity)
        rna.add_nulls(self.resistance)

        
    def initialise_permeability(self):
        """
        initialise permeability and hydraulic resistance based on 
        connections set up in resistivity array                           
        
        """
        if not hasattr(self,'resistivity'):
            self.initialise_resistivity()
        


        self.hydraulic_resistance,self.permeability = \
        rnap.get_hydraulic_resistance(self.aperture_hydraulic,
                                     self.permeability_matrix,
                                     self.cellsize,
                                     mu = self.fluid_viscosity,
                                     matrix_flow=self.matrix_flow)
        rna.add_nulls(self.permeability)
        rna.add_nulls(self.hydraulic_resistance)



    def solve_resistor_network(self):
        """
        generate and solve a random resistor network
        properties = string or list containing properties to solve for,
        'current','fluid' or a combination e.g. 'currentfluid'
        direction = string containing directions, 'x','y','z' or a combination
        e.g. 'xz','xyz'
        'x' solves x y and z currents for flow in the x (into page) direction
        'y' solves x y and z currents for flow in the y (horizontal) direction
        'z' solves x y and z currents for flow in the z (vertical) direction
        
        resulting current/fluid flow array:
      x currents  ycurrents  zcurrents
               |      |      |
               v      v      v
            [[xx,    xy,    xz], <-- current modelled in x direction
             [yx,    yy,    yz], <-- current y
             [zx,    zy,    zz]] <-- current z
        
        """
        # set kfactor to divide hydraulic conductivities by so that matrix
        # solving is more accurate. 
#        kfactor = 1e10
        
        property_arrays = {}
        if 'current' in self.solve_properties:
#            if not hasattr(self,'resistance'):
#                self.initialise_resistivity()
            property_arrays['current'] = self.resistance
        if 'fluid' in self.solve_properties:
#            if not hasattr(self,'hydraulic_resistance'):
#                self.initialise_permeability()
            property_arrays['fluid'] = self.hydraulic_resistance 
   
        dx,dy,dz = self.cellsize
        


        for pname in list(property_arrays.keys()):
            nz,ny,nx = np.array(np.shape(property_arrays[pname]))[:-1] - 2
            oa = np.zeros([nz+2,ny+2,nx+2,3,3])#*np.nan

            for dname, nn in [['x',nx],['y',ny],['z',nz]]:
                if dname in self.solve_direction:
                    if nn == 0:
                        self.solve_direction = self.solve_direction.strip(dname)
                        print("not solving {} as there are no resistors in this direction".format(dname))

            if 'x' in self.solve_direction:
                prop = 1.*property_arrays[pname].transpose(2,1,0,3)
                prop = prop[:,:,:,::-1]
                matrix,b = rnmb.build_matrix3d(prop)
                c = rnms.solve_matrix(matrix,b)
                nz,ny,nx = np.array(np.shape(prop))[:-1] - 2
                nfx,nfy,nfz = rnmb.get_nfree([nx,ny,nz])
                oa[1:,1:,:,0,0] = c[-nfz:].reshape(nz+2,ny+1,nx+1).transpose(2,1,0)
                oa[1:,1:-1,1:,0,1] = c[nfx:-nfz].reshape(nz+1,ny,nx+1).transpose(2,1,0)
                oa[1:-1,1:,1:,0,2] = c[:nfx].reshape(nz+1,ny+1,nx).transpose(2,1,0)               
                
            if 'y' in self.solve_direction:
                # transpose array as y direction is now locally the z direction
                prop = 1.*property_arrays[pname].transpose(1,0,2,3)
                # need to swap position of z and y values in the arrays
                prop[:,:,:,1:] = prop[:,:,:,1:][:,:,:,::-1]
                matrix,b = rnmb.build_matrix3d(prop)
                c = rnms.solve_matrix(matrix,b)
                nz,ny,nx = np.array(np.shape(prop))[:-1] - 2
                nfx,nfy,nfz = rnmb.get_nfree([nx,ny,nz])
                oa[1:,1:,1:-1,1,0] = c[:nfx].reshape(nz+1,ny+1,nx).transpose(1,0,2)
                oa[1:,:,1:,1,1] = c[-nfz:].reshape(nz+2,ny+1,nx+1).transpose(1,0,2)
                oa[1:-1,1:,1:,1,2] = c[nfx:-nfz].reshape(nz+1,ny,nx+1).transpose(1,0,2)  
            
            if 'z' in self.solve_direction:
                prop = 1.*property_arrays[pname]
                matrix,b = rnmb.build_matrix3d(prop)
                c = rnms.solve_matrix(matrix,b)
                nz,ny,nx = np.array(np.shape(prop))[:-1] - 2
                nfx,nfy,nfz = rnmb.get_nfree([nx,ny,nz])
                oa[1:,1:,1:-1,2,0] = c[:nfx].reshape(nz+1,ny+1,nx)
                oa[1:,1:-1,1:,2,1] = c[nfx:-nfz].reshape(nz+1,ny,nx+1)
                oa[:,1:,1:,2,2] = c[-nfz:].reshape(nz+2,ny+1,nx+1)  
            self.matrix = matrix
            self.b = b

            if 'current' in pname:
                self.current = 1.*oa
                self.resistivity_bulk, self.resistance_bulk = \
                rnap.get_bulk_resistivity(self.current,self.cellsize,1.)
    
            if 'fluid' in pname:
                self.flowrate=1.*oa
                self.permeability_bulk, self.hydraulic_resistance_bulk  = \
                rnap.get_bulk_permeability(self.flowrate,self.cellsize,self.fluid_viscosity,1.)

        
    
    def solve_resistor_network2(self, Vstart=None, Vsurf=0., Vbase=1., 
                                method = 'direct', itstep=100, tol=0.1,
                                solve_properties=None,solve_direction=None):
        """
        generate and solve a random resistor network by solving for potential
        or pressure rather than current/flow rate.
 
        properties = string or list containing properties to solve for,
        'current','fluid' or a combination e.g. 'currentfluid'
        direction = string containing directions, 'x','y','z' or a combination
        e.g. 'xz','xyz'
        'x' solves x y and z currents for flow in the x (into page) direction
        'y' solves x y and z currents for flow in the y (horizontal) direction
        'z' solves x y and z currents for flow in the z (vertical) direction
        
        resulting current/fluid flow array:
      x currents  ycurrents  zcurrents
               |      |      |
               v      v      v
            [[xx,    xy,    xz], <-- current modelled in x direction
             [yx,    yy,    yz], <-- current y
             [zx,    zy,    zz]] <-- current z
        
        """
        
        if solve_properties is not None:
            self.solve_properties = solve_properties
        if solve_direction is not None:
            self.solve_direction = solve_direction
            
        

        property_arrays = {}
        if 'current' in self.solve_properties:
            property_arrays['current'] = self.resistivity
        if 'fluid' in self.solve_properties:
            property_arrays['fluid'] = rnap.get_hydraulic_resistivity(self.hydraulic_resistance,self.cellsize)
   
        dx,dy,dz = self.cellsize
        nx,ny,nz = self.ncells
        
        for pname in list(property_arrays.keys()):
            output_array = np.zeros([nz+2,ny+2,nx+2,3,3])
            

            for sd in self.solve_direction:
                R = property_arrays[pname].copy()
                # transpose and reorder the conductivity arrays. Default solve
                # direction is z, if it's x or y we need to transpose and 
                # reorder the array. Call the transposed array Rm.
                if sd == 'x':
                    # transpose, and swap x and z in the array by reversing the order
                    Rm = R.copy().transpose(2,1,0,3)[:,:,:,::-1]
#                    if Vstart is not None:
#                        Vstart = Vstart.transpose(2,1,0)
                elif sd == 'y':
                    Rm = R.copy().transpose(1,0,2,3)
                    # swap the order of y and z in the array
                    Rm[:,:,:,-2:] = Rm[:,:,:,-2:][:,:,:,::-1]
#                    if Vstart is not None:
#                        Vstart = Vstart.transpose(1,0,2)
                elif sd == 'z':
                    Rm = R.copy()
                
                if Vstart is not None:
                    if sd == 'x':
                        Vstart = Vstart.transpose(1,2,0)
                    elif sd == 'z':
                        Vstart = Vstart.transpose(1,0,2)
                
                Vn = rnms.solve_matrix2(Rm,self.cellsize,Vsurf=Vsurf,Vbase=Vbase,Vstart=Vstart,
                                        method=method,tol = tol, itstep=itstep)
                if sd == 'x':
                    Vn = Vn.transpose(2,1,0)
                    i = 0
                if sd == 'y':
                    Vn = Vn.transpose(1,0,2)
                    i = 1
                elif sd == 'z':
                    i = 2
                    
                output_array[1:-1,1:,1:,i,2] = (Vn[1:]-Vn[:-1])*dx*dy/(R[1:-1,1:,1:,2]*dz)
                output_array[1:,1:-1,1:,i,1] = (Vn[:,1:]-Vn[:,:-1])*dx*dz/(R[1:,1:-1,1:,1]*dy)
                output_array[1:,1:,1:-1,i,0] = (Vn[:,:,1:]-Vn[:,:,:-1])*dy*dz/(R[1:,1:,1:-1,0]*dx)
                
                for i1,i2 in [[0,1],[-1,-2]]:
                    output_array[i1,:,:,2,2] = output_array[i2,:,:,2,2]
                    output_array[:,i1,:,1,1] = output_array[:,i2,:,1,1]
                    output_array[:,:,i1,0,0] = output_array[:,:,i2,0,0]

            if pname == 'current':
                self.current = output_array*1.
                self.voltage[:,:,:,i] = Vn
                self.resistivity_bulk, self.resistance_bulk = \
                rnap.get_bulk_resistivity(self.current,self.cellsize,Vbase-Vsurf)
                # limit maximum values to resistivity of matrix
                self.resistivity_bulk[self.resistivity_bulk > self.resistivity_matrix] =\
                    self.resistivity_matrix
            elif pname == 'fluid':
                self.pressure[:,:,:,i] = Vn
                self.flowrate = output_array*1.
                self.permeability_bulk, self.hydraulic_resistance_bulk  = \
                rnap.get_bulk_permeability(self.flowrate,self.cellsize,self.fluid_viscosity,Vbase-Vsurf)   
                # limit minimum k to matrix k
                self.permeability_bulk[self.permeability_bulk < self.permeability_matrix] =\
                    self.permeability_matrix
                
        

    def get_effective_apertures(self):
        """
        get effective apertures for a single planar fault down the centre
        of the volume.
        
        calculates a 3x3 array:
        opening in:
      xdirection  ydirection zdirection
       (yz plane) (xz plane) (xy plane)
               |      |      |
               v      v      v
            [[nan,    x(y),  x(z)], <-- x connectors
             [y(x),   nan,   y(z)], <-- y connectors
             [z(x),   z(y),   nan]]  <-- z connectors        
        
        """

        
        if type(self.cellsize) in [int,float]:
            self.cellsize = [self.cellsize]*3
        rhof,rhom = self.resistivity_fluid,self.resistivity_matrix
        km = self.permeability_matrix
        
        self.effective_hydraulic_aperture = np.ones((3,3))*np.nan
        self.effective_electric_aperture = np.ones((3,3))*np.nan
        for i in range(3):
            if 'xyz'[i] in self.solve_direction:
                for odir in range(3):
                    if odir != i:
                        width = self.cellsize[odir]*(self.ncells[odir]+1.)
                        if 'current' in self.solve_properties:
                            rhoeff = self.resistivity_bulk[i]
                            self.effective_electric_aperture [i,odir] = \
                            rnap.get_electric_aperture(width,rhoeff,rhof,rhom)
#                            rnap.get_electric_aperture(width,rhoeff,rhof)

                        if 'fluid' in self.solve_properties:
                            keff = self.permeability_bulk[i]
                            self.effective_hydraulic_aperture[i,odir] = \
                            rnap.get_hydraulic_aperture(width,keff,km)
#                            rnap.get_hydraulic_aperture(keff)

        
