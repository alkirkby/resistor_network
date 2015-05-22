# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:35:11 2015

@author: a1655681
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import rnpy.functions.assignfaults as rnaf
import rnpy.functions.assignproperties as rnap
import rnpy.functions.matrixbuild as rnmb
import rnpy.functions.matrixsolve as rnms
import rnpy.functions.array as rna
import rnpy.functions.faultaperture as rnfa
import sys
import time

class Rock_volume():
    """
    ***************Documentation last updated 8 October 2014*******************
    
    Class to contain volumes to be modelled as a random resistor network.
    workdir = working directory
    ncells = list containing number of nodes in the x,y and z direction, 
             default is [10,10,10]
    pconnectionx = probability of connection in the x direction if random faults, default 0.5
    pconnectiony = probability of connection in the y direction if random faults, default 0.5
    pconnectionz = probability of connection in the z direction if random faults, default 0.5
    cellsize = size of cells in x,y and z directions
    res_type =  string describing how to calculate the resistivity structure;
                options are "ones" (default; fully connected network), 
                            "random" (random network with some high resistivity bonds
                                      assigned according to px,py,pz),
                            "array" (resistivity network given as a numpy array)
                            "file" !!!! not yet implemented !!!! (network given by file) 
    resistivity_matrix = resistivity of the low conductivity matrix
    resistivity_fluid = resistivity of the high conductivity fluid. Used with 
                        fracture diameter to calculate the resistance of 
                        connected bonds
    resistivity = if res_type above is array, provide the resistivity array
    permeability_matrix = permeability of low electrical conductivity matrix
    fracture_diameter = diameter of fractures for connected cells
    fluid_viscosity = fluid viscosity, default for freshwater at 20 degrees
    faultlength_max = maximum fault length if res_type is "random"
    faultlength_decay = decay factor to describe shape of fault length
                        distribution function, default 5
                 
    """
    
    def __init__(self, **input_parameters):
        self.workdir = '.' # working directory
        self.ncells = [10,10,10] #ncells in x, y and z directions
        self.cellsize = np.array([1.,1.,1.])
        self.pconnectionx = 0.5
        self.pconnectiony = 0.5
        self.pconnectionz = 0.5
        self.resistivity_matrix = 1000.
        self.resistivity_fluid = 0.1
        self.resistivity = None
        self.permeability_matrix = 1.e-18
        self.fluid_viscosity = 1.e-3 #default is for freshwater at 20 degrees 
        self.fault_dict = dict(fractal_dimension=2.5,
                               fault_separation = 1e-4,
                               offset = 0,
                               length_max = 10.,
                               length_decay = 5.,
                               mismatch_wavelength_cutoff = None,
                               elevation_standard_deviation = 1e-4,
                               aperture_type = 'random',
                               fault_surfaces = None,
                               correct_aperture_for_geometry = True)
        self.fault_array = None
        self.fault_uvw = None                
        self.fault_edges = None
        self.fault_assignment = 'random' # how to assign faults, 'random' or 'list'
        self.aperture_array = None
        self.aperture_correction_c = None
        self.aperture_correction_f = None
        self.solve_properties = 'currentfluid'
        self.solve_direction = 'xyz'
        self.build = True
        update_dict = {}
        #correcting dictionary for upper case keys
        input_parameters_nocase = {}
        for key in input_parameters.keys():
            # only assign if it's a valid attribute
            if hasattr(self,key):
                input_parameters_nocase[key.lower()] = input_parameters[key]
            else:
                for dictionary in [self.fault_dict]:
                    if key in dictionary.keys():
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
                    if key in self.fault_dict.keys():
                        try:
                            value = float(update_dict[key])
                        except:
                            value = update_dict[key]
                        self.fault_dict[key] = value
                except:
                    continue 
        
        if type(self.cellsize) in [float,int]:
            self.cellsize = np.ones(3)*self.cellsize

        if self.build:
            self.build_faults()
            self.build_aperture()
            self.initialise_electrical_resistance()
            self.initialise_permeability()

    def build_faults(self):
        """
        initialise a faulted volume. 
        
        """
        nx,ny,nz = self.ncells
        if self.fault_array is None:
            fault_array = np.zeros([nz+2,ny+2,nx+2,3])
            fault_array = rna.add_nulls(fault_array)
            fault_uvw = []
        
            if self.fault_assignment == 'list':
                if self.fault_edges is not None:
                    if np.shape(self.fault_edges)[-2:] == (3,2):
                        if len(np.shape(self.fault_edges)) == 2:
                            self.fault_edges = [self.fault_edges]
                        for fedge in self.fault_edges:
                            fault_array = rnaf.add_fault_to_array(fedge,fault_array)
                            fuvwi = rnaf.minmax2uvw(fedge)
                            fault_uvw.append(fuvwi)
                else:
                    print "Can't assign faults, no fault list provided, use random"\
                    " assignment or provide fault_edges"
            elif self.fault_assignment =='random':
                pc = [self.pconnectionx,self.pconnectiony,self.pconnectionz]
                fault_array,fault_uvw = \
                rnaf.build_random_faults(self.ncells,
                                         pc,
                                         faultlengthmax = self.fault_dict['length_max'],
                                         decayfactor = self.fault_dict['length_decay'])
            else:
                print "Can't assign faults, invalid fault assignment type provided"
                return

            self.fault_array = fault_array
            self.fault_uvw = np.array(fault_uvw)
            
    def build_aperture(self):
        
        if self.aperture_array is None:
            if self.fault_dict['aperture_type'] == 'random':
#                if self.fault_dict['fault_surfaces'] is None:
#                    self.fault_dict['fault_surfaces'] = []
#                    for nn in self.fault_uvw:
#                        u0,v0,w0 = np.amin(nn, axis=(1,2))
#                        u1,v1,w1 = np.amax(nn, axis=(1,2))
#                        duvw = np.array([u1-u0,v1-v0,w1-w0])
#                        size = rnaf.get_faultsize(duvw,self.fault_dict['offset'])
#                        faultpair_inputs = dict(D=self.fault_dict['fractal_dimension'],
#                                                std=self.fault_dict['elevation_standard_deviation'],
#                                                cs=self.cellsize[0],
#                                                lc=self.fault_dict['mismatch_wavelength_cutoff'])
#                        self.fault_dict['fault_surfaces'].append(rnfa.build_fault_pair(size,**faultpair_inputs))
                    
                aperture_input = {}
                for key in ['fractal_dimension','fault_separation','offset',
                            'elevation_standard_deviation', 'fault_surfaces',
                            'mismatch_wavelength_cutoff',
                            'correct_aperture_for_geometry']:
                                aperture_input[key] = self.fault_dict[key]
                
                self.aperture_array,self.aperture_correction_f, \
                self.aperture_correction_c,self.fault_dict['fault_surfaces'] = \
                rnaf.assign_fault_aperture(self.fault_array,self.fault_uvw,**aperture_input)
            else:
                self.aperture_array = self.fault_array*self.fault_dict['fault_separation']
                self.aperture_array[(self.aperture_array < 1e-50)] = 1e-50
                self.fault_dict['fault_heights'] = np.ones()
                self.aperture_correction_f,self.aperture_correction_c = \
                [np.ones_like(self.aperture_array)]*2
        
        # get the aperture values from the faulted part of the volume to do some calculations on
        faultapvals = self.aperture_array[(self.fault_array.astype(bool))&np.isfinite(self.aperture_array)]
        self.aperture_mean = np.average(faultapvals)
        self.contact_area = float(len(faultapvals[faultapvals <= 1e-50]))/np.size(faultapvals)
 
        if self.aperture_correction_f is None:
            self.aperture_correction_f = np.ones_like(self.aperture_array)
        if self.aperture_correction_c is None:
            self.aperture_correction_c = np.ones_like(self.aperture_array)
        

    def initialise_electrical_resistance(self):
        """
        initialise a resistivity array

        """
        
        self.resistance = \
        rnap.get_electrical_resistance(self.aperture_array*self.aperture_correction_c,
                                      self.resistivity_matrix,
                                      self.resistivity_fluid,
                                      self.cellsize)
        
        
    def initialise_permeability(self):
        """
        initialise permeability and hydraulic resistance based on 
        connections set up in resistivity array                           
        
        """
        if not hasattr(self,'resistivity'):
            self.initialise_resistivity()
        

        self.permeability = \
        rnap.get_permeability(self.aperture_array*self.aperture_correction_f,
                             self.permeability_matrix,
                             self.cellsize)
        self.hydraulic_resistance = \
        rnap.get_hydraulic_resistance(self.aperture_array*self.aperture_correction_f,
                                     self.permeability_matrix,
                                     self.cellsize,
                                     mu = self.fluid_viscosity)


    def solve_resistor_network(self):
        """
        generate and solve a random resistor network
        properties = string or list containing properties to solve for,
        'current','fluid' or a combination e.g. 'currentfluid'
        direction = string containing directions, 'x','y','z' or a combination
        e.g. 'xz','xyz'
        'x' solves x y and z currents for flow in the x (horizontal) direction
        'y' solves x y and z currents for flow in the y direction (into page)
        'z' solves x y and z currents for flow in the z (vertical) direction
        
        resulting current/fluid flow array:
      x currents  ycurrents  zcurrents
               |      |      |
               v      v      v
            [[xx,    xy,    xz], <-- flow modelled in x direction
             [yx,    yy,    yz], <-- flow y
             [zx,    zy,    zz]] <-- flow z
        
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
   
        dx,dy,dz = [float(n) for n in self.cellsize]      

        for pname in property_arrays.keys():
            nz,ny,nx = np.array(np.shape(property_arrays[pname]))[:-1] - 2
            oa = np.zeros([nz+2,ny+2,nx+2,3,3])#*np.nan

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
            

            flow = np.array([np.sum(oa[:,:,-1,0,0]),
                             np.sum(oa[:,-1,:,1,1]),
                             np.sum(oa[-1,:,:,2,2])])
             
            factor = np.array([dz*dy*(ny+1)*(nz+1)/(dx*nx),
                               dz*dx*(nx+1)*(nz+1)/(dy*ny),
                               dy*dx*(nx+1)*(ny+1)/(dz*nz)])

            if 'current' in pname:
                self.current = 1.*oa
                self.resistance_bulk = 1./flow
                self.resistivity_bulk = factor*self.resistance_bulk
    
            if 'fluid' in pname:
                self.flowrate = 1.*oa
                self.hydraulic_resistance_bulk = 1./flow
                self.permeability_bulk = self.fluid_viscosity/(self.hydraulic_resistance_bulk*factor)
            
