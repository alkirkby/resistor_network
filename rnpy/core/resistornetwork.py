# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:35:11 2015

@author: a1655681
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import resistornetworkfunctions3d as rnf
import sys
import time

class Rock_volume():
    """
    ***************Documentation last updated 8 October 2014*******************
    
    Class to contain volumes to be modelled as a random resistor network.
    wd = working directory
    ncells = list containing number of nodes in the x,y and z direction, 
             default is [10,10,10]
    pconnection = probabilities of connection in the x,y and z directions,
                  default is [0.5,0.5,0.5]
    pembedded_fault
    pembedded_matrix
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
    mu = fluid viscosity
    faultlength_max = maximum fault length if res_type is "random"
    faultlength_decay = decay factor to describe shape of fault length
                        distribution function, default 5
                 
    """
    
    def __init__(self, **input_parameters):
        self.wd = '.' # working directory
        self.ncells = [10,10,10] #ncells in x, y and z directions
        self.cellsize = np.array([1.,1.,1.])
        self.pconnection = np.array([0.5,0.5,0.5])
        self.pembedded_fault = np.array([1.,1.,1.])
        self.pembedded_matrix = np.array([0.,0.,0.])
        self.resistivity_matrix = 1000.
        self.resistivity_fluid = 0.1
        self.resistivity = None
        self.permeability_matrix = 1.e-18
        self.mu = 1.e-3 #default is for freshwater at 20 degrees 
        self.fault_dict = dict(fractal_dimension=2.5,
                               fault_separation = 1e-4,
                               offset = 0,
                               length_max = None,
                               length_decay = 5.,
                               mismatch_frequency_cutoff = None,
                               elevation_standard_deviation = 1e-4,
                               aperture_assignment = 'random',
                               fault_surfaces = None,
                               correct_aperture_for_geometry = True)
        self.fault_array = None                       
        self.fault_edges = None
        self.fault_assignment = 'random' # how to assign faults, 'random' or 'list'
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

        self.build_faults()
        self.initialise_electrical_resistance()
        self.initialise_permeability()

    def build_faults(self):
        """
        initialise a faulted volume. 
        
        """
        nx,ny,nz = self.ncells
        fault_array = np.zeros([nz+2,ny+2,nx+2,3])
        fault_array = rnf._add_nulls(fault_array)
        fault_uvw = []
        
        if self.fault_assignment == 'list':
            if self.fault_edges is not None:
                if np.shape(self.fault_edges)[-2:] == (3,2):
                    if len(np.shape(self.fault_edges)) == 2:
                        self.fault_edges = [self.fault_edges]
                    for fedge in self.fault_edges:
                        fault_array, fuvwi = rnf.add_fault_to_array(fedge,fault_array)
                        fault_uvw.append(fuvwi)
            else:
                print "Can't assign faults, no fault list provided, use random"\
                " assignment or provide fault_edges"
        elif self.fault_assignment =='random':
            fault_array,fault_uvw = \
            rnf.build_random_faults(self.ncells,
                                    self.pconnection,
                                    faultlengthmax = self.fault_dict['length_max'],
                                    decayfactor = self.fault_dict['length_decay'])
        else:
            print "Can't assign faults, invalid fault assignment type provided"
            return
        
        if self.fault_dict['aperture_assignment'] == 'random':
            aperture_input = {}
            for key in ['fractal_dimension','fault_separation','offset',
                        'elevation_standard_deviation', 'fault_surfaces',
                        'mismatch_frequency_cutoff',
                        'correct_aperture_for_geometry']:
                            aperture_input[key] = self.fault_dict[key]
            self.aperture_array,self.aperture_correction_f, \
            self.aperture_correction_c,bvals = \
            rnf.assign_fault_aperture(fault_array,fault_uvw,**aperture_input)
        else:
            self.aperture_array = fault_array*self.fault_dict['fault_separation']
            self.aperture_array[self.aperture_array < 1e-50] = 1e-50
            self.aperture_correction_f,self.aperture_correction_c = \
            [np.ones_like(self.aperture_array)]*2
        self.fault_array = fault_array
        self.fault_uvw = np.array(fault_uvw)
        

    def initialise_electrical_resistance(self):
        """
        initialise a resistivity array

        """
        
        self.resistance = \
        rnf.get_electrical_resistance(self.aperture_array*self.aperture_correction_c,
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
        rnf.get_permeability(self.aperture_array*self.aperture_correction_f,
                             self.permeability_matrix,
                             self.cellsize)
        self.hydraulic_resistance = \
        rnf.get_hydraulic_resistance(self.aperture_array*self.aperture_correction_f,
                                     self.permeability_matrix,
                                     self.cellsize,
                                     mu = self.mu)


    def solve_resistor_network(self,properties,direction):
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
        if 'current' in properties:
#            if not hasattr(self,'resistance'):
#                self.initialise_resistivity()
            property_arrays['current'] = self.resistance
        if 'fluid' in properties:
#            if not hasattr(self,'hydraulic_resistance'):
#                self.initialise_permeability()
            property_arrays['fluid'] = self.hydraulic_resistance 
   
        dx,dy,dz = [float(n) for n in self.cellsize]      

        for pname in property_arrays.keys():
            nz,ny,nx = np.array(np.shape(property_arrays[pname]))[:-1] - 2
            oa = np.zeros([nz+2,ny+2,nx+2,3,3])#*np.nan

            if 'x' in direction:
                prop = 1.*property_arrays[pname].transpose(2,1,0,3)
                prop = prop[:,:,:,::-1]
                matrix,b = rnf.build_matrix3d(prop)
                c = rnf.solve_matrix(matrix,b)
                nz,ny,nx = np.array(np.shape(prop))[:-1] - 2
                nfx,nfy,nfz = rnf.get_nfree([nx,ny,nz])
                oa[1:,1:,:,0,0] = c[-nfz:].reshape(nz+2,ny+1,nx+1).transpose(2,1,0)
                oa[1:,1:-1,1:,0,1] = c[nfx:-nfz].reshape(nz+1,ny,nx+1).transpose(2,1,0)
                oa[1:-1,1:,1:,0,2] = c[:nfx].reshape(nz+1,ny+1,nx).transpose(2,1,0)               
            
            if 'y' in direction:
                # transpose array as y direction is now locally the z direction
                prop = 1.*property_arrays[pname].transpose(1,0,2,3)
                # need to swap position of z and y values in the arrays
                prop[:,:,:,1:] = prop[:,:,:,1:][:,:,:,::-1]
                matrix,b = rnf.build_matrix3d(prop)
                c = rnf.solve_matrix(matrix,b)
                nz,ny,nx = np.array(np.shape(prop))[:-1] - 2
                nfx,nfy,nfz = rnf.get_nfree([nx,ny,nz])
                oa[1:,1:,1:-1,1,0] = c[:nfx].reshape(nz+1,ny+1,nx).transpose(1,0,2)
                oa[1:,:,1:,1,1] = c[-nfz:].reshape(nz+2,ny+1,nx+1).transpose(1,0,2)
                oa[1:-1,1:,1:,1,2] = c[nfx:-nfz].reshape(nz+1,ny,nx+1).transpose(1,0,2)  
            
            if 'z' in direction:
                prop = 1.*property_arrays[pname]
                matrix,b = rnf.build_matrix3d(prop)
                c = rnf.solve_matrix(matrix,b)
                nz,ny,nx = np.array(np.shape(prop))[:-1] - 2
                nfx,nfy,nfz = rnf.get_nfree([nx,ny,nz])
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
                self.permeability_bulk = self.mu/(self.hydraulic_resistance_bulk*factor)
            