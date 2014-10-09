# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:05:42 2014

@author: Alison Kirkby

Modelling random resistor networks using python.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import resistornetworkfunctions3d as rnf

class Resistivity_volume():
    """
    ***************Documentation last updated 8 October 2014*******************
    
    Class to contain volumes to be modelled as a random resistor network.
    wd = working directory
    nx,ny,nz = number of nodes in the x,y and z direction, default is 10
    px,py,pz = probabilities of connection in the x,y and z directions
    dx,dy,dz = size of cells in x,y and z directions
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
        self.nx = 10
        self.ny = 10
        self.nz = 10
        self.px = 0.5
        self.py = 0.5
        self.pz = 0.5
        self.dx = 1.
        self.dy = 1.
        self.dz = 1.
        self.res_type = 'ones'
        self.resistivity_matrix = 1000
        self.resistivity_fluid = 0.1
        self.resistivity = None
        self.permeability_matrix = 1.e-18
        self.fracture_diameter = 1.e-3
        self.mu = 1.e-3 #default is for freshwater at 20 degrees 
        self.faultlength_max = None
        self.faultlength_decay = 5.                         

        
        update_dict = {}
        #correcting dictionary for upper case keys
        input_parameters_nocase = {}
        for key in input_parameters.keys():
            if hasattr(self,key):
                input_parameters_nocase[key.lower()] = input_parameters[key]

        update_dict.update(input_parameters_nocase)

        for key in update_dict:
            try:
                value = getattr(self,key)
                if type(value) == str:
                    try:
                        value = float(update_dict[key])
                    except:
                        value = update_dict[key]
                else:
                    value = update_dict[key]
                setattr(self,key,value)
            except:
                continue 

        if self.res_type == 'array':
            if self.resistivity is None:
                print "Please provide resistivity array or specify a different res_type"
                return
            else:
                self.nz,self.ny,self.nx = [int(i) for i in np.shape(self.resistivity)]

        self.initialise_resistivity()
        self.initialise_permeability()



    def initialise_resistivity(self):
        """
        initialise a resistivity array

        """
                                                                                                                                                                                    
        if self.res_type == 'ones':
            self.resistivity = np.ones([self.nz+2,self.ny+2,self.nx+2,3])*np.nan
            self.resistivity[1:,1:,1:-1,0] = 1.
            self.resistivity[1:,1:-1,1:,1] = 1.
            self.resistivity[1:-1,1:,1:,2] = 1.

        elif self.res_type == "random":
            self.resistivity,self.faults = \
            rnf.assign_random_resistivity([self.nx,self.ny,self.nz],
                                          [self.px,self.py,self.pz],
                                           self.resistivity_matrix,
                                           self.resistivity_fluid,
                                           faultlengthmax=self.faultlength_max,
                                           decayfactor = self.faultlength_decay)
            
        elif self.res_type == 'array':
            # resistivity fed in as a variable in initialisation so don't need
            # to create it
            self.resistivity_matrix = np.amax(self.resistivity[np.isfinite(self.resistivity)])
            self.resistivity_fluid = np.amin(self.resistivity[np.isfinite(self.resistivity)])
        else:
            print "res type {} not supported, please redefine".format(self.res_type)
            return
            
        d = [self.dx,self.dy,self.dz]
        self.resistance = rnf.get_electrical_resistance(self.resistivity,
                                                        self.resistivity_matrix,
                                                        self.resistivity_fluid,
                                                        d,
                                                        self.fracture_diameter)
        self.phi = sum(rnf.get_phi(d,self.fracture_diameter))


    def initialise_permeability(self):
        """
        initialise permeability and hydraulic resistance based on 
        connections set up in resistivity array                           
        
        """
        if not hasattr(self,'resistivity'):
            self.initialise_resistivity()
        
        d = [self.dz,self.dy,self.dx]
        
        self.permeability = rnf.get_permeability(self.resistivity,
                                                 self.permeability_matrix,
                                                 self.fracture_diameter)
        self.hydraulic_resistance = \
        rnf.get_hydraulic_resistance(self.permeability,
                                     self.permeability_matrix,
                                     d,
                                     self.fracture_diameter,
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
             resx resy resz
               |   |   |
               v   v   v
            [[xx, xy, xz], <-- flow modelled in x direction
             [yx, yy, yz], <-- flow y
             [zx, zy, zz]] <-- flow z
        
        """
        # set kfactor to divide hydraulic conductivities by so that matrix
        # solving is more accurate. 
#        kfactor = 1e10
        
        property_arrays = {}
        if 'current' in properties:
            if not hasattr(self,'resistance'):
                self.initialise_resistivity()
            property_arrays['current'] = self.resistance
        if 'fluid' in properties:
            if not hasattr(self,'hydraulic_resistance'):
                self.initialise_permeability()
            property_arrays['fluid'] = self.hydraulic_resistance 
        
   
        dx,dy,dz = [float(n) for n in self.dx,self.dy,self.dz]      

        for pname in property_arrays.keys():
            nz,ny,nx = np.array(np.shape(property_arrays[pname]))[:-1] - 2
            nfx,nfy,nfz = rnf.get_nfree([nx,ny,nz])
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
                self.permeability_bulk = flow*self.mu/factor
                self.hydraulic_resistance_bulk = 1./flow
