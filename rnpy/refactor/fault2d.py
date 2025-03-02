# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:10:04 2025

@author: alisonk
"""
import numpy as np


class Fault2d():
    def init(self,**inputs):
        self.workdir = '.' # working directory
        self.ncells = 10 # number of cells in x, y directions
        self.cellsize = 1e-3
        self.update_cellsize_tf = True # option to update cellsize if fault is wider than cellsize, only works if there are only faults in one direction.
        self.resistivity_matrix = 1000.
        self.resistivity_fluid = 0.1
        self.permeability_matrix = 1.e-18
        self.permeability_gouge = 1.e-14
        self.porosity_gouge = 0.3
        self.fluid_viscosity = 1.e-3 #default is for freshwater at 20 degrees 
        self.fault_dict = dict(fault_surfaces = None,
                               fractal_dimension=2.5,
                               fault_separation = 1e-4,
                               offset = 0,
                               deform_fault_surface=False,
                               mismatch_wavelength_cutoff = None,
                               elevation_scalefactor = 1e-3,
                               aperture_list = None,
                               random_numbers_dir=None,
                               correct_aperture_for_geometry = False,
                               preserve_negative_apertures = False)
        self.solve_direction = 'xy'
        self.build_arrays = True
        
        update_dict = {}
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
            self.ncells = (np.ones(2)*self.ncells).astype(int)
            

            
            
    def build_aperture(self):
        
        # build two random fault surfaces
        
        
        # 
        
        
