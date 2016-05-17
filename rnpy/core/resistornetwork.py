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
    ***************Documentation last updated 8 October 2014*******************
    
    Class to contain volumes to be modelled as a random resistor network.
    workdir = working directory
    ncells = list containing number of nodes in the x,y and z direction, 
             default is [10,10,10]
    pconnection = list of probability of connection in the x,y, and z direction if random faults, default 0.5
    cellsize = size of cells, same in x,y and z directions
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
        self.cellsize = 1e-3
        self.pconnection = [0.5,0.5,0.5]
        self.resistivity_matrix = 1000.
        self.resistivity_fluid = 0.1
        self.resistivity = None
        self.permeability_matrix = 1.e-18
        self.fluid_viscosity = 1.e-3 #default is for freshwater at 20 degrees 
        self.fault_dict = dict(fractal_dimension=2.5,
                               fault_separation = 1e-4,
                               offset = 0,
                               faultlength_max = np.amax(self.cellsize)*np.amax(self.ncells),
                               faultlength_min = np.amax(self.cellsize),
                               alpha = 10.,
                               a = 3.5,
                               mismatch_wavelength_cutoff = None,
                               elevation_scalefactor = 1e-3,
                               aperture_type = 'random',
                               fault_surfaces = None,
                               correct_aperture_for_geometry = True,
                               fault_spacing = 2)
        self.fault_array = None      
        self.fault_edges = None
        self.fault_assignment = 'single_yz' # how to assign faults, 'random' or 'list', or 'single_yz'
        self.aperture = None
        self.aperture_electric = None
        self.aperture_hydraulic = None
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
        
        if type(self.ncells) in [float,int]:
            self.ncells = np.ones(3)*self.ncells
        if type(self.cellsize) in [float,int]:
            self.cellsize = np.ones(3)*self.cellsize
        else:
            if (type(self.fault_assignment) == 'single_yz') and len(self.cellsize == 3):
                if self.cellsize[1] != self.cellsize[2]:
                    print "y cellsize not equal to z cellsize, updating z cellsize"
                    self.cellsize[2] = self.cellsize[1]
                else:
                    self.cellsize = [np.amin(self.cellsize)]*3

        if self.build:
       #     print "building faults"
            self.build_faults()
       #     print "building aperture"
            self.build_aperture()
       #     print "initialising electrical resistance"
            self.initialise_electrical_resistance()
       #     print "initialising permeability"
            self.initialise_permeability()

    def build_faults(self):
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
            if self.fault_array.shape != (nz,ny,nx,3,3):
                print "Fault array does not conform to dimensions of network, creating a new array!"
                self.fault_array= None
                
        
        if self.fault_array is None:
            print "initialising a new array"
            # initialise a fault array
            self.fault_array = np.zeros([nz+2,ny+2,nx+2,3,3])
            # add nulls to the edges
            self.fault_array = rna.add_nulls(self.fault_array)
            
            addfaults = False

            # option to specify fault edges as a list
            if self.fault_assignment == 'list':
                if self.fault_edges is not None:
                    # check the dimensions of the fault edges
                    if np.shape(self.fault_edges)[-2:] == (2,2,3):
                        if len(np.shape(self.fault_edges)) == 3:
                            self.fault_edges = np.array([self.fault_edges])
                        addfaults = True
                    
                        
            # option to specify a single fault in the centre of the yz plane
            if self.fault_assignment == 'single_yz':
                ix = int(nx/2) + 1
                iy0, iy1 = 1, ny + 1
                iz0, iz1 = 1, nz + 1
                self.fault_edges = np.array([[[ix,iy0,iz0],[ix,iy1,iz0]],
                                             [[ix,iy0,iz1],[ix,iy1,iz1]]])
                addfaults = True
            
            elif self.fault_assignment == 'multiple_yz':
                if self.fault_dict['fault_spacing'] > ny/2:
                    self.fault_dict['fault_spacing'] = ny/2
                self.fault_dict['fault_spacing'] = int(self.fault_dict['fault_spacing'])
                iy0, iy1 = 1, ny + 1
                iz0, iz1 = 1, nz + 1
                self.fault_edges = np.array([[[[ix,iy0,iz0],[ix,iy1,iz0]],
                                              [[ix,iy0,iz1],[ix,iy1,iz1]]] for ix in range(1,nx + 1,self.fault_dict['fault_spacing'])])
                addfaults = True                
    
    
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
                fracturecoords = rnaf.get_fracture_coords(lvals,networksize,pxyz,return_Nf = False,a=3.5,alpha=10.)
                self.fault_edges = rnaf.coords2indices(fracturecoords,networksize,[nx,ny,nz])

                addfaults = True
            if addfaults:
                rnaf.add_faults_to_array(self.fault_array,self.fault_edges)
            else:
                print "Can't assign faults, invalid fault assignment type or invalid fault edges list provided"
                return

            
    def build_aperture(self):
        
        if self.fault_assignment == 'single_yz':
            # if (and only if) all faults are along yz plane cellsize perpendicular 
            # is allowed to be different from cell size along fault.
            cellsize = self.cellsize[1]
        else:
            cellsize = np.amin(self.cellsize)
        
        if self.aperture is None:
            if self.fault_dict['aperture_type'] == 'random':
                    
                aperture_input = {}
            #    print "getting fault pair defaults"
                self.fault_dict['mismatch_wavelength_cutoff'], fc = \
                rnfa.get_faultpair_defaults(cellsize,
                                            self.fault_dict['mismatch_wavelength_cutoff'] 
                                            )
           #     print "getting keys"
                for key in ['fractal_dimension','fault_separation','offset',
                            'elevation_scalefactor', 'fault_surfaces',
                            'mismatch_wavelength_cutoff',
                            'correct_aperture_for_geometry']:
                                aperture_input[key] = self.fault_dict[key]
          #      print "assigning fault aperture"
                self.aperture,self.aperture_hydraulic, \
                self.aperture_electric,self.fault_dict['fault_surfaces'] = \
                rnaf.assign_fault_aperture(self.fault_array,self.fault_edges,**aperture_input)
            else:
         #       print "no need to assign new aperture array as aperture already provided"
                self.aperture = self.fault_array*self.fault_dict['fault_separation']
                self.aperture[(self.aperture < 1e-50)] = 1e-50
                self.fault_dict['fault_heights'] = np.ones()
                self.aperture_hydraulic,self.aperture_electric = \
                [self.aperture.copy()]*2
        
        # get the aperture values from the faulted part of the volume to do some calculations on
        #print "getting fault aperture values"
        faultapvals = [self.aperture[:,:,:,i][(self.fault_array[:,:,:,i].astype(bool))&(np.isfinite(self.aperture[:,:,:,i]))] \
                      for i in range(3)]
#        print "faultapvals size",[np.size(fv) for fv in faultapvals],"mean aperture",
        #print "calculating mean ap and contact area"
        self.aperture_mean = [np.mean(faultapvals[i]) for i in range(3)]
#        print self.aperture_mean,"separation",self.fault_dict['fault_separation']
        self.contact_area = []
        for i in range(3):
            if np.size(faultapvals[i]) > 0:
                self.contact_area.append(float(len(faultapvals[i][faultapvals[i] <= 1e-50]))/np.size(faultapvals[i]))
            else:
                self.contact_area.append(0.) 
        if self.aperture_hydraulic is None:
            self.aperture_hydraulic = self.aperture.copy()
        if self.aperture_electric is None:
            self.aperture_electric = self.aperture.copy()
        
        # update cellsize so it is at least as big as the largest fault aperture
        # but only if it's a 2d network or there are only faults in one direction
        if (('yz' in self.fault_assignment) or (min(self.ncells)==0) or \
            (np.count_nonzero(self.pconnection) == 1)):
            for i in range(3):
                apih = self.aperture_hydraulic[:,:,:,:,i][np.isfinite(self.aperture_hydraulic[:,:,:,:,i])]
                apie = self.aperture_electric[:,:,:,:,i][np.isfinite(self.aperture_electric[:,:,:,:,i])]
    
                for api in [apih,apie]:
                    if len(api) > 0:
                        apmax = np.amax(api)
                        if self.cellsize[i] < apmax:
                            rounding = -int(np.ceil(np.log10(self.cellsize[i])))+2
                            self.cellsize[i] = round(apmax,rounding)
            

    def initialise_electrical_resistance(self):
        """
        initialise a resistivity array

        """
        
        self.resistance,self.resistivity = \
        rnap.get_electrical_resistance(self.aperture_electric,
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
        


        self.hydraulic_resistance,self.permeability = \
        rnap.get_hydraulic_resistance(self.aperture_hydraulic,
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
        


        for pname in property_arrays.keys():
            nz,ny,nx = np.array(np.shape(property_arrays[pname]))[:-1] - 2
            oa = np.zeros([nz+2,ny+2,nx+2,3,3])#*np.nan

            for dname, nn in [['x',nx],['y',ny],['z',nz]]:
                if dname in self.solve_direction:
                    if nn == 0:
                        self.solve_direction = self.solve_direction.strip(dname)
                        print "not solving {} as there are no resistors in this direction".format(dname)

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
                rnap.get_bulk_resistivity(self.current,self.cellsize)
    
            if 'fluid' in pname:
                self.flowrate=1.*oa
                self.permeability_bulk, self.hydraulic_resistance_bulk  = \
                rnap.get_bulk_permeability(self.flowrate,self.cellsize,self.fluid_viscosity)

    
    def solve_resistor_network2(self, Vstart=None, Vsurf=0., Vbase=1., 
                                method = 'direct', itstep=100, tol=0.1,
                                solve_properties=None):
        """
        generate and solve a random resistor network using the relaxation method
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

        property_arrays = {}
        if 'current' in self.solve_properties:
            property_arrays['current'] = self.resistivity
        if 'fluid' in self.solve_properties:
            property_arrays['fluid'] = rnap.get_hydraulic_resistivity(self.hydraulic_resistance,self.cellsize)
   
        dx,dy,dz = self.cellsize
        nx,ny,nz = self.ncells

        for pname in property_arrays.keys():
            output_array = np.zeros([nz+2,ny+2,nx+2,3,3])

            for sd in self.solve_direction:
                R = property_arrays[pname]
                # transpose and reorder the conductivity arrays. Default solve
                # direction is z, if it's x or y we need to transpose and 
                # reorder the array
                if sd == 'x':
                    Rx = R.copy().transpose(2,1,0,3)[:,:,:,::-1]
                    C = rnmb.Conductivity(Rx)
                    lnz,lny,lnx = np.array(Rx.shape[:-1])-2
                    ldx,ldy,ldz = dz,dy,dx
                elif sd == 'y':
                    Ry = R.copy().transpose(1,0,2,3)
                    # swap y and z in the array
                    Ry[:,:,:,-2:] = Ry[:,:,:,-2:][:,:,:,::-1]
                    C = rnmb.Conductivity(Ry)
                    # "local" nx, ny, nz now that we've transposed the array
                    lnz,lny,lnx = np.array(Ry.shape[:-1])-2
                    ldx,ldy,ldz = dx,dz,dy
                elif sd == 'z':
                    C = rnmb.Conductivity(R)
                    lnz,lny,lnx = np.array(R.shape[:-1])-2
                    ldx,ldy,ldz = dx,dy,dz
                # initialise a default starting array if needed
                if Vstart is None:
                    Vo = np.zeros((lnx+1,lny+1,lnz+1))
                    Vo[:,:,:] = np.linspace(Vsurf,Vbase,lnz+1)
                    # transpose so that array is ordered by y, then z, then x
                    Vo = Vo.transpose(1,2,0)
                else:
                    Vo = Vstart.copy()
            
                A,D = rnmb.buildmatrix(C,ldx,ldy,ldz)
                b = rnmb.buildb(C,ldz,Vsurf,Vbase)
                
                Vn = rnms.solve_matrix2(Vo,C,A,D,b,[lnx,lny,lnz],[ldx,ldy,ldz], 
                                        method=method,tol = tol, w = 1.3, itstep=itstep)
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
                self.resistivity_bulk, self.resistance_bulk = \
                rnap.get_bulk_resistivity(self.current,self.cellsize)
            elif pname == 'fluid':
                self.flowrate = output_array*1.
                self.permeability_bulk, self.hydraulic_resistance_bulk  = \
                rnap.get_bulk_permeability(self.flowrate,self.cellsize,self.fluid_viscosity)                
        

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
                
        
        
        
        