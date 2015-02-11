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
                               aperture_assignment = 'random')
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
                        'elevation_standard_deviation',
                        'mismatch_frequency_cutoff']:
                            aperture_input[key] = self.fault_dict[key]
            self.aperture_array,bvals = \
            rnf.assign_fault_aperture(fault_array,fault_uvw,**aperture_input)
        else:
            self.aperture_array = fault_array*self.fault_dict['fault_separation']
        self.fault_array = fault_array
        self.fault_uvw = np.array(fault_uvw)
        

    def initialise_electrical_resistance(self):
        """
        initialise a resistivity array

        """
        
        self.resistance = \
        rnf.get_electrical_resistance(self.aperture_array,
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
        rnf.get_permeability(self.aperture_array,
                             self.permeability_matrix,
                             self.cellsize)
        self.hydraulic_resistance = \
        rnf.get_hydraulic_resistance(self.aperture_array,
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
            

class RandomResistorSuite():
    """
    organise and run a suite of resistivity/fluid flow runs and save results
    to a text file
    
    Author: Alison Kirkby
    """
    def __init__(self, **input_parameters):
        self.wd = 'outputs' # working directory
        self.ncells = [10,10,10]
        self.cellsize = np.array([1.,1.,1.])
        self.pconnection = np.array([[0.5,0.5,0.5]])
        self.pembedded_fault = np.array([[1.,1.,1.]])
        self.pembedded_matrix = np.array([[0.,0.,0.]])
        self.repeats = 1
        self.resistivity_matrix = 1000.
        self.resistivity_fluid = 0.1
        self.resistivity = None
        self.permeability_matrix = 1.e-18
        self.mu = 1.e-3 #default is for freshwater at 20 degrees 
        self.fault_dict = dict(fractal_dimension=2.5,
                               fault_separation = [1e-4],
                               offset = 0,
                               length_max = None,
                               length_decay = 5.,
                               mismatch_frequency_cutoff = None,
                               elevation_standard_deviation = [1e-4],
                               aperture_assignment = 'random')
        self.fault_array = None                       
        self.fault_edges = None
        self.fault_assignment = 'random' # how to assign faults, 'random' or 'list'
        self.outfile = 'outputs'                        
        self.arguments = sys.argv[1:]
        self.solve_properties = 'currentfluid'
        self.solve_directions = 'xyz'  
        self.input_parameters = input_parameters

        if len(self.arguments) > 0:
            self.read_arguments()
            

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

        
        self.setup_and_run_suite()

    def read_arguments(self):
        """
        takes list of command line arguments obtained by passing in sys.argv
        reads these and updates attributes accordingly
        """
        
        import argparse
        
                
        parser = argparse.ArgumentParser()
        parser.add_argument('-n','--ncells',
                            help = 'number of cells x,y and z direction',
                            nargs = 3,
                            type = int)
        parser.add_argument('-c','--cellsize',
                            help = 'number of cells x,y and z direction',
                            nargs = 3,
                            type = float)
        parser.add_argument('-p','--pconnection',
                            help = 'probability of connection in x y and z direction',
                            nargs = '*',
                            type = float)
        parser.add_argument('-pef','--pembedded_fault',
                            help = 'probability of embedment in a connected '\
                            'cell in x y and z direction',
                            nargs = '*',
                            type = float)
        parser.add_argument('-pem','--pembedded_matrix',
                            help = 'probability of embedment in an unconnected'\
                            ' cell in x y and z direction',
                            nargs = '*',
                            type = float)
        parser.add_argument('-pf','--probabilityfile',
                            help = 'space delimited text file containing '\
                            'probabilities (space delimited, order as follows'\
                            ': px py pz pefx pefy pefz pemx pemy pemz), '\
                            'alternative to command line entry, overwrites'\
                            'command line inputs')
        parser.add_argument('-r','--repeats',
                            help='number of repeats at each probability value',
                            type=int)
        parser.add_argument('-rm','--resistivity_matrix',
                            type=float)
        parser.add_argument('-rf','--resistivity_fluid',
                            type=float)
        parser.add_argument('-km','--permeability_matrix',
                            type=float)
        parser.add_argument('-fs','--fault_separation',
                            type=float)
        parser.add_argument('-mu','--fluid_viscosity',
                            type=float)
        parser.add_argument('-flm','--faultlength_max',
                            type=float)
        parser.add_argument('-fld','--faultlength_decay',
                            type=float)
                            
        for arg in ['fractal_dimension',
                    'separation',
                    'offset',
                    'length_max',
                    'length_decay',
                    'mismatch_frequency_cutoff',
                    'elevation_standard_deviation',
                    'aperture_assignment']:
                        parser.add_argument('--'+arg, type=float, nargs = '*')
        parser.add_argument('--fault_edges',type=int,nargs='*')    
        parser.add_argument('--aperture_assignment',
                            help='type of aperture assignment, random or constant')
        parser.add_argument('-wd',
                            help='working directory')
        parser.add_argument('-o','--outfile',
                            help='output file name')
        parser.add_argument('-sp','--solve_properties',
                            help='which property to solve, current, fluid or currentfluid')
        parser.add_argument('-sd','--solve_direction',
                            help='which direction to solve, x, y, z or a combination')

        args = parser.parse_args(self.arguments)

        if (hasattr(args,'probabilityfile') and (args.probabilityfile is not None)):
            try:
                pvals = np.loadtxt(args.probabilityfile)
                self.input_parameters['pconnection'] = pvals[:,:3]
                self.input_parameters['pembedded_fault'] = pvals[:,3:6]
                self.input_parameters['pembedded_matrix'] = pvals[:,6:9]
            except IOError:
                print "Can't read probability file"
        
        for at in args._get_kwargs():
            if at[1] is not None:
                if (at[0] in ['pconnection','pembedded_fault','pembedded_matrix']):
                    # make sure number of values is divisible by 3 by repeating the last value
                    while np.size(at[1])%3 != 0:
                        at[1].append(at[1][-1])
                    # reshape
                    at[1] = np.array(at[1]).reshape(len(at[1])/3,3)
                if at[0] == 'fault_edges':
                    if np.size(at[1])%6 != 6:
                        pass
                    else:
                        fault_edges = []
                        for i in range(np.size(at[1])/6):
                            fault_edges.append(np.array(at[0][i*6:(i+1)*6]).reshape(3,2))
                        fault_edges = np.array(fault_edges)
                self.input_parameters[at[0]] = at[1]


    def initialise_inputs(self):
        """
        make a list of run parameters
        """        

        list_of_inputs = []
        parameter_list = [v for v in dir(self) if v[0] != '_']

        for r in range(self.repeats):
            for pc in self.pconnection:
                for pef in self.pembedded_fault:
                    for pem in self.pembedded_matrix:
                        for sd in self.fault_dict['elevation_standard_deviation']:
                            for fs in self.fault_dict['fault_separation']:
                                input_dict = {} 
                                for key in parameter_list:
                                    if key in ['fracture_diameter',
                                               'permeability_matrix',
                                               'resistivity_matrix',
                                               'resistivity_fluid',
                                               'wd',
                                               'res_type',
                                               'mu',
                                               'outfile',
                                               'ncells',
                                               'cellsize']:
                                        input_dict[key] = getattr(self,key)
                        input_dict['fault_separation'] = fs
                        input_dict['elevation_standard_deviation'] = sd
                        input_dict['pconnection'] = pc
                        input_dict['pembedded_fault'] = pef
                        input_dict['pembedded_matrix'] = pem
                        list_of_inputs.append(input_dict)
        
        return list_of_inputs

        self.fault_array = None                       
        self.fault_edges = None
        self.fault_assignment = 'random'

        
        
    def run(self,list_of_inputs,rank,wd):
        """
        generate and run a random resistor network
        takes a dictionary of inputs to be used to create a resistivity object
        """
        
        r_objects = []

        r = 0
        for input_dict in list_of_inputs:
            # initialise random resistor network
            ro = Rock_volume(**input_dict)
            # solve the network
            ro.solve_resistor_network(self.solve_properties,self.solve_directions)
            # append result to list of r objects
            r_objects.append(ro)
            print "run {} completed".format(r)
            for prop in ['resistivity','permeability',
                         'current','flowrate']:
                np.save(os.path.join(wd,'{}{}_{}'.format(prop,rank,r)),
                        getattr(ro,prop)
                        )
            r += 1
        return r_objects
        
        
    def setup_and_run_suite(self):
        """
        set up and run a suite of runs in parallel using mpi4py
        """
        
        from mpi4py import MPI
        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        name = MPI.Get_processor_name()
        print 'Hello! My name is {}. I am process {} of {}'.format(name,rank,size)
       
        if rank == 0:
            list_of_inputs = self.initialise_inputs()
            inputs = rnf.divide_inputs(list_of_inputs,size)
        else:
            list_of_inputs = None
            inputs = None

        if rank == 0:
            if not os.path.exists(self.wd):
                os.mkdir(self.wd)
            self.wd = os.path.abspath(self.wd)

        wd2 = os.path.join(self.wd,'arrays')

        if rank == 0:
            if not os.path.exists(wd2):
                os.mkdir(wd2)
        else:
            # wait for wd2 to be created
            while not os.path.exists(wd2):
                time.sleep(1)
                print '.',

        inputs_sent = comm.scatter(inputs,root=0)
        r_objects = self.run(inputs_sent,rank,wd2)
        outputs_gathered = comm.gather(r_objects,root=0)
         
        if rank == 0:

            # flatten list, outputs currently a list of lists
            og2 = []
            for group in outputs_gathered:
                for ro in group:
                    og2.append(ro)
                    
            results = {}
            for prop in ['resistivity_bulk','permeability_bulk']:
                if hasattr(ro,prop):
                    results[prop] = np.vstack([np.hstack([ro.pconnection,
                                                         [ro.fault_dict['fault_separation']],
                                                         [ro.fault_dict['elevation_standard_deviation']],
                                                         getattr(ro,prop)]) for ro in og2])
            # save results to text file
            # first define header
            header  = '# resistor network models - results\n'
            header += '# resistivity_matrix (ohm-m) {}\n'.format(self.resistivity_matrix)
            header += '# resistivity_fluid (ohm-m) {}\n'.format(self.resistivity_fluid)
            header += '# permeability_matrix (m^2) {}\n'.format(self.permeability_matrix)
            header += '# fracture diameter (m) {}\n'.format(self.fault_dict['fault_separation'])
            header += '# fluid viscosity {}\n'.format(self.mu)
            header += '# fracture max length {}\n'.format(self.faultlength_max)
            header += '# fracture length decay {}\n'.format(self.faultlength_decay)
            header += '# ncells {} {} {}\n'.format(self.ncells[0],
                                                   self.ncells[1],
                                                   self.ncells[2])
            header += '# cellsize (metres) {} {} {}\n'.format(self.cellsize[0],
                                                              self.cellsize[1],
                                                              self.cellsize[2])
            header += ' '.join(['# px','py','pz','fault_sep','elev_sd','propertyx','propertyy','propertyz'])

            for rr in results.keys():
                np.savetxt(os.path.join(self.wd,rr+'.dat'),np.array(results[rr]),
                           comments='',
                           header = header,
                           fmt=['%4.2f','%4.2f','%4.2f',
                                '%6.3e','%6.3e','%6.3e'])
                       
                       
if __name__ == "__main__":
    RandomResistorSuite()
