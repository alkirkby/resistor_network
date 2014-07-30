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
import resistornetworkfunctions as rnf

class Resistivity_volume():
    """
    Class to contain volumes to be modelled as a random resistor network.
    wd = working directory
    nx = number of nodes in the x direction, default is 10
    nz = number of nodes in the z direction, default is 10
    res_type =  string describing how to calculate the resistivity structure;
                options are "ones" (fully connected network: res = 1 everywhere), 
                            "random" (random network with some high resistivity bonds), and 
                            "file" (network given by file)
                            "array" (resistivity network given as a numpy array)
    matrix_resistivity = list containing [x,z] resistivities for the low conductivity matrix
    resistivity_x, resistivity_z = array of resistivities in x and z directions
    resistivity_file_x,resistivity_file_z = filenames for resistivity arrays if self.restype == "file"
    linearity_factor = factor to adjust probabilities according to value in previous row
                       to make linear structures
                       e.g. if linearity_factor == 2: a given cell is twice as likely to be
                       connected if the corresponding cell in the previous row is connected.
                       probabilities are normalised so that overall probability in each row
                       is equal to pz
    px,pz = probabilities of connection in the x and z directions
    dx,dz = size of cells
                 
    """
    
    def __init__(self, **input_parameters):
        self.wd = '.' # working directory
        self.nx = 10
#        self.ny = 1
        self.nz = 10
        self.res_type = 'ones'
        self.resistivity_matrix = 1000
        self.resistivity_fluid = 0.1
        self.resistivity_x = None
#        self.resistivity_y = None
        self.resistivity_z = None
        self.permeability_matrix = 1.e-18
        self.fracture_diameter = 1.e-3
#        self.fluid_viscosity = 1.e-3 #default is for freshwater at 20 degrees 
        self.file_x = None
        self.file_z = None
        self.linearity_factor = 1.# 
                            
        self.px = 0.5
        self.pz = 0.5
        
        self.dx = 1.
        self.dz = 1.
        
        update_dict = {}

        #correcting dictionary for upper case keys
        input_parameters_nocase = {}
        for key in input_parameters.keys():
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
            
        self.initialise_resistivity()
        self.initialise_permeability()



    def initialise_resistivity(self):
        """
        initialise a resistivity array based on keywords
        options are "ones" - a fully connected network
                    "random" - a network with some bonds broken
                    "file" - read in array from file.
                    "array" - resistivity network given as a numpy array
                            
        
        """
        if self.res_type == 'ones':
            resx = np.ones((self.nz+1,self.nx))
            resz = np.ones((self.nz,self.nx+1))
        elif self.res_type == "random":
            r_matrix = float(self.resistivity_matrix)
            r_fluid = float(self.resistivity_fluid)

            # how much more likely there is to be a fault if previous cell holds a fault
            linearity_factor = float(self.linearity_factor)
            
            # create z resistivity
            if self.pz != 1.:
                resz = rnf.assign_random_resistivity([self.nx,self.nz],
                                                     [self.px,self.pz],
                                                     r_matrix,r_fluid,
                                                     linearity_factor)
            # create z resistivity
            if self.px != 1.:
                resx = rnf.assign_random_resistivity([self.nz,self.nx],
                                                     [self.pz,self.px],
                                                     r_matrix,r_fluid,
                                                     linearity_factor).T                                                     
        elif self.res_type == 'file':
            try:
                resx = np.loadtxt(os.path.join(self.wd,self.file_x))
                resz = np.loadtxt(os.path.join(self.wd,self.file_z))
                r_matrix = max([np.amax(resx),np.amax(resz)])
                r_fluid = min([np.amin(resx),np.amin(resz)])
            except IOError:
                print "Cannot find resistivity file"
        elif self.res_type == 'array':
            resx = self.resistivity_x
            resz = self.resistivity_z
            r_matrix = max([np.amax(resx[np.isfinite(resx)]),
                            np.amax(resz[np.isfinite(resz)])])
            r_fluid = min([np.amin(resx[np.isfinite(resx)]),
                           np.amin(resz[np.isfinite(resz)])])
        else:
            print "res type {} not supported, please redefine".format(self.res_type)
            return

        self.resistivity = np.zeros([self.nx+1,self.nz+1,2])
        
        for i in range(2):
            self.resistivity[:,:,i] = [resx,resz][i]
        d = [self.dz,self.dx]
        
        res2x,res2z = [rnf.get_electrical_resistance(dd,
                                                     self.fracture_diameter,
                                                     res,
                                                     r_matrix,
                                                     r_fluid)\
                           for dd,res in zip([d,d[::-1]],[resx,resz])]

        self.resistance = np.zeros([self.nx+1,self.nz+1,2])
       
        for i in range(2):
            self.resistance[:,:,i] = [res2x,res2z][i]
            
        d = [self.dz,self.dx]         
        self.phi = sum([rnf.get_phi(dd,self.fracture_diameter) for dd in d])
    

    def initialise_permeability(self):
        """
        initialise permeability and hydraulic resistance based on 
        connections set up in resistivity array                           
        
        """
        if self.resistivity_z is None:
            self.initialise_resistivity()
        
        d = [self.dz,self.dx]
        
        kx,kz = [rnf.get_permeability(res,
                                      self.resistivity_fluid,
                                      self.permeability_matrix,
                                      self.fracture_diameter) for res in \
                                      [self.resistivity[:,:,i] for i in [0,1]]]
        self.permeability = np.zeros([self.nx+1,self.nz+1,2])
        for i in range(2):
            self.permeability[:,:,i] = [kx,kz][i]
            
            
        hrx,hrz = [rnf.get_hydraulic_resistance(dd,k,
                                                self.permeability_matrix,
                                                self.fracture_diameter) \
                                     for dd,k in zip([d,d[::-1]],
                                                     [kx,kz])]
                                                     
        self.hydraulic_resistance = np.zeros([self.nx+1,self.nz+1,2])
        for i in range(2):
            self.hydraulic_resistance[:,:,i] = [hrx,hrz][i]


    def solve_resistor_network(self,properties,direction):
        """
        generate and solve a random resistor network
        properties = string or list containing properties to solve for,
        'current','fluid' or 'current_fluid'
        direction = string containing directions, 'x','z' or 'xz'
        
        """
        # set kfactor to divide hydraulic conductivities by so that matrix
        # solving is more accurate. 
        kfactor = 1e8
        
        property_arrays = []
        if 'current' in properties:
            if not hasattr(self,'resistance'):
                self.initialise_resistivity()
            property_arrays.append([[self.resistance[:,:,i]\
                                     for i in [0,1]],'current'])
        if 'fluid' in properties:
            if not hasattr(self,'hydraulic_resistance'):
                self.initialise_resistivity()
            property_arrays.append([[self.hydraulic_resistance[:,:,i]/kfactor\
                                     for i in [0,1]],'fluid']) 
        
        input_arrays = []
        for prop,pname in property_arrays:
            px,pz = prop
            px = px[:,:-1]
            pz = pz[:-1]
#            print px,pz
            if 'x' in direction:
                input_arrays.append([pz.T,px.T,pname+'x'])
            if 'z' in direction:
                input_arrays.append([px,pz,pname+'z'])
            
        current = np.zeros([self.nz+2,self.nx+2,2,2])
        flow = np.zeros([self.nz+2,self.nx+2,2,2])

        for propx,propz,pname in input_arrays:
            A = rnf.build_matrix(propx,propz)
            b = rnf.build_sums(np.shape(A)[0],[self.nx,self.nz])
            c = rnf.solve_matrix(A,b)
            if 'fluid' in pname:
                c = c/kfactor
            

            nx,nz = len(propx[0]),len(propz)
            cx = c[:nx*(nz+1)].reshape(nz+1,nx)
            cz = c[nx*(nz+1):].reshape(nz+2,nx+1)
            if 'current' in pname:
                if 'x' in pname:
                    # dealing with x direction current flow
                    current[1:,:,0,0] = cz.T
                    current[1:-1,1:,1,0] = cx.T
                if 'z' in pname:
                    # dealing with z direction current flow
                    current[1:,1:-1,0,1] = cx                    
                    current[:,1:,1,1] = cz
            if 'fluid' in pname:

                if 'x' in pname:
                    # dealing with x direction current flow
                    flow[1:,:,0,0] = cz.T
                    flow[1:-1,1:,1,0] = cx.T
                if 'z' in pname:
                    # dealing with z direction current flow
                    flow[1:,1:-1,0,1] = cx                    
                    flow[:,1:,1,1] = cz                
        
        self.current = current
        self.flowrate = flow
      
      


class Run_suite():
    """
    organise and run a suite of resistivity runs. Inherits Resistivity_volume
    Author: Alison Kirkby
    
    nx,nz = number of cells in x and z direction for individual runs
    px,pz = list of px,pz values to run
    linearity_factors = list of linearity factors to run
    """
    
    def __init__(self, **input_parameters):
        
        self.nx = 10
        self.nz = 10
        self.px = None
        self.pz = None
        self.linearity_factors = None
        self.n_repeats = 1
        self.arguments = []
        self.resistivity_matrix = 1000.
        self.resistivity_fluid = 0.1
        self.output_bn = 'rrn'
        self.wd = '.'
        update_dict = {}

        #correcting dictionary for upper case keys
        input_parameters_nocase = {}
        for key in input_parameters.keys():
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
            
        if len(self.arguments) > 0:
            self.read_arguments()
            
    def read_arguments(self):
        """
        takes list of command line arguments obtained by passing in sys.argv
        reads these and updates attributes accordingly
        """
        
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-nx',help='number of cells in horizontal direction',
                            type=int,default=self.nx)
        parser.add_argument('-nz',help='number of cells in vertical direction',
                            type=int,default=self.nz)
        parser.add_argument('-npx',help='number of probability values in horizontal direction',
                            type=int,default=11)
        parser.add_argument('-npz',help='number of probability values in vertical direction',
                            type=int,default=11)
        parser.add_argument('-lf',help='linearity factor to apply to resistor networks',type=int)
        parser.add_argument('-nlf',help='alternative to lf, can specify a number of linearity factors to run. Use this option if running a range of linearity factors')

        parser.add_argument('-r',help='number of repeats at each probability value',type=int,
                            default=self.n_repeats)
        parser.add_argument('-wd',help='working directory',default=self.wd)
        parser.add_argument('-o',help='output file name',default=self.output_bn)

        args = parser.parse_args()
        if args.lf is not None:
            setattr(self,'linearity_factors',[args.lf])
        elif args.nlf is not None:
            setattr(self,'linearity_factors',list(np.linspace(1,self.nz,args.nlf).astype(int)))

        if args.npx % 2 == 0:
            args.npx += 1
        if args.npz % 2 == 0:
            args.npz += 1
            
        setattr(self,'nx',args.nx)
        setattr(self,'nz',args.nx)
        setattr(self,'px',np.linspace(0.,1.,args.npx))
        setattr(self,'pz',np.linspace(0.,1.,args.npz))
        setattr(self,'n_repeats',args.r)
        setattr(self,'wd',args.wd)
        setattr(self,'output_bn',args.o)
    
    def initialise_inputs(self):
        """
        make a list of run parameters
        """        
        
        list_of_inputs = []        
        
        for ppx in self.px:
            for ppz in self.pz:
                for lf in self.linearity_factors:
                    for r in range(self.n_repeats):
                        list_of_inputs.append([ppx,ppz,lf,r])
        
        return list_of_inputs
        
    
    def divide_inputs(self,work_to_do,size):
        """
        divide list of inputs into chunks to send to each processor
        
        """
        
        chunks = [[] for _ in range(size)]
        for i,d in enumerate(work_to_do):
            chunks[i%size].append([round(dd,2) for dd in d])

        return chunks
        
        
    def run(self,list_of_inputs):
        """
        generate and run a random resistor network
        takes a list of inputs, each row in the list has the following values:
        [px,pz,linearity_factor,repeat number]
        """
        currents = np.zeros(len(list_of_inputs))
        anisotropy = np.zeros(len(list_of_inputs))
        r_objects = []

        r = 0
        for px,pz,lf,rr in list_of_inputs:
            # initialise random resistor network
            R = Resistivity_volume(nx = self.nx, nz = self.nz,
                              px = px, pz = pz,
                              linearity_factor=lf,
                              res_type = 'random',
                              resistivity_matrix = self.resistivity_matrix,
                              resistivity_fluid = self.resistivity_fluid)
            # solve the network
            R.generate_resistor_network_a(messages=False)
            
            # append result to list of r objects
            r_objects.append(R)
            # append the total current in the bottom layer to a temp array
            currents[r] = np.sum(R.current_z[-1])
            anisotropy[r] = R.anisotropy
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
            inputs = self.divide_inputs(list_of_inputs,size)
        else:
            list_of_inputs = None
            inputs = None
    
        inputs_sent = comm.scatter(inputs,root=0)
        r_objects = self.run(inputs_sent)
        #for ro in r_objects:
        
        outputs_gathered = comm.gather(outputs,root=0)

        #print outputs_gathered
        if rank == 0:
            results = []
            # make an n x 4 list to store inputs
            for ii in inputs:
                results += ii
            # add results to each element in results list
            for r,result in enumerate(results):
                result += list(np.vstack(outputs_gathered)[r])
                
            # save results to text file
            # first define header
            header  = '# resistor network models - results\n'
            header += '# resistivity_matrix (ohm-m) {}\n'.format(self.resistivity_matrix)
            header += '# resistivity_fluid (ohm-m) {}\n'.format(self.resistivity_fluid)
            header += '# nx {}\n'.format(self.nx)
            header += '# nz {}\n'.format(self.nz)
            header += '# dx (metres) {}\n'.format(self.dx)
            header += '# dz (metres) {}\n'.format(self.dz)
            header += ' '.join(['px','pz','lf','r','cz','anisotropy'])
            fn = os.path.join(self.wd,self.output_bn+'.dat')
            i = 1
            while os.path.exists(fn):
                fn = os.path.join(self.wd,self.output_bn+'%03i.dat'%i)
                i += 1
            np.savetxt(os.path.join(self.wd,self.output_bn),np.array(results),
                       comments='',
                       header = header,
                       fmt=['%4.2f','%4.2f','%4i','%2i','%5.3f','%5.3f'])


class Stochastic_outputs():
    """
    class to deal with analysis and plotting of outputs from stochastic run
    """
    
    def __init__(self,**input_parameters):
        self.wd = '.'
        self.fn = 'output.dat'
        update_dict = {}        
        
        #correcting dictionary for upper case keys
        input_parameters_nocase = {}
        for key in input_parameters.keys():
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

    def read_header(self):
        """
        read header info
        """

        outfile = open(os.path.join(self.wd,self.fn))
        
        line = outfile.readline()
        while '#' in line:
            lsplit = line.strip().split()
            if 'resistivity' in lsplit[1]:
                setattr(self,lsplit[1],float(lsplit[-1]))
            else:
                setattr(self,lsplit[1],int(lsplit[-1]))



    def read_models(self):
        """
        read model contents into an array and reshape according to different 
        varied parameters. If repeats were done, calculate an average also.
        
        """
        
        # read in as a structured array for sorting
        skiprows = 0
        f = open(os.path.join(self.wd,self.fn))
        while '#' in f.readline():
            skiprows += 1
        models_str = np.genfromtxt(os.path.join(self.wd,self.fn),names=True,skiprows=skiprows)
        nrows = len(f.readline().strip().split())
        
        # sort results
        models_str.sort(order=('px','pz','lf','r'))

        
        models = models_str.view(float).reshape(len(models_str),len(models_str.dtype))
        
        self.px = np.unique(models[:,0])
        self.pz = np.unique(models[:,1])
        self.linearity_factors = np.unique(models[:,2])
        self.n_repeats = len(np.unique(models[:,3]))
        
        npx = len(self.px)
        npz = len(self.pz)
        nlf = len(self.linearity_factors)
        nrp = self.n_repeats
        
        self.models = models.reshape(npx,npz,nlf,nrp,nrows)
        
        self.models_average = np.average(self.models,axis=3)
        self.models_std = np.std(self.models,axis=3)
        

    def plot_p_vs_a(self,lf=None):
        
        self.read_models()
        
        if lf is None:
            lf = self.linearity_factors[0]
        
        aniso = self.models_average[:,:,:,-1][self.models_average[:,:,:,2]==lf].reshape(len(self.px),len(self.pz))
        plt.contour(self.px,self.pz,aniso,20,norm=LogNorm(vmin=np.amin(aniso),vmax=np.amax(aniso)),levels=np.logspace(-3,3,19))

        plt.colorbar()
        plt.show()
