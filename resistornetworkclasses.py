# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:05:42 2014

@author: Alison Kirkby

Modelling random resistor networks using python.

"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import string
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
            except IOError:
                print "Cannot find resistivity file"
        elif self.res_type == 'array':
            resx = self.resistivity_x
            resz = self.resistivity_z
        else:
            print "res type {} not supported, please redefine".format(self.res_type)
            return

        self.resistivity = np.zeros([self.nx+1,self.nz+1,2])
        
        for i in range(2):
            self.resistivity[:,:,i] = [resx,resz][i]
        d = [self.dz,self.dx]
        
        
        self.resistance = [rnf.get_electrical_resistance(dd,
                                                         self.fracture_diameter,
                                                         res,
                                                         r_matrix,
                                                         r_fluid)\
                           for dd,res in zip([d,d[::-1],[resx,resz]])]
         
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

        self.hydraulic_resistance = [rnf.get_hydraulic_resistance(dd,k,
                                           self.permeability_matrix,
                                           self.fracture_diameter) \
                                     for dd,k in zip([d,d[::-1]],
                                                     [kx,kz])]
        

    def solve_resistor_network(self,properties,direction):
        """
        generate and solve a random resistor network
        properties = string or list containing properties to solve for,
        'current','fluid' or 'current_fluid'
        direction = string containing directions, 'x','z' or 'xz'
        
        """
        property_arrays = []
        if 'current' in properties:
            if not hasattr(self,'resistance'):
                self.initialise_resistivity()
            property_arrays.append([self.resistance,'current'])
        if 'fluid' in properties:
            if not hasattr(self,'hydraulic_resistance'):
                self.initialise_resistivity()
            property_arrays.append([self.hydraulic_resistance,'fluid'])        
        
        input_arrays = []
        for prop,pname in property_arrays:
            px,pz = [prop[:,:,i] for i in [0,1]]
            if 'x' in direction:
                input_arrays.append([pz.T,px.T,pname+'x'])
            if 'z' in direction:
                input_arrays.append([px,pz,pname+'z'])
            
        current = np.zeros([self.nz+2,self.nx+2,2,2])
        flow = np.zeros([self.nz+2,self.nx+2,2,2])

        for propx,propz,pname in input_arrays:
            A = rnf.build_matrix(propx,propz)
            b = rnf.build_sums(len(A[0]),[self.nx,self.nz])
            c = rnf.solve_matrix(A,b)
            nx = len(propx[0]),len(propz)
            cx = c[:self.nx*(self.nz+1)].reshape(self.nz+1,self.nx)
            cz = c[self.nx*(self.nz+1):].reshape(self.nz+2,self.nx+1)
            if 'current' in pname:
                if 'x' in pname:
                    # dealing with x direction current flow
                    current[1:-1,:,0,0] = cz.T
                    current[:,:,1,0] = cx.T
                if 'z' in pname:
                    # dealing with z direction current flow
                    current[:,:,0,1] = cx                    
                    current[:,:,1,1] = cz
                
                
      
      
    def solve_resistor_network_a(self,messages=True):
        """
        generate and solve a random resistor network in two directions
        to get anisotropy in resistivity
        
        """

        self.solve_resistor_network()
        resx = self.resistance_x
        resz = self.resistance_z
        current_x = self.current_x*1.
        current_z = self.current_z*1.
#        print current_z[-1],resz[-1]
        self.nx,self.nz,self.dx,self.dz = self.nz,self.nx,self.dz,self.dx
        self.resistance_x = resz.T
        self.resistance_z = resx.T
        self.res_type = 'array'
        self.solve_resistor_network()
#        print self.current_z[-1],self.resistance_z[-1]
        self.anisotropy = np.sum(current_z[-1])/np.sum(self.current_z[-1])
        self.currentx_x = self.current_z.T
        self.currentx_z = self.current_x.T
        self.current_x = current_x
        self.current_z = current_z
        self.resistance_x = resx
        self.resistance_z = resz
        self.nx,self.nz,self.dx,self.dz = self.nz,self.nx,self.dz,self.dx
        
        if messages:
            print "Anisotropy factor is {}".format(self.anisotropy)


    def solve_resistor_network_fluid_a(self,messages=True):
        """
        generate and solve a random resistor network in two directions
        to get anisotropy in resistivity
        
        """

        self.solve_resistor_network(prop = 'fluid')
        resx = self.hresistance_x
        resz = self.hresistance_z
        flowrate_x = self.flowrate_x*1.
        flowrate_z = self.flowrate_z*1.
#        print flowrate_z[-1],resz[-1]
        self.nx,self.nz,self.dx,self.dz = self.nz,self.nx,self.dz,self.dx
        self.hresistance_x = resz.T
        self.hresistance_z = resx.T
        self.res_type = 'array'
        self.solve_resistor_network(prop = 'fluid')
#        print self.flowrate_z[-1],self.resistance_z[-1]
#        print np.sum(flowrate_z[-1]),np.sum(self.flowrate_z[-1])
        self.anisotropy_k = np.sum(flowrate_z[-1])/np.sum(self.flowrate_z[-1])
        self.flowratex_x = self.flowrate_z.T
        self.flowratex_z = self.flowrate_x.T
        self.flowrate_x = flowrate_x
        self.flowrate_z = flowrate_z
        self.hresistance_x = resx
        self.hresistance_z = resz
        self.nx,self.nz,self.dx,self.dz = self.nz,self.nx,self.dz,self.dx
        
        if messages:
            print "Hydraulic anisotropy factor is {}".format(self.anisotropy_k)    
           

     

    
class Plot_network():
    """
    plot the results of a model using plt.quiver
    """
    
    def __init__(self, Resistivity_volume):
        
        self.Resistivity_volume = Resistivity_volume
        self.cmap = 'jet'
        self.clim_percent_fluids = (10,95)
        self.clim_percent_current = (0,100)
        self.arrowwidthscale = 0.2
        self.plot_arrowheads = False

    def get_meshlocations(self):
        """
        get locations of nodes for plotting
        
        """
        
        dx = self.Resistivity_volume.dx
        dz = self.Resistivity_volume.dz
        nx = self.Resistivity_volume.nx
        nz = self.Resistivity_volume.nz
        
        xx = np.linspace(0.,dx*(nx-1.),nx)
        xy = np.linspace(0.,dz*nz,nz+1)
        yx = np.linspace(0.,dx*nx,nx+1)
        yy = np.linspace(-dz,dz*nz,nz+2)
                
        self.plot_x = np.meshgrid(xx,xy)
        self.plot_z = np.meshgrid(yx,yy)

        xx = np.linspace(-dx,dx*nx,nx+2)
        xy = np.linspace(0.,dz*nz,nz+1)
        yx = np.linspace(0.,dx*nx,nx+1)
        yy = np.linspace(0.,dz*(nz-1),nz)

        self.plotx_x = np.meshgrid(xx,xy)
        self.plotx_z = np.meshgrid(yx,yy)
        
    def get_direction(self,prop):
        """
        get the direction of the current and fluid flow for plotting.
        
        1 means flow is down/right or zero
        -1 means flow is up/left        
        
        """
        
        direction = prop/(np.abs(prop))
        direction[np.isnan(direction)] = 1.
        
        return direction
            
            
    def get_quiver_origins(self):
        """
        get the locations of the origins of the quiver plot arrows.
        These are slightly different from the mesh nodes as where arrows are 
        negative, the arrow goes in the opposite direction so needs to be 
        shifted by dx/dy
        
        """
        
        # check if mesh locations have been calculated
        if not hasattr(self,'plotx'):
            self.get_meshlocations()
            
            
        dx = self.Resistivity_volume.dx
        dz = self.Resistivity_volume.dz
        
        qplot_x_current = np.zeros_like(self.plot_x)
        qplot_z_current = np.zeros_like(self.plot_z)                   
        qplot_x_fluid = np.zeros_like(self.plot_x)
        qplot_z_fluid = np.zeros_like(self.plot_z)

        qplotx_x_current = np.zeros_like(self.plotx_x)
        qplotx_z_current = np.zeros_like(self.plotx_z)                   
        qplotx_x_fluid = np.zeros_like(self.plotx_x)
        qplotx_z_fluid = np.zeros_like(self.plotx_z)
        
        qplot_x_current[0][self.get_direction(self.Resistivity_volume.current_x) < 0.] += dx
        qplot_z_current[1][self.get_direction(self.Resistivity_volume.current_z) > 0.] += dz        
        qplot_x_fluid[0][self.get_direction(self.Resistivity_volume.flowrate_x) < 0.] += dx
        qplot_z_fluid[1][self.get_direction(self.Resistivity_volume.flowrate_z) > 0.] += dz

        qplotx_x_current[0][self.get_direction(self.Resistivity_volume.currentx_x) < 0.] += dx
        qplotx_z_current[1][self.get_direction(self.Resistivity_volume.currentx_z) > 0.] += dz        
        qplotx_x_fluid[0][self.get_direction(self.Resistivity_volume.flowratex_x) < 0.] += dx
        qplotx_z_fluid[1][self.get_direction(self.Resistivity_volume.flowratex_z) > 0.] += dz
        
        self.qplot_x_current = qplot_x_current + self.plot_x
        self.qplot_z_current = qplot_z_current + self.plot_z
        self.qplot_x_fluid = qplot_x_fluid + self.plot_x
        self.qplot_z_fluid = qplot_z_fluid + self.plot_z
   
        self.qplotx_x_current = qplotx_x_current + self.plotx_x
        self.qplotx_z_current = qplotx_z_current + self.plotx_z
        self.qplotx_x_fluid = qplotx_x_fluid + self.plotx_x
        self.qplotx_z_fluid = qplotx_z_fluid + self.plotx_z                                       

    def quiverplot(self,X,Z,U,V,clim,fault_val=None):
        """
        make a plot using pyplot quiver
        
        """
#        print X,Z,U,V
        dx = self.Resistivity_volume.dx
        dz = self.Resistivity_volume.dz
        nx = self.Resistivity_volume.nx
        nz = self.Resistivity_volume.nz
        if fault_val is None:
            fault_val = max([np.amax(U),np.amax(V)])
        xlim = (min(X[0][:,0]),min(X[0][:,-1]+dx))
        
#        print xlim,zlim
        w = min(nx,nz)
        
        clim = (min([np.percentile(np.abs(U),clim[0]),np.percentile(np.abs(V),clim[0])]),
                max([np.percentile(np.abs(U),clim[1]),np.percentile(np.abs(V),clim[1])]))

        direction_u = self.get_direction(U)
        direction_v = self.get_direction(V)

        if self.plot_arrowheads:
            hw,hl,hal = 3,5,4.5
            hw,hl,hal = 1,2,1.5
        else:
            hw,hl,hal = 0,0,0

        if len(np.unique(U)) < 3:
            X[0][np.abs((fault_val-U)/U) <= 1e-1] = -99999.
        
        if np.median(V) > np.median(U):
            plt.quiver(X[0],X[1],
                       (np.zeros_like(U)+dx)*direction_u,
                       np.zeros_like(U),np.abs(U),
                       scale=dx*(nx+2),
                       width=self.arrowwidthscale/w,
                       cmap=self.cmap,
                       headwidth=hw,headlength=hl,headaxislength=hal)           
            plt.clim(clim)
        
        plotV = (np.zeros_like(V)-dz)*direction_v
        plotC = np.abs(V)
#        print plotV

        # for resistivity/permeability arrays, shift all the resistive arrows
        # off the plot, otherwise they overwrite X arrows and look funny.
        if len(np.unique(V)) < 3:            
            Z[0][np.abs((fault_val-V)/V) <= 1e-1] = -99999.

            
            
        plt.quiver(Z[0],Z[1],
                   np.zeros_like(V),
                   plotV,plotC,
                   scale=dx*(nx+2),
                   width=self.arrowwidthscale/w,
                   cmap=self.cmap,
                   headwidth=hw,headlength=hl,headaxislength=hal)
        plt.clim(clim)
        
        if np.median(U) >= np.median(V):
            plt.quiver(X[0],X[1],
                       (np.zeros_like(U)+dx)*direction_u,
                       np.zeros_like(U),np.abs(U),
                       scale=dx*(nx+2),
                       width=self.arrowwidthscale/w,
                       cmap=self.cmap,
                       headwidth=hw,headlength=hl,headaxislength=hal)           
            plt.clim(clim)
            
        plt.legend(prop={'size':1})

        plt.ylim(-dz,dz*(nz+1))
        plt.xlim(-dx,dx*(nx+1))
        plt.gca().set_aspect('equal')
        cbar = plt.colorbar(orientation='horizontal',fraction=0.05,pad=0.15)

        return plt.gca(), cbar

    
    def plot_resistivity_current(self):
        """
        plot resistivity and current
        """
        
        fig = plt.figure(figsize=(12,8))
        self.get_quiver_origins()
        
        cmap = self.cmap
        unit = 'ohms'        
        
        X = self.plot_x
        Y = self.plot_z

        clim = self.clim_percent_current
        
        self.cmap=self.cmap.rstrip('_r')
        for i,attribute in enumerate(['resistance','current','currentx']):
            plt.subplot(1,3,i+1)
            U = getattr(self.Resistivity_volume,attribute+'_x')
            V = getattr(self.Resistivity_volume,attribute+'_z')
            if attribute == 'resistance':
                Vnew = np.zeros([len(V)+2,len(V[0])])+np.amax(U)
                Vnew[1:-1] = V
                V = Vnew
                self.cmap = 'gray'
                plt.ylabel('Vertical distance, m')
            ax, cbar = self.quiverplot(X,Y,U,V,clim)
            cbar.set_label(str.capitalize(attribute)+', '+unit)
            if 'current' in attribute:
                
                newticks = [np.percentile(abs(V),self.clim_percent_current[0]),
                            np.percentile(abs(V),self.clim_percent_current[1])]
                newticks.insert(1,np.median(newticks))

                cbar.set_ticks(newticks)
                cbar.ax.set_xticklabels(['%2.1e'%i for i in newticks])
#                cbar.ax.set_xticklabels(['%2.1e'%i for i in cbar.ax.get_xticks()])
            plt.xlabel('Horizontal distance, m')
            self.cmap = cmap
            X = self.qplot_x_current
            Y = self.qplot_z_current
            unit = 'Amperes'
        
        plt.subplots_adjust(left=0.0625,bottom=0.05,top=0.95,right=0.95)

    def plot_permeability_flow(self):
        """
        plot permeability and fluid flow
        """
        fig = plt.figure(figsize=(12,8))
        self.get_quiver_origins()
        
        cmap = self.cmap
        unit = '$m^2$'
        
        X = self.plot_x
        Y = self.plot_z
#        self.get_clim()
        clim = self.clim_percent_current
        
        self.cmap=self.cmap.rstrip('_r')
        for i,attribute in enumerate(['hresistance','flowrate','flowratex']):
            plt.subplot(1,3,i+1)
            U = getattr(self.Resistivity_volume,attribute+'_x')
            V = getattr(self.Resistivity_volume,attribute+'_z')
            if 'hresistance' in attribute:
                Vnew = np.zeros([len(V)+2,len(V[0])])+np.amax(U)
                Vnew[1:-1] = V
                V = Vnew
                self.cmap = 'gray'
                plt.ylabel('Vertical distance, m')
            ax, cbar = self.quiverplot(X,Y,U,V,clim)
            cbar.set_label(str.capitalize(attribute)+', '+unit)
            if attribute == 'flowrate':
                
                newticks = [np.percentile(abs(V),self.clim_percent_current[0]),
                            np.percentile(abs(V),self.clim_percent_current[1])]
                newticks.insert(1,np.median(newticks))

                cbar.set_ticks(newticks)
                cbar.ax.set_xticklabels(['%2.1e'%i for i in newticks])

            plt.xlabel('Horizontal distance, m')
            self.cmap = cmap
            X = self.qplot_x_current
            Y = self.qplot_z_current    
            unit = '$m^3/s$'
        plt.subplots_adjust(left=0.0625,bottom=0.05,top=0.95,right=0.95)
                                                                    
    def plot_all(self):
        fig = plt.figure(figsize=(12,9))
        self.get_quiver_origins()
        
        cmap = self.cmap
        unit = 'ohms'        
        
        clim = self.clim_percent_current        
        
        self.cmap=self.cmap.rstrip('_r')
        for i,attribute in enumerate(['resistance','current','currentx',
                                      'permeability','flowrate','flowratex']):
            plt.subplot(2,3,i+1)
            U = getattr(self.Resistivity_volume,attribute+'_x')*1.
            V = getattr(self.Resistivity_volume,attribute+'_z')*1.
            fault_val = np.amax(V)

            if 'current' in attribute:
                unit = 'Amperes'
                if attribute[-1] == 'x':
                    X = [self.qplotx_x_current[0]*1.,self.qplotx_x_current[1]*1.]
                    Y = [self.qplotx_z_current[0]*1.,self.qplotx_z_current[1]*1.]
                else:
                    X = [self.qplot_x_current[0]*1.,self.qplot_x_current[1]*1.]
                    Y = [self.qplot_z_current[0]*1.,self.qplot_z_current[1]*1.]

            elif 'flowrate' in attribute:
                unit = 'm^3/s'
                if attribute[-1] == 'x':
                    X = [self.qplotx_x_fluid[0]*1.,self.qplotx_x_fluid[1]*1.]
                    Y = [self.qplotx_z_fluid[0]*1.,self.qplotx_z_fluid[1]*1.]
                else:
                    X = [self.qplot_x_fluid[0]*1.,self.qplot_x_fluid[1]*1.]
                    Y = [self.qplot_z_fluid[0]*1.,self.qplot_z_fluid[1]*1.]
                clim = self.clim_percent_fluids

        
            if 'resistance' in attribute:
#                Vnew = np.zeros([len(V)+2,len(V[0])])+np.amax(U)
#                Vnew[1:-1] = V
#                V = Vnew

                self.cmap = 'gray'
                plt.ylabel('Vertical distance, m')
                X = [1.*self.plot_x[0],1.*self.plot_x[1]]
                Y = [1.*self.plot_z[0][1:-1],
                     1.*self.plot_z[1][1:-1]+self.Resistivity_volume.dz]
            if 'permeability' in attribute:
#                Vnew = np.zeros([len(V)+2,len(V[0])])+np.amax(U)
#                Vnew[1:-1] = V
#                V = Vnew

                self.cmap = 'gray_r'
                plt.ylabel('Vertical distance, m')
                fault_val = np.amin(V)
                X = [1.*self.plot_x[0],1.*self.plot_x[1]]
                Y = [1.*self.plot_z[0][1:-1],
                     1.*self.plot_z[1][1:-1]+self.Resistivity_volume.dz]
                print np.shape(Y),np.shape(V)
                unit = 'm^2'

            ax, cbar = self.quiverplot(X,Y,U,V,clim,fault_val=fault_val)
            plt.text(-20*self.Resistivity_volume.dx,plt.ylim()[1],string.ascii_lowercase[i],fontsize=16)
            cbar.set_label('${},\ {}$'.format(str.capitalize(attribute),unit))
            if ('current' in attribute) or ('flowrate' in attribute):
                newticks = [min([np.percentile(np.abs(U),clim[0]),np.percentile(np.abs(V),clim[0])]),
                max([np.percentile(np.abs(U),clim[1]),np.percentile(np.abs(V),clim[1])])]

                newticks.insert(1,np.median(newticks))

                cbar.set_ticks(newticks)
                cbar.ax.set_xticklabels(['%2.1e'%i for i in newticks])
#                cbar.ax.set_xticklabels(['%2.1e'%i for i in cbar.ax.get_xticks()])
            plt.xlabel('Horizontal distance, m')
            self.cmap = cmap
            
        plt.subplots_adjust(left=0.0625,bottom=0.05,top=0.95,right=0.95)
    
def old_plot(self,prop,cscale='gray_r'):
    """
    plot the results of a model using plt.scatter
    """
    property_x = prop[0]
    property_z = prop[1]


    dx = self.dx
    dz = self.dz

    sx = np.shape(property_x)
    sz = np.shape(property_z)
    
    
    nx = sx[1]
    nz = sz[0]
    offset = (sz[0] - (sx[0] - 1))
    
    # define the centres of the lines representing x property
    # z centres align with nodes; x centres are halfway between nodes
    centres_xx = dx*(np.arange(nx)+0.5)
    centres_xz = dz*(np.arange(int(nz+1-offset)))
    centres_x = np.meshgrid(centres_xx,centres_xz)
    
    # define the centres of the lines representing z property
    # x centres align with nodes; z centres are halfway between nodes            
    centres_zx = dx*np.arange(int(nx+1))
    centres_zz = dz*(np.arange(nz)+0.5-offset/2.)
    centres_z = np.meshgrid(centres_zx,centres_zz)
    
    # define colour normalisation
    norm = max(np.percentile(property_x,95),np.percentile(property_z,95))

    # define marker size
    msx = 100./(nx**0.5)
    msz = 100./(nz**0.5)
    
    # create the plot
    plt.scatter(centres_x[0].flatten(),centres_x[1].flatten(),s=msx,c=property_x,marker = '_',cmap=cscale,lw=0.5)
    plt.clim(0.,norm)
    plt.scatter(centres_z[0].flatten(),centres_z[1].flatten(),s=msz,c=property_z,marker = '|',cmap=cscale,lw=0.5)
    plt.clim(0.,norm)
    plt.xlim(0.,self.dx*self.nx)
    plt.ylim(0.,self.dz*self.nz)
    ax = plt.gca()
    ax.set_xticklabels([str(float(i)) for i in ax.get_xticks()])
    ax.set_yticklabels([str(float(i)) for i in ax.get_yticks()])
    plt.gca().set_aspect('equal')


def old_plot_res_current(self,cscale='gray',save=False,fn=None):
    """
    plot the resistor network and the current alongside each other
    need to run a model first
    """
    
    if hasattr(self,'current_x'):
        plt.Figure(figsize=(50,25))
        xpad = 0.05
        ypad = xpad
        title = 'Resistance'
        for i, prop in enumerate([[self.resistance_x,self.resistance_z],[self.current_x,self.current_z]]):
            ax = plt.axes([xpad+i*0.5*(1.-3.5*xpad),ypad,(0.5-3.*xpad),1.-2.*ypad])
            self.plot(prop,cscale = cscale)
            ax.set_aspect('equal')
            if i == 0:
                cscale += '_r'
            plt.title(title)
            title = 'Current'

        ax = plt.axes([1.-5*xpad,ypad,2*xpad,1.-2.*ypad])
        ax.set_visible(False)
        cbar = plt.colorbar()
        cbar.set_label('Amperes')
        cbar.ax.set_yticklabels(['%2.1e'%i for i in cbar.ax.get_yticks()])
        if save:
            if fn is None:
                fn = os.path.join(self.wd,'rrn_res_current.png')
            plt.savefig(os.path.join(self.wd,fn),dpi=300)
            plt.close()
        plt.show()



def old_plot_k_fluidflow(self,cscale='gray',save=False,fn=None):
    """
    plot the resistor network and the fluid flow alongside each other
    need to run a model first
    """
    
    if hasattr(self,'flowrate_x'):
        plt.Figure(figsize=(50,25))
        xpad = 0.05
        ypad = xpad
        title = 'Permeability'
        for i, prop in enumerate([[self.hresistance_x,self.hresistance_z],[self.flowrate_x,self.flowrate_z]]):
            ax = plt.axes([xpad+i*0.5*(1.-3.5*xpad),ypad,(0.5-3.*xpad),1.-2.*ypad])
            self.plot(prop,cscale = cscale)
            ax.set_aspect('equal')
            if i == 0:
                cscale += '_r'
            plt.title(title)
            title = 'Flow rate'

        ax = plt.axes([1.-5*xpad,ypad,2*xpad,1.-2.*ypad])
        ax.set_visible(False)
        cbar = plt.colorbar()
        cbar.set_label('m^3/s')
#            cbar.ax.set_yticklabels(['%2.1e'%i for i in cbar.ax.get_yticks()])
        if save:
            if fn is None:
                fn = os.path.join(self.wd,'rrn_k_fluidflow.png')
            plt.savefig(os.path.join(self.wd,fn),dpi=300)
            plt.close()
        plt.show()


        


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
