# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:05:42 2014

@author: a1655681

modelling random resistor networks using python

"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import string

class Resistivity_volume():
    """
    class to contain volumes to be modelled as a random resistor network.
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
            mres = float(self.resistivity_matrix)
            fres = float(self.resistivity_fluid)

            factor = float(self.linearity_factor) # how much more likely there is to be a fault if previous cell holds a fault

         
            resz = np.ones((self.nz,self.nx+1))*fres
            resx = np.ones((self.nz+1,self.nx)).T*fres
            if self.pz != 1.:
                resz0 = np.random.random(size=(self.nx+1))*mres
                resz[0,resz0>=self.pz*mres] = mres
                
                for i in range(1,self.nz):
                    # figure out number of fractures in previous row
                    nf = len(resz[i,resz[i-1]==fres])
                    # number of matrix cells in previous row
                    nm = self.nx-nf 
                    # multiplication factor to apply to matrix probabilities
                    f = self.nx/(factor*nf+nm) 
                    # probability of fracture if the above cell is a matrix cell
                    pmz = f*self.pz 
                    # probability of fracture if the above cell is a fracture cell
                    pfz = factor*pmz 
                    # make a new row
                    reszi = np.random.random(size=(self.nx+1))*mres 
                    # if adjacent cell on previous row had a fracture, 
                    # assign matrix cell to this row with probability 1-pfz
                    resz[i,(reszi>=pfz*mres)&(resz[i-1]==fres)] = mres
                    # if adjacent cell on previous row had no fracture, 
                    # assign matrix cell to this row with probability 1-pmz
                    resz[i,(reszi>=pmz*mres)&(resz[i-1]==mres)] = mres

            # repeat above for x direction
            if self.px != 1.:
                resx0 = np.random.random(size=(self.nz+1))*mres
                resx[0,resx0>=self.px*mres] = mres

                for i in range(1,self.nx):
                    nf = len(resx[i,resx[i-1]==fres])
                    nm = self.nz-nf
                    f = self.nx/(factor*nf+nm)
                    pmx = f*self.px
                    pfx = factor*pmx
                    resxi = np.random.random(size=(self.nz+1))*mres
                    resx[i,(resxi>=pfx*mres)&(resx[i-1]==fres)] = mres
                    resx[i,(resxi>=pmx*mres)&(resx[i-1]==mres)] = mres
            resx = resx.T
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

        self.resistivity_x = resx*1.
        self.resistivity_z = resz*1.

        # use fracture diameter and cellsize dx and dz to define fracture
        # volume percent in x and z directions
        phix = self.fracture_diameter/self.dx
        phiz = self.fracture_diameter/self.dz
        
        # convert resistivities to resistances
        resx[resx!=mres] = 1./(phix/fres + (1.-phix)/mres)
        resz[resz!=mres] = 1./(phiz/fres + (1.-phiz)/mres)
        resx[resx==mres] = mres*self.dx/self.dz
        resz[resz==mres] = mres*self.dz/self.dx
               
        self.resistance_x = resx
        self.resistance_z = resz

    
    def initialise_permeability(self):
        """
        initialise a permeability array based on 
        connections set up in resistivity array                           
        
        """
        if self.resistivity_z is None:
            self.initialise_resistivity()
        
        km = self.permeability_matrix
        d = self.fracture_diameter
#        mu = self.fluid_viscosity
        dx = self.dx
        dz = self.dz
        
        self.permeability_x = np.ones_like(self.resistivity_x)*km         
        self.permeability_z = np.ones_like(self.resistivity_z)*km
        self.hresistance_x = (1./self.permeability_x)*(dx/dz)
        self.hresistance_z = (1./self.permeability_z)*(dz/dx)
        
        self.permeability_x[self.resistivity_x==self.resistivity_fluid] = d**2/12.
        self.permeability_z[self.resistivity_z==self.resistivity_fluid] = d**2/12.
        self.hresistance_x[self.resistivity_x==self.resistivity_fluid] = 12.*dx/(d**3)
        self.hresistance_z[self.resistivity_z==self.resistivity_fluid] = 12.*dz/(d**3)               


    def solve_resistor_network(self, prop = 'current'):
        """
        generate and solve a random resistor network
        
        """
        
        self.build_matrix(prop = prop)
        self.build_sums()
        self.solve_matrix(prop = prop)
      
      
    def solve_resistivity(self,messages=True):
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


    def solve_permeability(self,messages=True):
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
        
    def build_matrix(self,prop = 'current'):
        """
        build the matrix to solve the currents at every link in the matrix
        
        """
        nx = self.nx
        nz = self.nz
        
        if prop == 'current':
            propertyz = self.resistance_z
            propertyx = self.resistance_x
        elif prop == 'fluid':
            propertyz = self.hresistance_z
            propertyx = self.hresistance_x
     
        # 1. construct part of matrix dealing with voltages (sum zero in each cell)
        #    a. construct x parts of loop
        #       define top part of loop - positive (positive right)
        d1v = propertyx[:-1].flatten()
        #       define bottom part of loop - negative (positive right)
        d2v = -propertyx[1:].flatten()
        #       construct matrix using above diagonals        
        xblock_v = sparse.diags([d1v,d2v],[0,nx],shape=(nx*nz,nx*(nz+1)))
        
        #    b. construct z parts of loop
        blocks = []
  
        for j in range(nz):
        #       construct dia1 - summing z voltage drops on lhs of loop, negative (positive down)
            dia1 = -propertyz[j,:nx]
        #       construct dia2 - summing z voltage drops on rhs of loop, positive (positive down)
            dia2 = propertyz[j,1:]
        #       construct diagonal matrix containing the above diagonals
            blocks.append(sparse.diags([dia1,dia2],[0,1],shape=(nx,nx+1)))
        #       construct matrix using above blocks
        yblock_v = sparse.block_diag(blocks)

        #    c. construct parts of matrix dealing with top and bottom z currents (zero voltage)
        yblock2_v = sparse.coo_matrix((nx*nz,nx+1))

        #    d. combine parts together to make voltage part of matrix
        m_voltage = sparse.coo_matrix(sparse.bmat([[xblock_v,yblock2_v,yblock_v,yblock2_v]]))

        
        # 2. construct part of matrix dealing with currents (sum zero in each node)
        #    need to skip a node in the middle of the matrix
        #    a. part dealing with x currents
        #       top and bottom parts of matrix
        
        onx = np.ones(nx)
        onx2 = np.ones(nx/2)

        xblock1 = sparse.diags([-onx,onx],offsets=[0,-1],shape=(nx+1,nx))
        
        #       middle part of matrix - one node is skipped in the middle so different pattern
        xblock2_s1 = sparse.diags([onx2,-onx2[:-1]],offsets=[0,1])
        xblock2_s2 = sparse.diags([-onx2,onx2[:-1]],offsets=[0,-1])
        #xblock2 = sparse.block_diag([xblock2_s1,xblock2_s2])

        #       build matrix from xblock1 and xblock2
        xblock = sparse.block_diag([xblock1]*int(nz/2)+[xblock2_s2]+[xblock2_s1]+[xblock1]*int(nz/2))
        
        #    b. part dealing with y currents
        #       block above skipped node (same as below)

        yblock1 = sparse.diags([np.ones(((nz/2)*(nx+1)+nx/2)),-np.ones(((nz/2)*(nx+1)+nx/2))],
                                offsets=[0,nx+1],
                                shape = (((nz/2)*(nx+1)+nx/2),((nz/2)*(nx+1)+nx/2)+nx+1))
        #       empty block to fill in the gap                  
        yblock2 = sparse.coo_matrix(((nz/2)*(nx+1)+nx/2,(nz/2)*(nx+1)+nx/2+1))
        
        #    c. combine the blocks together
        yblock = sparse.bmat([[sparse.bmat([[yblock1,yblock2]])],[sparse.bmat([[yblock2,yblock1]])]])
        
        #    d. combine x and y blocks together
        m_current = sparse.hstack([xblock,yblock])
        
        # 3. current in = current out
        m_cicu = np.hstack([np.zeros(nx*(nz+1)),np.ones(nx+1),np.zeros((nx+1)*nz),-np.ones(nx+1)])
        
        # 4. normalisation
        norm1a = sparse.coo_matrix((nx+1,(nz+1)*nx+nx+1))
        norm1b_sub = []
        for i in range(nz):
            norm1b_sub.append(sparse.diags(propertyz[i],0))
        norm1b = sparse.hstack(norm1b_sub)
        norm1c = sparse.coo_matrix((nx+1,nx+1))
        norm1 = sparse.hstack([norm1a,norm1b,norm1c])
        
        norm2a = sparse.diags(propertyx[-1],nx*nz,shape=(nx,nx*(nz+1)))
        norm2b_sub = []
        for i in range(nz):
            norm2b_sub.append(sparse.diags(propertyz[i,:nx],0,shape=(nx,nx+1)))
        norm2b = sparse.hstack(norm2b_sub)    
        
        norm2c = sparse.coo_matrix((nx,nx+1))
        norm2 = sparse.hstack([norm2a,norm2c,norm2b,norm2c])
        
        m_norm = sparse.vstack([norm1,norm2])
        
        # 5. combine all matrices together.
        m = sparse.csr_matrix(sparse.vstack([m_voltage,m_current,m_cicu,m_norm]))

        if prop == 'current':
            self.matrix_c = m
        elif prop == 'fluid':
            self.matrix_f = m
        print "matrix built, property = ",prop
        self.nfree = np.shape(m)[0]

    def build_sums(self):
        """
        builds the matrix b to solve the matrix equation Ab = C
        where A is the matrix defined in build_matrix
        and C is the electrical current values.
        
        """
        if hasattr(self,"nfree"):
            nfree = self.nfree
        else:
            print "build matrix A first"
            return
        
        b_dense = np.zeros(nfree)
        b_dense[-(2*self.nx + 1):] = float(self.nz)/(float(self.nx)+1.)
        
        self.b = sparse.csr_matrix(b_dense)        

      
    def solve_matrix(self, prop = 'current'):
        """
        """
        
        if prop == 'current':
            A = self.matrix_c
        elif prop == 'fluid':
            A = self.matrix_f
        b = self.b
        
        c = linalg.spsolve(A,b)
        
        if prop == 'current':
            self.current_x = c[:(self.nx*(self.nz+1))].reshape(self.nz+1,self.nx)
            self.current_z = c[(self.nx*(self.nz+1)):].reshape(self.nz+2,self.nx+1)
        if prop == 'fluid':
            self.flowrate_x = c[:(self.nx*(self.nz+1))].reshape(self.nz+1,self.nx)
            self.flowrate_z = c[(self.nx*(self.nz+1)):].reshape(self.nz+2,self.nx+1)
        
        print "matrix solved, property = ",prop

    
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
