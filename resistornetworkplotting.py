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