# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:05:42 2014

@author: Alison Kirkby

Modelling random resistor networks using python.

"""

import numpy as np
import matplotlib.pyplot as plt
import resistornetworkfunctions as rnf 

class Plot_network():
    """
    plot the results of a model using plt.quiver
    """
    
    def __init__(self, Resistivity_volume, **input_parameters):
        
        self.Resistivity_volume = Resistivity_volume
        self.cmap = dict(fluid='gray_r', current='gray_r',
                         permeability='gray',resistance='gray')
        self.clabels = dict(resistance = 'Resistance, $\Omega$',
                            permeability = 'Permeability, $m^2$',
                            current='Current, $Amps$',
                            fluid='Fluid flow rate, $m^2/s$')
        self.clim_percent = dict(fluid=[0.,90.], current=[0.,90.],
                                 permeability=[0.,100.],resistance=[0.,100.])
        self.clim = {}
        self.cbar_dict = dict(orientation='horizontal',fraction=0.04,
                              no_cticks = 2)
        self.plot_range = dict(fluid=[], current=[],
                               permeability = [], resistance = [])
        self.plot_arrowheads = False
        self.arrow_dict = {'widthscale':0.1}
        self.parameters = 'all'
        self.plot_tf = True
        self.figsize = (12,8)
        self.hspace = 0.3
        self.wspace = 0.2
    
        update_dict = {}

        #correcting dictionary for upper case keys
        input_parameters_nocase = {}
        for key in input_parameters.keys():
            input_parameters_nocase[key.lower()] = input_parameters[key]

        update_dict.update(input_parameters_nocase)

        for key in update_dict.keys():
            try:
                value = getattr(self,key)
                if type(value) == dict:
                    try:
                        newdict = update_dict[key]
                        for key2 in newdict.keys():
                            try:
                                value[key2] = newdict[key2]
                            except:
                                pass
                    except:
                        pass
                setattr(self,key,value)
            except:
                pass
        
        self.initialise_parameter_arrays()
        if self.plot_range['permeability'] == []:
            k = self.Resistivity_volume.permeability
            self.plot_range['permeability'] = [1.1*np.amin(k[np.isfinite(k)]),
                                               1.1*np.amax(k[np.isfinite(k)])]
        if self.plot_range['resistance'] == []:
            r = self.Resistivity_volume.resistance
            self.plot_range['resistance'] = [0.9*np.amin(r[np.isfinite(r)]),
                                             0.9*np.amax(r[np.isfinite(r)])]
        
        if self.plot_tf:
            self.initialise_plot_parameters()
            self.plot()



    
    def _set_axis_params(self):
        """
        
        """
        
    def initialise_parameter_arrays(self):

        RV = self.Resistivity_volume
        allowed_params = ['resistance','current','permeability','fluid']

        if type(self.parameters) == str:      
            self.parameters = [self.parameters]
        if type(self.parameters) == list:
            tmplist = []
            for p in allowed_params:
                if p in self.parameters:
                    tmplist.append(p)
            self.parameters = tmplist

        else:
            self.parameters = 'all'
        if self.parameters == []:
            print "invalid parameters list, plotting all parameters"
            self.parameters = 'all'
        if self.parameters == 'all':
            self.parameters = allowed_params
            
        parameter_arrays = {}    
        for p in self.parameters:
            if p == 'fluid':
                parameter_arrays['fluid'] = [RV.flowrate[:,:,:,i][::-1] for i in [0,1]]
            elif p == 'current':
                parameter_arrays['current'] = [RV.current[:,:,:,i][::-1] for i in [0,1]]              
            elif p == 'resistance':
                parameter_arrays[p] = [RV.resistance[::-1]]
            elif p == 'permeability':
                parameter_arrays[p] = [RV.permeability[::-1]]
            
        self.parameter_arrays = parameter_arrays
        
        
    def initialise_plot_parameters(self):
        
        self.plot_xzuwc = {}
        RV = self.Resistivity_volume      
        plotxz = rnf.get_meshlocations([RV.dx,RV.dz],
                                       [RV.nx,RV.nz])
        n_subplots = 0
        
        for key in self.parameter_arrays.keys():
            self.plot_xzuwc[key] = []
            value_list = self.parameter_arrays[key]
            for value in value_list:
                X,Z = rnf.get_quiver_origins([RV.dx,RV.dz],
                                                 plotxz,
                                                 value)
                try: pr = self.plot_range[key]
                except: pr = None
                U,W,C = rnf.get_quiver_UW(value,plot_range=pr)

                self.plot_xzuwc[key].append([X,Z,U,W,C])
                n_subplots += 1
        self.n_subplots = n_subplots
                
    def plot(self):
        """
        
        """
        
        if not hasattr(self,'plot_xzuwc'):
            self.initialise_plot_parameters()
        
        if self.plot_arrowheads:
            hw,hl,hal = 1,2,1.5
        else:
            hw,hl,hal = 0,0,0
        
        RV = self.Resistivity_volume 
        w = min((RV.nx,RV.nz))
            
        sp = 1
        if self.n_subplots < 4:
            sx,sy = 1,self.n_subplots
        elif self.n_subplots == 4:
            sx,sy = 2,2
        elif self.n_subplots <= 6:
            sx,sy = 2,3

        # initialise xy limits for plot and clim
        clim = {}
        n = 0
        for key in self.parameters:
            for X,Z,U,W,C in self.plot_xzuwc[key]:
                if n == 0:
                    xlim = [np.amin(np.array(X)),np.amax(np.array(X))+RV.dx]
                    ylim = [np.amin(np.array(Z))-RV.dz,np.amax(np.array(Z))]
                else:
                    xlim = [min(np.amin(np.array(X)),xlim[0]),max(np.amax(np.array(X))+RV.dx,xlim[1])]
                    ylim = [min(np.amin(np.array(Z))-RV.dz,ylim[0]),max(np.amax(np.array(Z)),ylim[1])]
                UW = np.hstack([cc[np.isfinite(cc)].flatten() for cc in C])
                if key == 'resistance':
                    fr = RV.resistance[np.isfinite(RV.resistance)]
                    clim[key] = (np.amin(fr),np.amax(fr))
                elif key == 'permeability':
                    clim[key] = [RV.fracture_diameter**2/12,1.1*RV.fracture_diameter**2/12]
                else:
                    if key in clim.keys():
                        clim[key] = (min(np.percentile(UW,self.clim_percent[key][0]),clim[key][0]),
                                     max(np.percentile(UW,self.clim_percent[key][1]),clim[key][1]))
                    else:
                        clim[key] = (np.percentile(UW,self.clim_percent[key][0]),
                                     np.percentile(UW,self.clim_percent[key][1]))
                n += 1

        fig = plt.figure(figsize=self.figsize)
        fig.patch.set_alpha(0.)
        for key in self.parameters:
            for X,Z,U,W,C in self.plot_xzuwc[key]:
                
                ax = plt.subplot(sx,sy,sp)
                
                # set order of plotting
                if np.average(C[0][np.isfinite(C[0])]) > \
                np.average(C[1][np.isfinite(C[1])]):
                    ii = [1,0]
                else:
                    ii = [0,1]
#                if key == 'resistance':
#                    ii = ii[::-1]
                for i in ii:
#                    print C[i]
                    if i == 1: scale = RV.nz + 2
                    else: scale = RV.nx + 2
                    plt.quiver(X[i],Z[i],U[i],W[i],C[i],
                               scale=scale,
                               width=self.arrow_dict['widthscale']/w,
                               cmap=self.cmap[key],
                               headwidth=hw,
                               headlength=hl,
                               headaxislength=hal)
                
                    plt.clim(clim[key][0],clim[key][1])
                plt.xlim(xlim[0],xlim[1])
                plt.ylim(ylim[0],ylim[1])
                plt.xlabel('Distance, m')
                plt.ylabel('Distance, m')

                cbar = plt.colorbar(fraction=self.cbar_dict['fraction'],
                                    orientation=self.cbar_dict['orientation'],
                                    pad=0.2)
                cticks = np.linspace(clim[key][0],clim[key][1],self.cbar_dict['no_cticks'])

                cbar.set_ticks(cticks)
                if key in ['permeability']:#,'resistance']:
                    cticks = np.linspace(RV.fracture_diameter**2/12,
                                         RV.permeability_matrix,
                                         self.cbar_dict['no_cticks'])
                cbar.set_ticklabels(['%.1e'%t for t in cticks])
                cbar.set_label(self.clabels[key])
                ax.set_aspect('equal')
                sp += 1
        plt.subplots_adjust(wspace=self.wspace,hspace=self.hspace)
        self.clim = clim