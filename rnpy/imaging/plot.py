# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:05:42 2014

@author: Alison Kirkby

Modelling random resistor networks using python.

"""
import rnpy.functions.readoutputs as rnro
import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as mplfm
import os.path as op
import os

def plot_flatplate(data1,idict,parameter_names,reference_width,rno=0,direction = 'z',color='k',plot_mm = True):
    """
    plot flat plate analytical result for and y parameter names specified (can
    be any two of aperture, resistivity, permeability)
    """    
    if type(reference_width) not in [int,float]:
        fp = dict(aperture = data1['aperture_mean'+direction][data1['repeat']==rno])
        fp['aperture'].sort()
    else:
        fp = dict(aperture = np.logspace(np.log10(np.amin(data1['aperture_mean'+direction])),
                                         np.log10(np.amax(data1['aperture_mean'+direction])),20))

    fp['resistivity'] = idict['resistivity_matrix']/rnro.resistivity(fp['aperture'],
                                                                     reference_width,idict['resistivity_fluid'],
                                                                     idict['resistivity_matrix'])
    fp['permeability'] = rnro.permeability(fp['aperture'],reference_width,idict['permeability_matrix'])
    flatplate = []
#    print reference_width,fp['aperture']
    for ii, pname in enumerate(parameter_names):
        for pname2 in ['aperture','resistivity','permeability']:
            if pname2 in pname:
                flatplate.append(fp[pname2])
                break
    if ((plot_mm) and ('aperture' in parameter_names[0])):
        flatplate[0] = np.array(flatplate[0])*1e3
        
    plt.plot(flatplate[0],flatplate[1],'--',c=color)

def plot_data(data, input_params, pnames, rnos, parameter_names, 
              plot_bold = None, reference_width = None, direction = 'z',plot_params = {},
              flatplate = True, plot_mm = True):
    """
    iterate through all input permutations and repeats and plot some x y points
    
    
    
    """
    pp = dict(c='k',lw=0.1)
        
    pp.update(plot_params)
    


    for vals in itertools.product(*(input_params.values())):
        
        idict = {}
        for i, key in enumerate(input_params.keys()):
            idict[key] = vals[i]
            
        # get the x and y parameters to get the percolation threshold on
        xall,yall,data1,kbulk,rbulk = rnro.get_xy(data,input_params.keys(),
                                                       vals,parameter_names[0],
                                                       parameter_names[1],
                                                       direction=direction, 
                                                       reference_width=reference_width)
        # cycle through repeat numbers and plot
        for rno in rnos:
            if rno in data1['repeat']:
                x,y,fsep,csx = [arr[data1['repeat']==rno] for arr in [xall,yall,data1['fault_separation'],data1['cellsizex']]]
                x,y,indices = rnro.sort_xy(x,y,fsep)
                if ((plot_mm) and ('aperture' in parameter_names[0])):
                    x *= 1e3
                plt.plot(x,y,**pp)#
                        
            if plot_bold is not None:
                if rno == plot_bold:
                    plt.plot(x,y,'k-')

        if reference_width is None:
            width = csx[indices]
        else:
            width = reference_width            

        if flatplate:
            plot_flatplate(data1,idict,parameter_names,width,rno=rno,
                           direction=direction,color = pp['c'],plot_mm=plot_mm)
            
                
def plot_rk_aperture(wd,filelist,reference_width = None, direction = 'z',plot_params=None,
                     input_params = {}, plot_bold = None, plot_mm = True, flatplate = True):
    """
    plot resistivity and permeability vs aperture
    plotxy: option to plot some xy data (format [[x0,x1,..,xn],[y10,y11,..,y1n],
    [y20,y21,..,y2n]] - two sets of y data for permeability and resistivity axes)
    
    
    """

    
    data, input_params2, fixed_params, pnames, rnos = rnro.read_data(wd,filelist)
    input_params2.update(input_params)

    plst = [['aperture','permeability'],['aperture','resistivity']]
    # calculate y limits so that they are plotted on same scale in log space for
    # resistivity and permeability
    ylim = dict(permeability=[1e-18,1e-8])
    ylim['resistivity'] = [1,10**(np.log10(ylim['permeability'][1])-np.log10(ylim['permeability'][0]))]
#    print xlim,ylim
#    ylabels = {'resistivity':'Resistivity ratio $\mathrm{\mathsf{R_{matrix}/R_{fracture}}}$',
#               'permeability':'Permeability, m$\mathrm{\mathsf{^2}}$'}
    ylabels = {'resistivity':'Resistivity ratio $M$',
               'permeability':'Permeability, m$\mathrm{\mathsf{^2}}$'}    
    
    if plot_params is None:
        plot_params = dict(resistivity = dict(c='0.5',lw=0.1),
                           permeability = dict(c='k',lw=0.1))
    

    apmin,apmax = np.amin(data['aperture_mean'+direction]), np.amax(data['aperture_mean'+direction])
    xlim = [aparray for aparray in [apmin,apmax]]
    if plot_mm:
        for i in range(2):
            xlim[i] *= 1e3
    

    firstplot = True
    ax1 = plt.gca()
    for parameter_names in plst:
        ppdict = plot_params[parameter_names[1]]
        plot_data(data,input_params2,pnames,rnos,parameter_names,
                  reference_width = reference_width, direction = direction,
                  plot_params = ppdict, flatplate = flatplate,
                  plot_bold = plot_bold, plot_mm = plot_mm)
        
        plt.xlim(*xlim)
        plt.xscale('log')
        plt.ylim(*ylim[parameter_names[1]])
        plt.yscale('log')
        plt.ylabel(ylabels[parameter_names[1]], color=ppdict['c'])
        plt.gca().tick_params(axis='y', colors=ppdict['c'])
        if firstplot:
            plt.xlabel('Mean aperture, mm')  
            ax2 = plt.twinx()
            firstplot = False
          

    return ax1,ax2

def plot_r_vs_k(wd,filelist,reference_width = None, direction = 'z',flatplate = True,
                input_params = {}, limits = None,plot_bold = None, color='k'):
    
    data, input_params2, fixed_params, pnames, rnos = rnro.read_data(wd,filelist)
    input_params2.update(input_params)

    parameter_names = ['resistivity','permeability']
    xlim = [1e0,1e3]
    ylim = [1e-18,1e-8]

#    labels = {'resistivity':r'Resistivity ratio $\mathrm{\mathsf{\rho_{matrix}/\rho_{fracture}}}$',
#               'permeability':'Permeability, m$\mathrm{\mathsf{^2}}$'}
    labels = {'resistivity':'Resistivity ratio $M$',
               'permeability':'Permeability, m$\mathrm{\mathsf{^2}}$'}    
    plot_params = dict(c=color,lw=0.1)

    plot_data(data,input_params2,pnames,rnos,parameter_names,
              reference_width = reference_width, direction = direction,
              plot_params = plot_params,flatplate = flatplate,
              plot_bold = plot_bold)
    ax1 = plt.gca()
    plt.xlim(*xlim)
    plt.xscale('log')
    plt.ylim(*ylim)
    plt.yscale('log')
    plt.xlabel(labels['resistivity'])     
    plt.ylabel(labels['permeability'])    

    return ax1 


def set_axes(n,o,cellsize):
    plt.gca().set_aspect('equal')
    plt.xlim(0.,cellsize*(o-2.))
    plt.ylim(0.,cellsize*(n-2.))
    
    
    plt.setp(plt.gca().get_xticklabels(),visible=False)
    plt.setp(plt.gca().get_yticklabels(),visible=False)
    

def scalearrow(length,centre,color='k',linewidth=1):
    arrowsize = length*0.4
    head_length = length*0.1
    head_width = head_length/2.
    for i in [-1.,1.]:
        plt.arrow(centre[0], centre[1], i*arrowsize, 0.0,
                  head_length=head_length,
                  head_width=head_width,
                  linewidth=linewidth,
                  fc=color)
    plt.text(centre[0],centre[1]+0.002,'%1i cm'%(length*100.),ha='center')


def plot_fluidcurrent(wd, searchlist, cellsize, cutf = 1e-19, cutc = 1e-9,  cmap = 'gray_r',
                      figsize = (8,5.5),plot_aperture=True,plot_labels=True,
                      scale = None):
    """
    plot aperture, fluid and current for a list of models along the yz plane.
    a lot is hard coded in here. will work on generalising it at some point
    
    searchlist = a string contained in all the output>
    
    """
    
    scaleu = dict(current=1e-5,fluid=1e-12)
    if type(scale) is dict:
        scaleu.update(scale)
    
    
    ii = 0
    amplitudes = []
    labels = 'abcdef'
    nrows = len(searchlist)
    if plot_aperture:
        ncols = 3
    else:
        ncols = 2
    subplots = np.arange(1,nrows*ncols+1)
    plt.figure(figsize=figsize)
    axes = []
    plots = []
    
    for search in searchlist:
        
        flist = [op.join(ff) for ff in os.listdir(wd) if search in ff]
        if plot_aperture:
            aperture_file = [op.join(wd,ff) for ff in flist if 'aperture' in ff][0]
            aperture = np.load(aperture_file)
            ax,ay,az = [1.*aperture[1:,1:,1,i,0] for i in range(3)]
            
            
        flow_file = [op.join(wd,ff) for ff in flist if 'flow' in ff][0]
        current_file = [op.join(wd,ff) for ff in flist if 'current' in ff][0]
        
        flow = np.load(flow_file)
        current = np.load(current_file)
        
        uf,vf,wf = [1.*flow[:,:,1,2,i] for i in range(3)]
        uc,vc,wc = [1.*current[:,:,1,2,i] for i in range(3)]

        n,o = np.array(np.shape(wf))-2.
        x,y = np.meshgrid(np.linspace(0.,(np.shape(vf)[1]-1)*cellsize,np.shape(vf)[1]),
                          np.linspace(0.,(np.shape(vf)[0]-1)*cellsize,np.shape(vf)[0]))
                     

        
        if ii == 0:
            if plot_aperture:
                clima = [0.,np.percentile(az[np.isfinite(az)],100)]

                
#            climc = np.percentile(np.log10(np.abs(wc[(np.isfinite(wc))&(wc>0.)])),[0,100])
#            climf = np.percentile(np.log10(np.abs(wf[(np.isfinite(wf))&(wf>0.)])),[0,100])
            climc,climf = [np.log10([cut,np.amax(ww)]) for cut,ww in [[cutc,wc],[cutf,wf]]]
            
            print climc,climf
    
        if plot_aperture:    
            axes.append(plt.subplot(nrows,ncols,subplots[ii]))
            az[az==0.] = np.nan
            plots.append(plt.imshow(az,interpolation='none',cmap=cmap,extent=[0.,(o-2.)*cellsize,0.,(n-2.)*cellsize]))
            plt.clim(*clima)
            set_axes(n,o,cellsize)
            scalearrow(0.05,(0.03,0.005))
            if plot_labels:
                plt.title(labels[ii],loc='left')    
            ii += 1
        
        for v,w,scale,clim,cut in [[vc,wc,scaleu['current'],climc,cutc],[vf,wf,scaleu['fluid'],climf,cutf]]:
            xp,yp = x.copy(), y.copy()
            amplitudes.append(np.abs((v**2+w**2)**0.5))
            amplitudes[-1][amplitudes[-1]==0.] = np.nan
            axes.append(plt.subplot(nrows,ncols,subplots[ii]))
            yp[(v<cut)&(w<cut)] = np.nan
            xp[(v<cut)&(w<cut)] = np.nan
            v[(v<cut)&(w<cut)] = np.nan
            w[np.isnan(v)&(w<cut)] = np.nan
            plots.append(plt.quiver(xp,yp,v,w,np.log10(amplitudes[-1]),scale=scale,pivot='tail',cmap=cmap))
            

#            if ii == 0:
#                clim = np.percentile(np.log10(np.abs(w[(np.isfinite(w))&(w>0.)])),[50,100])
            print clim,cut,np.amax(w[np.isfinite(w)]),np.amin(w[np.isfinite(w)])
            plt.clim(*clim)
            set_axes(n,o,cellsize)
            if plot_labels:
                plt.title(labels[ii],loc='left')
            ii += 1
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    return plots,axes
    
def plot_pt_vs_res(ptfilelist,rratio_max = None,colors = ['0.5'],plot_fit = True,
                   rratiofitrange = None, textloc = [100,2],plot_permeability=False,
                   stderr=False,labels = None,fmt='-'):
    """
    """
    if type(ptfilelist) == str:
        ptfilelist = [ptfilelist]

    
    first = True
    for i,ptfile in enumerate(ptfilelist):

        rratios,data_median,data_std = rnro.average_perc_thresholds(ptfile,rratio_max = rratio_max,stderr=stderr)
        
        data = np.load(ptfile)
    
        if labels is None:
            labels = ['Percolation threshold']
        if (first or ((type(labels)==list) and (len(labels)==len(ptfilelist)))):
            plt.errorbar(rratios,data_median['x0'],yerr=data_std['x0'],fmt =fmt,ecolor=colors[i],c=colors[i], label = labels[i])
#            plt.xlabel(r'Resistivity ratio $\mathrm{\mathsf{\rho_{matrix}/\rho_{fluid}}}$')
#            plt.ylabel(r'Resistivity ratio $\mathrm{\mathsf{\rho_{matrix}/\rho_{fracture}}}$')
            plt.xlabel(r'Resistivity ratio $m$')
            plt.ylabel(r'${M_{PT}}$')
            plt.yscale('log')
            plt.xscale('log')
            ax = plt.gca()
            plt.legend(fontsize=12,loc='upper left')
            first = False
    
        else:
            plt.errorbar(rratios,data_median['x0'],yerr=data_std['x0'],fmt = fmt,ecolor=colors[i],c=colors[i])
            
        plt.errorbar(rratios,data_median['x1'],fmt =':',c=colors[i])#yerr=data_std['x1'],
    
        if plot_fit:
            if type(rratiofitrange) == list:
                if len(rratiofitrange) == 2:
                    rmin,rmax = rratiofitrange
                elif len(rratiofitrange) > 2:
                    rmin,rmax = rratiofitrange[:2]
                else:
                    rmin,rmax=[np.amin(rratios),np.amax(rratios)]
            else:
                rmin,rmax=[np.amin(rratios),np.amax(rratios)]
            if rmin is None:
                rmin = np.amin(rratios)
            if rmax is None:
                rmax = np.amax(rratios)
#            condition = (data['rm/rf'] >= rmin)&(data['rm/rf'] <= rmax)&(data['x0']>0.)
#            x,y = data['rm/rf'][condition],data['x0'][condition]
            condition = (data_median['rm/rf'] >= rmin)&(data_median['rm/rf'] <= rmax)&(data_median['x0']>0.)
            x,y = data_median['rm/rf'][condition],data_median['x0'][condition]            
            [m,c],cov = np.polyfit(np.log10(x),np.log10(y),1,cov=True)
            merr = cov[0,0]**0.5
            # need to transform c back into linear space
            cerr = np.average([np.abs(10.**(c+cov[1,1]**0.5)-10.**c),np.abs(10.**c - 10.**(c-cov[1,1]**0.5))])
            
            plt.plot(rratios,10.**c*rratios**m,'k--',label = 'Best fit line for \npercolation threshold')
            plt.text(textloc[0],textloc[1],'$y={:0.2f}\pm{:0.2f}'.format(10**c,cerr)+'x^{'+'{:.3f}\pm{:.3f}'.format(m,merr)+'}$',color='k',fontsize=14)

        if plot_permeability:
            x01 = np.hstack([data_median['x0'],data_median['x1']])
            y01 = np.log10(np.hstack([data_median['y0'],data_median['y1']]))
                
            plt.scatter(np.hstack([rratios]*2),x01,c=y01,lw=0)
            clim = [round(np.amin(y01)),round(np.amax(y01))]
            plt.clim(clim)
            cbar = plt.colorbar()
            cbar.set_label('log$_{10}$(Permeability, m$^2$)')
            cbar.set_ticks(clim)
            cbar.set_ticklabels(['%1i'%i for i in clim])
        

    return ax
    

def plot_pt_vs_offset(ptfilelist, offset_vals, rratio_values, offset_units='mm',
                      color = '0.5', plot_permeability=False):
    """
    """
    
    rratios,data_median,data_std = rnro.average_perc_thresholds(ptfilelist[0])    
    
    for ptfile in ptfilelist[1:]:
        rratios2,data_median2,data_std2 = rnro.average_perc_thresholds(ptfile)
        data_median = np.hstack([data_median,data_median2])
        data_std = np.hstack([data_std,data_std2])
        
    for rratio_value in rratio_values:
        
        data1_median = data_median[data_median['rm/rf']==rratio_value]
        data1_std = data_std[data_median['rm/rf']==rratio_value]
#        print data1_median['x0'],data1_std['x0']
        plt.errorbar(offset_vals,data1_median['x0'],yerr=data1_std['x0'],fmt ='-',c=color, label = 'Percolation threshold')
    ax = plt.gca()
    plt.yscale('log')
    plt.xlabel('Offset, '+offset_units)
    plt.ylabel(r'Resistivity ratio ($\mathrm{\mathsf{\rho_{matrix}/\rho_{fracture}}}$)')

#    if plot_permeability:
#        x01 = np.hstack([data_median['x0'],data_median['x1']])
#        y01 = np.log10(np.hstack([data_median['y0'],data_median['y1']]))
#            
#        plt.scatter(np.hstack([rratios]*2),x01,c=y01,lw=0)
#        clim = [round(np.amin(y01)),round(np.amax(y01))]
#        plt.clim(clim)
#        cbar = plt.colorbar()
#        cbar.set_label('log$_{10}$(Permeability, m$^2$)')
#        cbar.set_ticks(clim)
#        cbar.set_ticklabels(['%1i'%i for i in clim])
#    plt.legend(fontsize=10,loc='upper left')                                                    

    return ax

    
def plot3dconnectors(connector_array,cellsize,connector_type=None,thresh=None):
    """
    plot connectors from a resistor network in 3d. can plot aperture, resistance
    or permeability
    
    Inputs:
    **connector_array** array containing connectors. shape (nz+2,ny+2,nx+2,3,3) if
    the connector_type is aperture, otherwise shape (nz+2,ny+2,nx+2,3)
    **cellsize** list, tuple or array containing cellsize in x,y and z directions
    **connector_type** string, either 'aperture', 'resistance' (hydraulic or electric),
    or 'permeability'
    
    
    """
    import mayavi.mlab as mlab
    
    # try to guess connector type if it's not provided
    if connector_type is None:
        if len(np.shape(connector_array)) < 5:
            if np.nanmax(connector_array) < 1.:
                connector_type = 'permeability'
            else:
                connector_type = 'resistance'
        else:
            connector_type = 'aperture'

    # make a copy of the array for plotting, if we're plotting apertures then sum the two directions
    if connector_type == 'aperture':
        ap = connector_array.sum(axis=4)
    elif connector_type == 'resistance':
        ap = 1./connector_array
    elif connector_type == 'permeability':
        ap = connector_array.copy()

    if thresh is None:
        thresh = 1.1*np.nanmin(ap[ap>0])            
    
    # set nan and small apertures to 0.
    ap[np.isnan(ap)] = 0.
    ap[ap<thresh] = 0.


    
    print 'connector_type',connector_type
    print 'thresh',thresh
    
    # get number of cells and cellsize
    nz,ny,nx = np.array(ap.shape[:3]) - 2.
    dx,dy,dz = cellsize
    
    # get x,y z points of apertures, need to transpose to get sorting by z, y and x direction in that order
    x,y,z = [arr.transpose(2,0,1)*1e3 for arr in np.meshgrid(np.linspace(0,dx*(nx+1),nx+2),
                                                             np.linspace(0,dy*(ny+1),ny+2),
                                                             np.linspace(0,dz*(nz+1),nz+2))]
    # maximum color value, same for each array
    vmax=ap.max()
    mlab.figure()
    ## xconnectors
    ux,vx,wx = [np.ones_like(x)*sc for sc in [dx*1e3,0.,0.]]
    ux[ap[:,:,:,0] == 0.] = 0.
    quiv = mlab.quiver3d(x,y,z,ux,vx,wx,mode='2ddash',scale_factor=0.9,scalars=ap[:,:,:,0],vmin=0.,vmax=vmax)
    quiv.glyph.color_mode = 'color_by_scalar'
    
    # y connectors
    uy,vy,wy = [np.ones_like(y)*sc for sc in [0.,dy*1e3,0.]]
    vy[ap[:,:,:,1] == 0.] = 0.
    quiv = mlab.quiver3d(x,y,z,uy,vy,wy,mode='2ddash',scale_factor=0.9,scalars=ap[:,:,:,1],vmin=0.,vmax=vmax)
    quiv.glyph.color_mode = 'color_by_scalar'
    
    # z connectors
    uz,vz,wz = [np.ones_like(z)*sc for sc in [0.,0.,dz*1e3]]
    wz[ap[:,:,:,2] == 0.] = 0.
    quiv = mlab.quiver3d(x,y,z,uz,vz,wz,mode='2ddash',scale_factor=0.9,scalars=ap[:,:,:,2],vmin=0.,vmax=vmax)
    quiv.glyph.color_mode = 'color_by_scalar'    


def plot3dflow(flow_array,cellsize,thresh=1e-40,model_direction=2,direction='xyz',scale_factor = None,arrows='single'):
    import mayavi.mlab as mlab
    # make a copy of the array for plotting
    arr = flow_array.copy()
    
    # set nan and small apertures to 0.
    arr[np.isnan(arr)] = 0.
    arr[np.abs(arr)<thresh] = 0.
    arrmax = np.nanmax(arr)
    arrmin = np.nanmin(arr)
    # get number of cells and cellsize
    nz,ny,nx = np.array(arr.shape[:3]) - 2.
    dx,dy,dz = cellsize
    
    # get x,y z points of apertures, need to transpose to get sorting by z, y and x direction in that order
    x,y,z = [arr1.transpose(2,0,1)*1e3 for arr1 in np.meshgrid(np.linspace(0,dx*(nx+1),nx+2),
                                                               np.linspace(0,dy*(ny+1),ny+2),
                                                               np.linspace(0,dz*(nz+1),nz+2))]
    
    mlab.figure()
    
    if 'x' in direction:
        ## x currents
        vx,wx = [np.zeros_like(x)]*2
        ux = arr[:,:,:,model_direction,0]/arrmax
        
    else:
        ux = np.zeros_like(x)
    if 'y' in direction:
        # y connectors
        uy,wy = [np.zeros_like(y)]*2
        vy = arr[:,:,:,model_direction,1]/arrmax
        
    else:
        vy = np.zeros_like(y)    
    if 'z' in direction:
        # z connectors
        uz,vz = [np.zeros_like(z)]*2
        wz = arr[:,:,:,model_direction,2]/arrmax
#        quiv = 
    else:
        wz = np.zeros_like(z)
    optargs = dict(mode='2darrow',vmin=arrmin,vmax=arrmax)
    if scale_factor is not None:
        optargs['scale_factor'] = scale_factor
    if arrows == 'single':
        
        mlab.quiver3d(x,y,z,ux,vy,wz,**optargs)
    elif arrows == 'broken':
        mlab.quiver3d(x,y,z,ux,vx,wx,**optargs)
        mlab.quiver3d(x,y,z,uy,vy,wy,**optargs)
        mlab.quiver3d(x,y,z,uz,vz,wz,**optargs)