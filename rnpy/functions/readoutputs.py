# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:10:19 2015

@author: a1655681
"""

import os.path as op
import numpy as np


def read_header(wd,fn):
    """
    read header info to get fixed parameters and property names
    
    """
    
    # read header to get fixed parameters
    with open(op.join(wd,fn)) as f:
        fixed_params = {}
        line = f.readline()
        while 'fixed parameters' not in line:
            line = f.readline()
        while 'variable parameters' not in line:
            line = f.readline()
            plst = line.strip().split()[1:]
            try:
                fixed_params[plst[0]] = float(plst[1])
            except:
                fixed_params[plst[0]] = plst[1]
        pnames = f.readline().strip().split()[1:]
    
        return fixed_params, pnames

def read_data(wd,filelist,input_params):
    """
    read a list of output files to get all data. Fixed parameters and property
    names are taken from the first file so need to make sure the columns match.

    
    """
    data, rnos = None, None    

    fixed_params,pnames = read_header(wd,filelist[0])
    
    
    for fn in filelist:
        
        # read data from the whole file
        data0 = np.genfromtxt(op.join(wd,fn),names=pnames)
    
        # get the special parameters defined in input_params dictionary
        for key in input_params.keys():
            if key in fixed_params.keys():
                # check the value is not already there
                if fixed_params[key] not in input_params[key]:
                    input_params[key].append(fixed_params[key])
                
            elif key in pnames:
                for uval in np.unique(data0[key]):
                    if uval not in input_params[key]:
                        input_params[key].append(uval)
            else:
                print "no {}!!!".format(key)
                break
        
        if rnos is None:
            rnos = np.unique(data0['repeat'])
        else:
            data0['repeat'] += np.amax(rnos) + 1
            rnos = np.unique(np.hstack([rnos,data0['repeat']]))
        
        if data is not None:
            data = np.hstack([data,data0])
        else:
            data = data0.copy()
        
    return data, input_params, rnos

def get_rratios(data,fixed_params):  
    if 'resistivity_fluid' in fixed_params.keys():
        rf = fixed_params['resistivity_fluid']
    else:
        rf = data['resistivity_fluid']
        

    if 'resistivity_matrix' in fixed_params.keys():
        rm = fixed_params['resistivity_matrix']
    else:
        rm = data['resistivity_matrix']        
        
    return np.unique(rm/rf)

def get_thresholds_fromfile(wd,thresholdfn,rratios):
    """
    find out if threshold file exists and if so read it in
    """
    get_thresholds = False
    
    if op.exists(op.join(wd,thresholdfn)):
        thresholds = np.loadtxt(op.join(wd,thresholdfn))
        for rval in rratios:
            if rval not in thresholds[:,0]:
                get_thresholds = True
                break
            elif thresholds[thresholds[:,0]==rval][0,1] == 0.:
                get_thresholds = True
                break                
    else:
        get_thresholds = True

    if get_thresholds:
        thresholds = np.zeros([len(rratios),2])
        thresholds[:,0] = rratios

        
    return thresholds, get_thresholds

def filter_data(data0,keys,values,pnames):
    """
    grab out data with values equal to those specified by keys and values,
    pnames is a list of property names for the array data
    
    """
    # define condition to select all data with parameters given by the unique
    # set of values in the list vals
    cflag = True
    for ii, iparam in enumerate(keys):
        if iparam in pnames:
            if cflag:
                icond = data0[iparam] == values[ii]
                cflag = False
            else:
                icond = icond&(data0[iparam] == values[ii])
    # apply the condition created above
    if not cflag:
        data = data0[icond]
    else:
        data = data0.copy()
    
    return data
    
    
def get_bulk_properties(data,reference_width,km,rm,
                        pname_k = 'permeability_bulkz',
                        pname_r = 'resistivity_bulkz',
                        cellsize=None):
    """
    get the bulk permeabilty and resistivity for a given reference_width
    km and rm are the matrix permeability and resistivity
    
    """

    if cellsize is None:
       width = data['cellsizex']
    
    kbulk = ((width)*data[pname_k] + (reference_width-width)*km)/(reference_width)
    rbulk = (reference_width)/((width)/data[pname_r] + (reference_width-width)/rm)
    
    return kbulk, rbulk
    
    
def resistivity(fault_aperture, network_size, rhof, rhom):
    """
    calculate bulk resistivity of a fault extending through a network 
    of size network_size (float, length of network normal to fault plane)
    rhof = resistivity of fracture fill
    rhom = resistivity of matrix
    """
    
    return network_size/(fault_aperture/rhof + (network_size-fault_aperture)/rhom)
    
    
def permeability(fault_aperture, network_size, km):
    """
    calculate bulk permeability of a fault extending through a network 
    of size network_size (float, length of network normal to fault plane)
    km = permeability of matrix
    """
    return fault_aperture**3/(12*network_size) + \
           km*(1.-fault_aperture/(network_size))
           
def get_xy(data,kbulk,rbulk,xparam,yparam,rm,direction):
    
    
    if 'aperture' in xparam:
        x = data['aperture_mean'+direction]
    elif 'resistivity' in xparam:
        x = rm/rbulk
    if 'resistivity' in yparam:
        y = rm/rbulk
    elif 'permeability' in yparam:
        y = kbulk
    return x,y
           
def sort_xy(x,y,faultsep):
    """
    get unique values of fault separation and sort x and y by these values.
    """
    fsepu,indices = np.unique(faultsep,return_index=True)
    
    return x[indices],y[indices],indices
    
    
def get_gradient_log(x,y):
    """
    get the gradient of two sets of x, y points in log space.
    """
    
    xl = np.log10(x)
    yl = np.log10(y)
    dx = xl[1:]-xl[:-1]
    dy = yl[1:]-yl[:-1]

    return dy/dx


def get_perc_thresh(x,y,gradient_threshold,kmax = 1e-6):
    """
    get x and y values over the percolation threshold defined according to 
    a minimum gradient.
    """
    x,y = [arr[y<kmax] for arr in [x,y]]
    dydx = get_gradient_log(x,y)
    
    xpt,ypt = [],[]
    for i in range(len(dydx)):
        if (dydx)[i] >= gradient_threshold:
            xpt.append(x[i])
            ypt.append(y[i])
            if i < len(dydx)-1:
                if (dydx)[i+1] < gradient_threshold:
                    xpt.append(x[i+1])
                    ypt.append(y[i+1])
                    
    return xpt,ypt
    
    
def get_gradient_threshold(wd,filelist,outfilename, variables, sdmultiplier=3.):
    """
    get gradient threshold values based on a set of model results. gradient
    threshold calculated as follows:
    
    1. get the gradient of the resistivity-permeability curve in log space
    2. get the maximum gradient and take the log of this gradient
    3. repeat for all repeats for the given variables
    4. take the median and standard deviation of the maximum gradients
    5. the gradient threshold is defined as:
    10**(median(log10(maxgradient)) - sdmultiplier*sd(log10(maxgradient)))
    if sdmultiplier is 3 then this should capture
    
    """    
    
    mg = []
    for rno in rnos:            
        if rno in data1['repeat']:
            x,y,fsep = [arr[data1['repeat']==rno] for arr in [xall,yall,data1['fault_separation']]]
            x,y, indices = ort.sort_xy(x,y,fsep) 
            mg.append(np.amax(np.log10(ort.get_gradient_log(x,y))))

    threshold[i] = [rrat,np.median(mg)-4.*np.std(mg)]
    np.savetxt(op.join(wd,thresholdfn),threshold,fmt = ['%.1f','%.3f'])
         
    
def get_pt_all(filelist, outfilename, variables, thresholdgradient, kmax = 1e-6):
    """
    get an array containing percolation threshold + variable parameters 
    specified and save to a 1 or more column text file.
    
    columns will contain:
    variables listed in columns
    percolation start and end, defined as the resistivity and permeability at
    the beginning and end of the section that has a gradient above the
    threshold gradient.
    cellsize x start and end, cellsize perpendicular to the fault plane used
    for averaging at the beginning and end of the percolation threshold
    
    
    variables is a list containing input variable names. e.g. if variables = 
    resistivity_matrix then percolation thresholds will be sorted out by 
    matrix resistivity.
    
    thresholdgradient is a 2 column text file containing threshold gradient 
    for each combination of variables listed above
    
    """
    return
    
    
    