# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:10:19 2015

@author: a1655681
"""

import os.path as op
import numpy as np
import itertools

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

def read_data(wd,filelist):
    """
    read a list of output files to get all data. Fixed parameters and property
    names are taken from the first file so need to make sure the columns match.

    variables: list containing input parameters to be varied. function
    gets out all the unique values in the data array and puts it into a 
    dictionary input_params
    
    
    """
    data, rnos = None, None
    variables = ['permeability_matrix','resistivity_matrix','resistivity_fluid']
        
    input_params = {}
    for vname in variables:
        input_params[vname] = []
    
    for fn in filelist:
        # read data from the whole file
        fixed_params,pnames = read_header(wd,fn)
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
        
    return data, input_params, fixed_params, pnames, rnos

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

def filter_data(data0,keys,values):
    """
    grab out data with values equal to those specified by keys and values,
    pnames is a list of property names for the array data
    
    """
    pnames = data0.dtype.names    

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
    
    
def get_bulk_properties(data,idict,direction=None,
                        reference_width=None,cellsize=None):
                            
    
    """
    get the bulk permeabilty and resistivity for a given reference_width
    km and rm are the matrix permeability and resistivity
    
    """

    if cellsize is None:
       width = data['cellsizex']

    
    pname_k = 'permeability_bulk'+direction
    pname_r = 'resistivity_bulk'+direction
    
    # get bulk parameters according to the reference width. If no reference
    # width we are looking at the parameters of the fracture only.
    if reference_width is None:
        kbulk,rbulk = [data[pname+'_bulk'+direction] for pname in ['permeability','resistivity']]
    else:
        kbulk = ((width)*data[pname_k] + (reference_width-width)*idict['permeability_matrix'])/(reference_width)
        rbulk = (reference_width)/((width)/data[pname_r] + (reference_width-width)/idict['resistivity_matrix'])
    
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




def get_xy(data,keys,vals,xparam,yparam,direction=None,reference_width=None):
    
    idict = {}
    for i, key in enumerate(keys):
        idict[key] = vals[i]
        
    data1 = filter_data(data,keys,vals)   
    
    kbulk,rbulk = get_bulk_properties(data1,idict,
                                      reference_width=reference_width,
                                      direction=direction)
    
    if 'aperture' in xparam:
        x = data1['aperture_mean'+direction]
    elif 'resistivity' in xparam:
        x = idict['resistivity_matrix']/rbulk
    if 'resistivity' in yparam:
        y = idict['resistivity_matrix']/rbulk
    elif 'permeability' in yparam:
        y = kbulk
        
        
    return x,y,data1,kbulk,rbulk
           
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
    
def get_perc_thresh2(x,y):
    """
    get gradient threshold using inflection point method, i.e. 
    """
    
def construct_outfilename(prefix,reference_width,sdmultiplier,parametername):
    """
    create a filename
    
    """
    # get reference width info for title
    if reference_width is None:
        rw,suf = 0,''
    else:
        for length,suf in [[1.,'m'],[1e-3,'mm'],[1e-6,'micm']]:
            rw = round(reference_width/length)
            if rw >= 1:
                break
        
    # construct an outfile name
    return prefix + 'rw%1i'%rw+suf + '_%1isd_'%sdmultiplier + parametername
   
def get_gradient_threshold(wd, filelist, outfilename = None,
                           sdmultiplier=3., reference_width = None, direction='z'):
    """
    get gradient threshold values based on a set of model results. gradient
    threshold calculated as follows:
    
    1. get the gradient of the parameters in log space
    2. get the maximum gradient and take the log of this gradient
    3. repeat for all repeats for the given variables
    4. take the median and standard deviation of the maximum gradients
    5. the gradient threshold is defined as:
    10**(median(log10(maxgradient)) - sdmultiplier*sd(log10(maxgradient)))
    if sdmultiplier is 3 then this should capture

    
    """
    parameter_names = ['resistivity','permeability']
    
    # get the array containing the data
    data, input_params, fixed_params, pnames, rnos = read_data(wd,filelist)


    thresholds = []
    # cycle through unique values
    for vals in itertools.product(*(input_params.values())):
        
        idict = {}
        for i, key in enumerate(input_params.keys()):
            idict[key] = vals[i]     
            
        # get the x and y parameters to get the percolation threshold on
        xall,yall,data1,kbulk,rbulk = get_xy(data,input_params.keys(),
                                                 vals,parameter_names[0],
                                                 parameter_names[1],
                                                 direction=direction, 
                                                 reference_width=reference_width)
        rrat = get_rratios(data1,fixed_params)

        maxgrad = []
        # go through repeats, sort by fault separation, and get out the maximum gradient in each
        for rno in rnos:            
            if rno in data1['repeat']:
                x,y,fsep = [arr[data1['repeat']==rno] for arr in [xall,yall,data1['fault_separation']]]
                x,y, indices = sort_xy(x,y,fsep)
                mg = np.log10(get_gradient_log(x,y))
                maxgrad.append(np.amax(mg[np.isfinite(mg)]))
    
        thresholds.append([rrat,np.median(maxgrad)-4.*np.std(maxgrad)])
    prefix = 'gt_o%02i'%fixed_params['offset']
    outfilename = construct_outfilename(prefix,reference_width,sdmultiplier,parameter_names[0])+'.dat'
    
    header = 'rmatrix/rfluid log10(gradient_threshold)'
    np.savetxt(op.join(wd,outfilename),np.array(thresholds),fmt = ['%.1f','%.3f'],header=header)
    
    return op.join(wd,outfilename)
         
    
def get_percolation_thresholds(wd,filelist, outfilename = None, gradientthresholdfn = None,
                               sdmultiplier=3., reference_width = None, direction='z', kmax = 1e-6):
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

    
    if gradientthresholdfn is None:
        gradientthresholdfn = get_gradient_threshold(wd, filelist, 
                               sdmultiplier=sdmultiplier, outfilename=outfilename,
                               reference_width = reference_width, 
                               direction=direction)
    gtvals = np.loadtxt(gradientthresholdfn)
    
    parameter_names = ['resistivity','permeability']

   
    # get the array containing the data
    data, input_params, fixed_params, pnames, rnos = read_data(wd,filelist)

    
    outputs = np.zeros([np.product([len(val) for val in input_params.values()]),len(rnos)],
                        dtype = [('rm/rf','f64'),('offset','f64'),('km','f64'),('repeat','i5'),
                                 ('x0','f64'),('x1','f64'),('y0','f64'),('y1','f64'),
                                 ('cs0','f64'),('cs1','f64'),('ap0','f64'),('ap1','f64'),
                                 ('ca0','f64'),('ca1','f64')])

    
    iv = 0
    for vals in itertools.product(*(input_params.values())):
    
        idict = {}
        for i, key in enumerate(input_params.keys()):
            idict[key] = vals[i]     
        
        # get the x and y parameters to get the percolation threshold on
        xall,yall,data1,kbulk,rbulk = get_xy(data,input_params.keys(),
                                                  vals,parameter_names[0],
                                                  parameter_names[1],
                                                  direction=direction, 
                                                  reference_width=reference_width) 
        rrat = get_rratios(data1,fixed_params)
        outputs['rm/rf'][iv] = rrat
        outputs['km'][iv] = idict['permeability_matrix']

        ir = 0
        for rno in rnos:
            if rno in data1['repeat']:
                x,y,fsep,csx,apm,ca = [arr[data1['repeat']==rno] for arr in \
                [xall,yall,data1['fault_separation'],data1['cellsizex'],
                 data1['aperture_mean'+direction],data1['contact_area'+direction]]]
                x,y,indices = sort_xy(x,y,fsep)
                csx = csx[indices]
                apm = apm[indices]
                ca = ca[indices]
                xpt,ypt = get_perc_thresh(x,y,10**gtvals[gtvals[:,0]==rrat][0,1],kmax = 1e-10)

                if len(xpt) > 0:
                    outputs['repeat'][iv,ir] = rno
                    outputs['x0'][iv,ir] = xpt[0]
                    outputs['x1'][iv,ir] = xpt[-1]
                    outputs['y0'][iv,ir] = ypt[0]
                    outputs['y1'][iv,ir] = ypt[-1]
                    outputs['cs0'][iv,ir] = csx[x==xpt[0]]
                    outputs['cs1'][iv,ir] = csx[x==xpt[-1]]
                    outputs['ap0'][iv,ir] = apm[x==xpt[0]]
                    outputs['ap1'][iv,ir] = apm[x==xpt[-1]]
                    outputs['ca0'][iv,ir] = ca[x==xpt[0]]
                    outputs['ca1'][iv,ir] = ca[x==xpt[-1]]
                    ir += 1

        iv += 1
    prefix = 'pt_o%02i'%fixed_params['offset']
    outfilename = construct_outfilename(prefix,reference_width,sdmultiplier,parameter_names[0])

    np.save(op.join(wd,outfilename),outputs)
        
    return op.join(wd,outfilename)
        
def average_perc_thresholds(ptfile,rratio_max = None,stderr=False):
    # load the data
    data = np.load(ptfile)
    # get rm/rf ratios
    rratios = np.unique(data['rm/rf'])
    if rratio_max is not None:
        rratios = rratios[rratios <= rratio_max]
    # create array to contain mean and standard deviation values
    data_median = np.zeros(len(rratios),dtype = data.dtype)
    data_std = np.zeros(len(rratios),dtype = data.dtype)
    # create averages for each rratio
    for i in range(len(rratios)):
        for dd in data.dtype.names:
            dtoavg = data[dd][(data[dd] != 0.)&(np.isfinite(data[dd]))&(data['rm/rf'] == rratios[i])]
            data_median[dd][i] = np.median(dtoavg)
            data_std[dd][i] = np.std(dtoavg)
            if stderr:
                data_std[dd][i] = np.std(dtoavg)/(len(data_std)**0.5)

    return rratios,data_median,data_std