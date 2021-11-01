# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:12:34 2021

@author: alisonk
"""

def interpolate_to_all_fs(outputs):
    data_dict1 = {}
    data_dict1['fs'] = np.unique(outputs[:,:,0])
    nrpts = outputs.shape[0]
    
    idx_dict = {'cf':1,'res':2,'k':3,'xcs':5}

    for pname in idx_dict.keys():
        i = idx_dict[pname]
        data_dict1[pname] = np.zeros((nrpts,data_dict1['fs'].shape[0]))
        for r in range(nrpts):
            func = interp1d(outputs[r,:,0],np.log10(outputs[r,:,i]),bounds_error=False)
            data_dict1[pname][r] = 10**func(data_dict1['fs'])
            
    return data_dict1