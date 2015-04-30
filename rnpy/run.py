# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:52:26 2015

@author: a1655681
"""

import numpy as np
import sys

arguments = sys.argv[1:]

def read_arguments(arguments):
    """
    takes list of command line arguments obtained by passing in sys.argv
    reads these and updates attributes accordingly
    """
    
    import argparse
    
            
    parser = argparse.ArgumentParser()
    
    # arguments to put into parser:
    # [longname,shortname,help,nargs,type]
    argument_names = [['ncells','n','number of cells x,y and z direction',3,int],
                      ['cellsize','c','cellsize in x,y and z direction',3,float],
                      ['pconnectionx','px','probability of connection in x direction','*',float],
                      ['pconnectiony','py','probability of connection in y direction','*',float],
                      ['pconnectionz','pz','probability of connection in z direction','*',float],
                      ['resistivity_matrix','rm','','*',float],
                      ['resistivity_fluid','rf','','*',float],
                      ['permeability_matrix','km','','*',float],
                      ['fluid_viscosity','mu','','*',float]
                      ['fault_assignment',None,'how to assign faults, random or list, '\
                                               'if list need to provide fault edges',1,str],
                      ['offset',None,'number of cells offset between fault surfaces','*',float],
                      ['length_max',None,'maximum fault length, if specifying random faults','*',float],
                      ['length_decay',None,'decay in fault length, if specifying random '\
                                           'fault locations, for each fault: '\
                                           'faultlength = length_max*exp(-length_decay*R)'\
                                           'where R is a random number in [0,1]','*',float],
                      ['mismatch_frequency_cutoff',None,
                      'frequency cutoff for matching between faults','*',float],
                      ['elevation_standard_deviation',None,
                      'standard deviation in elevation of fault surfaces','*',float],
                      ['fractal_dimension',None,
                      'fractal dimension of fault surfaces, recommended values in range (2.0,2.5)',
                      '*',float],
                      ['fault_separation',None,'amount to separate faults by, in metres','*',float],
                      ['fault_edges','fe','indices of fault edges in x,y,z directions '\
                                          'xmin xmax ymin ymax zmin zmax',6,int],
                      ['aperture_assignment',None,'type of aperture assignment, random or constant',1,str],
                      ['workdir','wd','working directory',1,str],
                      ['outfile','o','output file name',1,str],
                      ['solve_properties','sp','which property to solve, current, fluid or currentfluid (default)',1,str],
                      ['solve_direction','sd','which direction to solve, x, y, z or a combination, e.g. xyz (default), xy, xz, y, etc',1,str]]
                      
    for longname,shortname,helpmsg,nargs,vtype in argument_names:
        if longname == 'fault_edges':
            action = 'append'
        else:
            action = 'store'
        longname = '--'+longname
        
        if shortname is None:
            parser.add_argument(longname,help=helpmsg,
                                nargs=nargs,type=vtype)
        else:
            shortname = '-'+shortname
            parser.add_argument(shortname,longname,help=helpmsg,
                                nargs=nargs,type=vtype)
    

    args = parser.parse_args(arguments)

    loop_parameters = {}
    fixed_parameters = {}    
    
    for at in args._get_kwargs():
        if at[1] is not None:
            if at[0] == 'fault_edges':
                nf = len(at[1])
                value = np.reshape(at[1],(nf,3,2))
            else:
                value = at[1]
                
            if type(at[1]) != list:
                fixed_parameters[at[0]] = value
            elif len(value) == 1:
                fixed_parameters[at[0]] = value[0]
            else:
                loop_parameters[at[0]] = value
    
    return fixed_parameters, loop_parameters


def initialise_inputs(fixed_parameters, loop_parameters):
    """
    make a list of run parameters
    """        

    list_of_inputs = []
    
    fixed_param_list =
    
    
    parameter_list = [v for v in dir(self) if v[0] != '_']
    fdkeys = [k for k in self.fault_dict.keys() if k not in ['fault_separation','elevation_standard_deviation']]
    parameter_list += fdkeys






    print self.fault_dict['fault_separation']
    for r in range(self.repeats):
        for pc in self.pconnection:
            for pef in self.pembedded_fault:
                for pem in self.pembedded_matrix:
                    for sd in self.fault_dict['elevation_standard_deviation']:
                        for fs in self.fault_dict['fault_separation']:
                            input_dict = {} 
                            for key in parameter_list:
                                if key in ['fault_assignment',
                                           'fault_edges',
                                           'permeability_matrix',
                                           'resistivity_matrix',
                                           'resistivity_fluid',
                                           'wd',
                                           'mu',
                                           'outfile',
                                           'ncells',
                                           'cellsize']:
                                    input_dict[key] = getattr(self,key)
                                elif key in fdkeys:
                                    input_dict[key] = self.fault_dict[key]
                            input_dict['fault_separation'] = fs
                            input_dict['elevation_standard_deviation'] = sd
                            input_dict['pconnection'] = pc
                            input_dict['pembedded_fault'] = pef
                            input_dict['pembedded_matrix'] = pem
                            list_of_inputs.append(input_dict)

    return list_of_inputs
    
    
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
                     'current','flowrate','aperture_array']:
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
        print "gathering outputs..",
        # flatten list, outputs currently a list of lists
        og2 = []
        count = 1
        for group in outputs_gathered:
            for ro in group:
                og2.append(ro)
                print count,
                count += 1
        print "gathered outputs into list, now sorting"
        results = {}
        for prop in ['resistivity_bulk','permeability_bulk']:
            if hasattr(ro,prop):
                results[prop] = np.vstack([np.hstack([ro.pconnection,
                                                     [ro.fault_dict['fault_separation']],
                                                     [ro.fault_dict['elevation_standard_deviation']],
                                                     getattr(ro,prop)]) for ro in og2])
        print "outputs sorted, now writing to a text file, define header"
        # save results to text file
        # first define header
        header  = '# resistor network models - results\n'
        header += '# resistivity_matrix (ohm-m) {}\n'.format(self.resistivity_matrix)
        header += '# resistivity_fluid (ohm-m) {}\n'.format(self.resistivity_fluid)
        header += '# permeability_matrix (m^2) {}\n'.format(self.permeability_matrix)
        header += '# fracture diameter (m) {}\n'.format(self.fault_dict['fault_separation'])
        header += '# fluid viscosity {}\n'.format(self.mu)
        header += '# fracture max length {}\n'.format(self.fault_dict['length_max'])
        header += '# fracture length decay {}\n'.format(self.fault_dict['length_decay'])
        header += '# ncells {} {} {}\n'.format(self.ncells[0],
                                               self.ncells[1],
                                               self.ncells[2])
        header += '# cellsize (metres) {} {} {}\n'.format(self.cellsize[0],
                                                          self.cellsize[1],
                                                          self.cellsize[2])
        header += ' '.join(['# px','py','pz','fault_sep','elev_sd','propertyx','propertyy','propertyz'])
        "header defined"
        for rr in results.keys():
            np.savetxt(os.path.join(self.wd,rr+'.dat'),np.array(results[rr]),
                       comments='',
                       header = header,
                       fmt=['%4.2f','%4.2f','%4.2f',
                            '%6.2e','%6.2e',
                            '%6.3e','%6.3e','%6.3e'])
                   
                   
if __name__ == "__main__":
RandomResistorSuite()

