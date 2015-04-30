# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:52:26 2015

@author: a1655681
"""

class RandomResistorSuite():
    """
    organise and run a suite of resistivity/fluid flow runs and save results
    to a text file
    
    Author: Alison Kirkby
    """
    def __init__(self, **input_parameters):
        self.wd = 'outputs' # working directory
        self.ncells = [10,10,10]
        self.cellsize = np.array([1.,1.,1.])
        self.pconnection = np.array([[0.5,0.5,0.5]])
        self.pembedded_fault = np.array([[1.,1.,1.]])
        self.pembedded_matrix = np.array([[0.,0.,0.]])
        self.repeats = 1
        self.resistivity_matrix = 1000.
        self.resistivity_fluid = 0.1
        self.resistivity = None
        self.permeability_matrix = 1.e-18
        self.mu = 1.e-3 #default is for freshwater at 20 degrees 
        self.fault_dict = dict(fractal_dimension=2.5,
                               fault_separation = [1e-4],
                               offset = 0,
                               length_max = None,
                               length_decay = 5.,
                               mismatch_frequency_cutoff = None,
                               elevation_standard_deviation = [1e-4],
                               aperture_assignment = 'random')
        self.fault_array = None                       
        self.fault_edges = None
        self.fault_assignment = 'random' # how to assign faults, 'random' or 'list'
        self.outfile = 'outputs'                        
        self.arguments = sys.argv[1:]
        self.solve_properties = 'currentfluid'
        self.solve_directions = 'xyz'  
        self.input_parameters = input_parameters

        if len(self.arguments) > 0:
            self.read_arguments()
            

        update_dict = {}
        #correcting dictionary for upper case keys
        input_parameters_nocase = {}
        for key in input_parameters.keys():
            # only assign if it's a valid attribute
            if hasattr(self,key):
                input_parameters_nocase[key.lower()] = input_parameters[key]
            else:
                for dictionary in [self.fault_dict]:
                    if key in dictionary.keys():
                        input_parameters_nocase[key] = input_parameters[key]
                

        update_dict.update(input_parameters_nocase)
        for key in update_dict:
            try:
                # original value defined
                value = getattr(self,key)
                if type(value) == str:
                    try:
                        value = float(update_dict[key])
                    except:
                        value = update_dict[key]
                elif type(value) == dict:
                    value.update(update_dict[key])
                else:
                    value = update_dict[key]
                setattr(self,key,value)
            except:
                try:
                    if key in self.fault_dict.keys():
                        try:
                            value = float(update_dict[key])
                        except:
                            value = update_dict[key]
                        self.fault_dict[key] = value
                except:
                    continue 
        
        if type(self.cellsize) in [float,int]:
            self.cellsize = np.ones(3)*self.cellsize

        if type(self.fault_dict['fault_separation']) == float:
            self.fault_dict['fault_separation'] = [self.fault_dict['fault_separation']]
        if type(self.fault_dict['elevation_standard_deviation']) == float:
            self.fault_dict['elevation_standard_deviation'] = [self.fault_dict['elevation_standard_deviation']]
        
        self.setup_and_run_suite()

    def read_arguments(self):
        """
        takes list of command line arguments obtained by passing in sys.argv
        reads these and updates attributes accordingly
        """
        
        import argparse
        
                
        parser = argparse.ArgumentParser()
        parser.add_argument('-n','--ncells',
                            help = 'number of cells x,y and z direction',
                            nargs = 3,
                            type = int)
        parser.add_argument('-c','--cellsize',
                            help = 'number of cells x,y and z direction',
                            nargs = 3,
                            type = float)
        parser.add_argument('-p','--pconnection',
                            help = 'probability of connection in x y and z direction',
                            nargs = '*',
                            type = float)
        parser.add_argument('-pef','--pembedded_fault',
                            help = 'probability of embedment in a connected '\
                            'cell in x y and z direction',
                            nargs = '*',
                            type = float)
        parser.add_argument('-pem','--pembedded_matrix',
                            help = 'probability of embedment in an unconnected'\
                            ' cell in x y and z direction',
                            nargs = '*',
                            type = float)
        parser.add_argument('-pf','--probabilityfile',
                            help = 'space delimited text file containing '\
                            'probabilities (space delimited, order as follows'\
                            ': px py pz pefx pefy pefz pemx pemy pemz), '\
                            'alternative to command line entry, overwrites'\
                            'command line inputs')
        parser.add_argument('-r','--repeats',
                            help='number of repeats at each probability value',
                            type=int)
        parser.add_argument('-rm','--resistivity_matrix',
                            type=float)
        parser.add_argument('-rf','--resistivity_fluid',
                            type=float)
        parser.add_argument('-km','--permeability_matrix',
                            type=float)
        parser.add_argument('--fault_assignment',
                            help = 'how to assign faults, random or list')
        parser.add_argument('-mu','--fluid_viscosity',
                            type=float)
                            
        for arg in ['fractal_dimension',
                    'fault_separation',
                    'offset',
                    'length_max',
                    'length_decay',
                    'mismatch_frequency_cutoff',
                    'elevation_standard_deviation']:
                        parser.add_argument('--'+arg, type=float, nargs = '*')
        parser.add_argument('--fault_edges',type=int,nargs='*')    
        parser.add_argument('--aperture_assignment',
                            help='type of aperture assignment, random or constant')
        parser.add_argument('-wd',
                            help='working directory')
        parser.add_argument('-o','--outfile',
                            help='output file name')
        parser.add_argument('-sp','--solve_properties',
                            help='which property to solve, current, fluid or currentfluid')
        parser.add_argument('-sd','--solve_direction',
                            help='which direction to solve, x, y, z or a combination')

        args = parser.parse_args(self.arguments)

        if (hasattr(args,'probabilityfile') and (args.probabilityfile is not None)):
            try:
                pvals = np.loadtxt(args.probabilityfile)
                self.input_parameters['pconnection'] = pvals[:,:3]
                self.input_parameters['pembedded_fault'] = pvals[:,3:6]
                self.input_parameters['pembedded_matrix'] = pvals[:,6:9]
            except IOError:
                print "Can't read probability file"
        
        for at in args._get_kwargs():
            if at[1] is not None:
                if (at[0] in ['pconnection','pembedded_fault','pembedded_matrix']):
                    # make sure number of values is divisible by 3 by repeating the last value
                    while np.size(at[1])%3 != 0:
                        at[1].append(at[1][-1])
                    # reshape
                    self.input_parameters[at[0]] = np.array(at[1]).reshape(len(at[1])/3,3)
                elif at[0] == 'fault_edges':
                    if np.size(at[1])%6 != 0:
                        print "fault edges are incorrect size!!"
                    else:
                        fault_edges = []
                        for i in range(int(np.size(at[1])/6)):
                            fault_edges.append(np.array(at[1][i*6:(i+1)*6]).reshape(3,2))
                            print fault_edges
                        self.input_parameters[at[0]] = np.array(fault_edges)
                else:
                    self.input_parameters[at[0]] = at[1]


    def initialise_inputs(self):
        """
        make a list of run parameters
        """        

        list_of_inputs = []
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
