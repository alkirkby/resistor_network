# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 09:25:37 2014

@author: Alison Kirkby
Tests for resistor network modelling

"""

import unittest
import resistornetwork3d as rnc
import numpy as np


class TestRNClasses(unittest.TestCase):
    def setUp(self):
        ra = np.ones([6,8,4,3])*np.nan
        ra2 = np.ones([4,4,4,3])*np.nan
#        for i in range(len(ra)):
        ra[1:-1,1:,1:,2] = 1.
        ra[1:,1:-1,1:,1] = 1.
        ra[1:,1:,1:-1,0] = 1.
        ra2[1:-1,1:,1:,2] = 1.
        ra2[1:,1:-1,1:,1] = 1.
        ra2[1:,1:,1:-1,0] = 1.

#        print ra
        self.res_ones_2x6x4array = ra
        self.res_ones_2x2x2array = ra2
#        print self.res_ones_2x2x2array
        
    
    def test_res_nanvalues(self):
        """
        tests setup of nan values in ones and random array
        
        """
        rf = 1.
        rm = 1000.
        for rtype in ['ones','random']:
            a = self.res_ones_2x6x4array*1.
            rv = rnc.Resistivity_volume(res_type=rtype,
                                        nx=2,
                                        ny=6,
                                        nz=4,
                                        resistivity_fluid=rf,
                                        resistivity_matrix=rm)
            r = rv.resistivity
            rr = rv.resistance
            b = r/r
            bb = rr/rr
            
            a[np.isnan(a)] = 0.
            
            for val in [b,bb]:
                val[np.isnan(val)] = 0.
                self.assertTrue(all(a==val))
    
        
        
    def test_res_random(self):
        """
        tests that the res array initialised by res_type = 'random' contains 
        only matrix, fluid or nan values
        
        """
        rv = rnc.Resistivity_volume(res_type='random',
                    nx=12,
                    ny=4,
                    nz=6,
                    resistivity_fluid=1.,
                    resistivity_matrix=1000.)
        a = np.array([0.,1.,1000.])
        b = rv.resistivity
        b[np.isnan(b)] = 0.
        b = np.unique(b)
        self.assertTrue(all(a==b))
        
        
    def test_res_ones(self):
        """
        Test initialisation of fully connected network
        """
        a = self.res_ones_2x6x4array*1.
        rv = rnc.Resistivity_volume(res_type='ones',
                                    nx=2,
                                    ny=6,
                                    nz=4,
                                    resistivity_fluid=1.,
                                    resistivity_matrix=1000.)
        b = rv.resistivity
        a[np.isnan(a)] = 0.
        b[np.isnan(b)] = 0.
        self.assertTrue(all(a==b))
        
        
    def test_res_array(self):
        rf,rm = 1.,1000.
        a = self.res_ones_2x6x4array*rf
        a[2,1:] = rm
        a[1:,3] = rm
        rv = rnc.Resistivity_volume(res_type='array',
                                    resistivity=a*1.,
                                    nx=4,
                                    nz=6)
        b = rv.resistivity
        a[np.isnan(a)] = 0.
        b[np.isnan(b)] = 0.
        # test that resistivities of matrix and fluid have been found correctly
        self.assertEqual(np.amax(a),rv.resistivity_matrix)
        self.assertEqual(np.amin(a[a!=0]),rv.resistivity_fluid)
        
        #test that the array has been transferred to the object correctly
        self.assertTrue(all(a==b))


        
    def test_res_unconnected(self):
        """
        Test 1
        
        Unconnected - resistivity = matrix resistivity (should equal matrix res)
          
        """
        n=[2,2,2]
        mres = 1000.

        rv = rnc.Resistivity_volume(res_type='array',
                                    fracture_diameter = 1e-3,
                                    resistivity = mres*self.res_ones_2x2x2array,
                                    permeability_matrix=1e-18,
                                    nx=n[0],ny=n[1],nz=n[2]
                                    )
#        print(rv.nx,rv.ny,rv.nz)
        
        rv.solve_resistor_network('current','xyz')
        for i in range(3):
            if i == 0:
                lx,ly,lz = n[2]+1,n[1]+1,n[0]
            if i == 1:
                lx,ly,lz = n[0]+1,n[2]+1,n[1]
            if i == 2:
                 lx,ly,lz = n[0]+1,n[1]+1,n[2]
            self.assertAlmostEqual(rv.resistivity_bulk[i],
                                   mres,
                                   delta=0.0001)
            self.assertAlmostEqual(rv.resistance_bulk[i],
                                   mres*lz/(lx*ly),
                                   delta=0.001
                                   )

    def test_res_unconnected2(self):
        """
        Test 1
        
        Unconnected - resistivity = matrix resistivity (should equal matrix res)
          
        """
        n=[2,6,4]
        mres = 1000.

        rv = rnc.Resistivity_volume(res_type='array',
                                    fracture_diameter = 1e-3,
                                    resistivity = mres*self.res_ones_2x6x4array,
                                    permeability_matrix=1e-18,
                                    nx=n[0],ny=n[1],nz=n[2]
                                    )

        rv.solve_resistor_network('current','xyz')
        for i in range(3):
            if i == 0:
                lx,ly,lz = n[2]+1,n[1]+1,n[0]
            if i == 1:
                lx,ly,lz = n[0]+1,n[2]+1,n[1]
            if i == 2:
                 lx,ly,lz = n[0]+1,n[1]+1,n[2]
            self.assertAlmostEqual(rv.resistivity_bulk[i],
                                   mres,
                                   delta=0.0001)
            self.assertAlmostEqual(rv.resistance_bulk[i],
                                   mres*lz/(lx*ly),
                                   delta=0.001
                                   )

    def test_res_1colconnected(self):
        """
        Test 2
        
        1 planar connector fully connected in xz plane:
        
             |  
             |
             |
             |
             |
             |
        
        """
        n = [2,6,4]
        # matrix and fluid resistivities
        mres = 100000.
        fres = 0.1
        # cell size, same in x, y and z directions
        d = 0.1
        a = mres*self.res_ones_2x6x4array
        a[:,3,:,0] = fres
        a[:,3,:,2] = fres
        # restore nulls
        a = a*self.res_ones_2x6x4array      
        # fracture diameter
        fd = 1e-3
        # fracture porosity
        phi = np.pi*(fd/(2*d))**2
        # resistivity of fault
        rf = 1./(phi/fres + (1-phi)/mres)
        # resistance of one connector, connected and unconnected
        rconn = rf/d
        ruc = mres/d

        rv = rnc.Resistivity_volume(res_type='array',
                                    fracture_diameter = fd,
                                    resistivity_matrix=mres,
                                    
                                    resistivity = a,
                                    dx=d*1.,dy=d*1.,dz=d*1.,
                                    nx=n[0],ny=n[1],nz=n[2]
                                    )    
        rv.solve_resistor_network('current','xyz')
        ixz = [((n[2]+1.)/n[0])*(1./rconn + n[1]/ruc),
               ((n[0]+1.)/n[2])*(1./rconn + n[1]/ruc)]#[::-1]

        self.assertAlmostEqual(rv.resistance_bulk[1],
                               mres*n[1]/((n[0]+1)*(n[2]+1)*d),
                               delta=0.01)
        self.assertAlmostEqual(rv.resistivity_bulk[1],
                               mres,
                               delta=0.01)
        for i,ii in enumerate([0,2]):
            self.assertAlmostEqual(rv.resistance_bulk[ii],
                                   1./ixz[i],
                                    delta=0.01)
            n1,n2 = [n[nn] for nn in range(3) if nn != ii]
            self.assertAlmostEqual(rv.resistivity_bulk[ii],
                                   ((n1+1)*(n2+1)*d)/(ixz[i]*n[ii]),
                                    delta=0.01)

    def test_res_1colconnected2(self):
        """
        Test 2
        
        1 planar connector fully connected in xz plane with different cell
        sizes in x,y and z directions:
        
             |  
             |
             |
             |
             |
             |
        
        """
        n = [2,6,4]
        # matrix and fluid resistivities
        mres = 10000000.
        fres = 0.1
        # cell size, same in x, y and z directions
        d = np.array([0.2,0.1,0.05])
        a = mres*self.res_ones_2x6x4array
        for col in range(1,n[1]):
            a = mres*self.res_ones_2x6x4array
            a[:,col,:,0] = fres
            a[:,col,:,2] = fres
            # restore nulls
            a = a*self.res_ones_2x6x4array      
            # fracture diameter
            fd = 1e-3
            # fracture porosity in x y and z directions
            area = np.around([np.product([dd for dd in d if dd!=ddd]) for ddd in d],
                              decimals=6)
    
            phi = np.pi*(fd/2)**2/area
            # resistivity of fault in x and z directions
            rf = 1./(phi/fres + (1-phi)/mres)
            # resistance of one connector, connected and unconnected
            rconn = rf*d/area
            ruc = mres*d/area
    
            rv = rnc.Resistivity_volume(res_type='array',
                                        fracture_diameter = fd,
                                        resistivity_matrix=mres,  
                                        resistivity = a,
                                        dx=d[0],dy=d[1],dz=d[2],
                                        nx=n[0],ny=n[1],nz=n[2]
                                        )
                                        
            rv.solve_resistor_network('current','xyz')
            ixz = [((n[2]+1.)/n[0])*(1./rconn[0] + n[1]/ruc[0]),
                   ((n[0]+1.)/n[2])*(1./rconn[2] + n[1]/ruc[2])]#[::-1]
    
            self.assertAlmostEqual(rv.resistance_bulk[1],
                                   mres*n[1]*d[1]/((n[0]+1)*(n[2]+1)*area[1]),
                                   delta=0.01)
            self.assertAlmostEqual(rv.resistivity_bulk[1],
                                   mres,
                                   delta=0.01)       
            for i,ii in enumerate([0,2]):
                self.assertAlmostEqual(rv.resistance_bulk[ii],
                                       1./ixz[i],
                                       delta=0.01)
                n1,n2 = [n[nn] for nn in range(3) if nn != ii]
                self.assertAlmostEqual(rv.resistivity_bulk[ii],
                                       ((n1+1)*(n2+1)*area[ii])/(ixz[i]*n[ii]*d[ii]),
                                        delta=0.01)

    
    
    def test_res_allcolconnected(self): 
        """
        Test 3
        
        all verticals connected in x direction (repeated for y and z):
        
        | | | | |
        | | | | |
        | | | | |
        | | | | |
        | | | | |
        | | | | |
          
        """
        n = [2,6,4]
        # matrix and fluid resistivities
        mres = 100000.
        fres = 0.1
        # cell size, same in x, y and z directions
        d = np.array([0.2,0.1,0.05])
        # fracture diameter
        fd = 1e-3
        # area of 1 cell in direction perpendicular to connected cells
        area = np.around([np.product([dd for dd in d if dd!=ddd]) for ddd in d],
                          decimals=6)
        # number of cells in direction perpendicular to connected cells
        nxcells = np.array([np.product([nn+1 for nn in n if nn!=nnn]) for nnn in n])
        # fracture porosity in x y and z directions
        phi = np.pi*(fd/2)**2/area
        # resistivity of fault in x y and z directions
        rf = 1./(phi/fres + (1-phi)/mres)

        
        for direction in [0,1,2]:

            a = mres*self.res_ones_2x6x4array
            a[:,:,:,direction] = fres
            # restore nulls
            a = a*self.res_ones_2x6x4array      
    
            rv = rnc.Resistivity_volume(res_type='array',
                                        fracture_diameter = fd,
                                        resistivity_matrix=mres,
                                        resistivity = a,
                                        dx=d[0],dy=d[1],dz=d[2],
                                        nx=n[0],ny=n[1],nz=n[2]
                                        )           
            rv.solve_resistor_network('current','xyz')
    
            for j in range(3):
                if j == direction:
                    res = rf[direction]
                else:
                    res = mres*1.
                self.assertAlmostEqual(rv.resistivity_bulk[j],
                                       res,
                                       delta=0.01)
                self.assertAlmostEqual(rv.resistance_bulk[j],
                                       res*n[j]*d[j]/(nxcells[j]*area[j]),
                                       delta=0.01)            


    def test_res_1colgap(self): 
        """
        Test 4
        
        all verticals connected except 1:
        
        | | | | |
        | | | | |

        | | | | |
        | | | | |
        | | | | |
          
        """
        n = [2,6,4]
        # matrix and fluid resistivities
        mres = 100000.
        fres = 0.1
        # cell size, same in x, y and z directions
        d = np.array([0.2,0.1,0.05])
        # fracture diameter
        fd = 1e-3
        # area of 1 cell in direction perpendicular to connected cells
        area = np.around([np.product([dd for dd in d if dd!=ddd]) for ddd in d],
                          decimals=6)
        # number of cells in direction perpendicular to connected cells
        nxcells = np.array([np.product([nn+1 for nn in n if nn!=nnn]) for nnn in n])
        # fracture porosity in x y and z directions
        phi = np.pi*(fd/2)**2/area
        # resistivity of fault in x y and z directions
        rf = 1./(phi/fres + (1-phi)/mres)

        
        for direction in [0,1,2]:
            for row in range(1,n[direction]):
                a = mres*self.res_ones_2x6x4array
                a[:,:,:,direction] = fres

                if direction == 0:
                    a[:,:,row,0] = mres
                elif direction == 1:
                    a[:,row,:,1] = mres            
                elif direction == 2:
                    a[row,:,:,2] = mres 
    
                # restore nulls
                a = a*self.res_ones_2x6x4array      
                rv = rnc.Resistivity_volume(res_type='array',
                                            fracture_diameter = fd,
                                            resistivity_matrix=mres,
                                            resistivity = a,
                                            dx=d[0],dy=d[1],dz=d[2],
                                            nx=n[0],ny=n[1],nz=n[2]
                                            )           
                rv.solve_resistor_network('current','xyz')
        
                for j in range(3):
                    if j == direction:
                        res = (rf[direction]*(n[direction]-1) + mres)/n[direction]
                    else:
                        res = mres*1.

                    self.assertAlmostEqual(rv.resistivity_bulk[j],
                                           res,
                                           delta=0.01)
                    self.assertAlmostEqual(rv.resistance_bulk[j],
                                           res*n[j]*d[j]/(nxcells[j]*area[j]),
                                           delta=0.01)  

    def test_k_nanvalues(self):
        """
        tests setup of nan values in ones and random array
        
        """
        fd = 1e-3
        km = 1e-18
        for rtype in ['ones','random']:
            a = self.res_ones_2x6x4array*1.
            rv = rnc.Resistivity_volume(res_type=rtype,
                                        nx=2,
                                        ny=6,
                                        nz=4,
                                        permeability_matrix=km,
                                        fracture_diameter = fd)
            r = rv.permeability
            rr = rv.hydraulic_resistance
            b = r/r
            bb = rr/rr
            
            a[np.isnan(a)] = 0.
            
            for val in [b,bb]:
                val[np.isnan(val)] = 0.
                self.assertTrue(all(a==val))
    
        
        
    def test_k_random(self):
        """
        tests that the k array initialised by res_type = 'random' contains 
        only matrix, fluid or nan values
        
        """
        fd = 1e-3
        km = 1e-18

        rv = rnc.Resistivity_volume(res_type='random',
                    nx=12,
                    ny=4,
                    nz=6,
                    permeability_matrix=km,
                    fracture_diameter = fd,
                    resistivity_fluid=1.,
                    resistivity_matrix=1000.)

        kf = fd**2/32.
        a = np.array([km,kf])
        b = np.unique(rv.permeability[np.isfinite(rv.permeability)])
        self.assertTrue(all((a-b)/a<0.001))
        
        
    def test_k_ones(self):
        """
        Test initialisation of fully connected network using res_type = 'ones'
        """
        fd = 1e-3
        kf = fd**2/32.
        a = self.res_ones_2x6x4array*kf
        rv = rnc.Resistivity_volume(res_type='ones',
                                    nx=2,
                                    ny=6,
                                    nz=4,
                                    resistivity_fluid=1.,
                                    resistivity_matrix=1000.)
        b = rv.permeability
        a[np.isnan(a)] = 0.
        b[np.isnan(b)] = 0.
        self.assertTrue(all(a==b))
        
        
    def test_k_array(self):
        """
        test initialisation of a network using res_type = 
        'array'
        """
        
        km = 1e-18
        fd = 1e-3
        kf = fd**2/32.
        rm,rf=1000.,1.
        a = self.res_ones_2x6x4array*rf
        a[2,1:] = rm
        a[1:,3] = rm
        rv = rnc.Resistivity_volume(res_type='array',
                                    resistivity=a*1.,
                                    fracture_diameter = fd,
                                    permeability_matrix = km
                                    )
        b = rv.permeability
        a[a==rm] = km
        a[a==rf] = kf
        a[np.isnan(a)] = 0.
        b[np.isnan(b)] = 0.
        
        
        # test that resistivities of matrix and fluid have been found correctly
        self.assertEqual(km,rv.permeability_matrix)
        self.assertEqual(kf,np.amax(b))
        
        #test that the array has been transferred to the object correctly
        self.assertTrue(all(a==b))


        
    def test_k_unconnected(self):
        """
        Test 1
        
        Unconnected - permeability = matrix permeability, different nx,ny,nz
        and dx,dy,dz.
          
        """
        n=[2,6,4]
        mres = 1000.
        km = 1e-18
        mu=1e-3
        d = [0.1,0.2,0.05]
        # area of 1 cell in direction perpendicular to connected cells
        area = np.around([np.product([d[dd] for dd in range(3) if dd!=ddd]) for ddd in range(3)],
                          decimals=6)
        # number of cells in direction perpendicular to connected cells
        nxcells = np.array([np.product([n[nn]+1 for nn in range(3) if nn!=nnn]) for nnn in range(3)])
        
        rv = rnc.Resistivity_volume(res_type='array',
                                    fracture_diameter = 1e-3,
                                    resistivity = mres*self.res_ones_2x6x4array,
                                    permeability_matrix=km,
                                    nx=n[0],ny=n[1],nz=n[2],
                                    dx=d[0],dy=d[1],dz=d[2]
                                    )
        rv.solve_resistor_network('fluid','xyz')
        for i in range(3):
            hr = d[i]*n[i]*mu/(area[i]*nxcells[i]*km)
            self.assertAlmostEqual(rv.permeability_bulk[i],
                                   km,
                                   delta=0.0001*km)
            self.assertAlmostEqual(rv.hydraulic_resistance_bulk[i],
                                   hr,
                                   delta=1e-10*hr
                                   )



    def test_k_connected(self):
        """
        Test 1
        
        Fully connected for fluid flow, different nx,ny,nz and dx,dy,dz.
          
        """
        n=[2,6,4]
        rm = 1000.
        rf = 1.
        km = 1e-18
        mu=1e-3
        fd = 1e-3
        d = np.array([0.1,0.2,0.05])
        # area of 1 cell in direction perpendicular to connected cells
        area = np.around([np.product([d[dd] for dd in range(3) if dd!=ddd]) for ddd in range(3)],
                          decimals=6)
        # number of cells in direction perpendicular to connected cells
        nxcells = np.array([np.product([n[nn]+1 for nn in range(3) if nn!=nnn]) for nnn in range(3)])
        
        # fluid flow through each fracture tube in x,y and z directions
        qf = np.pi*fd**4/(128*mu*d*n)
        # total fluid flow in each direction:
        qtot = qf*nxcells        
        
        rv = rnc.Resistivity_volume(res_type='array',
                                    fracture_diameter = 1e-3,
                                    resistivity = rf*self.res_ones_2x6x4array,
                                    resistivity_fluid = rf,
                                    resistivity_matrix = rm,
                                    permeability_matrix=km,
                                    nx=n[0],ny=n[1],nz=n[2],
                                    dx=d[0],dy=d[1],dz=d[2]
                                    )
        rv.solve_resistor_network('fluid','xyz')
        for i in range(3):
            kb = qtot[i]*mu*d[i]*n[i]/(nxcells[i]*area[i])
            self.assertAlmostEqual(rv.permeability_bulk[i],
                                   kb,
                                   delta=0.0001*kb)
            self.assertAlmostEqual(rv.hydraulic_resistance_bulk[i],
                                   1./qtot[i],
                                   delta=0.0001/qtot[i]
                                   )


    def test_k_1colconnected(self):
        """
        Test 2
        
        1 planar connector fully connected in xz plane:
        
             |  
             |
             |
             |
             |
             |
        
        """
        n = [2,6,4]
        # matrix and fluid resistivities
        mres = 100000.
        fres = 0.1
        mu = 1e-3
        # matrix permeability
        km = 1e-18
        # cell size in x, y and z directions
        d = np.array([0.1,0.2,0.05])
        # fracture diameter
        fd = 1e-3
        # area of 1 cell in direction perpendicular to connected cells
        area = np.around([np.product([d[dd] for dd in range(3) if dd!=ddd]) for ddd in range(3)],
                          decimals=6)
                          
        nxcells = np.array([np.product([n[nn]+1 for nn in range(3) if nn!=nnn]) for nnn in range(3)])
        
        for row in range(1,n[1]):
            # resistivity matrix
            a = mres*self.res_ones_2x6x4array
            a[:,row,:,0] = fres
            a[:,row,:,2] = fres
            # restore nulls
            a = a*self.res_ones_2x6x4array      
    
    
            # hydraulic resistance of one connected cell
            rhc = (128*mu*d/(np.pi*fd**4))
    
            # hydraulic resistance of one unconnected cell
            ruc = mu*d/(area*km)
    
    
            rv = rnc.Resistivity_volume(res_type='array',
                                        fracture_diameter = fd,
                                        resistivity_matrix=mres,
                                        resistivity_fluid=fres,
                                        resistivity = a,
                                        permeability_matrix =km,
                                        dx=d[0],dy=d[1],dz=d[2],
                                        nx=n[0],ny=n[1],nz=n[2]
                                        )    
            rv.solve_resistor_network('fluid','xyz')
            # flow rate in x y and z directions, approximately (to 5 dp) equals 
            #total flow through fractures only in x and z directions
            q = [(n[2]+1.)/(rhc[0]*n[0]),
                 nxcells[1]/(ruc[1]*n[1]),
                 (n[0]+1.)/(rhc[2]*n[2])]
                 
            for i in range(3):
                # hydraulic resistance should be inverse of total current
                self.assertAlmostEqual(rv.hydraulic_resistance_bulk[i],
                                       1./q[i],
                                        delta=1e-5/q[i])
                # kval depends on direction of flow, in y direction should
                # equal matrix value, in x and z directions should equal
                # mu*L/RA
                if i == 1:
                    kval = km
                else:
                    kval = mu*d[i]*n[i]*q[i]/(nxcells[i]*area[i])
                self.assertAlmostEqual(rv.permeability_bulk[i],
                                       kval,
                                       delta=kval*1e-5)                          


    
    
    def test_k_allcolconnected(self): 
        """
        Test 3
        
        all verticals connected in x direction (repeated for y and z):
        
        | | | | |
        | | | | |
        | | | | |
        | | | | |
        | | | | |
        | | | | |
          
        """
        n=[2,6,4]
        rm = 1000.
        rf = 1.
        km = 1e-18
        mu=1e-3
        fd = 1e-3
        d = np.array([0.1,0.2,0.05])
        # area of 1 cell in direction perpendicular to connected cells
        area = np.around([np.product([d[dd] for dd in range(3) if dd!=ddd]) for ddd in range(3)],
                          decimals=6)
        # number of cells in direction perpendicular to connected cells
        nxcells = np.array([np.product([n[nn]+1 for nn in range(3) if nn!=nnn]) for nnn in range(3)])
        
        # fluid flow through each fracture tube in x,y and z directions
        qf = np.pi*fd**4./(128.*mu*d*n)
        # total fluid flow in each direction:
        qtot = qf*nxcells        
        for direction in range(3):
            a = rf*self.res_ones_2x6x4array
            for ii in [dd for dd in range(3) if dd != direction]:
                a[:,:,:,ii] = rm
            
            rv = rnc.Resistivity_volume(res_type='array',
                                        fracture_diameter = 1e-3,
                                        resistivity = a,
                                        resistivity_fluid = rf,
                                        resistivity_matrix = rm,
                                        permeability_matrix=km,
                                        nx=n[0],ny=n[1],nz=n[2],
                                        dx=d[0],dy=d[1],dz=d[2]
                                        )
            rv.solve_resistor_network('fluid','xyz')
            for i in range(3):
                if i == direction:
                    kb = qtot[i]*mu*d[i]*n[i]/(nxcells[i]*area[i])
                    self.assertAlmostEqual(rv.permeability_bulk[i],
                                           kb,
                                           delta=1e-5*kb)
                    self.assertAlmostEqual(rv.hydraulic_resistance_bulk[i],
                                           1./qtot[i],
                                           delta=1e-5/qtot[i]
                                           )
                else:
                    self.assertAlmostEqual(rv.permeability_bulk[i],
                                           km,
                                           delta=0.0001*km)
#
#
#    def test_res_1colgap(self): 
#        """
#        Test 4
#        
#        all verticals connected except 1:
#        
#        | | | | |
#        | | | | |
#
#        | | | | |
#        | | | | |
#        | | | | |
#          
#        """
#        n = [2,6,4]
#        # matrix and fluid resistivities
#        mres = 100000.
#        fres = 0.1
#        # cell size, same in x, y and z directions
#        d = np.array([0.2,0.1,0.05])
#        # fracture diameter
#        fd = 1e-3
#        # area of 1 cell in direction perpendicular to connected cells
#        area = np.around([np.product([dd for dd in d if dd!=ddd]) for ddd in d],
#                          decimals=6)
#        # number of cells in direction perpendicular to connected cells
#        nxcells = np.array([np.product([nn+1 for nn in n if nn!=nnn]) for nnn in n])
#        # fracture porosity in x y and z directions
#        phi = np.pi*(fd/2)**2/area
#        # resistivity of fault in x y and z directions
#        rf = 1./(phi/fres + (1-phi)/mres)
#
#        
#        for direction in [0,1,2]:
#            for row in range(1,n[direction]):
#                a = mres*self.res_ones_2x6x4array
#                a[:,:,:,direction] = fres
#
#                if direction == 0:
#                    a[:,:,row,0] = mres
#                elif direction == 1:
#                    a[:,row,:,1] = mres            
#                elif direction == 2:
#                    a[row,:,:,2] = mres 
#    
#                # restore nulls
#                a = a*self.res_ones_2x6x4array      
#                rv = rnc.Resistivity_volume(res_type='array',
#                                            fracture_diameter = fd,
#                                            resistivity_matrix=mres,
#                                            resistivity = a,
#                                            dx=d[0],dy=d[1],dz=d[2],
#                                            nx=n[0],ny=n[1],nz=n[2]
#                                            )           
#                rv.solve_resistor_network('current','xyz')
#        
#                for j in range(3):
#                    if j == direction:
#                        res = (rf[direction]*(n[direction]-1) + mres)/n[direction]
#                    else:
#                        res = mres*1.
#
#                    self.assertAlmostEqual(rv.resistivity_bulk[j],
#                                           res,
#                                           delta=0.01)
#                    self.assertAlmostEqual(rv.resistance_bulk[j],
#                                           res*n[j]*d[j]/(nxcells[j]*area[j]),
#                                           delta=0.01)  
    
if __name__ == '__main__':
    unittest.main(exit=False)

