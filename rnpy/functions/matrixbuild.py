# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681
"""

from __future__ import division, print_function
import numpy as np
import scipy.sparse as sparse
    

def get_nfree(n):
    nx,ny,nz = n
    return [nx*(ny+1)*(nz+1),ny*(nx+1)*(nz+1),(nx+1)*(ny+1)*(nz+2)]
    
def get_nnodes(n):
    nx,ny,nz = n
    return (nx+1)*(ny+1)*(nz+1)

def get_ncells(n):
    nx,ny,nz = n
    ncxz = nx*(ny+1)*nz # number of cells in the xz plane
    ncyz = (nx+1)*ny*nz # number of cells in the yz plane
    return [ncxz,ncyz] # number of cells
    


def build_sums(nfree,n):
    """
    
    builds the matrix b to solve the matrix equation Ab = C
    where A is the matrix defined in build_matrix
    and C is the electrical current values.
    
    nfree = length of C (equal to length of each axis in A)
    n = list containing number of nodes in x and z direction [nx,nz]
    
    """
    
    b_dense = np.zeros(nfree)
    # apply a unit voltage
    b_dense[-(2*n[0] + 1):] = 1.
    
    return sparse.csr_matrix(b_dense)
    
def buildmatrix3d_kirchhoff(n):
    """
    calculate numbers to populate matrix and their row and column, relating
    to kirchhoff's law for electrical current and equivalent for fluid flow
    (i.e., sum of total current in and out of each node is zero)
    
    ==============================inputs=======================================
    n = number of cells in the x (horizontal), y (into the plane)
        and z (vertical) directions [nx,ny,nz]
    ===========================================================================
    """
    nx,ny,nz = n
    nfx,nfy,nfz = get_nfree(n)
    nn = get_nnodes(n)
    
    #   a. x connectors
    data1a = np.hstack([-np.ones(nfx),np.ones(nfx)])
    rows1as = np.hstack([np.arange(nx)]*(ny+1)*(nz+1)) \
            + np.hstack([np.ones(nx)*(nx+1)*i for i in range((ny+1)*(nz+1))])
    rows1a = np.hstack([rows1as,rows1as + 1])
    cols1a = np.hstack([np.arange(nfx)]*2)
    
    #   b. y connectors
    data1b = np.hstack([-np.ones(nfy),np.ones(nfy)])
    rows1bs = np.hstack([np.arange(ny*(nx+1))]*(nz+1)) \
            + np.hstack([np.ones(ny*(nx+1))*(nx+1)*(ny+1)*i for i in range(nz+1)])
    rows1b = np.hstack([rows1bs,rows1bs + nx + 1])
    cols1b = np.hstack([np.arange(nfy)]*2)+nfx
    
    #   c. z connectors
    data1c = np.hstack([np.ones(nn),-np.ones(nn)])
    cols1cs = np.arange(nn) + nfx + nfy
    cols1c = np.hstack([cols1cs,cols1cs + (nx+1)*(ny+1)])
    rows1c = np.hstack([np.arange(nn)]*2)    
    
    return np.hstack([data1a,data1b,data1c]),np.hstack([rows1a,rows1b,rows1c]),\
           np.hstack([cols1a,cols1b,cols1c])
      
      
def buildmatrix3d_potential(resistance):
    """
    calculate numbers to populate matrix and their row and column, relating
    to conservation of potential and equivalent for fluid flow
    (i.e., potential is conservative in each elementary cell)
    
    ==============================inputs=======================================
    resistivity = array containing resistivities in the x,y,z directions
     
    ===========================================================================
    """

    nz,ny,nx = [int(i-2) for i in np.shape(resistance)[:3]]
    n = [nx,ny,nz]
    nfx,nfy,nfz = get_nfree(n)
    nn = get_nnodes(n)
    ncxz,ncyz = get_ncells(n)
    nc = ncxz + ncyz # number of cells

    resx = resistance[1:,1:,1:-1,0]
    resy = resistance[1:,1:-1,1:,1]
    resz = resistance[1:-1,1:,1:,2]    

    #    a. x connectors
    data2a = np.hstack([np.ones(ncxz)*resx.flatten()[:ncxz], 
                        np.ones(ncxz)*(-resx.flatten()[-ncxz:])])
    rows2a = np.hstack([np.arange(ncxz)+nn]*2)
    cols2a = np.hstack([np.arange(ncxz), np.arange(ncxz) + nx*(ny+1)])
    
    #    b. y connectors
    data2b = np.hstack([np.ones(ncyz)*resy.flatten()[:ncyz], 
                        np.ones(ncyz)*(-resy.flatten()[-ncyz:])])
    rows2b = np.hstack([np.arange(ncyz) + nn + ncxz]*2)
    cols2b = np.hstack([np.arange(ncyz) + nx*(ny+1)*(nz+1),
                        np.arange(ncyz) + nx*(ny+1)*(nz+1) + ny*(nx+1)])
    
    #    c. z connectors
    data2c = np.hstack([np.ones(nc)*np.hstack([-resz[:,:,:-1].flatten(),-resz[:,:-1,:].flatten()]),
                        np.ones(nc)*np.hstack([resz[:,:,1:].flatten(),resz[:,1:,:].flatten()])])
                        
    rows2c = np.hstack([np.arange(nc) + nn]*2)#nfx + nfy + (nx+1)*(ny+1)]*2)
    cols2c1 = np.hstack([np.arange(nx)]*(ny+1)*nz) \
            + np.hstack([np.ones(nx)*(nx+1)*i for i in range((ny+1)*nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
    cols2c2 = np.hstack([np.arange((nx+1)*ny)]*nz) \
            + np.hstack([np.ones((nx+1)*ny)*(nx+1)*(ny+1)*i for i in range(nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
    cols2c = np.hstack([cols2c1,cols2c2,cols2c1+1,cols2c2+nx+1])
    
    return np.hstack([data2a,data2b,data2c]),np.hstack([rows2a,rows2b,rows2c]),\
           np.hstack([cols2a,cols2b,cols2c])
    
    
def buildmatrix3d_normalisation(resistance):
    """
    calculate numbers to populate matrix and their row and column, relating
    to normalisation across the network (i.e., total voltage drop across
    entry and exit nodes), also add one row that forces currents flowing
    into the network to equal currents exiting the network
    
    ==============================inputs=======================================
    resistivity = array containing resistivities in the x,y,z directions as for
    buildmatrix3d_potential
    ===========================================================================
    """
    
    nz,ny,nx = [int(i-2) for i in np.shape(resistance)[:3]]
    n = [nx,ny,nz]
    nfx,nfy,nfz = get_nfree(n)
    nfree = sum([nfx,nfy,nfz])
    nn = get_nnodes(n)
  
    resx = resistance[1:,1:,1:-1,0]
    resy = resistance[1:,1:-1,1:,1]
    resz = resistance[1:-1,1:,1:,2]    
    
    ncxz,ncyz = get_ncells([nx,ny,nz])
    nc = ncxz + ncyz # number of cells
 
    #    a. x connectors
    data3a = np.ones(nx*(ny+1))*resx[-1].flatten()
    rows3a = np.arange(nx*(ny+1)) + nn + nc + (nx+1)*(ny+1)
    cols3a = np.arange(nx*(ny+1)) + ncxz
    
    #    b. y connectors
    data3b = np.ones((nx+1)*ny)*resy[-1].flatten()
    rows3b = np.arange((nx+1)*ny) + nn + nc + (2*nx+1)*(ny+1)
    cols3b = np.arange((nx+1)*ny) + nfx + (nx+1)*ny*nz
    
    #    c. z connectors
    data3c1 = np.ones((nx+1)*(ny+1)*nz)*resz.flatten()
    rows3c1 = np.hstack([np.arange((nx+1)*(ny+1))]*nz) + nn + nc
    cols3c1 = np.hstack([np.arange((nx+1)*(ny+1))]*nz) \
            + np.hstack([np.ones((nx+1)*(ny+1))*(nx+1)*(ny+1)*i for i in range(nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
    
    data3c2 = np.ones(ncxz)*resz[:,:,:-1].flatten()
    cols3c2 = np.hstack([np.arange(nx)]*(ny+1)*nz) \
            + np.hstack([np.ones(nx)*(nx+1)*i for i in range(ny+1)*nz]) \
            + np.hstack([np.ones(nx*(ny+1))*(nx+1)*(ny+1)*i for i in range(nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
            
    rows3c2 = np.hstack([np.arange(nx*(ny+1))]*nz) \
            + nn + nc + (nx+1)*(ny+1)
    
    data3c3 = np.ones((nx+1)*ny*nz)*resz[:,:-1,:].flatten()
    rows3c3 = np.hstack([np.arange((nx+1)*ny)]*nz) \
            + nn + nc + (2*nx+1)*(ny+1)     
    cols3c3 = np.hstack([np.arange((nx+1)*ny)]*nz) \
            + np.hstack([np.ones((nx+1)*ny)*(nx+1)*(ny+1)*i for i in range(nz)]) \
            + nfx + nfy + (nx+1)*(ny+1)
    
    data3c = np.hstack([data3c1,data3c2,data3c3])
    rows3c = np.hstack([rows3c1,rows3c2,rows3c3])
    cols3c = np.hstack([cols3c1,cols3c2,cols3c3]) 
 
    # 4. Current in = current out
    data4 = np.hstack([np.ones((nx+1)*(ny+1)),-np.ones((nx+1)*(ny+1))])
    rows4 = np.ones((nx+1)*(ny+1)*2)*(nfree)
    cols4 = np.hstack([np.arange((nx+1)*(ny+1)) + nfx + nfy,
                       np.arange((nx+1)*(ny+1)) + nfx + nfy + (nx+1)*(ny+1)*(nz+1)])
                       
    return np.hstack([data3a,data3b,data3c,data4]),\
           np.hstack([rows3a,rows3b,rows3c,rows4]),\
           np.hstack([cols3a,cols3b,cols3c,cols4])  

def build_matrix3d(resistance):
    """
    """
    nx,ny,nz = np.array(np.shape(resistance)[:-1][::-1])-2
    n = [nx,ny,nz]
    nn = get_nnodes(n)
    nc = sum(get_ncells(n))
    nfree = sum(get_nfree(n))
    data1,rows1,cols1 = buildmatrix3d_kirchhoff([nx,ny,nz])
    data2,rows2,cols2 = buildmatrix3d_potential(resistance)
    data3,rows3,cols3 = buildmatrix3d_normalisation(resistance)
    
    data,rows,cols = np.hstack([data1,data2,data3]),\
                     np.hstack([rows1,rows2,rows3]),\
                     np.hstack([cols1,cols2,cols3])
    m = sparse.coo_matrix((data,(rows,cols)), shape=(nfree+1,nfree))
    # take out central row to make the matrix square
    mc = sparse.bmat([[m.tocsr()[:int(nn/2)]],[m.tocsr()[int(nn/2)+1:]]]).tocsr()
    b = np.zeros(nfree)
    b[nn+nc-1:-1] = 1.
    
    return mc,b
