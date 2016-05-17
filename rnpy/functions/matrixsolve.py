# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:45:56 2015

@author: a1655681
"""

import scipy.sparse.linalg as slinalg
import scipy.linalg as linalg
import scipy.sparse as ssparse
import rnpy.functions.matrixbuild as rnmb
import numpy as np
    
def solve_matrix(A,b):
    return slinalg.spsolve(A,b)


def solve_matrix2(Vo,C,A,D,b,ncells,cellsize,method='direct',tol = 0.1, w = 1.3, itstep=100):

    nx,ny,nz = ncells
    dx,dy,dz = cellsize

    
    if method == 'direct':
        Vn = Vo.copy()
        Vn[:,1:-1] = slinalg.spsolve(A,b).reshape(ny+1,nz-1,nx+1)
        r = 0
        
    elif method in ['jacobi','gauss','ssor']:
    
    
        c = 1
    
        L = -ssparse.tril(A,k=-1)
        U = L.T
        Di = rnmb.get_dinverse(D)        
        
        if method in ['gauss','ssor']:
            Dmd = ssparse.diags(D,0).todense()
            Ud = U.todense()
            Ld = L.todense()

        
        Vof = Vo[:,1:-1].flatten()
        Vn = Vo.copy()
        if method == 'jacobi':
            mult,cst = Di.dot(U+L), Di.dot(b)
        elif method == 'gauss':
            
            dli = linalg.inv(Dmd - Ld)
            mult,cst = dli.dot(Ud),dli.dot(b)
        elif method == 'ssor':
            ilid = linalg.solve_triangular(Dmd - w*Ld,np.identity(len(D)),lower=True)
            mult,cst = ilid.dot((1.-w)*Dmd + w*Ud), w*ilid.dot(b)
            
        print "mult, cst calculated"


        while 1:

            Vnf = np.array(mult.dot(Vof)).flatten() + cst

    
            if c == 1e5:
                print 'Reached maximum number of iterations'
                r = rnmb.residual(dx,dy,dz,Vn,C)
                break
            
            if c % itstep == 0:
                Vn[:,1:-1] = Vnf.reshape(ny+1,nz-1,nx+1)
                r = rnmb.residual(dx,dy,dz,Vn,C)                

#                print ' %.6f'%(np.amax(r))
                if np.amax(r) < tol:
                 
                    print ' Completed in %i iterations,'%c,'max residual %1e'%np.amax(r)
                    break    
    
            Vof = Vnf
            c = c + 1

        Vn[:,1:-1] = Vnf.reshape(ny+1,nz-1,nx+1)
    

    Vn = Vn.transpose(1,0,2)
    
    return Vn