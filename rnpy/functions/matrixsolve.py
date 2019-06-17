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




def solve_matrix2(R,cellsize,Vsurf=0.,Vbase=1.,Vstart=None,method='direct',
                  tol = 0.1, w = 1.3, itstep=100, return_termination = False):

    dx,dy,dz = cellsize
    C = rnmb.Conductivity(R)
    nz,ny,nx = np.array(R.shape[:-1])-2

    # initialise a default starting array if needed
    if Vstart is None:
        Vo = np.zeros((nx+1,ny+1,nz+1))
        Vo[:,:,:] = np.linspace(Vsurf,Vbase,nz+1)
        # transpose so that array is ordered by y, then z, then x
        Vo = Vo.transpose(1,2,0)
    else:
        Vo = Vstart.copy()

    A,D = rnmb.buildmatrix(C,dx,dy,dz)
    b = rnmb.buildb(C,dz,Vsurf,Vbase)

    
    if method == 'direct':
        Vn = Vo.copy()
        Vn[:,1:-1] = slinalg.spsolve(A,b).reshape(ny+1,nz-1,nx+1)
        r = 0
        
    elif method == 'bicg':
        Vn = Vo.copy()
        Vn[:,1:-1] = slinalg.bicg(A,b)[0].reshape(ny+1,nz-1,nx+1)
        
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
        
        r = np.abs(rnmb.residual(dx,dy,dz,Vn,C))#/Vn[1:-1,1:-1,1:-1]
        #vsum = ((Vn[1:]-Vn[:-1])*C.u[1:]).sum()
        while 1:

            Vnf = np.array(mult.dot(Vof)).flatten() + cst

    
            if c == 1e6:
                print(('Reached maximum number of iterations','mean residual %1e'%np.mean(r),'median residual %1e'%np.median(r)))
                r = rnmb.residual(dx,dy,dz,Vn,C)
                termination = 0
                break
            
            if c % itstep == 0:
                Vn[:,1:-1] = Vnf.reshape(ny+1,nz-1,nx+1)
                rnew = np.abs(rnmb.residual(dx,dy,dz,Vn,C))#/Vn[1:-1,1:-1,1:-1]
                #rnew[np.isinf(rnew)] = 0.

    
                #vsumnew = ((Vn[1:]-Vn[:-1])*C.u[1:]).sum()
                #dvchange = (max(vsumnew/vsum,vsum/vsumnew) - 1.)/itstep
#                print "sum of last row",vsumnew,'% change',dvchange*100,"residual",np.mean(rnew)
                print(("residual",np.mean(rnew)))
                #vsum = vsumnew
#                if ((np.nanmean(r) < tol) or (rchange < tol)):
                if np.mean(rnew) < tol:# or (dvchange < tol)):
                    print((' Completed in %i iterations,'%c,'mean residual %1e'%np.mean(rnew),'median residual %1e'%np.median(rnew)), end=' ')

                    if np.mean(rnew) < tol:
                        print("reached tol")
                    else:
                        print("change less than threshold")
                    termination = 1
                    break
                r = rnew

            Vof = Vnf
            c = c + 1

        Vn[:,1:-1] = Vnf.reshape(ny+1,nz-1,nx+1)
    

    Vn = Vn.transpose(1,0,2)
    
    if return_termination:
        return Vn,termination
    else:
        return Vn


def solve_matrix3(R,cellsize,Vsurf=0.,Vbase=1.,Vstart=None,method='direct',
                  tol = 0.1, w = 1.3, itstep=100, return_termination = False):
    
    dx,dy,dz = cellsize
    C = rnmb.Conductivity(R)
    nz,ny,nx = np.array(R.shape[:-1])-2

    # initialise a default starting array if needed
    if Vstart is None:
        Vo = np.zeros((nx+1,ny+1,nz+1))
        Vo[:,:,:] = np.linspace(Vsurf,Vbase,nz+1)
#        Vo[:,:,0] = 1.
        # transpose so that array is ordered by y, then z, then x
        Vo = Vo.transpose(1,2,0)
    else:
        Vo = Vstart.copy()    
    
    tol = 1e-4
    
    c = 1
    while 1:
        Vn = Vsolve3d(nx,dx,ny,dy,nz,dz,Vo,C,w)
#        Vn = Vsolve3d2(Vo,C,w,dx,dy,dz)
        if c == 1e5:
            print(('Reached maximum number of iterations',))
            break
        if c % itstep == 0:
    #        change = np.amax(np.abs(V`n - Vo))
    #        print change
            r = np.abs(rnmb.residual(dx,dy,dz,Vn,C))
            print(' %.6f'%(np.amax(r)))
            if np.amax(r) < tol:
    #            r = residual(dx,dz,Vn)
                
                print((' Completed in %i iterations'%c,))
                break    

        Vo = Vn
        c = c + 1    
        
    return Vn.transpose(1,0,2)
   
def Vsolve(nx,dx,nz,dz,Vo,C,w):
    
    V = Vo.copy()
    for i in range(1,nx-1):
        for j in range(1,nz-1):
            V[j,i] = (1-w)*V[j,i] + \
            w*((C.r[j,i]*V[j,i+1] + C.l[j,i]*V[j,i-1])*dz**2. \
            + (C.d[j,i]*V[j+1,i] + C.u[j,i]*V[j-1,i])*dx**2.) \
            / ((C.r[j,i] + C.l[j,i])*dz**2. \
            + (C.u[j,i] + C.d[j,i])*dx**2.)
    
    V[:,0] = V[:,1]
    V[:,-1] = V[:,-2]
            
    return V
    
def Vsolve3d(nx,dx,ny,dy,nz,dz,Vo,C,w):
    V = Vo.copy()
    for i in range(1,nx):
        for j in range(1,nz):
            for k in range(1,ny):
                V[k,j,i] = (1.-w)*V[k,j,i] +\
               w*((C.r[k,j,i]*V[k,j,i+1] + C.l[k,j,i]*V[k,j,i-1])*(dz**2.)*(dy**2.) \
                + (C.d[k,j,i]*V[k,j+1,i] + C.u[k,j,i]*V[k,j-1,i])*(dx**2.)*(dy**2.) \
                + (C.i[k,j,i]*V[k+1,j,i] + C.o[k,j,i]*V[k-1,j,i])*(dx**2.)*(dz**2.)) \
                /((C.r[k,j,i] + C.l[k,j,i])*dz**2.*dy**2.\
                + (C.u[k,j,i] + C.d[k,j,i])*dx**2.*dy**2.\
                + (C.o[k,j,i] + C.i[k,j,i])*dx**2.*dz**2.)
    V[0] = V[1]
    V[-1] = V[-2]
    V[:,:,0] = V[:,:,1]
    V[:,:,-1] = V[:,:,-2]
    
    return V

def Vsolve3d2(Vo,C,dx,dy,dz):
    
    V = Vo.copy()
    V[1:-1,1:-1,1:-1] = \
     ((C.r[1:-1,1:-1,1:-1]*V[1:-1,1:-1,2:] + C.l[1:-1,1:-1,1:-1]*V[1:-1,1:-1,:-2])*(dz**2.)*(dy**2.) \
    + (C.d[1:-1,1:-1,1:-1]*V[1:-1,2:,1:-1] + C.u[1:-1,1:-1,1:-1]*V[1:-1,:-2,1:-1])*(dx**2.)*(dy**2.) \
    + (C.i[1:-1,1:-1,1:-1]*V[2:,1:-1,1:-1] + C.o[1:-1,1:-1,1:-1]*V[:-2,1:-1,1:-1])*(dx**2.)*(dz**2.)) \
    /((C.r[1:-1,1:-1,1:-1] + C.l[1:-1,1:-1,1:-1])*dz**2.*dy**2.\
    + (C.u[1:-1,1:-1,1:-1] + C.d[1:-1,1:-1,1:-1])*dx**2.*dy**2.\
    + (C.o[1:-1,1:-1,1:-1] + C.i[1:-1,1:-1,1:-1])*dx**2.*dz**2.)    
    
    V[0] = V[1]
    V[-1] = V[-2]
    V[:,:,0] = V[:,:,1]
    V[:,:,-1] = V[:,:,-2]    
    
    return V