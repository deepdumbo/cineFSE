'''@package docstring
@author: Jyh-Miin Lin  (Jimmy), Cambridge University
@address: jyhmiinlin@gmail.com
Initial created date: 2013/1/21
================================================================================
    This file is part of pynufft.

    pynufft is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pynufft is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with pynufft.  If not, see <http://www.gnu.org/licenses/>.
================================================================================

    Remark

    pynufft is the fast program aims to do constraint-inversion
    of irregularly sampled data.

    Among them, nufft.py was translated from NUFFT in MATLAB of
    Jeffrey A Fessler et al, University of Michigan
    which was a BSD-licensed work.

    However, there are several important modifications. In
    particular, the scaling factor adjoint NUFFT,
    as only the Kaiser-Bessel window is realized.

    Please cite J A Fessler, Bradley P Sutton.
    Nonuniform fast Fourier transforms using min-max interpolation.
    IEEE Trans. Sig. Proc., 51(2):560-74, Feb. 2003.  

    3. Other relevant works:
     NFFT is a c-library with gaussian interpolator 
     (http://www-user.tu-chemnitz.de/~potts/nfft/)
    *fortran version: http://www.cims.nyu.edu/cmcl/nufft/nufft.html
    at alpha/beta stage
    *MEX-version 
    http://www.mathworks.com/matlabcentral/fileexchange/
         25135-nufft-nufft-usffft
'''

import numpy
import scipy.sparse
from scipy.sparse.csgraph import _validation  # for cx_freeze debug
# import sys
# import scipy.fftpack
import numpy.fft as fftpack

# try:
#     import pyfftw
# except:
#     pass
# try:
#     from numba import jit
# except:
#     pass

# def mydot(A,B):
#     return numpy.dot(A,B)

# def mysinc(A):
#     return numpy.sinc(A)
#     print('no pyfftw, use slow fft')
dtype = numpy.complex64
# try:
#     from numba import autojit  
# except:
#     pass
#     print('numba not supported')

# @jit
########################
import scipy.signal
# myN = 64*32+1
# bpass = scipy.signal.remez(myN, [0, 8-2, 8+2, 256], [ 1, 0 ],[1, 10], Hz=512,
# maxiter=25)
# # bpass = scipy.signal.remez(myN, [0, 0.2 ,3.0, 256], [1,0 ], Hz=512,
# # maxiter=25)
# # 
# # freq, response = scipy.signal.freqz(bpass)
# # ampl = numpy.abs(response)
#  
# 
# import nitime.algorithms.spectral
#     n_bases = 5
#     v,e = nitime.algorithms.spectral.dpss_windows(256,1, n_bases )
#     for jj in xrange(0,n_bases):
#         v[jj,:]=v[jj,:]*e[jj]
# t = numpy.arange(0, 8000)
# t = (t - 3999)/80.0
# # print('t',t)
# bpass = scipy.signal.get_window(('chebwin', 6000.0), 8001)
# bpass = bpass[1:]
# bpass = bpass/numpy.max(bpass)
# import scipy.interpolate
# dirichlet = scipy.interpolate.interp1d(t, bpass)
#########################
# def correct_phase_of_gridding(V,   Kd   ):
#     prod_Kd= numpy.prod(Kd)
#     
#     
#     import nitime.algorithms.spectral
#     import scipy.signal
#     n_bases = 5
#     N = Kd[0]
# #     v,e = nitime.algorithms.spectral.dpss_windows( N , 0.15, n_bases )
#      
# #     for jj in xrange(0,n_bases):
# #         v[jj,:]=v[jj,:]*e[jj]
#     import scipy.interpolate
# #     t =  numpy.arange(0, 8000)
# #     t = (t- 3999 ) /80.0
#     v = scipy.signal.kaiser( N, 0.7)
# 
# #     v = scipy.interpolate.interp1d(t, v)
# #     print(numpy.shape(v))
#     import matplotlib.pyplot
#     
#     matplotlib.pyplot.plot(v)
#     matplotlib.pyplot.show()
#     dpss_x =  v#numpy.sum(v,0)
#     dpss_x = numpy.tile(numpy.reshape(dpss_x,   (Kd[0] , 1),order='F'), (1, Kd[1] ))
#     dpss_y =  v#numpy.sum(v,0)
#     dpss_y = numpy.tile(numpy.reshape(dpss_y,   (1 , Kd[1]),order='F'), (Kd[0],1 ))
#    
#     dpss = dpss_x.flatten() * dpss_y.flatten()
#     matplotlib.pyplot.plot(dpss)
#     matplotlib.pyplot.show()
# #     dpss = numpy.reshape( dpss, (prod_Kd,1) )
#     print('dpss.shape',dpss.shape)
#     V2 = scipy.sparse.diags( dpss ,0)
#     V = V.dot( V2)  
#     
# #     from scipy.sparse.linalg import lsqr
# # #     import scipy.sparse
# #     b =numpy.ones((M,1),dtype = numpy.complex64)
# #     
# # #     kk = V.dot(b)
# # #     import matplotlib.pyplot
# # #     matplotlib.pyplot.plot(numpy.abs(kk))
# # #     matplotlib.pyplot.show()
# # #     rat0 = numpy.max(numpy.abs(V.dot(b)))
# # #     b0 = numpy.ones((prod_Kd,1),dtype = numpy.complex64)
# # #     
# # #     x3 = V.conj().T.dot(V.dot(b0))
# # #     tmp_phase = x3
# # #     for ppp in xrange(0,2):
# #     x0 = lsqr(V, b , iter_lim=10)
# #     
# #     print( (x0[0]))
# #     tmp_phase = x0[0]
# # 
# # #     print(zz.shape,tmp_phase.shape)
# # 
# # #     tmp_phase = (tmp_phase +1e-7)/ (numpy.abs(tmp_phase)+1e-7)
# # 
# # #     tmp_phase = tmp_phase * tmp_phase[0]
# # #     rat =numpy.max(numpy.abs(tmp_phase))
# # #     print('rat',rat)
# #     V2 = scipy.sparse.diags(tmp_phase    ,0)
# #     V= V.dot( V2)
# # #     V = scipy.sparse.diags(zz,0).dot(V)
# # #     V2= V.tolil()
# # #     for pp in xrange(0,prod_Kd):
# # #         V2[:,pp] = V2[:,pp]*x0[0][pp]
#     
#     return V
import scipy.linalg

try: 
    xrange # python3 compatibility
except NameError: 
    xrange = range # python2 

# from numpy import arange as xrange

def custom_schur(A):
    [Rk, Qk] = scipy.linalg.schur(A)

    P = numpy.round(Qk) # find the permutation matrix if A is very close to diagonal
    Q = Qk.dot(P.T)  # back-permuate the orthogonal transform Qk 
    R = P.dot(Rk.dot(P.T.conj())) 
    
    return (Q,R)

# def trim_mat(V, M, Jd, Kd):
#     import scipy.sparse
#     print('M,Jd,Kd',M,Jd,Kd)
# # # 
#     last_indx= V.shape[1]
#     print('last_indx',last_indx)
# #     block_size = 1200
# 
# #     import scipy.linalg
#     V=V.tolil()
#     square_size = 10
# #     start_idx = 0 #Kd[0]*Kd[1] -  Kd[0]*Jd[1] 
#     for start_idx in xrange( Kd[0]*(Kd[1]/2 -square_size/2) - square_size/2 , Kd[0]*(Kd[1]/2 -square_size/2) - square_size/2 + 1):#Kd[0]*Kd[1] -  Kd[0]*Jd[1] ):
#         print('start_idx',start_idx)
#         indices =[]
#         for ppp in xrange(0, square_size):
#             print(ppp)
# #             tmp_list = numpy.arange(start_idx + Kd[1]*ppp ,start_idx+ Jd[0]+ Kd[1]*ppp )
#             tmp_list = numpy.arange(start_idx + Kd[1]*ppp ,start_idx+  square_size + Kd[1]*ppp )
#             indices = numpy.concatenate( (indices, tmp_list ))
#             
# #             print('indices=',indices)
#         V_crop = V[:,indices]
#     #     for pp in xrange(0,10):
#     #         print('pp = ',pp)
#     #         start_idx = pp
#         B=V_crop.getH().dot(V_crop)#V[:,start_idx:start_idx+block_size].getH().dot(V[:,start_idx:start_idx+block_size]).todense()
#         tmp_ddd = B.todense()
#         [Q,R] = custom_schur(tmp_ddd)
# #         tmp_ddd = R
#     # #      
#         V[:,indices] = V[:,indices].dot(Q)
#     V= V.tocsc()
#     V_crop = V[:,indices]
#     tmp_ddd = V_crop.getH().dot(V_crop)
# #     print(B[0,0], B[2,0], B[0,2], B[2,2])
#     import matplotlib.pyplot
# # #     matplotlib.pyplot.imshow(numpy.abs( - tmp_ddd),cmap = matplotlib.cm.gray)
# # #     matplotlib.pyplot.show()
# #     matplotlib.pyplot.imshow(numpy.imag(tmp_ddd),cmap = matplotlib.cm.gray)
# #     matplotlib.pyplot.show()    
# #     matplotlib.pyplot.imshow(numpy.real(tmp_ddd),cmap = matplotlib.cm.gray)
# #     matplotlib.pyplot.show()   
# #     x = tmp_ddd[:,0:3].A
#     
# #     print(x)
# #     matplotlib.pyplot.plot( x  )
# #     matplotlib.pyplot.show()       
#     return V 


# def refine_k_density(V,W ,Kd):
#     from scipy.sparse.linalg import lsqr, lsmr
#     V1=V.getH()
    
#     b = numpy.ones( (numpy.prod(Kd),1),dtype  = numpy.complex64)
#     x0 = lsmr(V.getH(), b , iter_lim=50, damp=0.1)
#     x0=numpy.reshape(x0[0],(numpy.size(x0[0]),1),order='F')
#     D = lsmr(V, W*x0 , iter_lim=50, damp=0.1)
#     V = V.dot( scipy.sparse.diags(D[0]    ,0))*2.0
     
#     return V

def pipe_density(V): 
    '''
    The original iterative density compensation method by J. Pipe 1999 .
    The iterative implementation by using Python is slow. 
    So I used a lsqr iterative solution which is faster than iterations in Python. 
    '''
    
    V1=V.getH()
#     E = V.dot( V1.dot(    W   )   )
#     W = W*(E+1.0e-17)/(E*E+1.0e-17)    
    b = numpy.ones( (V.get_shape()[0] ,1) ,dtype  = numpy.complex64)  
    from scipy.sparse.linalg import lsqr, lsmr
        
#     x1 =  lsqr(V, b , iter_lim=10, calc_var = True, damp = 0.01)
    x1 =  lsmr(V, b , maxiter=12,  damp = 0.1)
    
    my_k_dens = x1[0]    # the first element is the answer
    
#     tmp_W =  lsqr(V1, my_k_dens, iter_lim=10, calc_var = True, damp = 0.01)
    tmp_W =  lsmr(V1, my_k_dens , maxiter=12,  damp = 0.1)
    
    W = numpy.reshape( tmp_W[0], (V.get_shape()[0] ,1),order='F' ) # reshape vector

#     for pppj in xrange(0,10):
# #             W[W>1.0]=1.0
# #             print(pppj)
#         E = V.dot( V1.dot(    W   )   )
#  
#         W = W*(E+1.0e-17)/(E**2+1.0e-17)
 
    return W
def checker(input_var,desire_size):
    '''
    check if debug = 1
    '''

    if input_var is None:
        print('input_variable does not exist!')
      
    if desire_size is None:
        print('desire_size does not exist!')  
             
    dd=numpy.size(desire_size)
    dims = numpy.shape(input_var)
#     print('dd=',dd,'dims=',dims)
    if numpy.isnan(numpy.sum(input_var[:])):
        print('input has NaN')
      
    if numpy.ndim(input_var) < dd:
        print('input signal has too few dimensions')
      
    if dd > 1:
        if dims[0:dd] != desire_size[0:dd]:
            print(dims[0:dd])
            print(desire_size)
            print('input signal has wrong size1')
    elif dd == 1:
        if dims[0] != desire_size:
            print(dims[0])
            print(desire_size)
            print('input signal has wrong size2')
       
    if numpy.mod(numpy.prod(dims),numpy.prod(desire_size)) != 0:
        print('input signal shape is not multiples of desired size!')
        
# import scipy.signal
# myN = 64*8+1
# bpass = scipy.signal.remez(myN, [0, 8-2,  8 + 2, 256], [ 1, 0 ], Hz=512,
# maxiter=25)
# # 
# # freq, response = scipy.signal.freqz(bpass)
# # ampl = numpy.abs(response)
# 
# t= numpy.arange(0,64*8+1)
# t = (t- (myN - 1)/2.0  ) /32.0
# 
# 
# bpass = bpass/numpy.max(bpass)
# import scipy.interpolate
# dirichlet = scipy.interpolate.interp1d(t, bpass)


#         
# def dirichlet(x):
#     return numpy.sinc(x )*(0.5+0.5*numpy.cos(x/3.0 )) #+ 1.0j*x/3.0 # approximate the lancos interpolator
#     return numpy.sinc(x)
#     x = x+1e-17
#     return -0.5j*(numpy.exp(1.0j*x) - numpy.exp(-1.0j*x))/x

def dirichlet2(x):
    return numpy.sin(numpy.pi *x+1e-20)/ (numpy.pi * x +1e-20)

def dirichlet(x):
    return numpy.sinc(x)

def outer_sum(xx,yy):
    nx=numpy.size(xx)
    ny=numpy.size(yy)
    
    arg1 = numpy.tile(xx,(ny,1)).T
    arg2 = numpy.tile(yy,(nx,1))
    #cc = arg1 + arg2
    
    return arg1 + arg2

def nufft_offset(om, J, K):
    '''
    For every om points(outside regular grids), find the nearest
    central grid (from Kd dimension)  
    '''
    gam = 2.0*numpy.pi/(K*1.0);
    k0 = numpy.floor(1.0*om / gam - 1.0*J/2.0) # new way
    return k0

def nufft_alpha_kb_fit(N, J, K):
    '''
    find out parameters alpha and beta
    of scaling factor st['sn']
    Note, when J = 1 , alpha is hardwired as [1,0,0...] 
    (uniform scaling factor)
    '''
    beta=1
    #chat=0
    Nmid=(N-1.0)/2.0
    
    if N > 40:
        #empirical L
        L= 13
    else:
        L=numpy.ceil(N/3)
        
    nlist = numpy.arange(0,N)*1.0-Nmid
#    print(nlist)
    (kb_a,kb_m)=kaiser_bessel('string', J, 'best', 0, K/N)
#    print(kb_a,kb_m)
    if J > 1:
        sn_kaiser = 1 / kaiser_bessel_ft(nlist/K, J, kb_a, kb_m, 1.0)
    elif J ==1:  # cases on grids
        sn_kaiser = numpy.ones((1,N),dtype=dtype)
#            print(sn_kaiser)
    gam = 2*numpy.pi/K;
    X_ant =beta*gam*nlist.reshape((N,1),order='F')
    X_post= numpy.arange(0,L+1)
    X_post=X_post.reshape((1,L+1),order='F') 
    X=numpy.dot(X_ant, X_post) # [N,L]
    X=numpy.cos(X)
    sn_kaiser=sn_kaiser.reshape((N,1),order='F').conj()
#    print(numpy.shape(X),numpy.shape(sn_kaiser))
#   print(X)
    #sn_kaiser=sn_kaiser.reshape(N,1)
    X=numpy.array(X,dtype=dtype)
    sn_kaiser=numpy.array(sn_kaiser,dtype=dtype)
    coef = numpy.linalg.lstsq(X,sn_kaiser)[0] #(X \ sn_kaiser.H);
#            print('coef',coef)
    #alphas=[]
    alphas=coef
    if J > 1:
        alphas[0]=alphas[0]
        alphas[1:]=alphas[1:]/2.0      
    elif J ==1: # cases on grids
        alphas[0]=1.0
        alphas[1:]=0.0                  
    alphas=numpy.real(alphas)
    return (alphas, beta)

def kaiser_bessel(x, J, alpha, kb_m, K_N):
    if K_N != 2 : 
        kb_m = 0
        alpha = 2.34 * J
    else:
        kb_m = 0    # hardwritten in Fessler's code, because it was claimed as the best!
        jlist_bestzn={2: 2.5, 
                        3: 2.27,
                        4: 2.31,
                        5: 2.34,
                        6: 2.32,
                        7: 2.32,
                        8: 2.35,
                        9: 2.34,
                        10: 2.34,
                        11: 2.35,
                        12: 2.34,
                        13: 2.35,
                        14: 2.35,
                        15: 2.35,
                        16: 2.33 }
        if J in jlist_bestzn:
#            print('demo key',jlist_bestzn[J])
            alpha = J*jlist_bestzn[J]
            #for jj in tmp_key:
            #tmp_key=abs(tmp_key-J*numpy.ones(len(tmp_key)))
#            print('alpha',alpha)
        else:
            #sml_idx=numpy.argmin(J-numpy.arange(2,17))
            tmp_key=(jlist_bestzn.keys())
            min_ind=numpy.argmin(abs(tmp_key-J*numpy.ones(len(tmp_key))))
            p_J=tmp_key[min_ind]
            alpha = J * jlist_bestzn[p_J]
            print('well, this is not the best though',alpha)
    kb_a=alpha
    return (kb_a, kb_m)

def kaiser_bessel_ft(u, J, alpha, kb_m, d):
    '''
    interpolation weight for given J/alpha/kb-m 
    '''
#     import types
    
    # scipy.special.jv (besselj in matlab) only accept complex
#     if u is not types.ComplexType:
#         u=numpy.array(u,dtype=numpy.complex64)
    u = u*(1.0+0.0j)
    import scipy.special
    z = numpy.sqrt( (2*numpy.pi*(J/2)*u)**2.0 - alpha**2.0 );
    nu = d/2 + kb_m;
    y = ((2*numpy.pi)**(d/2))* ((J/2)**d) * (alpha**kb_m) / scipy.special.iv(kb_m, alpha) * scipy.special.jv(nu, z) / (z**nu)
    y = numpy.real(y);
    return y

def nufft_scale1(N, K, alpha, beta, Nmid):
    '''
    calculate image space scaling factor
    '''
#     import types
#     if alpha is types.ComplexType:
    alpha=numpy.real(alpha)
#         print('complex alpha may not work, but I just let it as')
        
    L = len(alpha) - 1
    if L > 0:
        sn = numpy.zeros((N,1))
        n = numpy.arange(0,N).reshape((N,1),order='F')
        i_gam_n_n0 = 1j * (2*numpy.pi/K)*( n- Nmid)* beta
        for l1 in xrange(-L,L+1):
            alf = alpha[abs(l1)];
            if l1 < 0:
                alf = numpy.conj(alf)
            sn = sn + alf*numpy.exp(i_gam_n_n0 * l1)
    else:
        sn = numpy.dot(alpha , numpy.ones((N,1),dtype=numpy.float32))
    return sn

def nufft_scale(Nd, Kd, alpha, beta):
    dd=numpy.size(Nd)
    Nmid = (Nd-1)/2.0
    if dd == 1:
        sn = nufft_scale1(Nd, Kd, alpha, beta, Nmid);
#    else:
#        sn = 1
#        for dimid in numpy.arange(0,dd):
#            tmp =  nufft_scale1(Nd[dimid], Kd[dimid], alpha[dimid], beta[dimid], Nmid[dimid])
#            sn = numpy.dot(list(sn), tmp.H)
    return sn
def mat_inv(A):
#     import scipy.linalg
#     q,r  = scipy.linalg.qr(A,mode='economic')
#     B =  scipy.linalg.inv(r).dot(q.T.conj())
# #   
    I = numpy.eye(A.shape[0],A.shape[1])
#     for pp in xrange(0,50):


    B = scipy.linalg.pinv2(A)  
#     diff = I - A.dot(B)
#     print(scipy.linalg.norm(diff))
#     B=A.T.conj()
#     B = B/scipy.linalg.norm(A)
#     print(A.shape)
# #     I = numpy.eye(A.shape[0],A.shape[1])
    for pp in xrange(0,2):
        diff = I - A.dot(B)
#         print(scipy.linalg.norm(diff))
        B=B + B.dot(diff)
    
#     B = scipy.linalg.pinv2(A)
    

    
    
    return B

def nufft_T(N, J, K , alpha, beta):
    '''
     equation (29) and (26)Fessler's paper
     create the overlapping matrix CSSC (diagonal dominent matrix)
     of J points 
     and then find out the pseudo-inverse of CSSC 
     '''

#     import scipy.linalg
    L = numpy.size(alpha) - 1
#     print('L = ', L, 'J = ',J, 'a b', alpha,beta )
    cssc = numpy.zeros((J,J));
    [j1, j2] = numpy.mgrid[1:J+1, 1:J+1]
    overlapping_mat = j2 - j1 
    
    for l1 in xrange(-L,L+1):
        for l2 in xrange(-L,L+1):
            alf1 = alpha[abs(l1)]
#             if l1 < 0: alf1 = numpy.conj(alf1)
            alf2 = alpha[abs(l2)]
#             if l2 < 0: alf2 = numpy.conj(alf2)
            tmp = overlapping_mat + beta * (l1 - l2)
#             tmp = numpy.sinc(1.0*tmp/(1.0*K/N)) # the interpolator
            tmp = dirichlet(1.0*tmp/(1.0*K/N))
            cssc = cssc + alf1 * numpy.conj(alf2) * tmp;

#     cssc = scipy.linalg.inv(cssc )
#     q,r  = scipy.linalg.qr(cssc,mode='full')
#     cssc =  r.conj().T.dot(scipy.linalg.inv(q))
#     cssc =  scipy.linalg.inv(r).dot(q.T.conj())
#     T,Z = scipy.linalg.schur(cssc)
#     cssc = Z.conj().T.dot(scipy.linalg.inv(T))*Z
    
    return mat_inv(cssc) 


def iterate_sum(rr, alf, r1):
    rr = rr + alf * r1;
    return rr


def iterate_l1(L, alpha, arg, beta, K, N, rr):
    oversample_ratio = (1.0*K/N) 
    
    for l1 in xrange(-L,L+1):
        alf = alpha[abs(l1)]*1.0
        if l1 < 0: alf = numpy.conj(alf) 
    #             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
        input_array = ( arg+1.0*l1*beta)/oversample_ratio
        r1 = dirichlet2( input_array.astype(numpy.float32)  )
        rr = iterate_sum(rr, alf, r1)
    return rr

def nufft_r(om, N, J, K, alpha, beta):
    '''
    equation (30) of Fessler's paper
    
    '''
  
    M = numpy.size(om) # 1D size
    gam = 2.0*numpy.pi / (K*1.0)
    nufft_offset0 = nufft_offset(om, J, K) # om/gam -  nufft_offset , [M,1]
    dk = 1.0*om/gam -  nufft_offset0 # om/gam -  nufft_offset , [M,1]
    arg = outer_sum( -numpy.arange(1,J+1)*1.0, dk)
    L = numpy.size(alpha) - 1
#     print('alpha',alpha)
    rr = numpy.zeros((J,M), dtype=numpy.float32)
    rr = iterate_l1(L, alpha, arg, beta, K, N, rr)
#     if L > 0: 
#         rr = numpy.zeros((J,M))
#                if J > 1:
#     for l1 in xrange(-L,L+1):
#         alf = alpha[abs(l1)]*1.0
#         if l1 < 0: alf = numpy.conj(alf) 
# #             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
#         r1 = dirichlet(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
#         rr = 1.0*rr + alf * r1;            # [J,M]
#                elif J ==1:
#                    rr=rr+1.0
#     else: #L==0
# #         rr = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
#         rr = dirichlet(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
    return (rr,arg)
# def SC(om, N, J, K, alpha, beta):
#     '''
#     equation (30) of Fessler's paper
#     
#     '''
#   
#     M = numpy.size(om) # 1D size
#     gam = 2.0*numpy.pi / (K*1.0)
#     nufft_offset0 = nufft_offset(om, J, K) # om/gam -  nufft_offset , [M,1]
#     dk = 1.0*om/gam -  nufft_offset0 # om/gam -  nufft_offset , [M,1] phase shifts for M points
#     arg = outer_sum( -numpy.arange(1,J+1)*1.0, dk) # phase shifts for JxM points, [J, M]
# #     print(numpy.shape(arg))
#     L = numpy.size(alpha) - 1
# #     print('alpha',alpha)
#     rr = numpy.zeros((J,M))
# #     if L > 0: 
# #         rr = numpy.zeros((J,M))
# #                if J > 1:
#     for l1 in xrange(-L,L+1):
#         alf = alpha[abs(l1)]*1.0
#         if l1 < 0: alf = numpy.conj(alf) 
# #             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
# #         r1 = dirichlet(1.0*( 1.0*l1*beta)/(1.0*K/N))
#         r1 = dirichlet(1.0*(1.0*l1*beta)/(1.0*K/N))
#         rr = 1.0*rr + alf * r1;            # [J,M]
# #                elif J ==1:
# #                    rr=rr+1.0
# #     else: #L==0
# # #         rr = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
# #         rr = dirichlet(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
#     SC = rr.conj().T # [M, J]
#     return (SC,arg)

def block_outer_prod(x1, x2):
    '''
    multiply interpolators of different dimensions
    '''
    (J1,M)=x1.shape
    (J2,M)=x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1,1,M),order='F') #[J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1,(1,J2,1)) #[J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1,J2,M),order='F') # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2,(J1,1,1)) # [J1 J2 M], emulating ndgrid
    
#     ang_xx1=xx1/numpy.abs(xx1)
#     ang_xx2=xx2/numpy.abs(xx2)
    
    y= xx1* xx2
#     y= ang_xx1*ang_xx2*numpy.sqrt(xx1*xx1.conj())*numpy.sqrt( xx2*xx2.conj())
    
    # RMS
    return y # [J1 J2 M]
 
def block_outer_sum(x1, x2):
    (J1,M)=x1.shape
    (J2,M)=x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1,1,M),order='F') #[J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1,(1,J2,1)) #[J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1,J2,M),order='F') # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2,(J1,1,1)) # [J1 J2 M], emulating ndgrid
    y= xx1+ xx2
    return y # [J1 J2 M]

def crop_slice_ind(Nd):
    return [slice(0, Nd[_ss]) for _ss in range(0,len(Nd))]

class nufft:

    '''      
    pyNuff is ported to Python and Refined 
    by Jyh-Miin Lin at Cambridge University
    
    DETAILS:
    __init__(self,om, Nd, Kd,Jd): Create the class with om/Nd/Kd/Jd
                om: the k-space points either on grids or outside 
                       grids
                Nd: dimension of images, e.g. (256,256) for 2D
                Kd: Normally you should use Kd=2*Nd,e.g. (512,512)
                of above example. However, used Kd=Nd when Jd=1
                Jd: number of adjacents grids on kspace to do 
                    interpolation
                self.st: the structure storing information
                self.st['Nd']: image dimension
                self.st['Kd']: Kspace dimension
                self.st['M']: number of data points(on k-space)
                self.st['p']: interpolation kernel in 
                self.st['sn']: scaling in image space 
                self.st['w']: precomputed Cartesian Density 
                                   (the weighting in k-space)
                                   
    X=self.forward(x): transforming the image x to X(points not on kspace
                   grids)
                   pseudo-code: X= st['p']FFT{x*st['sn'],Kd}/sqrt(prod(KD))
    
    x2=self.backward(self,X):adjoint (conjugated operator) of forward
                    also known as (regridding)
                    pseudo-code: x = st['sn']*IFFT{X*st['p'].conj() ,Kd}*sqrt(prod(KD))

    
    Note: distinguishable major modification:
    1. modified of coefficient:

        The coefficient of J. Fessler's version may be problematic. 
                
        While his forward projection is not scaled, and backward 
        projection is scaled up by (prod(Kd))-- as is wrong for 
        iterative reconstruction, because the result will be 
        scaled up by (prod(Kd))
        
        The above coefficient is right in the sense of "adjoint" 
        operator, but it is wrong for iterative reconstruction!!
        
        
    2. Efficient backwardforward():
        see pyNufft_fast
        
     3. Slice over higher dimension
        The extraordinary property of pyNufft is the following: 
        x = x[[slice(0, Nd[_ss]) for _ss in range(0,numpy.size(Nd))]]
        This sentence is exclusive for python, and it can scope 
        high dimensional array.
        
      4.Support points on grids with Jd == 1:
      when Jd = (1,1), the scaling factor st['sn'] = 1 
      
    REFERENCES 
    
    I didn't reinvented the program: it was translated from 
    the NUFFT in MATLAB of Jeffrey A Fessler, University of Michigan.
    
    However, several important modifications are listed above. In 
    particular, the scaling factor st['scale']
    
    Yet only the Kaiser-Bessel window was implemented.
    
    Please refer to 
    "Nonuniform fast Fourier transforms using min-max interpolation." 
    IEEE Trans. Sig. Proc., 51(2):560-74, Feb. 2003.         

    '''


    def __init__(self,om, Nd, Kd,Jd,n_shift):
        '''
       constructor of pyNufft
        
        '''       

        '''
        Constructor: Start from here
        '''
        self.debug = 0 # debug          
        Nd=tuple(Nd) # convert Nd to tuple for consistent structure 
        Jd=tuple(Jd) # convert Jd to tuple for consistent structure
        Kd=tuple(Kd) # convert Kd to tuple for consistent structure
        # n_shift: the fftshift position, it must be at center
#         n_shift=tuple(numpy.array(Nd)/2)
          

        # dimensionality of input space (usually 2 or 3)
        dd=numpy.size(Nd)
        
    #=====================================================================
    # check input errors
    #=====================================================================
        st={}
        ud={}
        kd={}
#         phase_kd = {} # phase of Kd grids
        st['sense']=0 # default sense control flag
        st['sensemap']=[]  # default sensemap is null
        st['n_shift']=n_shift
    #=======================================================================
    # First, get alpha and beta: the weighting and freq
    # of formula (28) of Fessler's paper 
    # in order to create slow-varying image space scaling 
    #=======================================================================
        for dimid in xrange(0,dd):

            (tmp_alpha,tmp_beta)=nufft_alpha_kb_fit(Nd[dimid], Jd[dimid], Kd[dimid])
                
            st.setdefault('alpha', []).append(tmp_alpha)
            st.setdefault('beta', []).append(tmp_beta)

        
        st['tol'] = 0
        st['Jd'] = Jd
        st['Nd'] = Nd
        st['Kd'] = Kd
        M = om.shape[0]
        st['M'] = M
        st['om'] = om
        st['sn'] = numpy.array(1.0+0.0j)
        dimid_cnt=1 
    #=======================================================================
    # create scaling factors st['sn'] given alpha/beta
    # higher dimension implementation
    #=======================================================================
        for dimid in xrange(0,dd):
            tmp = nufft_scale(Nd[dimid], Kd[dimid], st['alpha'][dimid], st['beta'][dimid])

            dimid_cnt=Nd[dimid]*dimid_cnt
    #=======================================================================
    # higher dimension implementation: multiply over all dimension
    #=======================================================================

            st['sn'] =  numpy.dot(st['sn'] , tmp.T )
            st['sn'] =   numpy.reshape(st['sn'],(dimid_cnt,1),order='F') # JML do not apply scaling

        #=======================================================================
        # if numpy.size(Nd) > 1: 
        #=======================================================================
            # high dimension, reshape for consistent out put
            # order = 'F' is for fortran order otherwise it is C-type array 
        st['sn'] = st['sn'].reshape(Nd,order='F') # [(Nd)]
        #=======================================================================
        # else:
        #     st['sn'] = numpy.array(st['sn'],order='F')
#         #=======================================================================
            
        st['sn']=numpy.real(st['sn']) # only real scaling is relevant 

        # [J? M] interpolation coefficient vectors.  will need kron of these later
        for dimid in xrange(0,dd): # loop over dimensions
            N = Nd[dimid]
            J = Jd[dimid]
            K = Kd[dimid]
            alpha = st['alpha'][dimid]
            beta = st['beta'][dimid]
            #===================================================================
            # formula 29 , 26 of Fessler's paper
            #===================================================================
            
            T = nufft_T(N, J, K, alpha, beta) # pseudo-inverse of CSSC using large N approx [J? J?]
            #==================================================================
            # formula 30  of Fessler's paper
            #==================================================================

            (r,arg)=  nufft_r(om[:,dimid], N, J, K, alpha, beta) # large N approx [J? M]
            
            #==================================================================
            # formula 25  of Fessler's paper
            #==================================================================            
            c=numpy.dot(T,r)

            #===================================================================
            # grid intervals in radius
            #===================================================================
            gam = 2.0*numpy.pi/(K*1.0);
            
            phase_scale = 1.0j * gam * (N-1.0)/2.0
            phase = numpy.exp(phase_scale * arg) # [J? M] linear phase
            ud[dimid] = phase * c
            # indices into oversampled FFT components
            # FORMULA 7
            koff=nufft_offset(om[:,dimid], J, K)
            # FORMULA 9, find the indexes on Kd grids, of each M point
            kd[dimid]= numpy.mod(outer_sum( numpy.arange(1,J+1)*1.0, koff),K)
            if self.debug > 0:
                print('kd[',dimid,']',kd[dimid].shape)
#             phase_kd[dimid] = - ( numpy.arange(0,1.0, 1.00/K) - 0.5 )*2.0*numpy.pi*n_shift[dimid]# new phase2
             
            if dimid > 0: # trick: pre-convert these indices into offsets!
    #            ('trick: pre-convert these indices into offsets!')
                kd[dimid] = kd[dimid]*numpy.prod(Kd[0:dimid])-1

        kk = kd[0] # [J1 M] # pointers to indices
        uu = ud[0] # [J1 M]
#         zz = phase_kd[0] # aperture 

#         phase_2 = numpy.meshgrid(phase_kd)  
#         pp_kd =phase_kd[0]
        Jprod = Jd[0]
        Kprod = Kd[0]
        for dimid in xrange(1,dd):
            Jprod = numpy.prod(Jd[:dimid+1])
            Kprod = numpy.prod(Kd[:dimid+1])
            if self.debug > 0:
                print('Kprod',Kprod)
            kk = block_outer_sum(kk, kd[dimid])+1 # outer sum of indices
            kk = kk.reshape((Jprod, M),order='F')
            
#             tmp_kd1, tmp_kd2= numpy.meshgrid(pp_kd, phase_kd[dimid])
#             pp_kd = tmp_kd1 + tmp_kd2
#             pp_kd = numpy.reshape(pp_kd,(Kprod,),'F')
              
            uu = block_outer_prod(uu, ud[dimid]) # outer product of coefficients
            uu = uu.reshape((Jprod, M),order='F')

            #now kk and uu are [*Jd M]
            #now kk and uu are [*Jd M]
    #    % apply phase shift
    #    % pre-do Hermitian transpose of interpolation coefficients
#         print('pp_kd',pp_kd.shape)
#         phase_om = numpy.reshape(  numpy.tile(phase,[numpy.prod(Jd),1])  , (Jprod*M,),order='F')# [*Jd M]
        uu = uu.conj()#*numpy.tile(phase,[numpy.prod(Jd),1]) #    product(Jd)xM
        mm = numpy.arange(0,M) # indices from 0 to M-1
        mm = numpy.tile(mm,[numpy.prod(Jd),1]) #    product(Jd)xM

        st['p0'] = scipy.sparse.coo_matrix( # build sparse matrix from uu, mm, kk
                                          (     numpy.reshape(uu,(Jprod*M,),order='F'), # convert array to list
                                                (numpy.reshape(mm,(Jprod*M,),order='F'), # row indices, from 1 to M convert array to list
                                                 numpy.reshape(kk,(Jprod*M,),order='F') # colume indices, from 1 to prod(Kd), convert array to list
                                                           )
                                                    ),
                                        shape=(M,numpy.prod(Kd)) # shape of the matrix
                                          ).tocsc() #  order = 'F' is just a convention which has nothing to do with sparse matrix
#         tmp =  numpy.abs((  st['p0'].conj().T.dot(numpy.ones((st['p0'].get_shape()[0], 1),dtype = numpy.float32))))#**2) ))
#         tmp_max = numpy.max(tmp[:])
#         st['p0'] = st['p0'] /tmp_max
        
        self.st=st
        self.Nd = self.st['Nd'] # backup
        self.sn = self.st['sn'] # backup
        self.prod_Nd = numpy.prod(self.st['Nd'])
        self.prod_Kd = numpy.prod(self.st['Kd'])
        
        self.linear_phase( n_shift  ) # calculate the linear phase thing
#         phase = self.linear_phase(om, n_shift, M) # calculate phase of n_shift
        #numpy.exp(1.0j* numpy.sum(om*numpy.tile(n_shift,(M,1)) , 1))  # add-up all the linear phasees in all axes,

#         st['p'] = scipy.sparse.diags(phase ,0).dot(st['p0'] ) # multiply the diagonal, linear phase before the gridding matrix
        self.pipe_density() # recalculate the density compensation


        self.finalization() # finalize the generation of NUFFT opject 

        

    def pipe_density(self):
        '''
        Create the density function by iterative solution 
        '''
        self.st['W'] = pipe_density(self.st['p'])
        self.st['w'] =  numpy.abs(( self.st['p'].conj().T.dot(numpy.ones(self.st['W'].shape,dtype = numpy.float32))))#**2) ))
        tmp_max = numpy.max(self.st['w'])
        self.st['w'] = self.st['w'] /tmp_max
#         data_max = numpy.max(numpy.abs(self.st['w'][:]))
#         self.st['w'] = self.st['w']/data_max
#         self.st['p'] = self.st['p']/data_max
#         self.st['p0'] = self.st['p0']/data_max
#         self.st['w'] = self.st['w'] / numpy.percentile(self.st['w'], 100)
#         print('shape of w',numpy.shape(self.st['w']))
#         self.st['w'] = self.st['p']
#         self.st['w']  =numpy.abs( self.st['p'].conj().T.multiply(self.st['p'].T).sum(axis= 1)) 
#         self.st['w'] = numpy.reshape( self.st['w'] , (numpy.prod( self.st['Kd']),1))
#         tmp_size = numpy.shape(self.st['w'])
#         print('shape of tmp',numpy.shape(self.st['w'] ))        
#         self.st['w'] = numpy.reshape(self.st['w'], tmp_size +(1,))
        if self.debug > 0:
            print('shape of tmp',numpy.shape(self.st['w'] ))
#         self.st['w'] = numpy.sum(self.st['p']  )
#     def reduce_fov(self,rfov): # method to reduce FOV
#        
#         self.st['Nd'] = 
#         self.st['sn'] = 
               
    def linear_phase(self, n_shift ):
        '''
        Select the center of FOV 
        '''
        om = self.st['om']
        M = self.st['M']
        phase=numpy.exp(1.0j* numpy.sum(om*numpy.tile( tuple( numpy.array(n_shift) + numpy.array(self.st['Nd'])/2  ),(M,1)) , 1))  # add-up all the linear phasees in all axes,
        
        self.st['p'] = scipy.sparse.diags(phase ,0).dot(self.st['p0'] ) # multiply the diagonal, linear phase before the gridding matrix

    def finalization(self):
        #         self.st=st
#         if self.debug==0:
#             pass
#         else:        
#             print('st sn shape',st['sn'].shape)
        self.gpu_flag=0
#         self.initialize_gpu() 

        self.pyfftw_flag = 0#self.__initialize_pyfftw()

        self.threads=2#multiprocessing.cpu_count()
#         return phase
    
    def __initialize_pyfftw(self):
        pyfftw_flag = 0
        try:
            import pyfftw
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(60) # keep live 60 seconds
            pyfftw_flag = 1
#             if self.debug > 0:
            print('use pyfftw')
        except:
#             if self.debug > 0:
            print('no pyfftw, use slow fft')
            pyfftw_flag = 0
        return pyfftw_flag  
    def initialize_gpu(self):
        try:
            import reikna.cluda as cluda
            from reikna.fft import FFT 

            data = numpy.zeros( self.st['Kd'],dtype=dtype)

            print('get_platform')
            api = cluda.ocl_api()

            print('api=',api== cluda.cuda_api())

            self.gpu_api =  'opencl'
                
            self.thr = api.Thread.create(async=True)   
            print('line 630')   
            self.data_dev = self.thr.to_device(data)

            axes=range(0,numpy.size(self.st['Kd']))
            print('line 635')   
            myfft=  FFT( data, axes=axes)
            print('line 640')   
            self.myfft = myfft.compile(self.thr,fast_math=True)
            print('line 640')   
            self.gpu_flag=1

            print('create gpu fft?',self.gpu_flag)
            print('line 642')#             self.data_rec = self.thr.to_device(data2)

                
            W= self.st['w'][...,0]

            print('line 645')   
                
            self.W = numpy.reshape(W, self.st['Kd'],order='C')
            

            print('line 647')
#             self.thr2 = api.Thread.create() 
            print('line 649')
            self.W_dev = self.thr.to_device(self.W.astype(dtype))
            self.W2_dev = self.thr.to_device(self.W.astype(dtype))
            self.tmp_dev = self.thr.to_device(self.W.astype(dtype)) # device memory
#             self.tmp2_dev = self.thr.to_device(1.0/self.W.astype(dtype)) # device memory
            self.gpu_flag=1      
#             if self.debug > 0:          
            print('line 652')
        except:
            self.gpu_flag=0
#             if self.debug > 0:              
            print('get error, using cpu')
        

   
    def forward(self,x):
        '''
        foward(x): method of class pyNufft
        
        Compute dd-dimensional Non-uniform transform of signal/image x
        where d is the dimension of the data x.
        
        INPUT: 
          case 1:  x: ndarray, [Nd[0], Nd[1], ... , Kd[dd-1] ]
          case 2:  x: ndarray, [Nd[0], Nd[1], ... , Kd[dd-1], Lprod]
            
        OUTPUT: 
          X: ndarray, [M, Lprod] (Lprod=1 in case 1)
                    where M =st['M']
        '''
        
        st=self.st
        Nd = st['Nd']
#         Kd = st['Kd']
#         dims = numpy.shape(x)
        dd = numpy.size(Nd)
    #    print('in nufft, dims:dd',dims,dd)
    #    print('ndim(x)',numpy.ndim(x[:,1]))
        # exceptions
        if self.debug==0:
            pass
        else:
            checker(x,Nd)
        
        if numpy.ndim(x) == dd:
            Lprod = 1
        elif numpy.ndim(x) > dd: # multi-channel data
            Lprod = numpy.size(x)/self.prod_Nd#numpy.prod(Nd)
        '''
        Now transform Nd grids to Kd grids(not be reshaped)
        '''
        Xk=self.Nd2Kd(x, 1)
        
        # interpolate using precomputed sparse matrix
        if Lprod > 1:
            X = numpy.reshape(st['p'].dot(Xk),(st['M'],)+( Lprod,),order='F')
        else:
            X = numpy.reshape(st['p'].dot(Xk),(st['M'],1),order='F')
        if self.debug==0:
            pass
        else:
            checker(X,st['M']) # check output
        return X
    
    def backward(self,X):
        '''
        backward(x): method of class pyNufft
        
        from [M x Lprod] shaped input, compute its adjoint(conjugate) of 
        Non-uniform Fourier transform 

        
        INPUT: 
          X: ndarray, [M, Lprod] (Lprod=1 in case 1)
            where M =st['M']
          
        OUTPUT: 
          x: ndarray, [Nd[0], Nd[1], ... , Kd[dd-1], Lprod]
            
        '''
#     extract attributes from structure
        st=self.st

        Nd = st['Nd']
#         Kd = st['Kd']
        if self.debug==0:
            pass
        else:
            checker(X,st['M']) # check X of correct shape

#         dims = numpy.shape(X)
#         Lprod= numpy.prod(dims[1:]) 
        # how many channel * slices

#         if numpy.size(dims) == 1:
#             Lprod = 1
#         else:
#             Lprod = dims[1]
#         print('Xshape',X.shape)
#         print('stp.shape',st['p'].shape)
        Xk_all = st['p'].getH().dot(X) 
        # Multiply X with interpolator st['p'] [prod(Kd) Lprod]
        '''
        Now transform Kd grids to Nd grids(not be reshaped)
        '''
        x = self.Kd2Nd(Xk_all, 1)
        
        if self.debug==0:
            pass
        else:        
            checker(x,Nd) # check output
        
        return x

    def Nd2Kd(self,x, weight_flag):
        '''
        Now transform Nd grids to Kd grids(not be reshaped)
        
        '''
        #print('661 x.shape',x.shape)        
        
        st=self.st
        Nd = st['Nd']
        Kd = st['Kd']
        dims = numpy.shape(x)
        dd = numpy.size(Nd)
    #    print('in nufft, dims:dd',dims,dd)
    #    print('ndim(x)',numpy.ndim(x[:,1]))
        # checker
        if self.debug==0:
            pass
        else:
            checker(x,Nd)

#         if numpy.ndim(x) == dd:
#             Lprod = 1
#         elif numpy.ndim(x) > dd: # multi-channel data
#             Lprod = numpy.size(x)/numpy.prod(Nd)
                    
        if numpy.ndim(x) == dd:   
            if weight_flag == 1:
                x = x * st['sn']
            else:
                pass 

            Xk = self.emb_fftn(x, Kd,range(0,dd))

            Xk = numpy.reshape(Xk, (self.prod_Kd,),order='F')
            
        else:# otherwise, collapse all excess dimensions into just one
            xx = numpy.reshape(x, [self.prod_Nd, numpy.prod(dims[(dd):])],order='F')  # [*Nd *L]
            L = numpy.shape(xx)[1]
    #        print('L=',L)
    #        print('Lprod',Lprod)
            Xk = numpy.empty( (self.prod_Kd, L),dtype=dtype) # [*Kd *L]
            for ll in xrange(0,L):
                xl = numpy.reshape(xx[:,ll], Nd,order='F') # l'th signal
                if weight_flag == 1:
                    xl = xl * st['sn'] # scaling factors
                else:
                    pass
                Xk[:,ll] = numpy.reshape(self.emb_fftn(xl, Kd,range(0,dd)),
                                         (self.prod_Kd,),order='F')
        if self.debug==0:
            pass
        else:                
            checker(Xk,self.prod_Kd)        
        return Xk
    
    def Kd2Nd(self,Xk_all,weight_flag):
        
        st=self.st

        Nd = st['Nd']
        Kd = st['Kd']
        dd = len(Nd)
        if self.debug==0:
            pass
        else:
            checker(Xk_all,self.prod_Kd) # check X of correct shape

        dims = numpy.shape(Xk_all)
        Lprod= numpy.prod(dims[1:]) # how many channel * slices

        if numpy.size(dims) == 1:
            Lprod = 1
        else:
            Lprod = dims[1]  
            
                  
        x=numpy.empty(Kd+(Lprod,),dtype=dtype)  # [*Kd *L]

#         if Lprod > 1:

        Xk_all = numpy.reshape(Xk_all, Kd+(Lprod,) , order='F')
        
        for ll in xrange(0,Lprod):  # ll = 0, 1,... Lprod-1
            x[...,ll] =  self.emb_ifftn(Xk_all[...,ll],Kd,range(0,dd))#.flatten(order='F'))

        x = x[crop_slice_ind(Nd)]


        if weight_flag == 0:
            pass
            
        else: #weight_flag =1 scaling factors
            snc = st['sn']#.conj()
            for ll in xrange(0,Lprod):  # ll = 0, 1,... Lprod-1    
                x[...,ll] = x[...,ll]*snc  #% scaling factors

        if self.debug==0:
            pass # turn off checker
        else:
            checker(x,Nd) # checking size of x divisible by Nd
        return x
    def gpufftn(self, data_dev):
        '''
        gpufftn: an interface to external gpu fftn:
        not working to date: awaiting more reliable gpu codes 
        '''
#         self.data_dev = self.thr.to_device(output_x.astype(dtype))
        self.myfft( data_dev,  data_dev)
        return data_dev#.get()   
    def gpuifftn(self, data_dev):
        '''
        gpufftn: an interface to external gpu fftn:
        not working to date: awaiting more reliable gpu codes 
        '''
#         self.data_dev = self.thr.to_device(output_x.astype(dtype))
        self.myfft( data_dev,  data_dev, inverse=1)
        return data_dev#.get()        
    def emb_fftn(self, input_x, output_dim, act_axes):
        '''
        embedded fftn: abstraction of fft for future gpu computing 
        '''
        output_x=numpy.zeros(output_dim, dtype=dtype)
        #print('output_dim',input_dim,output_dim,range(0,numpy.size(input_dim)))
#         output_x[[slice(0, input_x.shape[_ss]) for _ss in range(0,len(input_x.shape))]] = input_x
        output_x[crop_slice_ind(input_x.shape)] = input_x
#         print('GPU flag',self.gpu_flag)
#         print('pyfftw flag',self.pyfftw_flag)
#         if self.gpu_flag == 1:
#             self.thr.to_device(output_x.astype(dtype), dest=self.data_dev)
#             output_x=self.gpufftn(self.data_dev).get()
             
#         except:
#         elif self.gpu_flag ==0:
#         elif self.pyfftw_flag == 1:
# #             try:
#     #                 print('using pyfftw interface')
#     #                 print('threads=',self.threads)
#             output_x=pyfftw.interfaces.scipy_fftpack.fftn(output_x, output_dim, act_axes, 
#                                                               threads=self.threads,overwrite_x=True)
# #             except: 
#         else:
    #                 print('using OLD interface')                
#                 output_x=scipy.fftpack.fftn(output_x, output_dim, act_axes,overwrite_x=True)
        output_x=fftpack.fftn(output_x, output_dim, act_axes)


        return output_x
#     def emb_ifftn(self, input_x, output_dim, act_axes):
#         '''
#         embedded ifftn: abstraction of ifft for future gpu computing 
#         '''
#         
# #         output_x=input_x
#         output_x=self.emb_fftn(input_x.conj(), output_dim, act_axes).conj()/numpy.prod(output_dim)
#     
#         return output_x    
    def emb_ifftn(self, input_x, output_dim, act_axes):
        '''
        embedded fftn: abstraction of fft for future gpu computing 
        '''
        output_x=numpy.zeros(output_dim, dtype=dtype)
        #print('output_dim',input_dim,output_dim,range(0,numpy.size(input_dim)))
#         output_x[[slice(0, input_x.shape[_ss]) for _ss in range(0,len(input_x.shape))]] = input_x
        output_x[crop_slice_ind(input_x.shape)] = input_x 
#         print('GPU flag',self.gpu_flag)
#         print('pyfftw flag',self.pyfftw_flag)
#         if self.gpu_flag == 1:
#             self.thr.to_device(output_x.astype(dtype), dest=self.data_dev)
#             output_x=self.gpuifftn(self.data_dev).get()
             
#         except:

    #                 print('using OLD interface')                
#                 output_x=scipy.fftpack.ifftn(output_x, output_dim, act_axes,overwrite_x=True)
        output_x=fftpack.ifftn(output_x, output_dim, act_axes)


        return output_x
     
def test_init():
    import cProfile     
    import numpy 
    import matplotlib.pyplot
    import copy
     
    cm = matplotlib.cm.gray
    # load example image    
 
    image = numpy.loadtxt('phantom_256_256.txt') 
    image[128,128]= 1.0   
    Nd =(256,256) # image space size
    Kd =(512,512) # k-space size   
    Jd =(6,6) # interpolation size
     
    # load k-space points
    om = numpy.loadtxt('om.txt')
     
    #create object
    n_shift=tuple(numpy.array(Nd)*0)
#         else:
#             n_shift=tuple(list(n_shift)+numpy.array(Nd)/2)
            
      
     
    NufftObj = nufft(om, Nd, Kd,Jd,n_shift)    
    NufftObj.initialize_gpu()
    
if __name__ == '__main__':
    import cProfile
    test_init()
#     cProfile.run('test_init()')
           
