'''@package docstring
@author: Jyh-Miin Lin  (Jimmy), Cambridge University
@address: jyhmiinlin@gmail.com
Created on 2013/1/21
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
          
    and 
    "CS-PROPELLER MRI with Parallel Coils Using NUFFT and Split-Bregman Method"(in progress 2013)
    Jyh-Miin Lin, Andrew Patterson, Hing-Chiu Chang, Tzu-Chao Chuang, Martin J. Graves, 
    which is planned to be published soon.
    
    2. Note the "better" results by min-max interpolator of J.A. Fessler et al
    3. Other relevant works:
    *c-version: http://www-user.tu-chemnitz.de/~potts/nfft/
    is a c-library with gaussian interpolator
    *fortran version: http://www.cims.nyu.edu/cmcl/nufft/nufft.html
    alpha/beta stage
    * MEX-version http://www.mathworks.com/matlabcentral/fileexchange/25135-nufft-nufft-usffft
'''


import numpy
import scipy.sparse
from scipy.sparse.csgraph import _validation # for cx_freeze debug
# import sys
import scipy.fftpack
try:
    import pyfftw
except:
    pass
try:
    from numba import jit
except:
    pass

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
def pipe_density(V,W):
    V1=V.conj().T
#     E = V.dot( V1.dot(    W   )   )
#     W = W*(E+1.0e-17)/(E*E+1.0e-17)    
    
    for pppj in xrange(0,10):
#             W[W>1.0]=1.0
#             print(pppj)
        E = V.dot( V1.dot(    W   )   )

        W = W*(E+1.0e-17)/(E**2+1.0e-17)
        
    return W
def checker(input_var,desire_size):

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
        L=13
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
    z = numpy.sqrt( (2*numpy.pi*(J/2)*u)**2 - alpha**2 );
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


def nufft_T(N, J, K, tol, alpha, beta):
    '''
     equation (29) and (26)Fessler's paper
     the pseudo-inverse of T  
     '''
    import scipy.linalg
    L = numpy.size(alpha) - 1
    cssc = numpy.zeros((J,J));
    [j1, j2] = numpy.mgrid[1:J+1, 1:J+1]
    for l1 in xrange(-L,L+1):
        for l2 in xrange(-L,L+1):
            alf1 = alpha[abs(l1)]
            if l1 < 0: alf1 = numpy.conj(alf1)
            alf2 = alpha[abs(l2)]
            if l2 < 0: alf2 = numpy.conj(alf2)
            tmp = j2 - j1 + beta * (l1 - l2)
            tmp = numpy.sinc(1.0*tmp/(1.0*K/N)) # the interpolator
            cssc = cssc + alf1 * numpy.conj(alf2) * tmp;
            #print([l1, l2, tmp ])
    u_svd, s_svd, v_svd= scipy.linalg.svd(cssc)
    smin=numpy.min(s_svd)
    if smin < tol:
        tol=tol
        print('Poor conditioning %g => pinverse', smin)
    else:
        tol= 0.0
    for jj in xrange(0,J):
        if s_svd[jj] < tol/10:
            s_svd[jj]=0
        else:
            s_svd[jj]=1/s_svd[jj]      
    s_svd= scipy.linalg.diagsvd(s_svd,len(u_svd),len(v_svd))
    cssc = numpy.dot(  numpy.dot(v_svd.conj().T,s_svd), u_svd.conj().T)
    return cssc 

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
    if L > 0: 
        rr = numpy.zeros((J,M))
#                if J > 1:
        for l1 in xrange(-L,L+1):
            alf = alpha[abs(l1)]*1.0
            if l1 < 0: alf = numpy.conj(alf) 
            r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
            rr = 1.0*rr + alf * r1;            # [J,M]
#                elif J ==1:
#                    rr=rr+1.0
    else: #L==0
        rr = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
    return (rr,arg)

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
#     y= ang_xx1*ang_xx2*numpy.sqrt(xx1*xx1.conj() + xx2*xx2.conj())
    
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
    return [slice(0, Nd[_ss]) for _ss in xrange(0,len(Nd))]

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
#             rms1= numpy.dot(st['sn'], tmp.T**0) 
#             rms2 = numpy.dot(st['sn']**0, tmp.T)
#             ang_rms1 = rms1/numpy.abs(rms1)
#             ang_rms2 = rms2/numpy.abs(rms2)
#             st['sn'] = ang_rms1*ang_rms2* numpy.sqrt( rms1*rms1.conj() +rms2*rms2.conj() )
#             # RMS
#             st['sn'] =  numpy.dot(st['sn'] , tmp.T )
#             st['sn'] =   numpy.reshape(st['sn'],(dimid_cnt,1),order='F')
#             if True: # do not apply scaling
#                 st['sn']= numpy.ones((dimid_cnt,1),dtype=numpy.complex64)
#             else:
            st['sn'] =  numpy.dot(st['sn'] , tmp.T )
            st['sn'] =   numpy.reshape(st['sn'],(dimid_cnt,1),order='F')**0.0 # JML do not apply scaling


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
            
            T = nufft_T(N, J, K, st['tol'], alpha, beta) # [J? J?]
            #==================================================================
            # formula 30  of Fessler's paper
            #==================================================================
            if self.debug==0:
                pass
            else:            
                print('dd',dd)
                print('dimid',dimid)
            
            (r,arg)=  nufft_r(om[:,dimid], N, J, K, alpha, beta) # [J? M]
            #==================================================================
            # formula 25  of Fessler's paper
            #==================================================================            
            c=numpy.dot(T,r)
#             
#             print('size of c, r',numpy.shape(c), numpy.shape(T),numpy.shape(r))
#             import matplotlib.pyplot
#             matplotlib.pyplot.plot(r[:,0:])
#             matplotlib.pyplot.show()
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
            # FORMULA 9
            kd[dimid]= numpy.mod(outer_sum( numpy.arange(1,J+1)*1.0, koff),K)
          
            if dimid > 0: # trick: pre-convert these indices into offsets!
    #            ('trick: pre-convert these indices into offsets!')
                kd[dimid] = kd[dimid]*numpy.prod(Kd[0:dimid])-1

        kk = kd[0] # [J1 M]
        uu = ud[0] # [J1 M]

        for dimid in xrange(1,dd):
            Jprod = numpy.prod(Jd[:dimid+1])

            kk = block_outer_sum(kk, kd[dimid])+1 # outer sum of indices
            kk = kk.reshape((Jprod, M),order='F')  
            uu = block_outer_prod(uu, ud[dimid]) # outer product of coefficients
            uu = uu.reshape((Jprod, M),order='F')
            #now kk and uu are [*Jd M]
    #    % apply phase shift
    #    % pre-do Hermitian transpose of interpolation coefficients
        phase = numpy.exp( 1.0j* numpy.dot(om, 1.0*numpy.array(n_shift,order='F'))).T # [1 M]
        uu = uu.conj()*numpy.tile(phase,[numpy.prod(Jd),1]) #[*Jd M]
        mm = numpy.arange(0,M)
        mm = numpy.tile(mm,[numpy.prod(Jd),1]) # [Jd, M]
#         print('shpae uu',uu[:])
#         print('shpae kk',kk[:])
#         print('shpae mm',mm[:])
#         sn_mask=numpy.ones(st['Nd'],dtype=numpy.float16)
    #########################################now remove the corners of sn############
#         n_dims= numpy.size(st['Nd'])
#  
#         sp_rat =0.0
#         for di in xrange(0,n_dims):
#             sp_rat = sp_rat + (st['Nd'][di]/2)**2
#   
#         sp_rat = sp_rat**0.5
#         x = numpy.ogrid[[slice(0, st['Nd'][_ss]) for _ss in xrange(0,n_dims)]]
# 
#         tmp = 0
#         for di in xrange(0,n_dims):
#             tmp = tmp + ( (x[di] - st['Nd'][di]/2.0)/(st['Nd'][di]/2.0) )**2
#         
#         tmp = (1.0*tmp)**0.5
#         
#         indx = tmp >=1.0
#     
#                 
#         st['sn'][indx] = numpy.mean(st['sn'][...])      
        #########################################now remove the corners of sn############
#         st['sn']=st['sn']*sn_mask

        st['p'] = scipy.sparse.csc_matrix(
                                          (numpy.reshape(uu,(numpy.size(uu),)),
                                           (numpy.reshape(mm,(numpy.size(mm),)), numpy.reshape(kk,(numpy.size(kk),)))),
                                          (M,numpy.prod(Kd))
                                          )

        ## Now doing the density compensation of Jackson ##
        W=numpy.ones((st['M'],1))
#         w=numpy.ones((numpy.prod(st['Kd']),1))
#         for pppj in xrange(0,100):
# #             W[W>1.0]=1.0
# #             print(pppj)
#             E = st['p'].dot( st['p'].conj().T.dot(    W   )   )
# 
#             W = W*(E+1.0e-17)/(E**2+1.0e-17)
        W = pipe_density(st['p'],W)
#             import matplotlib.pyplot
#             matplotlib.pyplot.subplot(2,1,1)
#             matplotlib.pyplot.plot(numpy.abs(E))
#             matplotlib.pyplot.subplot(2,1,2)
#             matplotlib.pyplot.plot(numpy.abs(W))
# #             matplotlib.pyplot.show()
#         matplotlib.pyplot.plot(numpy.abs(W))
#         matplotlib.pyplot.show()    
        st['W'] = W
#         st['w'] = 
        ## Finish the density compensation of Jackson ##
#         st['q'] = st['p']
#         st['T'] = st['p'].conj().T.dot(st['p']) # huge memory leak>5G
#         p_herm = st['p'].conj().T.dot(st['W'])
#         print('W',numpy.shape(W))
#         print('p',numpy.shape(st['p']))
#         temp_w = numpy.tile(W,[1,numpy.prod(st['Kd'])])
#         print('temp_w',numpy.shape(temp_w))
#         st['q'] = st['p'].conj().multiply(st['p'])
#         st['q'] = st['p'].conj().T.dot(p_herm.conj().T).diagonal()

#         st['q'] = scipy.sparse.diags(W[:,0],offsets=0).dot(st['q'])
        
#         st['q'] = st['q'].sum(0)
#         
#         st['q'] = numpy.array(st['q'] )
# #         for pp in range(0,M):
# #             st['q'][pp,:]=st['q'][pp,:]*W[pp,0]
# 
#         st['q']=numpy.reshape(st['q'],(numpy.prod(st['Kd']),1),order='F').real
       
        st['w'] =  numpy.abs(( st['p'].conj().T.dot(numpy.ones(st['W'].shape,dtype = numpy.float32))))#**2) ))
#         st['q']=numpy.max(st['w'])*st['q']/numpy.max(st['q'])        
        import matplotlib.pyplot
#         matplotlib.pyplot.imshow(numpy.reshape(st['w'],st['Kd']), 
#                                 cmap=matplotlib.cm.gray,
#                                 norm=matplotlib.colors.Normalize(vmin=0.0, vmax=3.0))
#         matplotlib.pyplot.plot(numpy.reshape(st['q'],st['Kd'])[:, 0])#, 
# #                                 cmap=matplotlib.cm.gray,
# #                                 norm=matplotlib.colors.Normalize(vmin=0.0, vmax=3.0))        
# #         matplotlib.pyplot.imshow(st['sn'], 
# #                                  cmap=matplotlib.cm.gray,
# #                                  norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0))
#         matplotlib.pyplot.show()
#         matplotlib.pyplot.plot(numpy.reshape(st['w'],st['Kd'])[:, 0])#, 
# #                                 cmap=matplotlib.cm.gray,
# #                                 norm=matplotlib.colors.Normalize(vmin=0.0, vmax=3.0))        
# #         matplotlib.pyplot.imshow(st['sn'], 
# #                                  cmap=matplotlib.cm.gray,
# #                                  norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0))
#         matplotlib.pyplot.show()        
        self.st=st
        if self.debug==0:
            pass
        else:        
            print('st sn shape',st['sn'].shape)
        self.gpu_flag=0
        self.__initialize_gpu() 
#         self.__initialize_gpu2() 
        self.pyfftw_flag =self.__initialize_pyfftw()
#         import multiprocessing
        self.threads=1#multiprocessing.cpu_count()
        self.st=st
#         print( 'optimize sn and p' )
#         temp_c = self.Nd2Kd(st['sn'],0)


#         self.st['w'] = w
        
    def __initialize_pyfftw(self):
        pyfftw_flag = 0
        try:
            import pyfftw
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(60) # keep live 60 seconds
            pyfftw_flag = 1
            print('use pyfftw')
        except:
            print('no pyfftw, use slow fft')
            pyfftw_flag = 0
        return pyfftw_flag  
    def __initialize_gpu(self):
        try:
            import reikna.cluda as cluda
            from reikna.fft import FFT 
#             dtype = dtype#numpy.complex64
            data = numpy.zeros( self.st['Kd'],dtype=numpy.complex64)
#             data2 = numpy.empty_like(data)
            api = cluda.ocl_api()
            self.thr = api.Thread.create(async=True)      
            self.data_dev = self.thr.to_device(data)
#             self.data_rec = self.thr.to_device(data2)
            axes=range(0,numpy.size(self.st['Kd']))
            myfft=  FFT( data, axes=axes)
            self.myfft = myfft.compile(self.thr,fast_math=True)
 
            self.gpu_flag=1
            print('create gpu fft?',self.gpu_flag)
            print('line 642')
            W= self.st['w'][...,0]
            print('line 645')
            self.W = numpy.reshape(W, self.st['Kd'],order='C')
            print('line 647')
#             self.thr2 = api.Thread.create() 
            print('line 649')
            self.W_dev = self.thr.to_device(self.W.astype(dtype))
            self.gpu_flag=1                
            print('line 652')
        except:
            self.gpu_flag=0              
            print('get error, using cpu')
#     def __initialize_gpu2(self):
#         try:
# #             import reikna.cluda as cluda
# #             from reikna.fft import FFT 
#             from pycuda.sparse.packeted import PacketedSpMV
#             spmv = PacketedSpMV(self.st['p'], options.is_symmetric, numpy.complex64)
# #             dtype = dtype#numpy.complex64
#             data = numpy.zeros( self.st['Kd'],dtype=numpy.complex64)
# #             data2 = numpy.empty_like(data)
#             api = cluda.ocl_api()
#             self.thr = api.Thread.create(async=True)      
#             self.data_dev = self.thr.to_device(data)
# #             self.data_rec = self.thr.to_device(data2)
#             axes=range(0,numpy.size(self.st['Kd']))
#             myfft=  FFT( data, axes=axes)
#             self.myfft = myfft.compile(self.thr,fast_math=True)
#  
#             self.gpu_flag=1
#             print('create gpu fft?',self.gpu_flag)
#             print('line 642')
#             W= self.st['w'][...,0]
#             print('line 645')
#             self.W = numpy.reshape(W, self.st['Kd'],order='C')
#             print('line 647')
# #             self.thr2 = api.Thread.create() 
#             print('line 649')
#             self.W_dev = self.thr.to_device(self.W.astype(dtype))
#             self.gpu_flag=1                
#             print('line 652')
#         except:
#             self.gpu_flag=0              
#             print('get error, using cpu')    
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
        Kd = st['Kd']
        dims = numpy.shape(x)
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
            Lprod = numpy.size(x)/numpy.prod(Nd)
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
        Kd = st['Kd']
        if self.debug==0:
            pass
        else:
            checker(X,st['M']) # check X of correct shape

        dims = numpy.shape(X)
        Lprod= numpy.prod(dims[1:]) 
        # how many channel * slices

        if numpy.size(dims) == 1:
            Lprod = 1
        else:
            Lprod = dims[1]
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

            Xk = numpy.reshape(Xk, (numpy.prod(Kd),),order='F')
            
        else:# otherwise, collapse all excess dimensions into just one
            xx = numpy.reshape(x, [numpy.prod(Nd), numpy.prod(dims[(dd):])],order='F')  # [*Nd *L]
            L = numpy.shape(xx)[1]
    #        print('L=',L)
    #        print('Lprod',Lprod)
            Xk = numpy.zeros( (numpy.prod(Kd), L),dtype=dtype) # [*Kd *L]
            for ll in xrange(0,L):
                xl = numpy.reshape(xx[:,ll], Nd,order='F') # l'th signal
                if weight_flag == 1:
                    xl = xl * st['sn'] # scaling factors
                else:
                    pass
                Xk[:,ll] = numpy.reshape(self.emb_fftn(xl, Kd,range(0,dd)),
                                         (numpy.prod(Kd),),order='F')
        if self.debug==0:
            pass
        else:                
            checker(Xk,numpy.prod(Kd))        
        return Xk
    
    def Kd2Nd(self,Xk_all,weight_flag):
        
        st=self.st

        Nd = st['Nd']
        Kd = st['Kd']
        dd = len(Nd)
        if self.debug==0:
            pass
        else:
            checker(Xk_all,numpy.prod(Kd)) # check X of correct shape

        dims = numpy.shape(Xk_all)
        Lprod= numpy.prod(dims[1:]) # how many channel * slices

        if numpy.size(dims) == 1:
            Lprod = 1
        else:
            Lprod = dims[1]  
            
                  
        x=numpy.zeros(Kd+(Lprod,),dtype=dtype)  # [*Kd *L]

#         if Lprod > 1:

        Xk = numpy.reshape(Xk_all, Kd+(Lprod,) , order='F')
        
        for ll in xrange(0,Lprod):  # ll = 0, 1,... Lprod-1
            x[...,ll] =  self.emb_ifftn(Xk[...,ll],Kd,range(0,dd))#.flatten(order='F'))

        x = x[crop_slice_ind(Nd)]


        if weight_flag == 0:
            pass
            
        else: #weight_flag =1 scaling factors
            snc = st['sn'].conj()
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
        try:
#             print('using GPU')
#             print('using GPU interface')  
#             self.data_dev = self.ctx.to_device(output_x.astype(dtype))
#             self.myfft(self.res_dev, self.data_dev, -1)
#             output_x=self.res_dev.get() 
#             self.data_dev = 
            self.thr.to_device(output_x.astype(dtype), dest=self.data_dev)
            output_x=self.gpufftn(self.data_dev).get()
             
        except:
#         elif self.gpu_flag ==0:
#         elif self.pyfftw_flag == 1:
            try:
    #                 print('using pyfftw interface')
    #                 print('threads=',self.threads)
                output_x=pyfftw.interfaces.scipy_fftpack.fftn(output_x, output_dim, act_axes, 
                                                              threads=self.threads,overwrite_x=True)
            except: 
#         else:
    #                 print('using OLD interface')                
                output_x=scipy.fftpack.fftn(output_x, output_dim, act_axes)


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
        try:
#             print('using GPU')
#             print('using GPU interface')  
#             self.data_dev = self.ctx.to_device(output_x.astype(dtype))
#             self.myfft(self.res_dev, self.data_dev, -1)
#             output_x=self.res_dev.get() 
#             self.data_dev = 
            self.thr.to_device(output_x.astype(dtype), dest=self.data_dev)
            output_x=self.gpuifftn(self.data_dev).get()
             
        except:
#         elif self.pyfftw_flag == 1:
            try:
    #                 print('using pyfftw interface')
    #                 print('threads=',self.threads)
                output_x=pyfftw.interfaces.scipy_fftpack.ifftn(output_x, output_dim, act_axes, 
                                                              threads=self.threads,overwrite_x=True)
            except:
#         else: 
    #                 print('using OLD interface')                
                output_x=scipy.fftpack.ifftn(output_x, output_dim, act_axes)


        return output_x 