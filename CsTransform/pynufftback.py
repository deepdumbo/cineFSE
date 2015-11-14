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

First, see test_1D(),test_2D(), test_3D(), examples 

'''

from nufft import *
import numpy
import scipy.fftpack
import numpy.random
import sys
#import matplotlib
#import matplotlib.numerix
#import matplotlib.numerix.random_array
# from cx import *
# Add the ptdraft folder path to the sys.path list
# sys.path.append('..')


#import CsTransform.pynufft as pf
# try:
#     import pycuda.gpuarray as gpuarray
#     import pycuda.driver as cuda
#     import pycuda.autoinit
#     import pycuda.cumath as cumath
#     gpu_flag = 1
# except:
#     print "No PyOpenCL/PyFFT detected"  
#     gpu_flag = 0  
    
#cmap=matplotlib.cm.gray
#norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
# def create_krnl(self,u): # create the negative 3D laplacian __kernel of size u.shape[0:3]
#     
#     krnl = numpy.zeros(numpy.shape(u)[0:3],dtype=numpy.complex64) 
#     krnl[0,0,0]=6
#     krnl[1,0,0]=-1
#     krnl[0,1,0]=-1
#     krnl[0,0,1]=-1
#     krnl[-1,0,0]=-1
#     krnl[0,-1,0]=-1
#     krnl[0,0,-1]=-1
#     krnl = self.ifft_kkf(krnl)
# 
#     return krnl # (256*256*16) 

def output(cc):
    print('max',numpy.max(numpy.abs(cc[:])))
    
def Normalize(D):
    return D/numpy.max(numpy.abs(D[:]))
def checkmax(x):
    max_val = numpy.max(numpy.abs(x[:]))
    print( max_val)
    return max_val
def appendmat(input_array,L):
    if numpy.ndim(input_array) == 1:
        input_shape = numpy.size(input_array)
        input_shape = (input_shape,)
    else:
        input_shape = input_array.shape
        
        
    Lprod= numpy.prod(input_shape)
    output_array=numpy.copy(input_array)
    
    
    output_array=numpy.reshape(output_array,(Lprod,1),order='F')
    
    output_array=numpy.tile(output_array,(1,L))
    
    output_array=numpy.reshape(output_array,input_shape+(L,),order='F')
    
    
    return output_array
def freq_gradient(x):# zero frequency at centre
    grad_x = numpy.copy(x)
    
    dim_x=numpy.shape(x)
    print('freq_gradient shape',dim_x)
    for pp in range(0,dim_x[2]):
        grad_x[...,pp,:]=grad_x[...,pp,:] * (-2.0*numpy.pi*(pp -dim_x[2]/2.0 )) / dim_x[2]

    return grad_x
def freq_gradient_H(x):
    return -freq_gradient(x)
# def shrink_core(s,LMBD):
# #     LMBD = LMBD + 1.0e-15
#     s = numpy.sqrt(s).real
#     ss = numpy.maximum(s-LMBD , 0.0)/(s+1e-7) # shrinkage
#     return ss

# def shrink(dd, bb,LMBD):
# 
#     n_dims=numpy.shape(dd)[0]
#  
#     xx=()
# 
#     s = numpy.zeros(dd[0].shape)
#     for pj in range(0,n_dims):    
#         s = s+ (dd[pj] + bb[pj])*(dd[pj] + bb[pj]).conj()   
#     s = numpy.sqrt(s).real
#     ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-7) # shrinkage
#     for pj in range(0,n_dims): 
#         
#         xx = xx+ (ss*(dd[pj]+bb[pj]),)        
#     
#     return xx
def shrink(dd, bb,LMBD):

#     n_dims=numpy.shape(dd)[0]
    n_dims = len(dd)
#     print('n_dims',n_dims)
#     print('dd shape',numpy.shape(dd))
     
    xx=()
#     ss = shrink1(n_dims,dd,bb,LMBD)
#     xx = shrink2(n_dims,xx,dd,bb,ss)
#     return xx
# def shrink1(n_dims,dd,bb,LMBD):
    s = numpy.zeros(dd[0].shape)
    for pj in range(0,n_dims):    
        s = s+ (dd[pj] + bb[pj])*(dd[pj] + bb[pj]).conj()   
    s = numpy.sqrt(s).real
    ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-7) # shrinkage
#     return ss
# def shrink2(n_dims,xx,dd,bb,ss):    
    for pj in range(0,n_dims): 
        
        xx = xx+ (ss*(dd[pj]+bb[pj]),)        
    
    return xx
def TVconstraint(xx,bb):

    n_xx = len(xx)
    n_bb =  len(bb)
    #print('n_xx',n_xx)
    if n_xx != n_bb: 
        print('xx and bb size wrong!')
    
    cons_shape = numpy.shape(xx[0])
    cons=numpy.zeros(cons_shape,dtype=numpy.complex64)
    
    for jj in range(0,n_xx):

        cons =  cons + get_Diff_H( xx[jj] - bb[jj] ,  jj)
    

    return cons 
def Dx(u):
    rows=numpy.shape(u)[0]
#     print('ushape',u.shape)
#     print('rows,size=',rows)
#     print('u.dtype',u.dtype)

#     slice1 = range(0,rows)
#     slice2 = range(0,rows)-1
    ind1 = numpy.arange(0,rows)
    ind2 = numpy.roll(ind1,1) 
#     d = numpy.zeros(u.shape,u.dtype)    
#     d[1:-1] = u[1:-1]-u[0:-2]
#     d[0] = u[0]-u[-1] 
#     return d
    return u[ind1,...]-u[ind2,...]

# def Dx(u):
#     rows=numpy.size(u)
# #     print('ushape',u.shape)
# #     print('rows,size=',rows)
# #     print('u.dtype',u.dtype)
# 
# #     slice1 = range(0,rows)
# #     slice2 = range(0,rows)-1
#     ind1 = numpy.arange(0,rows)
#     ind2 = numpy.roll(ind1,1) 
# #     d = numpy.zeros(u.shape,u.dtype)    
# #     d[1:-1] = u[1:-1]-u[0:-2]
# #     d[0] = u[0]-u[-1] 
# #     return d
#     return u[ind1]-u[ind2]


def get_Diff_H(x,axs): # hermitian operator of get_Diff(x,axs)
    if axs > 0:
        # transpose the specified axs to 0
        # and use the case when axs == 0
        # then transpose back
        mylist=list(numpy.arange(0,x.ndim)) 
        (mylist[0], mylist[axs])=(mylist[axs],mylist[0])
        tlist=tuple(mylist[:])
        #=======================================================================
        dcxt=numpy.transpose(
                        get_Diff_H(numpy.transpose(x,tlist),0),
                                    tlist)   
    elif axs == 0:
#        x=x[::-1,...]
        #x=numpy.flipud(x)
        dcxt=-get_Diff(x, 0)
         
        #dcxt=numpy.flipud(dcxt)# flip along axes
#        dcxt=dcxt[::-1,...]       
        dcxt=numpy.roll(dcxt, axis=0, shift=-1) 
 
#        dcxt=-get_Diff(x,0)
#        dcxt=numpy.roll(dcxt,shift=2, axis=0)
    return dcxt
 
def get_Diff(x,axs):
    #calculate the 1D gradient of images
    if axs > 0:
        # transpose the specified axs to 0
        # and use the case when axs == 0
        # then transpose back
        mylist=list(numpy.arange(0,x.ndim)) 
        (mylist[0], mylist[axs])=(mylist[axs],mylist[0])
        tlist=tuple(mylist[:])
        #=======================================================================
        dcx=numpy.transpose(
                        get_Diff(numpy.transpose(x,tlist),0),
                                    tlist)         
    elif axs == 0:  
#         xshape=numpy.shape(x)
 
#        dcy=numpy.empty(numpy.shape(y),dtype=numpy.complex64)
#         ShapeProd=numpy.prod(xshape[1:])
#         x = numpy.reshape(x,xshape[0:1]+(ShapeProd,),order='F')
#         dcx=numpy.empty(numpy.shape(x),dtype=x.dtype)
         
#        dcx=Dx(x)
#         for ss in numpy.arange(0,ShapeProd):
#             dcx[:,ss] = Dx(x[:,ss]) # Diff operators
        dcx = Dx(x)
#            dcy[:,:,ll] = Dyt(y[:,:,ll]-by[:,:,ll]) # Hermitian of Diff operators
#         dcx=numpy.reshape(dcx, xshape ,order='F')
    return dcx 

def CombineMulti(multi_coil_data,axs):
    U=numpy.mean(multi_coil_data,axs)
    U = appendmat(U,multi_coil_data.shape[axs])

    return U

def CopySingle2Multi(single_coil_data,n_tail):
   
    U=numpy.copy(single_coil_data)
    
    U = appendmat(U, n_tail)
    
    return U

class pynufft(nufft):
    def __init__(self,om, Nd, Kd,Jd):
        nufft.__init__(self,om, Nd, Kd,Jd)
        self.st['q'] = self.st['p']
        self.st['q'] = self.st['q'].conj().multiply(self.st['q'])
        self.st['q'] = self.st['q'].sum(0)
        self.st['q'] = numpy.array(self.st['q'] )
        self.st['q']=numpy.reshape(self.st['q'],(numpy.prod(self.st['Kd']),1),order='F').real

#         self.st['q']=self.st['p'].getH().dot(self.st['p']).diagonal()  # slow version  
#  
#         self.st['q']=numpy.reshape(self.st['q'],(numpy.prod(self.st['Kd']),1),order='F')
#         

                
    def forwardbackward(self,x):

        st=self.st
        Nd = st['Nd']
#         Kd = st['Kd'] # unused
#         dims = numpy.shape(x) #unused
        dd = numpy.size(Nd)
    #    print('in nufft, dims:dd',dims,dd)
    #    print('ndim(x)',numpy.ndim(x[:,1]))
        # checker
        checker(x,Nd)
        
        if numpy.ndim(x) == dd:
            Lprod = 1
            x = numpy.reshape(x,Nd+(1,),order='F')
        elif numpy.ndim(x) > dd: # multi-channel data
            Lprod = numpy.size(x)/numpy.prod(Nd)
            Lprod = Lprod.astype(int)
        '''
        Now transform Nd grids to Kd grids(not be reshaped)
        '''
        Xk = self.Nd2Kd(x,0) #

        for ii in range(0,Lprod):
        
            Xk[...,ii] = st['q'][...,0]*Xk[...,ii]
        '''
        Now transform Kd grids to Nd grids(not be reshaped)
        '''
        x= self.Kd2Nd(Xk,0) #
        
        checker(x,Nd) # check output
        return x    


    def inverse(self,data, mu, LMBD, gamma, nInner, nBreg): # main function of solver
        self.f = data
        self.mu = mu
        self.LMBD = LMBD
        self.gamma = gamma
        self.nInner= nInner
        self.nBreg= nBreg
#         print(numpy.size(data) , self.st['M'] )
                
        if numpy.size(data) == self.st['M']:
            self.st['senseflag'] = 0
#             print(numpy.size(data) )
            print('turn-off sense recon')
        
        try: 
            if  self.st['senseflag']==0:
                self.st = self.__create_mask()                
                pass
            else:
                raise
        except:
            self.LMBD=self.LMBD*1.0
    
            self.st['senseflag']=0 # turn-off sense, to get sensemap
            
            #precompute highly constrainted images to guess the sensitivity maps 
            (u0,dump)=self.__kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 1,2)
    #===============================================================================
    # mask
    #===============================================================================
            self.st = self.__create_mask()
            if numpy.size(u0.shape) > numpy.size(self.st['Nd']):
                for pp in range(2,numpy.size(u0.shape)):
                    self.st['mask'] = appendmat(self.st['mask'],u0.shape[pp] )
    #===============================================================================
            
            #estimate sensitivity maps by divided by rms images
            self.st = self.__make_sense(u0) # setting up sense map in st['sensemap']
    
            self.st['senseflag']=1 # turn-on sense, to get sensemap
      
    
            #scale back the __constrainted factor LMBD
            self.LMBD=self.LMBD/1.0
        #CS reconstruction
        
        (self.u, self.u_stack)=self.__kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 
                          self.nInner,self.nBreg)

        
#         for jj in range(0,self.u.shape[-1]):
#             self.u[...,jj] = self.u[...,jj]*(self.st['sn']**0.7)# rescale the final image intensity
#         
        if self.u.shape[-1] == 1:
            if numpy.ndim(self.u) != numpy.ndim(self.st['Nd']):  # alwasy true?          
                self.u = self.u[...,0]

#         self.u = Normalize(self.u)

        
        return self.u 
#        self.u=1.5*self.u/numpy.max(numpy.real(self.u[:]))
    def __kernel(self, f, st , mu, LMBD, gamma, nInner, nBreg):
        L= numpy.size(f)/st['M'] 
        image_dim=st['Nd']+(L,)
         
        if numpy.ndim(f) == 1:# preventing row vector
            f=numpy.reshape(f,(numpy.shape(f)[0],1),order='F')
#         f0 = numpy.copy(f) # deep copy to prevent scope f0 to f
#         unused
#        u = numpy.zeros(image_dim,dtype=numpy.complex64)



    #===========================================================================
    # check whether sense is used
    # if senseflag == 0, create an all-ones mask
    # if sensflag size is wrong, create an all-ones mask (shouldn't occur)
    #===========================================================================
        if st['senseflag'] == 0:
            st['sensemap'] = numpy.ones(image_dim,dtype=numpy.complex64)
        elif numpy.shape(st['sensemap']) != image_dim: #(shouldn't occur)
            st['sensemap'] = numpy.ones(image_dim,dtype=numpy.complex64)
        else:
            pass # correct, use existing sensemap
    #=========================================================================
    # check whether mask is used  
    #=========================================================================
#         if st.has_key('mask'):
        if 'mask' in st:
            if numpy.shape(st['mask']) != image_dim:
                st['mask'] = numpy.ones(image_dim,dtype=numpy.complex64)
        else:
            st['mask'] = numpy.ones(image_dim,dtype=numpy.complex64)

    #===========================================================================
    # update sensemap so we don't need to add ['mask'] in the iteration
    #===========================================================================
        st['sensemap'] = st['sensemap']*st['mask']  
 


        #=======================================================================
        # RTR: k-space sampled density
        #      only diagonal elements are relevant (on k-space grids)
        #=======================================================================
        RTR=self.__create_kspace_sampling_density()

#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to __constraint
#===============================================================================
        uker = self.__create_laplacian_kernel()

        #=======================================================================
        # uker: deconvolution kernel in k-space, 
        #       which will be divided in k-space in iterations
        #=======================================================================

    #===========================================================================
    # initial estimation u, u0, uf
    #===========================================================================

        u = st['sensemap'].conj()*(self.backward(f))
#         c = numpy.max(numpy.abs(u.flatten())) # Rough coefficient

        for jj in range(0,u.shape[-1]):
            u[...,jj] = u[...,jj]/self.st['sn']
            
        print('senseflag',st['senseflag'])
        if st['senseflag'] == 1:
            
            u=CombineMulti(u,-1)[...,0:1] # summation of multicoil images
            

        u0 = numpy.copy(u)
        self.thresh_scale= numpy.max(numpy.abs(u0[:]))           
        self.u0=numpy.copy(u0)
#        else:
#            print('existing self.u, so we use previous u and u0')
#            u=numpy.copy(self.u) # using existing initial values
#            u0=numpy.copy(self.u0)
#        if st['senseflag'] == 1:
#            print('u.shape line 305',u.shape)
#            u == u[...,0:1]
#            print('u.shape line 307',u.shape)
#===============================================================================
    #   Now repeat the uker to L slices e.g. uker=512x512x8 (if L=8)
    #   useful for later calculation 
#===============================================================================
        #expand 2D/3D kernel to desired dimension of kspace

        uker = self.__expand_deconv_kernel_dimension(uker,u.shape[-1])

        RTR = self.__expand_RTR(RTR,u.shape[-1])

        uker = self.mu*RTR - LMBD*uker + gamma
        print('uker.shape line 319',uker.shape)
                
        (xx,bb,dd)=self.__make_split_variables(u)        

        uf = numpy.copy(u0)  # only used for ISRA, written here for generality 
          
        murf = numpy.copy(u) # initial values 
#    #===============================================================================
        u_stack = numpy.empty(st['Nd']+(nBreg,),dtype=numpy.complex)
        self.err =1.0e+13
        u_k_1=0
        for outer in numpy.arange(0,nBreg):
            for inner in numpy.arange(0,nInner):
                # update u
                print('iterating',[inner,outer])
                #===============================================================
#                 update u  # simple k-space deconvolution to guess initial u
                u = self.__update_u(murf,u,uker,xx,bb)
                for jj in range(0,u.shape[-1]):
                    u[...,jj] = u[...,jj]*(self.st['sn']**1)
                    # Temporally scale the image for softthresholding                
                
                c = numpy.max(numpy.abs(u.flatten())) # Rough coefficient
                # to correct threshold of nonlinear shrink
                
            #===================================================================
            # # update d
            #===================================================================
            #===================================================================
            # Shrinkage: remove tiny values "in somewhere sparse!"
            # dx+bx should be sparse! 
            #===================================================================
            # shrinkage 
            #===================================================================
                dd=self.__update_d(u,dd)

                xx=self.__shrink( dd, bb, c/LMBD/numpy.sqrt(numpy.prod(st['Nd'])))
                #===============================================================
            #===================================================================
            # # update b
            #===================================================================

                bb=self.__update_b(bb, dd, xx)
                for jj in range(0,u.shape[-1]):
                    u[...,jj] = u[...,jj]/(self.st['sn']**1)
                    #  Temporally scale the image for softthresholding  
#            if outer < nBreg: # do not update in the last loop
            if st['senseflag']== 1:
                u = appendmat(u[...,0],L)
         
            (u, murf, uf, u_k_1)=self.__external_update(u, uf, u0, u_k_1, outer) # update outer Split_bregman
#             err = (checkmax(tmpuf) -checkmax(u0) )/checkmax(u0)
#             r = u0  - tmpuf
# #         r = u0  - tmpuf
#             p = r 
# #         err = (checkmax(tmpuf)- checkmax(u0))/checkmax(u0) 
#             err= numpy.abs(err)
#             print('err',err,self.err)
# #         if (err < self.err):
# #             uf = uf+p*err*0.1            
#             if err < self.err:
#                 uf = uf + p*err*0.1*(outer+1)
#                 self.err = err
# 
#                 u_k_1 = u
#             else: 
#                 err = self.err
#                 print('no function')
#                 u = u_k_1
#             murf = uf 
                           
            if st['senseflag']== 1:
                u = u[...,0:1]
                murf = murf[...,0:1]

            u_stack[...,outer] = (u[...,0]*(self.st['sn']**1))
#            u_stack[...,outer] =u[...,0] 
        for jj in range(0,u.shape[-1]):
            u[...,jj] = u[...,jj]*(self.st['sn']**1)# rescale the final image intensity

        return (u,u_stack)
  
    def __update_u(self,murf,u,uker,xx,bb):
        #print('inside __update_u')
#        checkmax(u)
#        checkmax(murf)
#        rhs = self.mu*murf + self.LMBD*self.get_Diff(x,y,bx,by) + self.gamma
        #=======================================================================
        # Trick: make "llist" for numpy.transpose 
        mylist = tuple(numpy.arange(0,numpy.ndim(xx[0]))) 
        tlist = mylist[1::-1]+mylist[2:] 
        #=======================================================================
        # update the right-head side terms
        rhs = (self.mu*murf + 
               self.LMBD*self.__constraint(xx,bb) +      
               self.gamma * u) 
        
        rhs = rhs * self.st['mask'][...,0:u.shape[-1]]
 
#        rhs=Normalize(rhs)
        #=======================================================================
#         Trick: make "flist" for fftn 
        flist = mylist[:-1:1]    
            
        u = self.__k_deconv(rhs, uker,self.st,flist,mylist)
#         print('max rhs u',numpy.max(numpy.abs(rhs[:])),numpy.max(numpy.abs(u[:])))
#         print('max,q',numpy.max(numpy.abs(self.st['q'][:])))
#        for jj in range(0,1):
#            u = u - 0.1*(self.k_deconv(u, 1.0/(RTR+self.LMBD*uker+self.gamma),self.st,flist,mylist) - rhs 
#                         )
#        checkmax(u)
#        checkmax(rhs)
#        checkmax(murf)
        
        #print('leaving __update_u')
        return u # normalization    
    def __k_deconv(self, u,uker,st,flist,mylist):
        u0=numpy.copy(u)
        
        u=u*st['mask'][...,0:u.shape[-1]]
        
#            u=scipy.fftpack.fftn(u, st['Kd'],flist)

        U=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
        
        for pj in range(0,u.shape[-1]):
            U[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) / uker[...,pj] # deconvolution
            U[...,pj]=self.emb_ifftn(U[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
         
        u = U[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
        
        # optional: one- additional Conjugated step to ensure the quality
        
#         for pp in range(0,3):
#             u = self.__cg_step(u0,u,uker,st,flist,mylist)
#         
        
        u=u*st['mask'][...,0:u.shape[-1]]
      
        return u
    def __cg_step(self, rhs, u, uker, st,flist,mylist):
        u=u#*st['mask'][...,0:u.shape[-1]]
#            u=scipy.fftpack.fftn(u, st['Kd'],flist)
        AU=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
#        print('U.shape. line 446',U.shape)
#        print('u.shape. line 447',u.shape)
        for pj in range(0,u.shape[-1]):
            AU[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
            AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
            
         
        ax0 = AU[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
          

        u=u#*st['mask'][...,0:u.shape[-1]]        
        r  = rhs - ax0
        p = r
        for running_count in range(0,1):
            
            upper_inner = r.conj()*r
            upper_inner = numpy.sum(upper_inner[:])
            
            AU=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
    #        print('U.shape. line 446',U.shape)
    #        print('u.shape. line 447',u.shape)
            for pj in range(0,u.shape[-1]):
                AU[...,pj]=self.emb_fftn(p[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
                AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
                
             
            Ap = AU[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
            
            lower_inner =  p.conj()*Ap
            lower_inner = numpy.sum(lower_inner[:])
            
            alfa_k =  upper_inner/ lower_inner
#             alfa_k = alfa_k*0.6
            
            u = u + alfa_k * p
            
            r2 = r - alfa_k *Ap
            beta_k = numpy.sum((r2.conj()*r2)[:])/numpy.sum((r.conj()*r)[:])
          
            r = r2
            
            p = r + beta_k*p
            
        return u
        
    def __constraint(self,xx,bb):
        '''
        include TVconstraint and others
        '''
        cons = TVconstraint(xx[:],bb[:])
        
        return cons
    
    def __shrink(self,dd,bb,thrsld):
        '''
        soft-thresholding the edges
        
        '''
        output_xx=shrink( dd[:], bb[:], thrsld)# 3D thresholding 
        
        return output_xx
        
    def __make_split_variables(self,u):
        x=numpy.zeros(u.shape)
        y=numpy.zeros(u.shape)
        bx=numpy.zeros(u.shape)
        by=numpy.zeros(u.shape)
        dx=numpy.zeros(u.shape)
        dy=numpy.zeros(u.shape)        
        xx= (x,y)
        bb= (bx,by)
        dd= (dx,dy)
        return(xx,bb,dd)
    def __make_sense(self,u0):
        st=self.st
        L=numpy.shape(u0)[-1]
        u0dims= numpy.ndim(u0)
        
        if u0dims-1 >0:
            rows=numpy.shape(u0)[0]
            dpss_rows = numpy.kaiser(rows, 100)     
            dpss_rows = numpy.fft.fftshift(dpss_rows)
            dpss_rows[3:-3] = 0.0
            dpss_fil = dpss_rows
            print('dpss shape',dpss_fil.shape)
        if u0dims-1 > 1:
                              
            cols=numpy.shape(u0)[1]
            dpss_cols = numpy.kaiser(cols, 100)            
            dpss_cols = numpy.fft.fftshift(dpss_cols)
            dpss_cols[3:-3] = 0.0
            
            dpss_fil = appendmat(dpss_fil,cols)
            dpss_cols  = appendmat(dpss_cols,rows)

            dpss_fil=dpss_fil*numpy.transpose(dpss_cols,(1,0))
            print('dpss shape',dpss_fil.shape)
        if u0dims-1 > 2:
            
            zag = numpy.shape(u0)[2]
            dpss_zag = numpy.kaiser(zag, 100)            
            dpss_zag = numpy.fft.fftshift(dpss_zag)
            dpss_zag[3:-3] = 0.0
            dpss_fil = appendmat(dpss_fil,zag)
                     
            dpss_zag = appendmat(dpss_zag,rows)
            
            dpss_zag = appendmat(dpss_zag,cols)
            
            dpss_fil=dpss_fil*numpy.transpose(dpss_zag,(1,2,0)) # low pass filter
            print('dpss shape',dpss_fil.shape)
        #dpss_fil=dpss_fil / 10.0
        
        rms=numpy.sqrt(numpy.mean(u0*u0.conj(),-1)) # Root of sum square
        st['sensemap']=numpy.ones(numpy.shape(u0),dtype=numpy.complex64)
        print('sensemap shape',st['sensemap'].shape, L)
        print('u0shape',u0.shape,rms.shape)

        #    print('L',L)
        #    print('rms',numpy.shape(rms))
        for ll in numpy.arange(0,L):
            st['sensemap'][...,ll]=(u0[...,ll]+1e-16)/(rms+1e-16)
            
            print('sensemap shape',st['sensemap'].shape, L)
            print('rmsshape', rms.shape) 
            st['sensemap'][...,ll] = scipy.fftpack.fftn(st['sensemap'][...,ll], 
                                              st['sensemap'][...,ll].shape,
                                                    range(0,numpy.ndim(st['sensemap'][...,ll]))) 
            st['sensemap'][...,ll] = st['sensemap'][...,ll] * dpss_fil
            st['sensemap'][...,ll] = scipy.fftpack.ifftn(st['sensemap'][...,ll], 
                                              st['sensemap'][...,ll].shape,
                                                    range(0,numpy.ndim(st['sensemap'][...,ll])))                             
#             st['sensemap'][...,ll]=scipy.fftpack.ifftn(scipy.fftpack.fftn(st['sensemap'][...,ll])*dpss_fil)
#         st['sensemap'] = Normalize(st['sensemap'])
        return st

    def __create_kspace_sampling_density(self):
            #=======================================================================
            # RTR: k-space sampled density
            #      only diagonal elements are relevant (on k-space grids)
            #=======================================================================
        RTR=self.st['q'] # see __init__() in class "nufft"
        
        return RTR 
    def __create_laplacian_kernel(self):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to constraint
#===============================================================================
        uker = numpy.zeros(self.st['Kd'][:],dtype=numpy.complex64)
        n_dims= numpy.size(self.st['Nd'])
        if n_dims == 1:
            uker[0] = -2.0
            uker[1] = 1.0
            uker[-1] = 1.0
        elif n_dims == 2:
            uker[0,0] = -4.0
            uker[1,0] = 1.0
            uker[-1,0] = 1.0
            uker[0,1] = 1.0
            uker[0,-1] = 1.0
        elif n_dims == 3:  
            uker[0,0,0] = -6.0
            uker[1,0,0] = 1.0
            uker[-1,0,0] = 1.0
            uker[0,1,0] = 1.0
            uker[0,-1,0] = 1.0
            uker[0,0,1] = 1.0
            uker[0,0,-1] = 1.0                      

        uker =self.emb_fftn(uker, self.st['Kd'][:], range(0,numpy.ndim(uker)))
        return uker
    def __expand_deconv_kernel_dimension(self, uker, L):

#         if numpy.size(self.st['Kd']) > 2:
#             for dd in range(2,numpy.size(self.st['Kd'])):
#                 uker = appendmat(uker,self.st['Kd'][dd])
        
        uker = appendmat(uker,L)
        
        
        return uker
    def __expand_RTR(self,RTR,L):
#         if numpy.size(self.st['Kd']) > 2:
#             for dd in range(2,numpy.size(self.st['Kd'])):
#                 RTR = appendmat(RTR,self.st['Kd'][dd])
                
        RTR= numpy.reshape(RTR,self.st['Kd'],order='F')
        
        RTR = appendmat(RTR,L)

        return RTR

    def __update_d(self,u,dd):

        out_dd = ()
        for jj in range(0,len(dd)) :
            out_dd = out_dd  + (get_Diff(u,jj),)

        return out_dd
    
    def __update_b(self, bb, dd, xx):
        ndims=len(bb)
        cc=numpy.empty(bb[0].shape)
        out_bb=()
        for pj in range(0,ndims):
            cc=bb[pj]+dd[pj]-xx[pj]
            out_bb=out_bb+(cc,)

        return out_bb
  

    def __create_mask(self):
        st=self.st

        st['mask']=numpy.ones(st['Nd'],dtype=numpy.float64)
        n_dims= numpy.size(st['Nd'])
 
        sp_rat =0.0
        for di in range(0,n_dims):
            sp_rat = sp_rat + (st['Nd'][di]/2)**2
  
        x = numpy.ogrid[[slice(0, st['Nd'][_ss]) for _ss in range(0,n_dims)]]

        tmp = 0
        for di in range(0,n_dims):
            tmp = tmp + ( x[di] - st['Nd'][di]/2 )**2
        indx = tmp/sp_rat >=1.0/n_dims
            
        st['mask'][indx] =0.0       
         
  
        return st   
    
    def __external_update(self,u, uf, u0, u_k_1, outer): # overload the update function

        
        tmpuf=self.st['sensemap'].conj()*(
                self.forwardbackward(
                        u*self.st['sensemap']))

        if self.st['senseflag'] == 1:
            tmpuf=CombineMulti(tmpuf,-1)
        err = (checkmax(tmpuf) -checkmax(u0) )/checkmax(u0)
        r = u0  - tmpuf
#         r = u0  - tmpuf
        p = r 
#         err = (checkmax(tmpuf)- checkmax(u0))/checkmax(u0) 
        err= numpy.abs(err)
        print('err',err,self.err)
#         if (err < self.err):
#             uf = uf+p*err*0.1            
        if numpy.abs(err) < numpy.abs(self.err):
            uf = uf + p*err*0.1*(outer+1)
            self.err = err

            u_k_1 = u
        else: 
            err = self.err
            print('no function')
            u = u_k_1
        murf = uf 


        print('leaving ext_update') 

        return (u, murf, uf, u_k_1)   
def show_3D():
    import mayavi.mlab
    raw = numpy.load('phantom_3D_128_128_128.npy')    
    reconreal = numpy.load('reconreal.npy')
    blurreal = numpy.load('blurreal.npy')
    reconreal[0:,0:80,0:64]=0
    raw[0:,0:80,0:64]=0  
    blurreal[0:,0:80,0:64]=0
    mayavi.mlab.contour3d(raw, contours=4, transparent=True)
    mayavi.mlab.show()
    mayavi.mlab.contour3d(blurreal, contours=4, transparent=True)
    mayavi.mlab.show()
    mayavi.mlab.contour3d(reconreal, contours=4, transparent=True)
    mayavi.mlab.show()
            
def test_3D():

#    cm = matplotlib.cm.gray
#   load raw data, which is 3D shapp-logan phantom
    raw = numpy.load('phantom_3D_128_128_128.npy')
#     numpy.save('testfile.npy',raw)
#     raw = numpy.load('testfile.npy')
# demonstrate the 64th slice
#    matplotlib.pyplot.imshow(raw[:,:,64],cmap=cm)
#    matplotlib.pyplot.show()
    print('max.image',numpy.max(raw[:]))
# load 3D k-space trajectory (sparse)   
    om = numpy.loadtxt('om3D2.txt')
# image dimension is 3D isotropic 
    Nd=(128,128,128)
    Kd=(128,128,128)
# Note: sparse sampling works best for Jd = 1    
    Jd=(1,1,1)
#     create Nufft Object
    MyNufftObj = pynufft(om, Nd, Kd, Jd)
# create data    
    K_data=MyNufftObj.forward(raw)
#     regridding and blurred images
    image_blur = MyNufftObj.backward(K_data)[...,0]
    
# turn off sense recon because it is not necessary        
    MyNufftObj.st['senseflag']=0
#  Now doing the reconstruction
    image1 = MyNufftObj.inverse(K_data,1.0, 0.05, 0.001, 3,20)
    
#    matplotlib.pyplot.subplot(2,3,1)
#    matplotlib.pyplot.imshow(raw[:,:,64],cmap=cm,interpolation = 'nearest')
#    matplotlib.pyplot.subplot(2,3,2)
#    matplotlib.pyplot.imshow(image_blur[:,:,64].real,cmap=cm,interpolation = 'nearest')
#    matplotlib.pyplot.subplot(2,3,3)
#    matplotlib.pyplot.imshow((image1[:,:,64].real),cmap=cm,interpolation = 'nearest')
#    matplotlib.pyplot.subplot(2,3,4)
#    matplotlib.pyplot.imshow(raw[:,:,96],cmap=cm,interpolation = 'nearest')
#    matplotlib.pyplot.subplot(2,3,5)
#    matplotlib.pyplot.imshow(image_blur[:,:,96].real,cmap=cm,interpolation = 'nearest')
#    matplotlib.pyplot.subplot(2,3,6)
#    matplotlib.pyplot.imshow((image1[:,:,96].real),cmap=cm,interpolation = 'nearest')    
#    matplotlib.pyplot.show() 
#    numpy.save('blurreal.npy',image_blur.real)
#    numpy.save('reconreal.npy',image1.real)
#     mayavi.mlab.imshow()  
def test_2D():

    import numpy 
#    import matplotlib#.pyplot
#    cm = matplotlib.cm.gray
    # load example image    

    image = numpy.loadtxt('phantom_256_256.txt') 
    image[128,128]= 1.0   
    Nd =(256,256) # image space size
    Kd =(512,512) # k-space size   
    Jd =(6,6) # interpolation size
    
    # load k-space points
    om = numpy.loadtxt('om.txt')
    
    #create object
    NufftObj = pynufft(om, Nd,Kd,Jd)   
    
    # simulate "data"
    data= NufftObj.forward(image )

    
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 

    image_recon = NufftObj.inverse(data, 1.0, 0.05, 0.01,3, 20)
    image_blur = NufftObj.backward(data)
    image_recon = Normalize(image_recon)

#    matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
#    matplotlib.pyplot.show()

#    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
#    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0e-1)
    # display images
#    matplotlib.pyplot.subplot(2,2,1)
#    matplotlib.pyplot.imshow(image,
#                             norm = norm,cmap =cm,interpolation = 'nearest')
#    matplotlib.pyplot.title('true image')   
#    matplotlib.pyplot.subplot(2,2,3)
#    matplotlib.pyplot.imshow(image_recon.real,
#                             norm = norm,cmap= cm,interpolation = 'nearest')
#    matplotlib.pyplot.title('recovered image')    
#    matplotlib.pyplot.subplot(2,2,2)
#    matplotlib.pyplot.imshow(image_blur[:,:,0].real,
#                             norm = norm,cmap= cm,interpolation = 'nearest')
#    matplotlib.pyplot.title('blurred image') 
#    matplotlib.pyplot.subplot(2,2,4)
#    matplotlib.pyplot.imshow(image_recon.real-image,
#                             norm = norm,cmap= cm,interpolation = 'nearest')
#    matplotlib.pyplot.title('residual error') 
    
#    matplotlib.pyplot.show() 
def test_1D():
# import several modules
    import numpy 
#    import matplotlib#.pyplot

#create 1D curve from 2D image
    image = numpy.loadtxt('phantom_256_256.txt') 
    image = image[:,128]
#determine the location of samples
    om = numpy.loadtxt('om1D.txt')
    om = numpy.reshape(om,(numpy.size(om),1),order='F')
# reconstruction parameters
    Nd =(256,) # image space size
    Kd =(256,) # k-space size
     
    Jd =(1,) # interpolation size
# initiation of the object
    NufftObj = pynufft(om, Nd,Kd,Jd)
# simulate "data"
    data= NufftObj.forward(image )
#adjoint(reverse) of the forward transform
    image_blur= NufftObj.backward(data)[:,0]
#inversion of data
    image_recon = NufftObj.inverse(data, 1.0, 1, 0.001,15,16)

#Showing histogram of sampling locations
#    matplotlib.pyplot.hist(om,20)
#    matplotlib.pyplot.title('histogram of the sampling locations')
#    matplotlib.pyplot.show()
#show reconstruction
#    matplotlib.pyplot.subplot(2,2,1)

#    matplotlib.pyplot.plot(image)
#    matplotlib.pyplot.title('original') 
#    matplotlib.pyplot.ylim([0,1]) 
           
#    matplotlib.pyplot.subplot(2,2,3)    
#    matplotlib.pyplot.plot(image_recon.real)
#    matplotlib.pyplot.title('recon') 
#    matplotlib.pyplot.ylim([0,1])
            
#    matplotlib.pyplot.subplot(2,2,2)

#    matplotlib.pyplot.plot(image_blur.real) 
#    matplotlib.pyplot.title('blurred')
#    matplotlib.pyplot.subplot(2,2,4)

#    matplotlib.pyplot.plot(image_recon.real - image) 
#    matplotlib.pyplot.title('residual')
#     matplotlib.pyplot.subplot(2,2,4)
#     matplotlib.pyplot.plot(numpy.abs(data))  
#    matplotlib.pyplot.show()  

            
if __name__ == '__main__':
    test_1D()
    test_2D()
    test_3D()
#    show_3D()
#     import cProfile
#     cProfile.run('test_2D()')    
#     cProfile.run('test_3D()')
