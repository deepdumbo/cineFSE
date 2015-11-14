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
try:
    from nufft import *
    import numpy
    import scipy.fftpack
#     import numpy.random
    import matplotlib.pyplot
    import matplotlib.cm
    # import matplotlib.numerix 
    # import matplotlib.numerix.random_array
#     import sys
#     import utils
except:
    print('faile to import modules')
    print('numpy, scipy, matplotlib are required')
    raise
dtype = numpy.complex64
#     
# try:
#     import llvm
# except:
#     print('llvm not supported')

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
# import utils    
cmap=matplotlib.cm.gray
norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
# try:
#     from numba import autojit  
# 
# except:
#     print('numba not supported')
 
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
# @autojit
def tailor_fftn(X):
    X = scipy.fftpack.fftshift(scipy.fftpack.fftn(scipy.fftpack.fftshift((X))))
    return X
def tailor_ifftn(X):
    X = scipy.fftpack.fftshift(scipy.fftpack.ifftn(scipy.fftpack.ifftshift(X)))
    return X
def output(cc):
    print('max',numpy.max(numpy.abs(cc[:])))
    
def Normalize(D):
    return D/numpy.max(numpy.abs(D[:]))
def checkmax(x,dbg):
    max_val = numpy.max(numpy.abs(x[:]))
    if dbg ==0:
        pass
    else:
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
#     print('freq_gradient shape',dim_x)
    for pp in xrange(0,dim_x[2]):
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
#     for pj in xrange(0,n_dims):    
#         s = s+ (dd[pj] + bb[pj])*(dd[pj] + bb[pj]).conj()   
#     s = numpy.sqrt(s).real
#     ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-7) # shrinkage
#     for pj in xrange(0,n_dims): 
#         
#         xx = xx+ (ss*(dd[pj]+bb[pj]),)        
#     
#     return xx
def shrink2(dd,bb,ss,n_dims):
    xx = tuple(ss*(dd[pj]+bb[pj]) for pj in xrange(0,n_dims))
    return xx 

def shrink1(dd,bb,n_dims):
#     s = numpy.zeros(numpy.shape(dd[0]),dtype = numpy.float)
#     c = numpy.empty_like(s) # only real
#     for pj in xrange(0,n_dims):  
#         c  = (dd[pj] + bb[pj]).real
#         s = s+ c**2
    s = sum((dd[pj] + bb[pj]).real**2 for pj in xrange(0,n_dims))
    s = s**0.5
    return s.real
def shrink(dd, bb,LMBD):

#     n_dims=numpy.shape(dd)[0]
    n_dims = len(dd)

    s = shrink1(dd,bb,n_dims)
    ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-15)# shrinkage
       
    xx = shrink2(dd,bb,ss,n_dims)
    
    return xx

def TVconstraint(xx,bb):

    try:    
        n_xx = len(xx)
#         n_bb =  len(bb)
#         cons_shape = numpy.shape(xx[0])
#         cons=numpy.zeros(cons_shape,dtype=numpy.complex64)
        
    #     cons = sum( get_Diff_H( xx[jj] - bb[jj] ,  jj) 
    #                 for jj in xrange(0,n_xx))
        
#         for jj in xrange(0,n_xx):
#             cons =  cons + get_Diff_H( xx[jj] - bb[jj] ,  jj)
        cons = sum(get_Diff_H( xx[jj] - bb[jj] ,  jj) for jj in xrange(0,n_xx))
    except:
        n_xx = len(xx)
        n_bb =  len(bb)
        if n_xx != n_bb: 
            print('xx and bb size wrong!')    

    return cons 



# def Dx(u):
#     shapes = numpy.shape(u)
#     rows=shapes[0]
#     ind1 = xrange(0,rows)
#     ind2 = numpy.roll(ind1,1,axis=0) 
#     u2= u[ind2,...]
#     u2[...]= u[...] - u2[...]
#     return u2#u[ind1,...]-u[ind2,...]

def Dx(u):
    u2=numpy.concatenate((u,u[0:1,...]),axis=0)
    u2=numpy.roll(u2,1,axis=0)
    u2=numpy.diff(u2,n=1,axis=0)
    return u2


def get_Diff_H(x,axs): # hermitian operator of get_Diff(x,axs)
    if axs > 0:
        # transpose the specified axs to 0
        # and use the case when axs == 0
        # then transpose back
        mylist=list(xrange(0,x.ndim)) 
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
        mylist=list(xrange(0,x.ndim)) 
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
#         for ss in xrange(0,ShapeProd):
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
#         self.st['q'] = self.st['p']
#         self.st['q'] = self.st['q'].conj().multiply(self.st['q'])
#         self.st['q'] = self.st['q'].sum(0)
#         self.st['q'] = numpy.array(self.st['q'] )
#         self.st['q']=numpy.reshape(self.st['q'],(numpy.prod(self.st['Kd']),1),order='F').real

#         self.st['q']=self.st['p'].getH().dot(self.st['p']).diagonal()  # slow version  
#  
#         self.st['q']=numpy.reshape(self.st['q'],(numpy.prod(self.st['Kd']),1),order='F')
#         
    def forwardbackward(self,x):
        if self.cuda_flag == 0:
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
    
            for ii in xrange(0,Lprod):
            
                Xk[...,ii] = st['q'][...,0]*Xk[...,ii]
            '''
            Now transform Kd grids to Nd grids(not be reshaped)
            '''
            x= self.Kd2Nd(Xk,0) #
            
            checker(x,Nd) # check output
            return x    
        elif self.cuda_flag == 1:
            return self.forwardbackward_gpu(x)
    def gpu_k_modulate(self):
        try:
            self.myfft(self.data_dev, self.data_dev,inverse=False)
            self.data_dev=self.W_dev*self.data_dev
            self.myfft(self.data_dev, self.data_dev,inverse=True)
            return 0
        except: 
            return 1       
#     def gpu_k_demodulate(self):
#         try:
#             self.myfft(self.data_dev, self.data_dev,inverse=False)
#             self.data_dev=self.data_dev/self.W_dev
#             self.myfft(self.data_dev, self.data_dev,inverse=True)
#             print('inside gpu_k_demodulate')
#             return 0
#         except: 
#             return 1   
    def Nd2KdWKd2Nd_gpu(self,x, weight_flag):
        '''
        Now transform Nd grids to Kd grids(not be reshaped)
        
        '''
        #print('661 x.shape',x.shape)        
#         x is Nd Lprod
        st=self.st
        Nd = st['Nd']
        Kd = st['Kd']
#         dims = numpy.shape(x)
#         dd = numpy.size(Nd)
        Lprod = numpy.shape(x)[-1]

        if self.debug==0:
            pass
        else:
            checker(x,Nd)
            
        snc = st['sn']
        output_x=numpy.zeros(Kd, dtype=numpy.complex64)
#         self.W_dev = self.thr.to_device(self.W.T.astype(dtype))
        for ll in xrange(0,Lprod):

            if weight_flag == 0:
                pass
            else:
                x[...,ll] = x[...,ll] * snc # scaling factors
            
            output_x=output_x*0.0
        
            output_x[crop_slice_ind(x[...,ll].shape)] = x[...,ll]
            self.data_dev = self.thr.to_device(output_x.astype(dtype))

            if self.gpu_k_modulate()==0:
                pass
            else:
                print('gpu_k_modulate error')
                break
            x[...,ll]=self.data_dev.get()[crop_slice_ind(Nd)]

            if weight_flag == 0:
                pass
            else: #weight_flag =1 scaling factors
                x[...,ll] = x[...,ll]*snc.conj() #% scaling factors

        if self.debug==0:
            pass # turn off checker
        else:
            checker(x,Nd) # checking size of x divisible by Nd
        return x    
    def forwardbackward_gpu(self,x):
#         print('inside forwardbackward_gpu ')
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
        elif numpy.ndim(x) > dd: # multi-channel data
            Lprod = numpy.size(x)/numpy.prod(Nd)
            Lprod = Lprod.astype(int)
        x = numpy.reshape(x,Nd+(Lprod,),order='F')
        '''
        Now transform Nd grids to Kd grids(not be reshaped)
        '''
        x = self.Nd2KdWKd2Nd_gpu(x,0) #

#         for ii in xrange(0,Lprod):
# #             tmp_Xk = self.Nd2Kd_gpu(x[...,ii],0)
#             Xk[...,ii] = st['q'][...,0]*Xk[...,ii]
#             x[...,ii]= self.Kd2Nd_gpu(tmp_Xk,0)
        '''
        Now transform Kd grids to Nd grids(not be reshaped)
        '''
#         x= self.Kd2Nd(Xk,0) #
        
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
                self.st = self._create_mask()                
                pass
            else:
                raise
        except:
            self.LMBD=self.LMBD*1.0
    
            self.st['senseflag']=0 # turn-off sense, to get sensemap
            
            #precompute highly constrainted images to guess the sensitivity maps 
            u0 =self._kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 1,2)
    #===============================================================================
    # mask
    #===============================================================================
            self.st = self._create_mask()
            if numpy.size(u0.shape) > numpy.size(self.st['Nd']):
                for pp in xrange(2,numpy.size(u0.shape)):
                    self.st['mask'] = appendmat(self.st['mask'],u0.shape[pp] )
                    self.st['mask2'] = appendmat(self.st['mask2'],u0.shape[pp] )
    #===============================================================================
            
            #estimate sensitivity maps by divided by rms images
            self.st['sensemap'] = self._make_sense(u0) # setting up sense map in st['sensemap']
    
#             for jj in xrange(0,self.st['sensemap'].shape[-1]):
#                 matplotlib.pyplot.subplot(2,2,jj+1)
#                 matplotlib.pyplot.imshow(self.st['sensemap'][...,jj].imag)
#             matplotlib.pyplot.show()
    
            self.st['senseflag']=1 # turn-on sense, to get sensemap
      
    
            #scale back the _constrainted factor LMBD
            self.LMBD=self.LMBD*1.0
        #CS reconstruction
        
        self.u = self._kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 
                          self.nInner,self.nBreg)

        
#         for jj in xrange(0,self.u.shape[-1]):
#             self.u[...,jj] = self.u[...,jj]*(self.st['sn']**0.7)# rescale the final image intensity
#         
        if self.u.shape[-1] == 1:
            if numpy.ndim(self.u) != numpy.ndim(self.st['Nd']):  # alwasy true?          
                self.u = self.u[...,0]

#         self.u = Normalize(self.u)

        
        return self.u 
#        self.u=1.5*self.u/numpy.max(numpy.real(self.u[:]))
    def _kernel(self, f, st , mu, LMBD, gamma, nInner, nBreg):
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
        if 'mask' in st: # condition in second step
            if (numpy.shape(st['mask']) != image_dim) :
                st['mask'] = numpy.reshape(st['mask'],image_dim,order='F')
#                 numpy.ones(image_dim,dtype=numpy.complex64)
        else: # condition in first step
            st['mask'] = numpy.ones(image_dim,dtype=numpy.complex64)
            
        if 'mask2' in st:
            if numpy.shape(st['mask2']) != image_dim:
                st['mask2'] = numpy.reshape(st['mask2'],image_dim,order='F')
        else:
            st['mask2'] = numpy.ones(image_dim,dtype=numpy.complex64)

    #===========================================================================
    # update sensemap so we don't need to add ['mask'] in the iteration
    #===========================================================================
        st['sensemap'] = st['sensemap']*st['mask']  
 


        #=======================================================================
        # RTR: k-space sampled density
        #      only diagonal elements are relevant (on k-space grids)
        #=======================================================================
        RTR=self._create_kspace_sampling_density()

#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to _constraint
#===============================================================================
        uker = self._create_laplacian_kernel()

        #=======================================================================
        # uker: deconvolution kernel in k-space, 
        #       which will be divided in k-space in iterations
        #=======================================================================

    #===========================================================================
    # initial estimation u, u0, uf
    #===========================================================================

        u = self.backward(f)*self.st['sensemap'].conj()#/(1e-10+self.st['sensemap'].conj())#st['sensemap'].conj()*(self.backward(f))
#         c = numpy.max(numpy.abs(u.flatten())) # Rough coefficient

        for jj in xrange(0,u.shape[-1]):
            u[...,jj] = u[...,jj]/self.st['sn']
        if self.debug ==0:
            pass
        else:  
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

        uker = self._expand_deconv_kernel_dimension(uker,u.shape[-1])

        RTR = self._expand_RTR(RTR,u.shape[-1])

        uker = self.mu*RTR - LMBD*uker + gamma
        if self.debug ==0:
            pass
        else:
            print('uker.shape line 319',uker.shape)
                
        (xx,bb,dd)=self._make_split_variables(u)        

        uf = numpy.copy(u0)  # only used for ISRA, written here for generality 
        alpha = 0.01
        u = u*alpha
        murf = numpy.copy(u) # initial values 
#    #===============================================================================
#         u_stack = numpy.empty(st['Nd']+(nBreg,),dtype=numpy.complex)
        self.err =1.0e+13
        u_k_1=0
        for outer in xrange(0,nBreg):
            for inner in xrange(0,nInner):
                # update u
                if self.debug==0:
                    pass
                else:
                    print('iterating',[inner,outer])
                #===============================================================
#                 update u  # simple k-space deconvolution to guess initial u
                u = self._update_u(murf,u,uker,xx,bb)
                for jj in xrange(0,u.shape[-1]):
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
                dd=self._update_d(u,dd)

                xx=self._shrink( dd, bb, c/LMBD/(numpy.prod(st['Nd'])**(1.0/len(st['Nd']))))
                #===============================================================
            #===================================================================
            # # update b
            #===================================================================

                bb=self._update_b(bb, dd, xx)
                for jj in xrange(0,u.shape[-1]):
                    u[...,jj] = u[...,jj]/(self.st['sn']**1)
                    #  Temporally scale the image for softthresholding  
#            if outer < nBreg: # do not update in the last loop
            if st['senseflag']== 1:
                u = appendmat(u[...,0],L)
         
            (u, murf, uf, u_k_1)=self._external_update(u, uf, u0, u_k_1, outer) # update outer Split_bregman

                           
            if st['senseflag']== 1:
                u = u[...,0:1]
                murf = murf[...,0:1]

#             u_stack[...,outer] = (u[...,0]*(self.st['sn']))
#            u_stack[...,outer] =u[...,0] 
        if self.st['senseflag'] == 1:
            fermi = scipy.fftpack.fftshift( self.st['fermi'] )
        
        for jj in xrange(0,u.shape[-1]):
            if self.st['senseflag'] == 1:
                u[...,jj] = scipy.fftpack.ifftn(scipy.fftpack.fftn(u[...,jj])*fermi ) # apply GE's fermi filter
                u[...,jj] = u[...,jj]*(self.st['sn'])*self.st['mask2'][...,jj]# rescale the final image intensity
            else:
                u[...,jj] = u[...,jj]*(self.st['sn'])*self.st['mask2'][...,jj]
#         matplotlib.pyplot.imshow(self.st['mask2'][:,:,0].real)
#         matplotlib.pyplot.show()
        return u#(u,u_stack)
  
    def _update_u(self,murf,u,uker,xx,bb):
        #print('inside _update_u')
#        checkmax(u)
#        checkmax(murf)
#        rhs = self.mu*murf + self.LMBD*self.get_Diff(x,y,bx,by) + self.gamma
        #=======================================================================
        # Trick: make "llist" for numpy.transpose 
        mylist = tuple(xrange(0,numpy.ndim(xx[0]))) 
        tlist = mylist[1::-1]+mylist[2:] 
        #=======================================================================
        # update the right-head side terms
        rhs = (self.mu*murf + 
               self.LMBD*self._constraint(xx,bb) +      
               self.gamma * u) 
        
        rhs = rhs * self.st['mask'][...,0:u.shape[-1]]
 
#        rhs=Normalize(rhs)
        #=======================================================================
#         Trick: make "flist" for fftn 
        flist = mylist[:-1:1]    
            
        u = self._k_deconv(rhs, uker,self.st,flist,mylist)
#         print('max rhs u',numpy.max(numpy.abs(rhs[:])),numpy.max(numpy.abs(u[:])))
#         print('max,q',numpy.max(numpy.abs(self.st['q'][:])))
#        for jj in xrange(0,1):
#            u = u - 0.1*(self.k_deconv(u, 1.0/(RTR+self.LMBD*uker+self.gamma),self.st,flist,mylist) - rhs 
#                         )
#        checkmax(u)
#        checkmax(rhs)
#        checkmax(murf)
        
        #print('leaving _update_u')
        return u # normalization    
    def _k_deconv(self, u,uker,st,flist,mylist):
        u0=numpy.copy(u)
        
        u=u*st['mask'][...,0:u.shape[-1]]
        
#            u=scipy.fftpack.fftn(u, st['Kd'],flist)
###
#         if self.cuda_flag == 1:
#             tmpU=numpy.zeros(st['Kd'],dtype=u.dtype)
#             self.W_dev = self.thr.to_device((uker[...,0]).astype(numpy.complex64))
#             for pj in xrange(0,u.shape[-1]):
#                 
#                 tmpU=tmpU*0.0
#             
#                 tmpU[crop_slice_ind(st['Nd'])] = u[...,pj]
#                 self.data_dev = self.thr.to_device(tmpU.astype(numpy.complex64))
#                 
#     #             self.myfft(self.data_dev,  self.data_dev,inverse=False)
#     #             self.data_dev=self.W_dev*self.data_dev
#     #             self.myfft(self.data_dev, self.data_dev,inverse=True)
#                 if self.gpu_k_demodulate()==0:
#                     pass
#                 else:
#                     print('gpu_k_modulate error')
#                     break
#                 u[...,pj]=self.data_dev.get()[crop_slice_ind(st['Nd'])]
# #             u = U[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
#             self.W_dev = self.thr.to_device(1.0/uker[...,0].astype(numpy.complex64))
#         elif self.cuda_flag == 0:
 
        U=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
        for pj in xrange(0,u.shape[-1]):
                     
            U[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) 
            U[...,pj]=U[...,pj]/uker[...,pj] # deconvolution
            U[...,pj]=self.emb_ifftn(U[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) 
          
        u = U[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
        
        # optional: one- additional Conjugated step to ensure the quality
        
#         for pp in xrange(0,3):
#             u = self._cg_step(u0,u,uker,st,flist,mylist)
#         
        
        u=u*st['mask'][...,0:u.shape[-1]]
      
        return u
    def _cg_step(self, rhs, u, uker, st,flist,mylist):
        u=u#*st['mask'][...,0:u.shape[-1]]
#            u=scipy.fftpack.fftn(u, st['Kd'],flist)
        AU=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
#        print('U.shape. line 446',U.shape)
#        print('u.shape. line 447',u.shape)
        for pj in xrange(0,u.shape[-1]):
            AU[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
            AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) 
            
         
        ax0 = AU[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
          

        u=u#*st['mask'][...,0:u.shape[-1]]        
        r  = rhs - ax0
        p = r
        for running_count in xrange(0,1):
            
            upper_inner = r.conj()*r
            upper_inner = numpy.sum(upper_inner[:])
            
            AU=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
    #        print('U.shape. line 446',U.shape)
    #        print('u.shape. line 447',u.shape)
            for pj in xrange(0,u.shape[-1]):
                AU[...,pj]=self.emb_fftn(p[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
                AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) 
                
             
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
        
    def _constraint(self,xx,bb):
        '''
        include TVconstraint and others
        '''
        cons = TVconstraint(xx[:],bb[:])
        
        return cons
    
    def _shrink(self,dd,bb,thrsld):
        '''
        soft-thresholding the edges
        
        '''
        output_xx=shrink( dd[:], bb[:], thrsld)# 3D thresholding 
        
        return output_xx
        
    def _make_split_variables(self,u):
        n_dims = len(self.st['Nd'])
        xx = ()
        bb = ()
        dd = ()
        for jj in xrange(0,n_dims):
            x=numpy.zeros(u.shape)
            bx=numpy.zeros(u.shape)
            dx=numpy.zeros(u.shape)
            xx = xx + (x,)
            bb = bb + (bx,)
            dd = dd + (dx,)
        
#         x=numpy.zeros(u.shape)
#         y=numpy.zeros(u.shape)
#         bx=numpy.zeros(u.shape)
#         by=numpy.zeros(u.shape)
#         dx=numpy.zeros(u.shape)
#         dy=numpy.zeros(u.shape)        
#         xx= (x,y)
#         bb= (bx,by)
#         dd= (dx,dy)
        return(xx,bb,dd)
    def _extract_svd(self,input_stack,L):
        C= numpy.copy(input_stack) # temporary array
        print('size of input_stack', numpy.shape(input_stack))
        C=C/numpy.max(numpy.abs(C))

        reps_acs = 16
        mysize = 16
        K= 5 # rank of 10 prevent singular? artifacts(certain disruption)
        half_mysize = mysize/2
        dimension = numpy.ndim(C) -1 # collapse coil dimension
        if dimension == 1:
            tmp_stack = numpy.empty((mysize,),dtype = numpy.complex64) 
            svd_size = mysize
            C_size = numpy.shape(C)[0]
            data = numpy.empty((svd_size,L*reps_acs),dtype=numpy.complex64)
#             for jj in xrange(0,L):
#                 C[:,jj]=tailor_fftn(C[:,jj])
#                 for kk in xrange(0,reps_acs):
#                     tmp_stack = numpy.reshape(tmp_stack,(svd_size,),order = 'F')
#                     data[:,jj] = numpy.reshape(tmp_stack,(svd_size,),order = 'F') 
        elif dimension == 2:
            tmp_stack = numpy.empty((mysize,mysize,),dtype = numpy.complex64) 
            svd_size = mysize**2
            data = numpy.empty((svd_size,L*reps_acs),dtype=numpy.complex64)
            C_size = numpy.shape(C)[0:2]
            for jj in xrange(0,L):
#                 matplotlib.pyplot.imshow(C[...,jj].real)
#                 matplotlib.pyplot.show()
#                 tmp_pt=(C_size[0]-reps_acs)/2
                C[:,:,jj]=tailor_fftn(C[:,:,jj])
                for kk in xrange(0,reps_acs):
                    a=numpy.mod(kk,reps_acs**0.5)
                    b=kk/(reps_acs**0.5)
                    tmp_stack = C[C_size[0]/2-half_mysize-(reps_acs**0.5)/2+a : C_size[0]/2+half_mysize-(reps_acs**0.5)/2+a,
                                  C_size[1]/2-half_mysize-(reps_acs**0.5)/2+b : C_size[1]/2+half_mysize-(reps_acs**0.5)/2+b,jj]
                    data[:,jj*reps_acs+kk] = numpy.reshape(tmp_stack,(svd_size,),order = 'F')
                                        
        elif dimension == 3:
            tmp_stack = numpy.empty((mysize,mysize,mysize),dtype = numpy.complex64) 
            svd_size = mysize**3
            data = numpy.empty((svd_size,L),dtype=numpy.complex64)
            C_size = numpy.shape(C)[0:3]
            for jj in xrange(0,L):
                C[:,:,:,jj]=tailor_fftn(C[:,:,:,jj])
                tmp_stack= C[C_size[0]/2-half_mysize:C_size[0]/2+half_mysize,
                                C_size[1]/2-half_mysize:C_size[1]/2+half_mysize,
                                C_size[2]/2-half_mysize:C_size[2]/2+half_mysize,
                                jj]
                data[:,jj] = numpy.reshape(tmp_stack,(svd_size,),order = 'F')   
                
#         OK, data is the matrix of size (mysize*n, L) for SVD
        import scipy.linalg      
        (s_blah,vh_blah) = scipy.linalg.svd(data)[1:3]

        for jj in xrange(0,numpy.size(s_blah)): # 
            if s_blah[jj] > 0.1*s_blah[0]: # 10% of maximum singular value to decide the rank
                K = jj+1
#                 pass
            else:
                break

        v_blah =vh_blah.conj().T

        C = C*0.0 # now C will be used as the output stack
        V_para = v_blah[:,0:K]
        print('shape of V_para',numpy.shape(V_para))
        V_para = numpy.reshape(V_para,(reps_acs**0.5,reps_acs**0.5,L, K),order='F')
         
        C2 = numpy.zeros((C.shape[0],C.shape[1],L,K),dtype=numpy.complex64)
        for jj in xrange(0,L): # coils  
            for kk in xrange(0,K): # rank
                C2[C.shape[0]/2-reps_acs**0.5/2:C.shape[0]/2+reps_acs**0.5/2,
                   C.shape[1]/2-reps_acs**0.5/2:C.shape[1]/2+reps_acs**0.5/2,
                   jj,kk]=V_para[:,:,jj,kk]
                C2[:,:,jj,kk]=tailor_fftn(C2[:,:,jj,kk])
#         C_value = numpy.empty_like(C)
        
        for mm in xrange(0,C.shape[0]): # dim 0  
            for nn in xrange(0,C.shape[1]): # dim 1
                G =   C2[mm,nn,:,:].T # Transpose (non-conjugated) of G
                Gh = G.conj().T # hermitian
                g = numpy.dot(Gh,G)  #construct g matrix for eigen-decomposition  
                w,v = numpy.linalg.eig(g) # eigen value:w, eigen vector: v
                
                ind = numpy.argmax(numpy.abs(w)) # find the maximum 
                the_eig = numpy.abs(w[ind]) # find the abs of maximal eigen value
                ref_angle=(numpy.sum(v[:,ind])/(numpy.abs(numpy.sum(v[:,ind]))))
                v[:,ind] = v[:,ind]/ref_angle # correct phase by summed value
                C[mm,nn,:] = v[:,ind]*the_eig 
#         for jj in xrange(0,L):   
#             matplotlib.pyplot.subplot(2,2,jj+1)  
#             matplotlib.pyplot.imshow((C[...,jj].real))
#         matplotlib.pyplot.show()   
#         for jj in xrange(0,L):   
#             matplotlib.pyplot.subplot(2,2,jj+1)  
#             matplotlib.pyplot.imshow((input_stack[...,jj].real))
#         matplotlib.pyplot.show() 
        return C/numpy.max(numpy.abs(C)) # normalize the coil sensitivities 
  
    def _make_sense(self,u0):
#         st=self.st
        L=numpy.shape(u0)[-1]
        try:
            coil_sense = self._extract_svd(u0,L)
#             st['sensemap']=u0
#             for jj in xrange(0,L):   
#                 matplotlib.pyplot.subplot(2,2,jj+1)  
#                 matplotlib.pyplot.imshow((st['sensemap'][...,jj].real))
#             matplotlib.pyplot.show() 
            return coil_sense
        except:
            u0dims= numpy.ndim(u0)
            beta=100
            
            if u0dims-1 >0:
                rows=numpy.shape(u0)[0]
                dpss_rows = numpy.kaiser(rows, beta)     
                dpss_rows = numpy.fft.fftshift(dpss_rows)
                dpss_rows[3:-3] = 0.0
                dpss_fil = dpss_rows
                if self.debug==0:
                    pass
                else:            
                    print('dpss shape',dpss_fil.shape)
            if u0dims-1 > 1:
                                  
                cols=numpy.shape(u0)[1]
                dpss_cols = numpy.kaiser(cols, beta)            
                dpss_cols = numpy.fft.fftshift(dpss_cols)
                dpss_cols[3:-3] = 0.0
                
                dpss_fil = appendmat(dpss_fil,cols)
                dpss_cols  = appendmat(dpss_cols,rows)
    
                dpss_fil=dpss_fil*numpy.transpose(dpss_cols,(1,0))
                if self.debug==0:
                    pass
                else:            
                    print('dpss shape',dpss_fil.shape)
            if u0dims-1 > 2:
                
                zag = numpy.shape(u0)[2]
                dpss_zag = numpy.kaiser(zag, beta)            
                dpss_zag = numpy.fft.fftshift(dpss_zag)
                dpss_zag[3:-3] = 0.0
                dpss_fil = appendmat(dpss_fil,zag)
                         
                dpss_zag = appendmat(dpss_zag,rows)
                
                dpss_zag = appendmat(dpss_zag,cols)
                
                dpss_fil=dpss_fil*numpy.transpose(dpss_zag,(1,2,0)) # low pass filter
                if self.debug==0:
                    pass
                else:            
                    print('dpss shape',dpss_fil.shape)
            #dpss_fil=dpss_fil / 10.0
    
    #         rms = numpy.mean((coil_sense),-1)
    
    #         rms = rms/numpy.max()
            coil_sense = numpy.copy(u0)

            rms=(numpy.mean( (coil_sense*coil_sense.conj()),-1))**0.5 # Root of sum square / OLD 
              
            for ll in xrange(0,L):
    #             st['sensemap'][...,ll]=(u0[...,ll]+1e-16)/(rms+1e-16) # / OLD
                coil_sense[...,ll]=(coil_sense[...,ll]+1e-16)/(rms+1e-16) # need SVD  
     
     
#             st['sensemap']=coil_sense
             
    #         st['sensemap']=numpy.empty(numpy.shape(u0),dtype=numpy.complex64)
            if self.debug==0:
                pass
            else:        
#                 print('sensemap shape',st['sensemap'].shape, L)
                print('u0shape',u0.shape,rms.shape)
     
     
            for ll in xrange(0,L):
    #             st['sensemap'][...,ll]=(u0[...,ll]+1e-16)/(rms+1e-16) # / OLD
    #             st['sensemap'][...,ll]=coil_sense[...,ll] # need SVD  
                if self.debug==0:
                    pass
                else:
                    print('sensemap shape',coil_sense.shape, L)
                    print('rmsshape', rms.shape) 
                coil_sense[...,ll]=  scipy.fftpack.fftshift(coil_sense[...,ll])
                if self.pyfftw_flag == 1:
                    if self.debug==0:
                        pass
                    else:                
                        print('USING pyfftw and thread is = ',self.threads)
                    coil_sense[...,ll] = pyfftw.interfaces.scipy_fftpack.fftn(coil_sense[...,ll])#, 
    #                                                   coil_sense[...,ll].shape,
    #                                                         range(0,numpy.ndim(coil_sense[...,ll])), 
    #                                                         threads=self.threads) 
                    coil_sense[...,ll] = coil_sense[...,ll] * dpss_fil
                    coil_sense[...,ll] = pyfftw.interfaces.scipy_fftpack.ifftn(coil_sense[...,ll])#, 
    #                                                   coil_sense[...,ll].shape,
    #                                                         range(0,numpy.ndim(coil_sense[...,ll])), 
    #                                                         threads=self.threads)                                                 
                else:                                                                    
                    coil_sense[...,ll] = scipy.fftpack.fftn(coil_sense[...,ll])#, 
    #                                                   coil_sense[...,ll].shape,
    #                                                         range(0,numpy.ndim(coil_sense[...,ll]))) 
                    coil_sense[...,ll] = coil_sense[...,ll] * dpss_fil
                    coil_sense[...,ll] = scipy.fftpack.ifftn(coil_sense[...,ll])#, 
    #                                                   coil_sense[...,ll].shape,
    #                                                         range(0,numpy.ndim(coil_sense[...,ll])))                             
    #             coil_sense[...,ll]=scipy.fftpack.ifftn(scipy.fftpack.fftn(coil_sense[...,ll])*dpss_fil)
    #         coil_sense = Normalize(coil_sense)
                coil_sense[...,ll]=  scipy.fftpack.ifftshift(coil_sense[...,ll])
                return coil_sense

    def _create_kspace_sampling_density(self):
            #=======================================================================
            # RTR: k-space sampled density
            #      only diagonal elements are relevant (on k-space grids)
            #=======================================================================
        RTR=self.st['q'] # see __init__() in class "nufft"
        
        return RTR 
    def _create_laplacian_kernel(self):
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
    def _expand_deconv_kernel_dimension(self, uker, L):

#         if numpy.size(self.st['Kd']) > 2:
#             for dd in xrange(2,numpy.size(self.st['Kd'])):
#                 uker = appendmat(uker,self.st['Kd'][dd])
        
        uker = appendmat(uker,L)
        
        
        return uker
    def _expand_RTR(self,RTR,L):
#         if numpy.size(self.st['Kd']) > 2:
#             for dd in xrange(2,numpy.size(self.st['Kd'])):
#                 RTR = appendmat(RTR,self.st['Kd'][dd])
                
        RTR= numpy.reshape(RTR,self.st['Kd'],order='F')
        
        RTR = appendmat(RTR,L)

        return RTR

    def _update_d(self,u,dd):
        out_dd = tuple(get_Diff(u,jj) for jj in xrange(0,len(dd)))
#         out_dd = ()
#         for jj in xrange(0,len(dd)) :
#             out_dd = out_dd  + (get_Diff(u,jj),)
        
        return out_dd
    
    def _update_b(self, bb, dd, xx):
        ndims=len(bb)
        cc=numpy.empty(bb[0].shape)
        out_bb=()
        for pj in xrange(0,ndims):
            cc=bb[pj]+dd[pj]-xx[pj]
            out_bb=out_bb+(cc,)

        return out_bb
  

    def _create_mask(self):
        st=self.st

        st['mask']=numpy.ones(st['Nd'],dtype=numpy.float32)
#         st['mask2']=numpy.ones(st['Nd'],dtype=numpy.float32)
        
        n_dims= numpy.size(st['Nd'])
 
        sp_rat =0.0
        for di in xrange(0,n_dims):
            sp_rat = sp_rat + (st['Nd'][di]/2)**2
  
        sp_rat = sp_rat**0.5
        x = numpy.ogrid[[slice(0, st['Nd'][_ss]) for _ss in xrange(0,n_dims)]]

        tmp = 0
        for di in xrange(0,n_dims):
            tmp = tmp + ( (x[di] - st['Nd'][di]/2.0)/(st['Nd'][di]/2.0) )**2
        
        tmp = (1.0*tmp)**0.5
        
        indx = tmp >=1.05
    
                
        st['mask'][indx] =0.0       
#         st['mask'] = 1.0/(1.0+numpy.exp( (tmp-1.05)/0.00001))
        st['fermi'] =1.0/(1.0+numpy.exp( (tmp-1.0)/(10.0/st['Nd'][0])))
        st['mask2'] =1.0/(1.0+numpy.exp( (tmp-1.05)/0.03))
#         matplotlib.pyplot.imshow( indx)
#         matplotlib.pyplot.show() 
#         matplotlib.pyplot.imshow( tmp)
#         matplotlib.pyplot.show()         
#         matplotlib.pyplot.imshow( st['mask'].real)
#         matplotlib.pyplot.show()
#         matplotlib.pyplot.imshow( st['mask2'].real)
#         matplotlib.pyplot.show()
#         matplotlib.pyplot.imshow( st['fermi'].real)
#         matplotlib.pyplot.show()
        return st   
    
    def _external_update(self,u, uf, u0, u_k_1, outer): # overload the update function

        
        tmpuf=(self.forwardbackward(
                        u*self.st['sensemap']))*(self.st['sensemap'].conj())

        if self.st['senseflag'] == 1:
            tmpuf=CombineMulti(tmpuf,-1)
        err = (checkmax(tmpuf,self.debug) -checkmax(u0,self.debug) )/checkmax(u0,self.debug)
#         err = (checkmax(tmpuf - u0,self.debug) )/checkmax(u0,self.debug)
        r = u0  - tmpuf
#         r = u0  - tmpuf
        p = r 
#         err = (checkmax(tmpuf)- checkmax(u0))/checkmax(u0) 
        err= numpy.abs(err)
#         if self.debug==0:
#             pass
#         else:        
        print('err',err,self.err)
#         if (err < self.err):
#             uf = uf+p*err*0.1            
        if numpy.abs(err) < numpy.abs(self.err):
            uf = uf + p#*err*(outer+1)
            self.err = err

            u_k_1 = u
        else: 
            err = self.err
            if self.debug==0:
                pass
            else:            
                print('no function')
            u = u_k_1
        murf = uf 

        if self.debug==0:
            pass
        else:
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

    cm = matplotlib.cm.gray
#   load raw data, which is 3D shapp-logan phantom
    raw = numpy.load('phantom_3D_128_128_128.npy')
#     numpy.save('testfile.npy',raw)
#     raw = numpy.load('testfile.npy')
# demonstrate the 64th slice
    matplotlib.pyplot.imshow(raw[:,:,64],cmap=cm)
    matplotlib.pyplot.show()
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
    MyNufftObj.st['senseflag']=1
#  Now doing the reconstruction

#     import pp
#     job_server = pp.Server()
# 
#     f1=job_server.submit(MyNufftObj.inverse,(K_data, 1.0, 0.1, 0.01,3, 5),
#                          modules = ('numpy','pyfftw','pynufft'),globals=globals())
#     f2=job_server.submit(MyNufftObj.inverse,(numpy.sqrt(K_data)*10+(0.0+0.1j), 1.0, 0.05, 0.01,3, 20),
#                          modules = ('numpy','pyfftw','pynufft'),globals=globals())

#     image1 = f1()    
#     image2 = f2()
    image1=MyNufftObj.inverse(K_data, 1.0, 0.1, 0.01,10,10)


    
#     image1 = MyNufftObj.inverse(K_data,1.0, 0.05, 0.001, 1,10)
#     matplotlib.pyplot.subplot(2,3,1)
#     matplotlib.pyplot.imshow(raw[:,:,64],cmap=cm,interpolation = 'nearest')
#     matplotlib.pyplot.subplot(2,3,2)
#     matplotlib.pyplot.imshow(image_blur[:,:,64].real,cmap=cm,interpolation = 'nearest')
#     matplotlib.pyplot.subplot(2,3,3)
#     matplotlib.pyplot.imshow((image2[:,:,64].real),cmap=cm,interpolation = 'nearest')
#     matplotlib.pyplot.subplot(2,3,4)
#     matplotlib.pyplot.imshow(raw[:,:,96],cmap=cm,interpolation = 'nearest')
#     matplotlib.pyplot.subplot(2,3,5)
#     matplotlib.pyplot.imshow(image_blur[:,:,96].real,cmap=cm,interpolation = 'nearest')
#     matplotlib.pyplot.subplot(2,3,6)
#     matplotlib.pyplot.imshow((image2[:,:,96].real),cmap=cm,interpolation = 'nearest')    
#     matplotlib.pyplot.show() 
    
    matplotlib.pyplot.subplot(2,3,1)
    matplotlib.pyplot.imshow(raw[:,:,64],cmap=cm,interpolation = 'nearest')
    matplotlib.pyplot.subplot(2,3,2)
    matplotlib.pyplot.imshow(image_blur[:,:,64].real,cmap=cm,interpolation = 'nearest')
    matplotlib.pyplot.subplot(2,3,3)
    matplotlib.pyplot.imshow((image1[:,:,64].real),cmap=cm,interpolation = 'nearest')
    matplotlib.pyplot.subplot(2,3,4)
    matplotlib.pyplot.imshow(raw[:,:,96],cmap=cm,interpolation = 'nearest')
    matplotlib.pyplot.subplot(2,3,5)
    matplotlib.pyplot.imshow(image_blur[:,:,96].real,cmap=cm,interpolation = 'nearest')
    matplotlib.pyplot.subplot(2,3,6)
    matplotlib.pyplot.imshow((image1[:,:,96].real),cmap=cm,interpolation = 'nearest')    
    matplotlib.pyplot.show() 
    numpy.save('blurreal.npy',image_blur.real)
    numpy.save('reconreal.npy',image1.real)
#     mayavi.mlab.imshow()  
def test_2D():

    import numpy 
    import matplotlib#.pyplot
    cm = matplotlib.cm.gray
    # load example image    

    image = numpy.loadtxt('phantom_256_256.txt') 
    image[128,128]= 1.0  
#     import scipy.misc 
#     image = scipy.misc.imresize(image,Nd)
    Nd =(256,256) # image space size
    Kd =(512,512) # k-space size   
    Jd =(6,6) # interpolation size
    
    # load k-space points
    om = numpy.loadtxt('om.txt')
    
    #create object
    NufftObj = pynufft(om, Nd,Kd,Jd)   
    NufftObj.st['senseflag']=1
    # simulate "data"
    data= NufftObj.forward(image )

    
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 

    image_recon = NufftObj.inverse(data, 1.0, 0.4, 0.01,10, 10)
    image_blur = NufftObj.backward(data)
    image_recon = Normalize(image_recon)

    matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
    matplotlib.pyplot.show()

    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0e-1)
    # display images
    matplotlib.pyplot.subplot(2,2,1)
    matplotlib.pyplot.imshow(image,
                             norm = norm,cmap =cm,interpolation = 'nearest')
    matplotlib.pyplot.title('true image')   
    matplotlib.pyplot.subplot(2,2,3)
    matplotlib.pyplot.imshow(image_recon.real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('recovered image')    
    matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.imshow(image_blur[:,:,0].real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('blurred image') 
    matplotlib.pyplot.subplot(2,2,4)
    matplotlib.pyplot.imshow(image_recon.real-image,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('residual error') 
    
    matplotlib.pyplot.show() 
def test_1D():
# import several modules
    import numpy 
    import matplotlib#.pyplot

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
    matplotlib.pyplot.hist(om,20)
    matplotlib.pyplot.title('histogram of the sampling locations')
    matplotlib.pyplot.show()
#show reconstruction
    matplotlib.pyplot.subplot(2,2,1)

    matplotlib.pyplot.plot(image)
    matplotlib.pyplot.title('original') 
    matplotlib.pyplot.ylim([0,1]) 
           
    matplotlib.pyplot.subplot(2,2,3)    
    matplotlib.pyplot.plot(image_recon.real)
    matplotlib.pyplot.title('recon') 
    matplotlib.pyplot.ylim([0,1])
            
    matplotlib.pyplot.subplot(2,2,2)

    matplotlib.pyplot.plot(image_blur.real) 
    matplotlib.pyplot.title('blurred')
    matplotlib.pyplot.subplot(2,2,4)

    matplotlib.pyplot.plot(image_recon.real - image) 
    matplotlib.pyplot.title('residual')
#     matplotlib.pyplot.subplot(2,2,4)
#     matplotlib.pyplot.plot(numpy.abs(data))  
    matplotlib.pyplot.show()  

# def test_Dx():
#     u = numpy.ones((128,128,128,1),dtype = numpy.complex64)
def test_2D_multiprocessing():

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
    
    
    NufftObj = pynufft(om, Nd,Kd,Jd)   
    NewObj = copy.deepcopy(NufftObj)
    # simulate "data"
    data= NufftObj.forward(image )
#     data2=data.copy()
#     data2 =numpy.sqrt(data2)*10+(0.0+0.1j)
    
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 

    import pp
    job_server = pp.Server()

    f1=job_server.submit(NewObj.inverse,(data, 1.0, 0.05, 0.01,3, 20),
                         modules = ('numpy','pyfftw','pynufft'),globals=globals())
    f2=job_server.submit(NewObj.inverse,(numpy.sqrt(data)*10+(0.0+0.1j), 1.0, 0.05, 0.01,3, 20),
                         modules = ('numpy','pyfftw','pynufft'),globals=globals())
    image_recon = f1()    
    image_recon2 = f2()
    
#     image_recon = NewObj.inverse(data, 1.0, 0.05, 0.01,3, 20)
    
    
    image_blur = NufftObj.backward(data)
    image_recon = Normalize(image_recon)

    matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
    matplotlib.pyplot.show()

    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0e-1)
    # display images
    matplotlib.pyplot.subplot(2,2,1)
    matplotlib.pyplot.imshow(image,
                             norm = norm,cmap =cm,interpolation = 'nearest')
    matplotlib.pyplot.title('true image')   
    matplotlib.pyplot.subplot(2,2,3)
    matplotlib.pyplot.imshow(image_recon.real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('recovered image')    
    matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.imshow(image_blur[:,:,0].real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('blurred image') 
    matplotlib.pyplot.subplot(2,2,4)
    matplotlib.pyplot.imshow(image_recon.real-image,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('residual error') 
    
    matplotlib.pyplot.show()      
    
    matplotlib.pyplot.subplot(2,2,1)
    matplotlib.pyplot.imshow(image,
                             norm = norm,cmap =cm,interpolation = 'nearest')
    matplotlib.pyplot.title('true image')   
    matplotlib.pyplot.subplot(2,2,3)
    matplotlib.pyplot.imshow(image_recon2.real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('recovered image')    
    matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.imshow(image_blur[:,:,0].real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('blurred image') 
    matplotlib.pyplot.subplot(2,2,4)
    matplotlib.pyplot.imshow(image_recon.real-image,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('residual error') 
    
    matplotlib.pyplot.show()  
if __name__ == '__main__':
    import cProfile
    test_1D()
    test_2D()
    test_3D()
#     show_3D()
#     test_Dx()

#     cProfile.run('test_2D()')    
#     cProfile.run('test_2D_multiprocessing()')