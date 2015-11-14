'''
package docstring
author: Jyh-Miin Lin  (Jimmy), Cambridge University
address: jyhmiinlin at gmail.com
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
    import numpy.random
    import matplotlib.pyplot
    import matplotlib.cm
    import scipy.linalg.lapack
    import pywt # pyWavelets
    
#     import pygasp.dwt.dwt as gasp
    # import matplotlib.numerix 
    # import matplotlib.numerix.random_array
    import sys
#     import pywt # pyWavelets
#     import utils
except:
    print('faile to import modules')
    print('numpy, scipy, matplotlib, pyWavelets are required')
    raise
 

sys.path.append('..')
sys.path.append('utils') 
from utils.utils import *

cmap=matplotlib.cm.gray
# norm=matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
dtype =  numpy.complex64
# try:
#     from numba import jit  
#  
# except:
#     print('numba not supported')
  
def DFT_slow(x):
    """
    Compute the discrete Fourier Transform of the 1D array x
    https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
    """
    x = numpy.asarray(x, dtype=float)
    N = x.shape[0]
    n = numpy.arange(N)
    k = n.reshape((N, 1))
    
    M = numpy.exp(-2j * numpy.pi * k * n / N)
    print('shape', numpy.shape(x),numpy.shape(n),numpy.shape(k),numpy.shape(M))
    return numpy.dot(M, x)
def DFT_point(x, k):
    """
    Compute the discrete Fourier Transform of the 1D array x
    https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
    """
    x = numpy.asarray(x, dtype=float)
    N = x.shape[0]
    n = numpy.arange(N)
#     k = n.reshape((N, 1))
    M = numpy.exp(-2j * numpy.pi * k * n / N)
    return numpy.dot(M, x)
class pynufft(nufft):

    def __init__(self,om, Nd, Kd,Jd,n_shift=None):
        if n_shift == None:
#             n_shift=tuple(numpy.array(Nd)/2)
            n_shift=tuple(numpy.array(Nd)*0)
#         else:
#             n_shift=tuple(list(n_shift)+numpy.array(Nd)/2)
            
        nufft.__init__(self,om, Nd, Kd,Jd,n_shift)
        self.nufft_type= 1 # SD: SD-NUFFT, T: T-NUFFT
    def initialize_gpu(self):
        nufft.initialize_gpu(self)
    def gpu_k_deconv(self):
#         try:
        self.myfft(self.tmp_dev, self.data_dev,inverse=False)
#         print('doing gpu_k_modulate')
#         self.tmp_dev=self.W_dev*self.data_dev

        # element-wise multiplication is subject to changes of API in pyopencl and pycuda ...
        # Be cautious!!
#         print('gpu_api', self.gpu_api)
        if self.gpu_api == 'cuda':
            self.tmp_dev._elwise_multiply( self.W2_dev, self.data_dev) # cluda.cuda_api()
        elif self.gpu_api == 'opencl':
            self.tmp_dev._elwise_multiply(  self.data_dev, self.tmp_dev, self.W2_dev) # cluda.ocl_api()
         
#         print('doing gpu_k_deconv')
        self.myfft(self.data_dev, self.data_dev,inverse=True)
#         print('doing gpu_k_modulate')
        return 0       
    def gpu_k_modulate(self):
#         try:
        self.myfft(self.tmp_dev, self.data_dev,inverse=False)
#         print('doing gpu_k_modulate')
#         self.tmp_dev=self.W_dev*self.data_dev

        # element-wise multiplication is subject to changes of API in pyopencl and pycuda ...
        # Be cautious!!
#         print('gpu_api', self.gpu_api)
        if self.gpu_api == 'cuda':
            self.tmp_dev._elwise_multiply( self.W_dev, self.data_dev) # cluda.cuda_api()
        elif self.gpu_api == 'opencl':
            self.tmp_dev._elwise_multiply(  self.data_dev, self.tmp_dev, self.W_dev) # cluda.ocl_api()
         
#         print('doing gpu_k_modulate')
        self.myfft(self.data_dev, self.data_dev,inverse=True)
#         print('doing gpu_k_modulate')
        return 0
#         except:  
#             return 1         
    def gpu_Nd2KdWKd2Nd(self,x, weight_flag): 
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
        output_x=numpy.zeros(Kd, dtype=dtype)
#         self.W_dev = self.thr.to_device(self.W.T.astype(dtype)) 
        for ll in xrange(0,Lprod):
            if weight_flag == 0: 
                pass  
            else:
                x[...,ll] = x[...,ll] * snc # scaling factors
  
            output_x=output_x*0.0
 
            output_x[crop_slice_ind(x[...,ll].shape)] = x[...,ll]
            self.data_dev=self.thr.to_device(output_x.astype(dtype))#,dest=self.data_dev ) 
            if self.gpu_k_modulate()==0:
                pass
#                 print('successful gpu')
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
    def gpu_forwardbackward(self,x):
 
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
        x = self.gpu_Nd2KdWKd2Nd(x,0) #    
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
    def true_forward(self, my_phantom):
        '''
        compute the exact NUFT without sparse approximation
        only for simulation
        '''
        om = self.st['om'] # [-3.14  3.14] 
        Nd = self.st['Nd'] # dimension in image domain
        M = self.st['M']
        data = numpy.zeros( (M,1), dtype = dtype)
        modulate_index_x = numpy.arange(0,Nd[0])*1.0 - Nd[0]/2#/Nd[0]
        modulate_index_x = numpy.reshape(modulate_index_x, (Nd[0],1))
        modulate_index_x = numpy.tile(modulate_index_x, (1, Nd[1]))
      
        modulate_index_y = numpy.arange(0,Nd[1])*1.0- Nd[1]/2#/Nd[1]
        modulate_index_y = numpy.reshape(modulate_index_y, (1, Nd[1]))
        modulate_index_y = numpy.tile( modulate_index_y, ( Nd[0],1))
#         matplotlib.pyplot.imshow(modulate_index_x)
#         matplotlib.pyplot.show()
#         matplotlib.pyplot.imshow(modulate_index_y)      
#         matplotlib.pyplot.show()
        for jj  in xrange(0, M): # iterate to create the exact data
            modulate_phase =  numpy.exp(   -1j * (modulate_index_x * om[jj,0]  + modulate_index_y * om[jj,1] ) ) 
            modulate_phase = modulate_phase *my_phantom
            
            data[jj,0] = numpy.sum(modulate_phase.flatten())
            print('running exact FFT', jj, 'of ', M, data[jj,0])
#             matplotlib.pyplot.imshow(modulate_phase.imag)      
#             matplotlib.pyplot.show()
        
        return data            
    def forwardbackward(self,x):

#         else:
        if self.gpu_flag == 0:
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
             
                Xk[...,ii] = Xk[...,ii]*st['w'][...,0]
#                 Xk[...,ii] = st['p'].conj().T.dot(st['p'].dot(Xk[...,ii]))
#                 Xk[...,ii] = st['T'].dot(Xk[...,ii])
            '''
            Now transform Kd grids to Nd grids(not be reshaped)
            '''
            x= self.Kd2Nd(Xk,0) #
             
            checker(x,Nd) # check output
            return x    
        elif self.gpu_flag == 1:
            return self.gpu_forwardbackward(x)
    def pseudoinverse2(self,data): # main function of solver
        '''
        density compensation 
        '''
        self.f = data
#         self.mu = mu
#         self.LMBD = LMBD
#         self.gamma = gamma
#         self.nInner= nInner
#         self.nBreg= nBreg
#         print(numpy.size(data) , self.st['M'] )
                 
        if numpy.size(data) == self.st['M']:
            self.st['senseflag'] = 0
#             print(numpy.size(data) )
            print('turn-off sense recon')
    #===============================================================================
    # mask
    #===============================================================================
            self.st = self._create_mask()

    #===============================================================================
#             u0, dummy_stack =self._kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 1,1)
            u0 =self.backward2(self.f)
    #===============================================================================
    # mask
    #===============================================================================
            self.st = self._create_mask()
            if numpy.size(u0.shape) > numpy.size(self.st['Nd']):
                for pp in xrange(2,numpy.size(u0.shape)):
                    self.st['mask'] = appendmat(self.st['mask'],1 )
                    self.st['mask2'] = appendmat(self.st['mask2'],u0.shape[pp] )
            self.st['senseflag'] = 0
            self.u = u0
    #===============================================================================         
        try: 
            if  self.st['senseflag']==0:
                self.st = self._create_mask()                
                pass
            else:
                raise
        except:
#             self.LMBD=self.LMBD*1.0
     
            self.st['senseflag']=0 # turn-off sense, to get sensemap
             
            #precompute highly constrainted images to guess the sensitivity maps 
    #===============================================================================
    # mask
    #===============================================================================
            self.st = self._create_mask()

    #===============================================================================
#             u0, dummy_stack =self._kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 2,5)
            u0 =self.backward2(self.f)
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
            self.st = self._make_sense(u0) # setting up sense map in st['sensemap']
             
            self.st['senseflag']=1 # turn-on sense, to get sensemap
       
     
            #scale back the _constrainted factor LMBD
#             self.LMBD=self.LMBD/1.0
        #CS reconstruction
            u0 = u0*self.st['sensemap'].conj()#/(self.st['sensemap'].conj()*self.st['sensemap']+1e-2)
            self.u = CombineMulti(u0,-1)[...,0]
#         self.u, self.ustack = self._kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 
#                           self.nInner,self.nBreg)
        fermi = scipy.fftpack.fftshift( self.st['fermi'] )
#         matplotlib.pyplot.imshow(fermi.real)
#         matplotlib.pyplot.show()
#         for jj in xrange(0,self.u.shape[-1]):
#             u[...,jj] = u[...,jj]*(self.st['sn']**1)# rescale the final image intensity
#             u[...,jj] = scipy.fftpack.ifftn(scipy.fftpack.fftn(u[...,jj])*fermi ) # apply GE's fermi filter
#             if st['senseflag']== 1:
        self.u = scipy.fftpack.ifftn(scipy.fftpack.fftn(self.u)*fermi )  
         
#         for jj in xrange(0,self.u.shape[-1]):
#             self.u[...,jj] = self.u[...,jj]*(self.st['sn']**0.7)# rescale the final image intensity
#         
        if self.u.shape[-1] == 1:
            if numpy.ndim(self.u) != numpy.ndim(self.st['Nd']):  # alwasy true?          
                self.u = self.u[...,0]
 
#         self.u = Normalize(self.u)
#         fermi = scipy.fftpack.fftshift( self.st['fermi'] )
# #         matplotlib.pyplot.imshow(fermi.real)
# #         matplotlib.pyplot.show()
# #         for jj in xrange(0,self.u.shape[-1]):
# #             u[...,jj] = u[...,jj]*(self.st['sn']**1)# rescale the final image intensity
# #             u[...,jj] = scipy.fftpack.ifftn(scipy.fftpack.fftn(u[...,jj])*fermi ) # apply GE's fermi filter
# #             if st['senseflag']== 1:
#         u[...,jj] = scipy.fftpack.ifftn(scipy.fftpack.fftn(u[...,jj])*fermi ) 
        self.u = low_pass_phase(self.u)
        return self.u   
    def pseudoinverse3(self,data, mu, LMBD, gamma, nInner, nBreg): # main function of solver
        image1 = self.pseudoinverse(data, mu, LMBD, gamma, nInner, nBreg)
        image2 = self.backward2(data)
        image1 = Normalize(image1)
        image2 = Normalize(image2)
        
        kdensity = self.W

        kdensity = scipy.misc.imresize(kdensity,self.st['Nd'])
        kdensity= Normalize(kdensity)
        
        kspace1 = scipy.fftpack.fft2(image1)
        kspace2 = scipy.fftpack.fft2(image2[...,0])
        kspace3 = kspace1 * kdensity  + kspace2 * ( 1.0 - kdensity)  
        image3  = scipy.fftpack.ifft2(kspace3)
        return image3
            
    def pseudoinverse(self,data, mu, LMBD, gamma, nInner, nBreg): # main function of solver
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
    #===============================================================================
    # mask
    #===============================================================================
            self.st = self._create_mask()

    #===============================================================================
#             u0, dummy_stack =self._kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 1,1)
            u0 =self.backward2(self.f)
    #===============================================================================
    # mask
    #===============================================================================
            self.st = self._create_mask()
#             if numpy.size(u0.shape) > numpy.size(self.st['Nd']):
            for pp in xrange(2,numpy.size(u0.shape)):
                self.st['mask'] = appendmat(self.st['mask'],1 )
                self.st['mask2'] = appendmat(self.st['mask2'],u0.shape[pp] )
            self.st['senseflag'] = 0
    #===============================================================================         
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
    #===============================================================================
    # mask
    #===============================================================================
            self.st = self._create_mask()

    #===============================================================================
#             u0, dummy_stack =self._kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 2,5)
            u0 =self.backward2(self.f)
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
            self.st = self._make_sense(u0) # setting up sense map in st['sensemap']
             
            self.st['senseflag']=1 # turn-on sense, to get sensemap
       
     
            #scale back the _constrainted factor LMBD
            self.LMBD=self.LMBD/1.0
        #CS reconstruction
         
        self.u, self.ustack = self._kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 
                          self.nInner,self.nBreg)
 
         
#         for jj in xrange(0,self.u.shape[-1]):
#             self.u[...,jj] = self.u[...,jj]*(self.st['sn']**0.7)# rescale the final image intensity
#         
        if self.u.shape[-1] == 1:
            if numpy.ndim(self.u) != numpy.ndim(self.st['Nd']):  # alwasy true?          
                self.u = self.u[...,0]
 
#         self.u = Normalize(self.u)
        if len(self.st['Nd'])==2:
            self.u = low_pass_phase(self.u)
        return self.u 
    def _kernel(self, f, st , mu, LMBD, gamma, nInner, nBreg):
        L= numpy.size(f)/st['M'] 
        image_dim=st['Nd']+(L,)
          
        if numpy.ndim(f) == 1:# preventing row vector
            f=numpy.reshape(f,(numpy.shape(f)[0],1),order='F')
#         f0 = numpy.copy(f) # deep copy to prevent scope f0 to f
#         unused
#        u = numpy.zeros(image_dim,dtype=dtype)
 
 
 
    #===========================================================================
    # check whether sense is used
    # if senseflag == 0, create an all-ones mask
    # if sensflag size is wrong, create an all-ones mask (shouldn't occur)
    #===========================================================================
        if st['senseflag'] == 0:
            st['sensemap'] = numpy.ones(image_dim,dtype=dtype)
        elif numpy.shape(st['sensemap']) != image_dim: #(shouldn't occur)
            st['sensemap'] = numpy.ones(image_dim,dtype=dtype)
        else:
            pass # correct, use existing sensemap
    #=========================================================================
    # check whether mask is used  
    #=========================================================================
#         if st.has_key('mask'):
#         if 'mask' in st:
#             if numpy.shape(st['mask']) != image_dim:
#                 st['mask'] = numpy.ones(image_dim,dtype=dtype)
#         else:
#             pass
        st['mask'] = numpy.ones(image_dim,dtype=dtype)
 
        if 'mask2' in st:
            if numpy.shape(st['mask2']) != image_dim:
                st['mask2'] = numpy.ones(image_dim,dtype=dtype)
#                 st['mask2'] = numpy.reshape(st['mask2'],image_dim,order='F')
        else:
            st['mask2'] = numpy.ones(image_dim,dtype=dtype)
 
 
        tmp_sense_fun = self.st['sensemap'].copy()
#         self.sense_fun = numpy.mean(numpy.abs( self.sense_fun),axis=(-1,))
      
        tmp_sense_fun2 = self.forwardbackward(tmp_sense_fun)
        
        self.sense_fun =numpy.mean( ( ( tmp_sense_fun2)),axis=(-1,))**(- 0.33)
        # final_sense2
        self.final_sense= 1.0 #(self.sense_fun)/(self.sense_fun*self.sense_fun.conj() + 1e-2)       
#         if st['senseflag'] == 1:
# #             matplotlib.pyplot.figure(40)
# # #             matplotlib.pyplot.subplot(2,3,1)
# #             matplotlib.pyplot.imshow(numpy.real(self.final_sense))
# #             matplotlib.pyplot.subplot(2,3,2)
# #             matplotlib.pyplot.figure(41)
# #             matplotlib.pyplot.imshow(numpy.imag(self.sense_fun))
# #             matplotlib.pyplot.show()
# #             
#             matplotlib.pyplot.figure(32)
#             matplotlib.pyplot.subplot(3,3,1)
#             matplotlib.pyplot.imshow(numpy.real(self.st['sensemap'][...,0])  )
#             matplotlib.pyplot.subplot(3,3,2)
#             matplotlib.pyplot.imshow(numpy.real(self.st['sensemap'][...,1])  )
#             matplotlib.pyplot.subplot(3,3,3)
#             matplotlib.pyplot.imshow(numpy.real(self.st['sensemap'][...,2])  ) 
#             matplotlib.pyplot.subplot(3,3,4)
#             matplotlib.pyplot.imshow(numpy.real(self.st['sensemap'][...,3])  ) 
#             matplotlib.pyplot.subplot(3,3,5)
#             matplotlib.pyplot.imshow(numpy.real(self.st['sensemap'][...,4])  ) 
#             matplotlib.pyplot.subplot(3,3,6)
#             matplotlib.pyplot.imshow(numpy.real(self.st['sensemap'][...,5])  ) 
#             matplotlib.pyplot.subplot(3,3,7)
#             matplotlib.pyplot.imshow(numpy.real(self.st['sensemap'][...,6])  ) 
#             matplotlib.pyplot.subplot(3,3,8)
#             matplotlib.pyplot.imshow(numpy.real(self.st['sensemap'][...,7])  ) 
# #             matplotlib.pyplot.show()  
 
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
 
        u = self.adjoint(f)
#         c = numpy.max(numpy.abs(u.flatten())) # Rough coefficient
 
        for jj in xrange(0,u.shape[-1]):
            u[...,jj] = u[...,jj]/self.st['sn'] # remove scaling factor in the first place
        if self.debug ==0:
            pass
        else:  
            print('senseflag',st['senseflag'])
        if st['senseflag'] == 1:
             
            u=CombineMulti(u,-1)[...,0:1] # summation of multicoil images
         
 
        u0 = numpy.copy(u)
        self.thresh_scale= numpy.mean(numpy.abs(u0[:]))#/numpy.max(self.st['w'])           
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
 
#         if len(self.st['Nd']) == 2:
#             uker = self.mu*RTR - LMBD*uker + gamma*2.0
#         else:
        max_of_convolution = numpy.max(numpy.abs(uker[:])) # maximum value of the convolution kernel
        uker = self.mu*RTR*33.85 / max_of_convolution - LMBD*uker   + gamma 
#             *35/max_of_convolution 
        if self.debug ==0:
            pass
        else:
            print('uker.shape line 319',uker.shape, 'uker maximum',numpy.max(numpy.abs(uker[:])),'RTR maximum',numpy.max(numpy.abs(RTR[:])))
                         
        (xx,bb,dd)=self._make_split_variables(u)        
 
        uf = numpy.copy(u0)  # only used for ISRA, written here for generality
#         if st['senseflag'] ==1: 
        alpha = self.maxrowsum()
        print( 'uker maximum',numpy.max(numpy.abs(uker[:])),'RTR maximum',numpy.max(numpy.abs(RTR[:])),'alpha',alpha)
        try:
            factor = self.factor
        except:
            factor = 0.1
        
        
        u = u*alpha*factor#numpy.max(st['q'][:]) # initial values
         
        murf = numpy.copy(u) # initial values 
#    #===============================================================================
        u_stack = numpy.empty(st['Nd']+(nBreg,),dtype=dtype)
        self.err =1.0e+13
        self.ctrl = 1
        u_k_1=0.0
        if self.gpu_flag == 1:
            self.thr.to_device((1.0/uker[...,0]).astype(dtype), dest = self.W2_dev)  
        for outer in numpy.arange(0,nBreg):
             
            for inner in numpy.arange(0,nInner):
                # update u
#                 if self.debug==0:
#                     pass
#                 else:
#                 print('iterating',[inner,outer])
                #===============================================================
#                 update u  # simple k-space deconvolution to guess initial u
#                 if self.ctrl == 1:
#                     print('iterating',[inner,outer])

#                     for jj in xrange(0,u.shape[-1]):
#                         u[...,jj] = u[...,jj]*self.st['sn']
#                         murf[...,jj] = murf[...,jj]*self.st['sn']
                u = self._update_u(murf,u,uker,xx,bb)
#                     for jj in xrange(0,u.shape[-1]):
#                         u[...,jj] = u[...,jj]/self.st['sn']
                 
#                     for jj in xrange(0,u.shape[-1]):
#                         u[...,jj] = u[...,jj]*self.st['sn']
                    # Temporally scale the image for softthresholding                
                 
#                     c = numpy.max(numpy.abs(u.flatten())) # Rough coefficient
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
#                 self.thresh_scale = numpy.percentile(numpy.abs(u), 100)*3.0644028/0.68602877855300903
                print('alpha=',alpha, ',self.thresh_scale',self.thresh_scale,self.thresh_scale*33.85653/1024)
                xx=self._shrink( dd, bb, 
                                 6.0*33.85653/1024)
                #===============================================================
            #===================================================================
            # # update b
            #===================================================================
 
                bb=self._update_b(bb, dd, xx)
#                     for jj in xrange(0,u.shape[-1]):
#                         u[...,jj] = u[...,jj]/self.st['sn']
#                         murf[...,jj] = murf[...,jj]/self.st['sn']
                    #  Temporally scale the image for softthresholding  
#                 else:
#                     pass
                    ############################################################
                     
            if outer < nBreg -1: # do not update in the last loop
 
#             if self.ctrl == 1: # err < self.err
                if st['senseflag']== 1:
                    u = appendmat(u[...,0],L)
          
 
                (u, murf, uf, u_k_1)=self._external_update(u, uf, u0, u_k_1, outer) # update outer Split_bregman
                if st['senseflag']== 1:
                    u = u[...,0:1]
                    murf = murf[...,0:1]
                 
                u_stack[...,outer] = (u[...,0]*self.st['sn'])
                stop_iter = outer
#            u_stack[...,outer] =u[...,0] 
            else: # self.ctrl = 0 # err > self.err
                pass
 
                            
 
 
 
        if st['senseflag']== 1:
            fermi = scipy.fftpack.fftshift( self.st['fermi'] )
#         matplotlib.pyplot.imshow(fermi.real)
#         matplotlib.pyplot.show()
        for jj in xrange(0,u.shape[-1]):
#             u[...,jj] = u[...,jj]*(self.st['sn']**1)# rescale the final image intensity
#             u[...,jj] = scipy.fftpack.ifftn(scipy.fftpack.fftn(u[...,jj])*fermi ) # apply GE's fermi filter
            if st['senseflag']== 1:
#                 u[...,jj] = scipy.fftpack.ifftn(scipy.fftpack.fftn(u[...,jj])*fermi ) # apply GE's fermi filter
                u[...,jj] = u[...,jj]*(self.st['sn'])#*self.st['mask2'][...,jj]# rescale the final image intensity
                u[...,jj] = u[...,jj]*self.final_sense
            else:
                pass
#                 u[...,jj] = scipy.fftpack.ifftn(scipy.fftpack.fftn(u[...,jj])*fermi ) # Don't apply GE's fermi filter
                u[...,jj] = u[...,jj]*(self.st['sn'])#*self.st['mask2'][...,jj]# rescale the final image intensity
                 
 
        return u, u_stack[...,0:stop_iter +1 ]#(u,u_stack)
#     def final_sense(self,input_x):
#         input_x = input_x*(self.sense_fun.conj()+ 0.01)/(self.sense_fun*self.sense_fun.conj() + 0.01)
# 
#         return input_x
    def _update_u(self,murf,u,uker,xx,bb):
        n_dims = len(self.st['Nd'])
        #print('inside _update_u')
#        checkmax(u)
#        checkmax(murf)
#        rhs = self.mu*murf + self.LMBD*self.get_Diff(x,y,bx,by) + self.gamma
        #=======================================================================
        # Trick: make "llist" for numpy.transpose 
        mylist = tuple(numpy.arange(0,numpy.ndim(xx[0]))) 
#         tlist = mylist[1::-1]+mylist[2:] 
        #=======================================================================
        # update the right-head side terms
        rhs = (self.mu*murf + 
               self._constraint(xx,bb)) 
        if n_dims == 2:
            rhs = rhs# + self.gamma * u
        else:
            rhs = rhs + self.gamma * u
         
        rhs = rhs * self.st['mask'][...,0:u.shape[-1]]
  
#        rhs=Normalize(rhs)
        #=======================================================================
#         Trick: make "flist" for fftn 
        flist = mylist[:-1:1]    
#         for jj in xrange(0,rhs.shape[-1]):
#             rhs  = rhs /self.st['sn']
#             murf[...,jj] = murf[...,jj]/self.st['sn']  


        u = self._k_deconv(rhs, uker,self.st,flist,mylist)
#         for jj in xrange(0,rhs.shape[-1]):
#             u  = u /self.st['sn']
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


        if self.gpu_flag == 0:
#         U=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)            
            for pj in xrange(0,u.shape[-1]):
    #             U[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) / uker[...,pj] # deconvolution
    #             U[...,pj]=self.emb_ifftn(U[...,pj], st['Kd'], range(0,numpy.size(st['Kd'])))
                tmp=self.emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) / uker[...,pj] # deconvolution
                u[...,pj]=self.emb_ifftn(tmp, st['Kd'], range(0,numpy.size(st['Kd'])))[[slice(0, st['Nd'][_ss]) for _ss in flist]]
    #         u = U[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
        
        elif self.gpu_flag ==1:
#             self.thr.to_device((1.0/uker[...,0]).astype(dtype), dest = self.W2_dev) 
         
            for pj in xrange(0,u.shape[-1]):
                output_u=numpy.zeros(st['Kd'], dtype=dtype)
                #print('output_dim',input_dim,output_dim,range(0,numpy.size(input_dim)))
        #         output_x[[slice(0, input_x.shape[_ss]) for _ss in range(0,len(input_x.shape))]] = input_x
                output_u[crop_slice_ind(u[...,pj].shape)] = u[...,pj]  
                self.data_dev=self.thr.to_device(output_u.astype(dtype))#, dest=self.data_dev)
                #         try:
                self.gpu_k_deconv()
#                 self.myfft(self.tmp_dev, self.data_dev,inverse=False)
#         #         print('doing gpu_k_modulate')
#         #         self.tmp_dev=self.W_dev*self.data_dev
#         
#                 # element-wise multiplication is subject to changes of API in pyopencl and pycuda ...
#                 # Be cautious!!
#                 print('gpu_api', self.gpu_api)
#                 if self.gpu_api == 'cuda':
#                     self.tmp_dev._elwise_multiply( self.W2_dev, self.data_dev) # cluda.cuda_api()
#                 elif self.gpu_api == 'opencl':
#                     self.tmp_dev._elwise_multiply(  self.data_dev, self.tmp_dev, self.W2_dev) # cluda.ocl_api()
#                  
#         #         print('doing gpu_k_modulate')
#                 self.myfft(self.data_dev, self.data_dev,inverse=True)
                u[...,pj] = self.data_dev.get()[[slice(0, st['Nd'][_ss]) for _ss in flist]]
                
#             U=numpy.empty(st['Kd'] ,dtype=u.dtype)

#         self.W2_dev = self.thr.to_device(uker[...,0].astype(dtype)) #uker[...,pj]




        # optional: one- additional Conjugated step to ensure the quality
         
#         for pp in xrange(0,5):
#             u = self._cg_step(u0,u,uker,st,flist,mylist)
#         
         
       
        return u
    def _cg_step(self, rhs, u, uker, st,flist,mylist):
        u=u#*st['mask'][...,0:u.shape[-1]]
#            u=scipy.fftpack.fftn(u, st['Kd'],flist)
        AU=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
#        print('U.shape. line 446',U.shape)
#        print('u.shape. line 447',u.shape)
        for pj in xrange(0,u.shape[-1]):
            AU[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
            AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
             
          
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
         
    def _constraint(self,xx,bb):
        '''
        include TVconstraint and others
        '''
#         try:
        n_dims = len(self.st['Nd'])
#         print('inside constraint',numpy.shape(xx),numpy.shape(bb))
#         print('inside constraint',len(xx),n_dims)
#         print('xx[n_dims]',xx[n_dims])
#         print('xx[n_dims]',bb[n_dims])
        cons = self.LMBD*TVconstraint(xx[0:n_dims],bb[0:n_dims]) 
        if len(xx) > n_dims:
            cons =cons + self.gamma * Wconstraint(xx[n_dims],bb[n_dims]) 
#         if n_dims == 2:
#             cons =cons + self.gamma * Wconstraint(xx[n_dims ],bb[n_dims ]) 
#         except:
#             print('error !')
#             print('error in _constraint, len(xx), len(bb)',len(xx),len(bb))
        return cons
     
    def _shrink(self,dd,bb,thrsld):
        '''
        soft-thresholding the edges
         
        '''
        n_dims = len(self.st['Nd'])
        
        output_xx=shrink( dd[:n_dims], bb[:n_dims], thrsld/self.LMBD)# 3D thresholding
        if len(dd) > n_dims:
            output_xx=output_xx+(shrink(dd[-1],bb[-1],thrsld/self.gamma),)
         
        return output_xx
         
    def _make_split_variables(self,u):
        n_dims = len(self.st['Nd'])
        if n_dims ==2:
            num_of_split_variables = 2 # add wavelets only when the dimension is 2 !
        else:
            num_of_split_variables = n_dims
        xx = ()
        bb = ()
        dd = ()
        
            
        for jj in xrange(0,num_of_split_variables ): # n_dims + 1 for wavelets
            x=numpy.zeros(u.shape,dtype = dtype)
            bx=numpy.zeros(u.shape,dtype = dtype)
            dx=numpy.zeros(u.shape,dtype = dtype)
            
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
        print('len(bb)',len(bb))
        print('len(dd)',len(dd))
        print('len(xx)',len(xx))
        return(xx,bb,dd)
#     def _extract_svd(self,input_stack,L):
#         C= numpy.copy(input_stack) # temporary array
#         print('size of input_stack', numpy.shape(input_stack))
#         C=C/numpy.max(numpy.abs(C))
#  
#         reps_acs = 16 #16
#         mysize = 4 #16
#         K= 3 # rank of 10 prevent singular? artifacts(certain disruption)
#         half_mysize = mysize/2
#         dimension = numpy.ndim(C) -1 # collapse coil dimension
#         if dimension == 1:
#             tmp_stack = numpy.empty((mysize,),dtype = dtype) 
#             svd_size = mysize
#             C_size = numpy.shape(C)[0]
#             data = numpy.empty((svd_size,L*reps_acs),dtype=dtype)
# #             for jj in xrange(0,L):
# #                 C[:,jj]=tailor_fftn(C[:,jj])
# #                 for kk in xrange(0,reps_acs):
# #                     tmp_stack = numpy.reshape(tmp_stack,(svd_size,),order = 'F')
# #                     data[:,jj] = numpy.reshape(tmp_stack,(svd_size,),order = 'F') 
#         elif dimension == 2:
#             tmp_stack = numpy.empty((mysize,mysize,),dtype = dtype) 
#             svd_size = mysize**2
#             data = numpy.empty((svd_size,L*reps_acs),dtype=dtype)
#             C_size = numpy.shape(C)[0:2]
#             for jj in xrange(0,L):
# #                 matplotlib.pyplot.imshow(C[...,jj].real)
# #                 matplotlib.pyplot.show()
# #                 tmp_pt=(C_size[0]-reps_acs)/2
#                 C[:,:,jj]=tailor_fftn(C[:,:,jj])
#                 for kk in xrange(0,reps_acs):
#                     a=numpy.mod(kk,reps_acs**0.5)
#                     b=kk/(reps_acs**0.5)
#                     tmp_stack = C[C_size[0]/2-half_mysize-(reps_acs**0.5)/2+a : C_size[0]/2+half_mysize-(reps_acs**0.5)/2+a,
#                                   C_size[1]/2-half_mysize-(reps_acs**0.5)/2+b : C_size[1]/2+half_mysize-(reps_acs**0.5)/2+b,jj]
#                     data[:,jj*reps_acs+kk] = numpy.reshape(tmp_stack,(svd_size,),order = 'F')
#                                          
#         elif dimension == 3:
#             tmp_stack = numpy.empty((mysize,mysize,mysize),dtype = dtype) 
#             svd_size = mysize**3
#             data = numpy.empty((svd_size,L),dtype=dtype)
#             C_size = numpy.shape(C)[0:3]
#             for jj in xrange(0,L):
#                 C[:,:,:,jj]=tailor_fftn(C[:,:,:,jj])
#                 tmp_stack= C[C_size[0]/2-half_mysize:C_size[0]/2+half_mysize,
#                                 C_size[1]/2-half_mysize:C_size[1]/2+half_mysize,
#                                 C_size[2]/2-half_mysize:C_size[2]/2+half_mysize,
#                                 jj]
#                 data[:,jj] = numpy.reshape(tmp_stack,(svd_size,),order = 'F')   
#                  
# #         OK, data is the matrix of size (mysize*n, L) for SVD
# #         import scipy.linalg
# #         import scipy.sparse.linalg            
#         (s_blah,vh_blah) = scipy.linalg.svd(data)[1:3]
#  
#         for jj in xrange(0,numpy.size(s_blah)): # 
#             if s_blah[jj] > 0.1*s_blah[0]: # 10% of maximum singular value to decide the rank
#                 K = jj+1
# #                 pass
#             else:
#                 break
#  
#         v_blah =vh_blah.conj().T
#  
#         C = C*0.0 # now C will be used as the output stack
#         V_para = v_blah[:,0:K]
#         print('shape of V_para',numpy.shape(V_para))
#         V_para = numpy.reshape(V_para,(reps_acs**0.5,reps_acs**0.5,L, K),order='F')
#           
#         C2 = numpy.zeros((C.shape[0],C.shape[1],L,K),dtype=dtype)
#         for jj in xrange(0,L): # coils  
#             for kk in xrange(0,K): # rank
#                 C2[C.shape[0]/2-reps_acs**0.5/2:C.shape[0]/2+reps_acs**0.5/2,
#                    C.shape[1]/2-reps_acs**0.5/2:C.shape[1]/2+reps_acs**0.5/2,
#                    jj,kk]=V_para[:,:,jj,kk]
#                 C2[:,:,jj,kk]=tailor_fftn(C2[:,:,jj,kk])
# #         C_value = numpy.empty_like(C)
#  
#         for mm in xrange(0,C.shape[0]): # dim 0  
#             for nn in xrange(0,C.shape[1]): # dim 1
#                 tmp_g=C2[mm,nn,:,:]
# #                 G =   C2[mm,nn,:,:].T # Transpose (non-conjugated) of G
# # #                 Gh = G.conj().T # hermitian
# #                 Gh=C2[mm,nn,:,:].conj()
# #                 G=
#                 
#                 g = numpy.dot(tmp_g.conj(),tmp_g.T)  #construct g matrix for eigen-decomposition  
# #                 w,v = scipy.linalg.eig(g.astype(dtype), overwrite_a=True,
# #                                        check_finite=False) # eigen value:w, eigen vector: v
#  
# #                 print('L=',L,numpy.shape(g))
# #                 w,v = scipy.sparse.linalg.eigs(g , 3)
#                 w,v = myeig(g.astype(dtype))
# 
#                 ind = numpy.argmax(numpy.abs(w)) # find the maximum 
# #                 print('ind=',ind)
# #                 the_eig = numpy.abs(w[ind]) # find the abs of maximal eigen value
#                 tmp_v = v[:,ind]#*the_eig 
# #                 ref_angle=(numpy.sum(v[:,ind])/(numpy.abs(numpy.sum(v[:,ind]))))
# #                 v[:,ind] = v[:,ind]/ref_angle # correct phase by summed value
# 
#                 ref_angle=numpy.sum(tmp_v)
#                 ref_angle=ref_angle/numpy.abs(ref_angle)
#                 
#                 C[mm,nn,:] = tmp_v/ref_angle # correct phase by summed value
#         C=C/numpy.max(numpy.abs(C))
# #         matplotlib.pyplot.figure(1)         
# #         for jj in xrange(0,L):
# #             matplotlib.pyplot.subplot(2,4,jj+1)  
# #             matplotlib.pyplot.imshow(abs(input_stack[...,jj]),
# #                                      norm=matplotlib.colors.Normalize(vmin=0.0, vmax=0.2),
# #                                      cmap=matplotlib.cm.gray)
# #             matplotlib.pyplot.subplot(2,4,jj+1+4)  
# #             matplotlib.pyplot.imshow(abs(C[...,jj]),
# #                                      norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
# #                                      cmap=matplotlib.cm.gray)
# #                                   
# # #             matplotlib.pyplot.subplot(2,8,jj+1+8)          
# # #             matplotlib.pyplot.imshow(numpy.log(C[...,jj]).imag, cmap=matplotlib.cm.gray)
# #                          
# #         matplotlib.pyplot.show()  
#   
# #         for jj in xrange(0,L):   
# #             matplotlib.pyplot.subplot(2,2,jj+1)  
# #             matplotlib.pyplot.imshow((input_stack[...,jj].real))
# #         matplotlib.pyplot.show() 
#          
#         return C # normalize the coil sensitivities 
       
    def _make_sense(self,u0):
        st=self.st
#         L=numpy.shape(u0)[-1]
        u0dims= numpy.ndim(u0)
#         st=self.st
        L=numpy.shape(u0)[-1]
        try:
            st['sensemap'] = extract_svd(u0,L)
            print('run svd')
#             st['sensemap']=u0
#             matplotlib.pyplot.figure(1)
#             for pp in xrange(0,256):
#                 for mm in xrange(0,256):
#                     if (pp-128)**2 +(mm-128)**2> 128**2: 
#                         st['sensemap'][pp,mm,...]=st['sensemap'][pp,mm,...]*0.0
#             st['sensemap'] = st['sensemap']*0.7/numpy.max(st['sensemap'])    
#             for jj in xrange(0,L):   
#                 
#                 matplotlib.pyplot.subplot(2,4,jj+1+4)  
#                 matplotlib.pyplot.imshow((st['sensemap'][...,jj].real),norm = norm, cmap = cmap)
#             matplotlib.pyplot.show() 
            return st
        except: 
            print('not runing svd')       
            if u0dims-1 >0:
                rows=numpy.shape(u0)[0]
                dpss_rows = numpy.kaiser(rows, 100)     
                dpss_rows = scipy.fftpack.fftshift(dpss_rows)
                dpss_rows[3:-3] = 0.0
                dpss_fil = dpss_rows
                if self.debug==0:
                    pass
                else:            
                    print('dpss shape',dpss_fil.shape)
            if u0dims-1 > 1:
                                   
                cols=numpy.shape(u0)[1]
                dpss_cols = numpy.kaiser(cols, 100)            
                dpss_cols = scipy.fftpack.fftshift(dpss_cols)
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
                dpss_zag = numpy.kaiser(zag, 100)            
                dpss_zag = scipy.fftpack.fftshift(dpss_zag)
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
             
            rms=numpy.sqrt(numpy.mean(u0*u0.conj(),-1)) # Root of sum square
            st['sensemap']=numpy.ones(numpy.shape(u0),dtype=dtype)
            if self.debug==0:
                pass
            else:        
                print('sensemap shape',st['sensemap'].shape, L)
                print('u0shape',u0.shape,rms.shape)
     
            #    print('L',L)
            #    print('rms',numpy.shape(rms))
            for ll in numpy.arange(0,L):
                st['sensemap'][...,ll]=(u0[...,ll]+1e-16)/(rms+1e-16)
                if self.debug==0:
                    pass
                else:
                    print('sensemap shape',st['sensemap'].shape, L)
                    print('rmsshape', rms.shape) 
                     
                if self.pyfftw_flag == 1:
                    if self.debug==0:
                        pass
                    else:                
                        print('USING pyfftw and thread is = ',self.threads)
                    st['sensemap'][...,ll] = pyfftw.interfaces.scipy_fftpack.fftn(st['sensemap'][...,ll], 
                                                      st['sensemap'][...,ll].shape,
                                                            range(0,numpy.ndim(st['sensemap'][...,ll])), 
                                                            threads=self.threads) 
                    st['sensemap'][...,ll] = st['sensemap'][...,ll] * dpss_fil
                    st['sensemap'][...,ll] = pyfftw.interfaces.scipy_fftpack.ifftn(st['sensemap'][...,ll], 
                                                      st['sensemap'][...,ll].shape,
                                                            range(0,numpy.ndim(st['sensemap'][...,ll])), 
                                                            threads=self.threads)                                                 
                else:                                                                    
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
 
    def _create_kspace_sampling_density(self):
            #=======================================================================
            # RTR: k-space sampled density
            #      only diagonal elements are relevant (on k-space grids)
            #=======================================================================
        RTR=self.st['w'] # see __init__() in class "nufft"
         
        return RTR 
    def _create_laplacian_kernel(self):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to constraint
#===============================================================================
        uker = numpy.zeros(self.st['Kd'][:],dtype=dtype)
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
        try:
            n_dims = len(self.st['Nd'])
            out_dd = ()
            for jj in xrange(0,n_dims) :
                out_dd = out_dd  + (get_Diff(u,jj),)
            if len(dd)>n_dims :
                out_dd =out_dd + (get_wavelet(u.real )+1.0j*get_wavelet(u.imag ),) # only 2D is supported
#         print('len(bb)',len(bb))
        except:
            print('error in _update_d')
            print('_update_d len(dd)',len(dd))
#         print('len(xx)',len(xx))
        return out_dd
     
    def _update_b(self, bb, dd, xx):
        try:
            ndims=len(bb)
            cc=numpy.empty(bb[0].shape)
            out_bb=()
            for pj in xrange(0,ndims):
                cc=bb[pj]+dd[pj]-xx[pj]
                out_bb=out_bb+(cc,)
        except:
            print('error in _update_b')
            print('len(bb)',len(bb))
            print('len(dd)',len(dd))
            print('len(xx)',len(xx))
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
        st['fermi'] =1.0/(1.0+numpy.exp( (tmp-20.0)/(20.0*2.0/st['Nd'][0])))
        st['mask2'] =1.0/(1.0+numpy.exp( (tmp-1.05)/0.01))
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
 
#     def _create_mask(self):
#         st=self.st
# 
#         st['mask']=numpy.ones(st['Nd'],dtype=numpy.float64)
#         n_dims= numpy.size(st['Nd'])
#  
#         sp_rat =0.0
#         for di in xrange(0,n_dims):
#             sp_rat = sp_rat + (st['Nd'][di]/2)**2
#   
#         x = numpy.ogrid[[slice(0, st['Nd'][_ss]) for _ss in xrange(0,n_dims)]]
# 
#         tmp = 0
#         for di in xrange(0,n_dims):
#             tmp = tmp + ( x[di] - st['Nd'][di]/2 )**2
#         indx = tmp/sp_rat >=1.0/n_dims
#             
#         st['mask'][indx] =0.0       
#          
#   
#         return st   
    def forwardbackward2(self,x): # pseudo_inverse second order
        '''
        Update the data-space
        '''
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
#         Xk = self.Nd2Kd(x,0) #
# 
#         if Lprod > 1:
#             Xk = numpy.reshape(st['p'].dot(Xk),(st['M'],)+( Lprod,),order='F')
#         else:
#             Xk = numpy.reshape(st['p'].dot(Xk),(st['M'],1),order='F')
        for ii in xrange(0,Lprod):
#             Xk[...,ii] = Xk[...,ii]*st['q'][...,0]
            x[...,ii] = x[...,ii]/st['sn'][...,0] 
 
        Xk = self.forward(x)
#         Xk = self.Nd2Kd(x,0)
 
        Xk = self.f - Xk
 
        err2 = checkmax(Xk, self.debug)/checkmax(self.f, self.debug)
 
        '''
        Now transform Kd grids to Nd grids(not be reshaped)
        '''
        Xk2 = st['p'].getH().dot(Xk) 
        x1= self.Kd2Nd(Xk2,0) #
         
        for ii in xrange(0,Lprod):
#             Xk[...,ii] = Xk[...,ii]*st['q'][...,0]
            Xk[...,ii] = Xk[...,ii]*st['W'][:,0]#(st['q'][...,0]**0.5+1e-1)/(st['q'][...,0]+1e-1)
        Xk3 = st['p'].getH().dot(Xk)#*numpy.max(st['w']) # scale up the inverse solution, and ma
        x2= self.Kd2Nd(Xk3,0) #
         
#         for ii in xrange(0,Lprod):
# #             Xk[...,ii] = Xk[...,ii]*st['q'][...,0]
#             x1[...,ii] = x1[...,ii]*st['sn'][...,0] 
#             x2[...,ii] = x2[...,ii]*(st['sn'][...,0]) 
              
#         x= x1 - x2
#         checker(x,Nd) # check output
        return x1,x2,err2
    def maxrowsum(self):
#         try:
#             factor = self.factor
#         except:
#             factor = 0.1
        
        
        c = self.forwardbackward(numpy.ones(self.st['Nd'],dtype=dtype))
        alpha = 1.0/numpy.max(numpy.abs(c[:]))
#         print('alpha = ',alpha)
#         print('alpha = ',alpha)
#         print('alpha = ',alpha)
        return alpha  
    def backward2(self,X):
        '''
        backward2(x): method of class pyNufft
        
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
        X2= numpy.copy(X)
        Nd = st['Nd']
#         Kd = st['Kd']
        if self.debug==0:
            pass
        else:
            checker(X2,st['M']) # check X of correct shape

        dims = numpy.shape(X2)
        Lprod= numpy.prod(dims[1:]) 
        # how many channel * slices

        if numpy.size(dims) == 1:
            Lprod = 1
        else:
            Lprod = dims[1]

        for jj in xrange(0,Lprod):
            X2[:,jj]=X2[:,jj]*self.st['W'][:,0]
        Xk_all = st['p'].getH().dot(X2)
#         print(numpy.shape(Xk_all),numpy.shape(st['q'])) 
#         for jj in xrange(0,Lprod):
#             Xk_all[...,jj]=Xk_all[...,jj]*self.st['w'][:,0]#(st['q'][...,0]**0.5/(st['q'][...,0]+1.0e-2))
        # Multiply X2 with interpolator st['p'] [prod(Kd) Lprod]
        '''
        Now transform Kd grids to Nd grids(not be reshaped)
        '''
        x = self.Kd2Nd(Xk_all, 1)
        
        if self.debug==0:
            pass
        else:        
            checker(x,Nd) # check output
        
        return x
    
    def adjoint(self,f):
        '''
        adjoint operator to calcualte AT*y
        '''
        st = self.st
        return self.backward(f)*st['sensemap'].conj()#/(st['sensemap'].conj()*st['sensemap'] + 1e-2)    
#        self.u=1.5*self.u/numpy.max(numpy.real(self.u[:]))
    def adjoint2(self,f):
        '''
        adjoint operator to calcualte AT*y
        '''
#         f2 = numpy.copy(f)
        st = self.st
        return self.backward2(f)*st['sensemap'].conj()#/(st['sensemap'].conj()*st['sensemap'] + 1e-2) 
    def _external_update_new(self,u, uf, u0, u_k_1, outer): # overload the update function
        '''
        add conjugated gradient 
        '''
        
        tmpuf=self.forwardbackward(
                u*self.st['sensemap'])*self.st['sensemap'].conj()
#         tmpuf=self.backward(self.forward(
#                         u*self.st['sensemap']) )*self.st['sensemap'].conj()  
                                   
        if self.st['senseflag'] == 1:
            tmpuf=CombineMulti(tmpuf,-1)
             
        lhs = tmpuf 
        rhs = u0
 
        r = rhs - lhs
        p = r
#         for pp in range(0,5):
        Ap = self.forwardbackward(
                p*self.st['sensemap'])*self.st['sensemap'].conj()
        if self.st['senseflag'] == 1:
            Ap=CombineMulti(Ap,-1)
        
#         Ap = Ap - self.LMBD*self.do_laplacian(p, uker) + 2.0*self.gamma*p # a small constraint
        
        residue_square =numpy.sum((r.conj()*r)[:])
        residue_projection = numpy.sum((p.conj()*Ap)[:])
        alfa_k = residue_square/residue_projection
        
        
        print('r',residue_square,'alpha_k',alfa_k)
        #alfa_k = 0.3
        u_k_1 = u
        
        
        uf = uf + alfa_k * p
        r2 = r - alfa_k * Ap
        
        beta_k = numpy.sum( (r2.conj()*r2)[:] ) / residue_square
        r = r2
        p = r + beta_k * p
        if self.debug==0:
            pass
        else:
            print('leaving ext_update')
#         uf = uf + p*ctrl_factor*err#*(outer+1)            
        murf = uf 
         
        return (u, murf, uf, u_k_1) 
    def _external_update(self,u, uf, u0, u_k_1, outer): # overload the update function
 
# #          
#         if self.nufft_type=='SD':
#         if self.nufft_type==0:
# #         else: #self.nufft_type=='T'
#         tmpuf=self.backward(self.forward(
#                         u*self.st['sensemap']) )*self.st['sensemap'].conj()  
#         else: #                   
        tmpuf=self.forwardbackward(
                u*self.st['sensemap'])*self.st['sensemap'].conj()

                                   
        if self.st['senseflag'] == 1:
            tmpuf=CombineMulti(tmpuf,-1)
             
#         err = (checkmax(tmpuf,self.debug)-checkmax(u0,self.debug) )/checkmax(u0,self.debug)
#         err = (checkmax(tmpuf - u0,self.debug) )/checkmax(u0,self.debug)
        err = numpy.abs( numpy.mean( (tmpuf - u0).flatten()**2 )/numpy.mean( u0.flatten()**2 ) )**0.5
        try:
            precondition = self.precondition
        except:
            precondition = 0
        
        if outer < precondition: # 
            x1, x2, err2=self.forwardbackward2(u*self.st['sensemap'])
            x1 = x1*self.st['sensemap'].conj()
            x2 = x2*self.st['sensemap'].conj()#/(self.st['sensemap'].conj()*self.st['sensemap']+1e-2)
            tmpuf_order2 = x1 - x2 
            if self.st['senseflag'] == 1:
                tmpuf_order2 = CombineMulti(tmpuf_order2,-1)
            self.err2 = err2#1.0/(outer+1)#/(1.0+numpy.exp(0.4*(1.0e-7+0.3 - uf)))
                    
        else:

            tmpuf_order2 = 0.0
            self.err2 = 0.0
            err2 = 0.0
                           
#         print('outer=',outer, 'err=',err, 'err2 = ',err2)
        print(err)
         
#         r = (u0  - tmpuf)*(1.0-self.err2) - tmpuf_order2*self.err2#/(outer+1)
        r = (u0  - tmpuf)   - tmpuf_order2*self.err2#/(outer+1)  

        p = r 
#         err = (checkmax(tmpuf)- checkmax(u0))/checkmax(u0) 
        err= numpy.abs(err)
        if self.debug==0:
            pass
        else:        
            print('err',err,self.err)
#         if (err < self.err):
#             uf = uf+p*err*0.1            
#         if numpy.abs(err) < numpy.abs(self.err):
        ctrl_factor = 1.0
        self.err = err
        self.ctrl = 1
        u_k_1 = u
#         else: 
#             ctrl_factor = 0.0#0.0#numpy.abs(self.err)
#             print('stopping .... at the ', outer, '-th iteration')
#             self.ctrl = 1
#             if self.debug==0:
#                 pass
#             else:            
#                 print('no function')
#             u = u_k_1
#         print('ctrl_factor=' ,ctrl_factor)
        uf = uf + p*ctrl_factor*err#*(outer+1)            
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
    raw = numpy.load('phantom_3D_128_128_128.npy')*2.0
#     numpy.save('testfile.npy',raw)
#     raw = numpy.load('testfile.npy')
# demonstrate the 64th slice
#     matplotlib.pyplot.imshow(raw[:,:,64],cmap=cm)
#     matplotlib.pyplot.show()
    print('max.image',numpy.max(raw[:]))
# load 3D k-space trajectory (sparse)   
    om = numpy.loadtxt('om3D2.txt')
    print('omshape',numpy.shape(om))
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
 
#     import pp
#     job_server = pp.Server()
# 
#     f1=job_server.submit(MyNufftObj.inverse,(K_data, 1.0, 0.1, 0.01,3, 5),
#                          modules = ('numpy','pyfftw','pynufft'),globals=globals())
# #     f2=job_server.submit(MyNufftObj.inverse,(numpy.sqrt(K_data)*10+(0.0+0.1j), 1.0, 0.05, 0.01,3, 20),
# #                          modules = ('numpy','pyfftw','pynufft'),globals=globals())
# 
#     image1 = f1()    
# #     image2 = f2()
    MyNufftObj.precondition = 0
    image1 = MyNufftObj.pseudoinverse(K_data, 1.0, 10, 0.01,  2, 5)
#     image1 = MyNufftObj.pseudoinverse2(K_data)
     
#     image1 = MyNufftObj.pseudoinverse(K_data,1.0, 0.05, 0.001, 1,10)
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
def test_prolate():
 
    import numpy 
    import matplotlib#.pyplot
    cm = matplotlib.cm.gray
    # load example image    
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)
#     image = numpy.loadtxt('phantom_256_256.txt') 
#     matplotlib.pyplot.imshow(image[224-128:224, 64:192] ,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.show()    
#     image[128,128]= 1.0   
    N =128
    Nd =(N,N) # image space size
    import phantom
    image = phantom.phantom(Nd[0])
    Kd =(2*N,2*N) # k-space size   
    Jd =(7,7) # interpolation size
    import KspaceDesign.Propeller
    nblade =8
    theta=180.0/nblade #111.75 #
    propObj=KspaceDesign.Propeller.Propeller( N ,32,  nblade,   theta,  1,  1)
    om = propObj.om
    # load k-space points
#     om = numpy.loadtxt('om.txt')
#     om = numpy.loadtxt('om.gold')
     
    #create object
    NufftObj = pynufft(om, Nd,Kd,Jd)
    
    
    
    
    
#     NufftObj = pynufft(om, (256,256),(512,512),Jd) 
    precondition = 1
    factor = 0.1
    NufftObj.factor = factor
    NufftObj.precondition = precondition
    # simulate "data"
    data= NufftObj.forward(image )
    
    B = numpy.ones(image.shape)#*1.0/(N**2)
    for pp in xrange(0,Nd[0]):
        for qq in xrange(0,Nd[1]):
            if (pp - Nd[0]/2.0)**2 +(qq - Nd[1]/2.0)**2 > ( N/2.0 )**2:
                B[pp,qq] = 0.0
    A = numpy.ones(NufftObj.st['Kd'])#*1.0/(N**2)
    for pp in xrange(0,NufftObj.st['Kd'][0]):
        for qq in xrange(0,NufftObj.st['Kd'][1]):
            if (pp - NufftObj.st['Kd'][0]/2.0)**2 +(qq - NufftObj.st['Kd'][1]/2.0)**2 > ( N )**2:
                A[pp,qq] = 0.0    
                
    A = numpy.reshape(A,(numpy.size(A),1),order='F')            
#     matplotlib.pyplot.imshow(B)
#     matplotlib.pyplot.show()

    
#     Now doing the Arnold Iteration to find out the largest eigen vector
#     for prolate
        
    import numpy.linalg

    R_1 = NufftObj.st['W']**0.0
    R_2 = NufftObj.st['W']


    bbb = NufftObj.st['W'].copy()# initialize the vector
    for restart_times in xrange(0,3):
        iter_times = 3
        q  =numpy.empty((NufftObj.st['M'],iter_times))
        qk_1 = bbb 
        for kkk in xrange(0,iter_times):
            print('iteration',kkk)
            IMAGE1=NufftObj.backward( R_1*qk_1)
      
            IMAGE1[:,:,0]=B*IMAGE1[:,:,0]
            SIG1 = NufftObj.forward( IMAGE1 )
            qk = R_1*SIG1
            q[:,kkk:kkk+1] = qk
            for jjj in xrange(0,kkk):
                hj_k_1 = numpy.sum(q[:,jjj:jjj+1].conj()*qk)
                qk = qk - hj_k_1 * q[:,jjj:jjj+1]
            hk_k_1 = numpy.linalg.norm(qk)
            qk = qk/hk_k_1
            print('error = ',numpy.linalg.norm(qk-qk_1))
            numpy.save('prolate',qk)
            qk_1 = qk
        bbb=qk
         
    try:
        qk = numpy.load('prolate.npy')
    except:
        pass
            
    NufftObj.st['W'] =   bbb**1.0*NufftObj.st['W']
    
#     bbb = numpy.ones((N,N,1)) # initialize the vector
#     R_1 = NufftObj.st['W']**0.5
#     R_2 = NufftObj.st['W']
# 
#     for restart_times in xrange(0,16):
#         iter_times = 3
#         q  =numpy.empty((N,N,iter_times))
#         qk_1 = bbb 
#         for kkk in xrange(0,iter_times):
#             print('iteration',kkk)
#             SIG1=NufftObj.Nd2Kd( bbb,0)
#             print(numpy.shape(SIG1))
#             SIG1 =A*SIG1 
#             IMAGE1 = NufftObj.Kd2Nd( SIG1 ,0)
#             qk = IMAGE1
#             q[:,:,kkk ] = qk[:,:,0]
#             for jjj in xrange(0,kkk):
#                 hj_k_1 = numpy.sum(q[:,:, jjj ].conj()*qk[:,:,0])
#                 qk[:,:,0] = qk[:,:,0] - hj_k_1 * q[:,:, jjj ]
#             hk_k_1 = numpy.linalg.norm(qk[:,:,0])
#             qk = qk/hk_k_1
#             print('error = ',numpy.linalg.norm(qk-qk_1))
#             numpy.save('prolate',qk)
#             qk_1 = qk
#         bbb=qk
#         
#     try:
#         qk = numpy.load('prolate.npy')
#     except:
#         pass
#            
#     NufftObj.st['W'] =   NufftObj.st['W']*NufftObj.forward(bbb)
    
        
    LMBD =5.0
    nInner=5
    nBreg = 14
    
#     mu=1.0
    gamma = 0.001
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 
    NufftObj.st['senseflag'] = 1
#     NufftObj = pynufft(om, (256,256),(512,512),Jd,n_shift=(192,96))
#     NufftObj = pynufft(om, (256,256),(512,512),Jd)  
    image_blur = NufftObj.backward2(data)  
    

     
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
#     image_recon = NufftObj.pseudoinverse2(data)
#     image_blur = NufftObj.backward2(data)
#     image_recon = Normalize(numpy.real(image_recon))*  1.1# *1.3
#     image_blur=Normalize(numpy.real(image_blur[...,0]))*1.15
#     print(numpy.shape(image_recon),numpy.shape(image_blur))
#     matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
#     matplotlib.pyplot.show()

    # display images

#     matplotlib.pyplot.subplot(2,2,1)
#     matplotlib.pyplot.imshow(image[85:215,73:203],
#                              norm = norm,cmap =cm,interpolation = 'nearest')
#     matplotlib.pyplot.imshow( numpy.abs(image_blur[85:215,73:203, 0].real-image[85:215,73:203]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.show()
    
#     matplotlib.pyplot.title('PICO')   
#     matplotlib.pyplot.subplot(2,2,3)
#     matplotlib.pyplot.figure(1)
#     matplotlib.pyplot.imshow((image_recon[224-128:224, 64:192]).real,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

  
#     matplotlib.pyplot.subplot(2,2,2)
#     matplotlib.pyplot.figure(2)
#     matplotlib.pyplot.imshow(numpy.real(image_blur[224-128:224, 64:192]).real,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PIPE density compensation')  
#     matplotlib.pyplot.show()
    matplotlib.pyplot.plot(numpy.log( bbb[:,0]),'b')
    matplotlib.pyplot.plot(numpy.log( R_2[:,0]),'r:')
    matplotlib.pyplot.show()
        
    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.imshow((image_recon).real,
                              cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

  
#     matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.figure(4)
    matplotlib.pyplot.imshow(numpy.real(image_blur[...,0]),
                              cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('PIPE density compensation')  
#     matplotlib.pyplot.show()    

#     matplotlib.pyplot.figure(5)
#     matplotlib.pyplot.imshow(numpy.abs( (image_recon).real -  image),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

#     matplotlib.pyplot.figure(6)
# #     matplotlib.pyplot.subplot(2,2,2)
#     matplotlib.pyplot.imshow( numpy.abs(numpy.real(image_blur).real -  image),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PIPE density compensation')  
# #     matplotlib.pyplot.show()
#     matplotlib.pyplot.figure(7)    
#     matplotlib.pyplot.imshow(numpy.abs(image_recon[224-128:224, 64:192].real - image[224-128:224, 64:192]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

#     matplotlib.pyplot.figure(8)  
# #     matplotlib.pyplot.subplot(2,2,2)
#     matplotlib.pyplot.imshow(numpy.abs(numpy.real(image_blur[224-128:224, 64:192]).real- image[224-128:224, 64:192] ),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PIPE density compensation')  
    matplotlib.pyplot.show()
    
#     matplotlib.pyplot.title('blurred image') 
#     matplotlib.pyplot.subplot(2,2,4)
#     matplotlib.pyplot.imshow( numpy.abs(image_recon[85:215,73:203].real-image[85:215,73:203]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('residual error') 
#      
#     matplotlib.pyplot.show()    
#     mayavi.mlab.imshow()  
def test_2D():
 
    import numpy 
    import matplotlib#.pyplot
    cm = matplotlib.cm.gray
    # load example image    
    norm=matplotlib.colors.Normalize(vmin=-0.0, vmax=1) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)
    import cPickle
#     image = numpy.loadtxt('phantom_256_256.txt') 
#     matplotlib.pyplot.imshow(image[224-128:224, 64:192] ,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.show()    
#     image[128,128]= 1.0   
    N = 512
    Nd =(N,N) # image space size
    import phantom
    image = phantom.phantom(Nd[0])
    Kd =(2*N,2*N) # k-space size   
    Jd =(6,6) # interpolation size
    import KspaceDesign.Propeller
    nblade =  20#numpy.round(N*3.1415/24/2)
#     print('nblade',nblade)
    theta=180.0/nblade #111.75 #
    propObj=KspaceDesign.Propeller.Propeller( N ,24,  nblade,   theta,  1,  -1)
    om = propObj.om
     
    NufftObj = pynufft(om, Nd,Kd,Jd )
#         
         
    f = open('nufftobj.pkl','wb')
    cPickle.dump(NufftObj,f,2)
    f.close()
# #      
    f = open('nufftobj.pkl','rb')
    NufftObj = cPickle.load(f)
    f.close()

#     f2 = open('dense_density.pkl','wb')
#     cPickle.dump(NufftObj.st['w'],f2,-1)
#     f2.close()
#     print('size of sparse amtrix',NufftObj.st['p'].data.nbytes + NufftObj.st['p'].indptr.nbytes + NufftObj.st['p'].indices.nbytes)
#     print('size of dense matrix', NufftObj.st['w'].nbytes)


#     NufftObj = pynufft(om, (64,64),(512,512),Jd) 
    precondition = 0
    factor = 0.001
    NufftObj.factor = factor
    NufftObj.precondition = precondition
    # simulate "data"
    data= NufftObj.forward(image )
#     data= NufftObj.true_forward(image )  
    
    data_shape = (numpy.shape(data))
    power_of_data= numpy.max(numpy.abs(data))
#     purpose = 2 # noisy
    purpose = 1 # noisy
#     purpose = 0 # noise-free
    data = data + purpose*1e-3*power_of_data*(numpy.random.randn(data_shape[0],data_shape[1])+1.0j*numpy.random.randn(data_shape[0],data_shape[1]))
    LMBD =1
    nInner=2
    nBreg = 25
    
    mu=1.0
    gamma = 0.001
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 
    NufftObj.st['senseflag'] = 1
    NufftObj.initialize_gpu() 
#     NufftObj = pynufft(om, (64,64),(96,96),Jd)#,n_shift=(192,96))
#     NufftObj = pynufft(om, (256,256),(512,512),Jd)  
    image_blur = NufftObj.backward2(data)  
    

#     image_recon_inner_10 = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma, 10, nBreg)
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
    image_recon = Normalize(image_recon.real)
    image_blur=Normalize(numpy.real(image_blur[...,0]))*1.2 #4.9

#     my_scale = numpy.mean( image_blur[round(25*N/256):round(46*N/256), round(109*N/256):round(149*N/256)].flatten())
#     image_blur =image_blur / my_scale /5.0 
    
    my_scale = numpy.mean( image_recon[round(25*N/256):round(46*N/256), round(109*N/256):round(149*N/256)].flatten())
    image_recon =image_recon / my_scale /5.0 
#     print(my_scale)
#     image_recon = NufftObj.pseudoinverse2(data)
#     image_blur = NufftObj.backward2(data)
#     if purpose == 0:
#         image_recon = Normalize(image_recon.real)*1.05#2.2 # *1.3
#     elif purpose == 1:
#         image_recon = Normalize(image_recon.real)*1.1#54#2.2 # *1.3
# #     image_recon_inner_10 = Normalize(numpy.real(image_recon_inner_10))#*  1.1# *1.3
#     image_blur=Normalize(numpy.real(image_blur[...,0]))*1.2#4.9
#     if purpose == 0:
#         image_recon = Normalize(image_recon.real)*1.3#2.2 # *1.3
#     elif purpose == 1:
#         image_recon = Normalize(image_recon.real)*1.0#54#2.2 # *1.3
#     elif purpose == 2:
#         image_recon = Normalize(image_recon.real)*1.1#54#2.2 # *1.3     

#     image_recon_inner_10 = Normalize(numpy.real(image_recon_inner_10))#*  1.1# *1.3
    disp_image = numpy.concatenate( 
                                       (
#                         numpy.concatenate( ((  image_recon.real) , numpy.real(-image_recon  + image) ), axis = 0 ),
                        numpy.concatenate( ((  image_recon.real) , numpy.abs(image_recon  - image) ), axis = 0 ),
                        numpy.concatenate( ((  image_blur.real) ,numpy.abs(image_blur  - image) ), axis = 0  ) ), axis =1 )
    import scipy.misc 
    scipy.misc.imsave('/home/sram/Cambridge_2012/WORD_PPTS/PROP/generalize_inverse/MRM/FIGS/noise_XX.png',disp_image)
    fig, ax = matplotlib.pyplot.subplots()
    cax = ax.imshow( disp_image 
                     , norm = norm,
            cmap= cm,interpolation = 'nearest')
    cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['< 0', '0.5', '> 1'])# vertically oriented colorbar
    matplotlib.pyplot.savefig('/home/sram/Cambridge_2012/WORD_PPTS/PROP/generalize_inverse/MRM/FIGS/noise_XX.eps', bbox_inches='tight')
    
#     matplotlib.pyplot.figure(1)
#     matplotlib.pyplot.subplot(n_x,n_y,tt)        
#     matplotlib.pyplot.imshow(numpy.abs(image_recon.real) ,
#                             norm = norm,
#             cmap= cm,interpolation = 'nearest')  
    matplotlib.pyplot.show()
    
#     matplotlib.pyplot.title('blurred image') 
#     matplotlib.pyplot.subplot(2,2,4)
#     matplotlib.pyplot.imshow( numpy.abs(image_recon[85:215,73:203].real-image[85:215,73:203]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('residual error') 
#      
#     matplotlib.pyplot.show() 

def rFOV_2D():
 
    import numpy 
    import matplotlib#.pyplot
    cm = matplotlib.cm.gray
    # load example image    
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)
#     image = numpy.loadtxt('phantom_256_256.txt') 
#     matplotlib.pyplot.imshow(image[224-128:224, 64:192] ,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.show()    
#     image[128,128]= 1.0   
    N = 384
    Nd =(N,N) # image space size
    import phantom
    image = phantom.phantom(Nd[0])
    Kd =(2*N,2*N) # k-space size   
    Jd =(5,5) # interpolation size
    import KspaceDesign.Propeller
    nblade = 24# numpy.round(N*3.1415/24)
#     print('nblade',nblade)
    theta=180.0/nblade #111.75 #
    propObj=KspaceDesign.Propeller.Propeller( N ,24,  nblade,   theta,  1,  -1)
    om = propObj.om
#     print('m',numpy.size(om))
    # load k-space points
#     om = numpy.loadtxt('om.txt')
#     om = numpy.loadtxt('om.gold')
     
    #create object
    NufftObj = pynufft(om, Nd,Kd,Jd )
#     matplotlib.pyplot.imshow(scipy.fftpack.fftshift( numpy.reshape( numpy.abs( NufftObj.st['p'].conj().T.dot( NufftObj.st['W']**3)),Kd,order='F'),axes=(0,1)),
#                               cmap= cm)
#     matplotlib.pyplot.show()
# save the sparse matrix 
#     import cPickle
#     f = open('spmatrix.pkl','wb')
#     cPickle.dump(NufftObj.st['p'],f,-1)
#     f.close()
#      
#     f2 = open('dense_density.pkl','wb')
#     cPickle.dump(NufftObj.st['w'],f2,-1)
#     f2.close()
#     print('size of sparse amtrix',NufftObj.st['p'].data.nbytes + NufftObj.st['p'].indptr.nbytes + NufftObj.st['p'].indices.nbytes)
#     print('size of dense matrix', NufftObj.st['w'].nbytes)


#     NufftObj = pynufft(om, (64,64),(512,512),Jd) 
    precondition = 2
    factor = 0.01
    NufftObj.factor = factor
    NufftObj.precondition = precondition
    # simulate "data"
    data= NufftObj.forward(image )
    LMBD =1
    nInner=2
    nBreg = 25
    
    mu=1.0
    gamma = 0.001
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 
    NufftObj.st['senseflag'] = 1


    rFOV = 256
    NufftObj = pynufft(om, (256,256),(512,512),Jd,n_shift=(0,0))
#     NufftObj.st['Nd'] = (rFOV,rFOV)
#     NufftObj.st['sn'] = NufftObj.st['sn'][N/2-rFOV/2:N/2+rFOV/2 ,N/2-rFOV/2:N/2+rFOV/2 ]
    
#     NufftObj.linear_phase(om, (32,32), NufftObj.st['M'])# = pynufft(om, (64,64),(512,512),Jd,n_shift=(0,32))
    NufftObj.linear_phase(  (64, 0) )# = pynufft(om, (64,64),(512,512),Jd,n_shift=(0,32))
   
    NufftObj.initialize_gpu() 
#     NufftObj = pynufft(om, (256,256),(512,512),Jd)  
    image_blur = NufftObj.backward2(data)  
    

     
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
#     NufftObj.linear_phase(om, (-0, 0), NufftObj.st['M'])# = pynufft(om, (64,64),(512,512),Jd,n_shift=(0,32))
#    
#     NufftObj.initialize_gpu()  
#     image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)   
#     image_recon = NufftObj.pseudoinverse2(data)
#     image_blur = NufftObj.backward2(data)
    image_recon = Normalize(numpy.real(image_recon))# *1.3
    image_blur=Normalize(numpy.real(image_blur[...,0]))#*1.15
#     print(numpy.shape(image_recon),numpy.shape(image_blur))
#     matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
#     matplotlib.pyplot.show()

    # display images

# #     matplotlib.pyplot.subplot(2,2,1)
# #     matplotlib.pyplot.imshow(image[85:215,73:203],
# #                              norm = norm,cmap =cm,interpolation = 'nearest')
# #     matplotlib.pyplot.imshow( numpy.abs(image_blur[85:215,73:203, 0].real-image[85:215,73:203]),
# #                              norm = norm2,cmap= cm,interpolation = 'nearest')
# #     matplotlib.pyplot.show()
#     
# #     matplotlib.pyplot.title('PICO')   
# #     matplotlib.pyplot.subplot(2,2,3)
# #     matplotlib.pyplot.figure(1)
# #     matplotlib.pyplot.imshow((image_recon[224-128:224, 64:192]).real,
# #                              norm = norm,cmap= cm,interpolation = 'nearest')
# #     matplotlib.pyplot.title('PICO')    
# #     matplotlib.pyplot.show()
# 
#   
# #     matplotlib.pyplot.subplot(2,2,2)
# #     matplotlib.pyplot.figure(2)
# #     matplotlib.pyplot.imshow(numpy.real(image_blur[224-128:224, 64:192]).real,
# #                              norm = norm,cmap= cm,interpolation = 'nearest')
# #     matplotlib.pyplot.title('PIPE density compensation')  
# #     matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.imshow((image_recon).real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

  
#     matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.figure(4)
    matplotlib.pyplot.imshow(numpy.real(image_blur).real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('PIPE density compensation')  
#     matplotlib.pyplot.show()    

#     matplotlib.pyplot.figure(5)
#     matplotlib.pyplot.imshow(numpy.abs( (image_recon).real -  image),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

#     matplotlib.pyplot.figure(6)
# #     matplotlib.pyplot.subplot(2,2,2)
#     matplotlib.pyplot.imshow( numpy.abs(numpy.real(image_blur).real -  image),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PIPE density compensation')  
# #     matplotlib.pyplot.show()
#     matplotlib.pyplot.figure(7)    
#     matplotlib.pyplot.imshow(numpy.abs(image_recon[224-128:224, 64:192].real - image[224-128:224, 64:192]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

#     matplotlib.pyplot.figure(8)  
# #     matplotlib.pyplot.subplot(2,2,2)
#     matplotlib.pyplot.imshow(numpy.abs(numpy.real(image_blur[224-128:224, 64:192]).real- image[224-128:224, 64:192] ),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PIPE density compensation')  
    matplotlib.pyplot.show()
    
#     matplotlib.pyplot.title('blurred image') 
#     matplotlib.pyplot.subplot(2,2,4)
#     matplotlib.pyplot.imshow( numpy.abs(image_recon[85:215,73:203].real-image[85:215,73:203]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('residual error') 
#      
#     matplotlib.pyplot.show() 
def test_radial():
 
    import numpy 
    import matplotlib#.pyplot
    cm = matplotlib.cm.gray
    # load example image    
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)
#     image = numpy.loadtxt('phantom_256_256.txt') 
#     matplotlib.pyplot.imshow(image[224-128:224, 64:192] ,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.show()    
#     image[128,128]= 1.0   
    N = 256
    Nd =(N,N) # image space size
    import phantom
    image = phantom.phantom(Nd[0])
    Kd =(2*N,2*N) # k-space size   
    Jd =(8,8) # interpolation size
    import KspaceDesign.Radial

    nblade =  48
    theta=  68.754 #
    propObj=KspaceDesign.Radial.Radial( N ,  nblade,   theta,   -1)
    om = propObj.om
    # load k-space points
#     om = numpy.loadtxt('om.txt')
#     om = numpy.loadtxt('om.gold')
     
    #create object
    NufftObj = pynufft(om, Nd,Kd,Jd)
#     NufftObj = pynufft(om, (256,256),(512,512),Jd) 
    precondition = 1
    factor = 0.1
    NufftObj.factor = factor
    NufftObj.nufft_type = 1
    NufftObj.precondition = precondition
    # simulate "data"
    data= NufftObj.forward(image )
    LMBD =5
    nInner=2
    nBreg =50
    
    mu=1.0
    gamma = 0.001
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 
    NufftObj.st['senseflag'] = 1
#     NufftObj = pynufft(om, (256,256),(512,512),Jd,n_shift=(192,96))
#     NufftObj = pynufft(om, (256,256),(512,512),Jd)  
    image_blur = NufftObj.backward2(data)  
    

     
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
#     image_recon = NufftObj.pseudoinverse2(data)
#     image_blur = NufftObj.backward2(data)
    image_recon = Normalize(numpy.real(image_recon))#*  1.1# *1.3
    image_blur=Normalize(numpy.real(image_blur[...,0]))#*1.15
    print(numpy.shape(image_recon),numpy.shape(image_blur))
    matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
    matplotlib.pyplot.show()
 

    # display images

#     matplotlib.pyplot.subplot(2,2,1)
#     matplotlib.pyplot.imshow(image[85:215,73:203],
#                              norm = norm,cmap =cm,interpolation = 'nearest')
#     matplotlib.pyplot.imshow( numpy.abs(image_blur[85:215,73:203, 0].real-image[85:215,73:203]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.show()
    
#     matplotlib.pyplot.title('PICO')   
#     matplotlib.pyplot.subplot(2,2,3)
#     matplotlib.pyplot.figure(1)
#     matplotlib.pyplot.imshow((image_recon[224-128:224, 64:192]).real,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

  
#     matplotlib.pyplot.subplot(2,2,2)
#     matplotlib.pyplot.figure(2)
#     matplotlib.pyplot.imshow(numpy.real(image_blur[224-128:224, 64:192]).real,
#                              norm = norm,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PIPE density compensation')  
#     matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.imshow((image_recon).real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

  
#     matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.figure(4)
    matplotlib.pyplot.imshow(numpy.real(image_blur).real,
                             norm = norm,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('PIPE density compensation')  
#     matplotlib.pyplot.show()    

#     matplotlib.pyplot.figure(5)
#     matplotlib.pyplot.imshow(numpy.abs( (image_recon).real -  image),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

#     matplotlib.pyplot.figure(6)
# #     matplotlib.pyplot.subplot(2,2,2)
#     matplotlib.pyplot.imshow( numpy.abs(numpy.real(image_blur).real -  image),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PIPE density compensation')  
# #     matplotlib.pyplot.show()
#     matplotlib.pyplot.figure(7)    
#     matplotlib.pyplot.imshow(numpy.abs(image_recon[224-128:224, 64:192].real - image[224-128:224, 64:192]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PICO')    
#     matplotlib.pyplot.show()

#     matplotlib.pyplot.figure(8)  
# #     matplotlib.pyplot.subplot(2,2,2)
#     matplotlib.pyplot.imshow(numpy.abs(numpy.real(image_blur[224-128:224, 64:192]).real- image[224-128:224, 64:192] ),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('PIPE density compensation')  
    matplotlib.pyplot.show()
    
#     matplotlib.pyplot.title('blurred image') 
#     matplotlib.pyplot.subplot(2,2,4)
#     matplotlib.pyplot.imshow( numpy.abs(image_recon[85:215,73:203].real-image[85:215,73:203]),
#                              norm = norm2,cmap= cm,interpolation = 'nearest')
#     matplotlib.pyplot.title('residual error') 
#      
#     matplotlib.pyplot.show() 
def test_1D():
# import several modules
    import numpy 
    import matplotlib#.pyplot
 
#create 1D curve from 2D image
    image = numpy.loadtxt('phantom_256_256.txt') 
    image = image[:,128]
#determine the location of samples
    om = numpy.loadtxt('om1D.txt')[0:]
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
    NufftObj.precondition = 1
    image_recon = NufftObj.pseudoinverse(data, 1.0, 10, 0.001,15,15)
 
#Showing histogram of sampling locations
#     matplotlib.pyplot.hist(om,20)
#     matplotlib.pyplot.title('histogram of the sampling locations')
#     matplotlib.pyplot.show()
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
    matplotlib.pyplot.axis([0 ,256 , -0.1, 0.1]) 
    matplotlib.pyplot.title('residual')
#     matplotlib.pyplot.subplot(2,2,4)
#     matplotlib.pyplot.plot(numpy.abs(data))  
    matplotlib.pyplot.show()  
 
# def test_Dx():
#     u = numpy.ones((128,128,128,1),dtype = dtype)
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
     
#     image_recon = NewObj.pseudoinverse(data, 1.0, 0.05, 0.01,3, 20)
     
     
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
# def get_wavelet(image):
# #     import pywt#,numpy,matplotlib.pyplot    
#     coeffs = pywt.wavedec2(image, 'haar',level = 3)
# #     new_coeffs =  (coeffs[0]*0.0,)+coeffs[1:]
#     cA, cH1, cH2, cH3 = coeffs
#     
#     print('shape of cA',cA.shape)
#     print('shape of cH1',numpy.shape(cH1))
# 
# #     new_image = pywt.idwt2(coeffs,'haar')
#     
#     wave_image0 = numpy.concatenate((cA, cH1[0]),axis = 0)
#     wave_image1 = numpy.concatenate((cH1[1],cH1[2]),axis = 0)
#     wave_image =  numpy.concatenate((wave_image0, wave_image1),axis = 1)
#     wave_image0 = numpy.concatenate((wave_image, cH2[0]),axis = 0)
#     wave_image1 = numpy.concatenate((cH2[1],cH2[2]),axis = 0)    
#     wave_image =  numpy.concatenate((wave_image0, wave_image1),axis = 1)
#     wave_image0 = numpy.concatenate((wave_image, cH3[0]),axis = 0)
#     wave_image1 = numpy.concatenate((cH3[1],cH3[2]),axis = 0)    
#     wave_image =  numpy.concatenate((wave_image0, wave_image1),axis = 1)   
#     return wave_image  
# def get_wavelet_H(wave_image):
#     N = numpy.shape(wave_image)[0]
# #     new_image = numpy.empty_like(wave_image) 
#     p0=  wave_image[N/2:, 0:N/2]
#     p1 =  wave_image[0:N/2, N/2:]
#     p2 =  wave_image[N/2:, N/2:]
#     cA = wave_image[0:N/2, 0:N/2]
#     cH3 = (p0,p1,p2)
# 
#     
#     
#     N = numpy.shape(cA)[0]
#     p0=  cA[N/2:, 0:N/2]
#     p1 =  cA[0:N/2, N/2:]
#     p2 =  cA[N/2:, N/2:]
#     cH2 = (p0,p1,p2)    
#     cA = cA[0:N/2, 0:N/2]
#     
#     N = numpy.shape(cA)[0]
#     p0=  cA[N/2:, 0:N/2]
#     p1 =  cA[0:N/2, N/2:]
#     p2 =  cA[N/2:, N/2:]
#     cH1 = (p0,p1,p2)    
#     cA = cA[0:N/2, 0:N/2]
#       
#     new_image =  pywt.waverec2((cA,cH1, cH2, cH3),'haar')
#      
#      
#     return new_image 
def test_wavelet():

    image = numpy.loadtxt('phantom_256_256.txt') 
    wave_image = get_wavelet(image)
    
    new_image = get_wavelet_H(wave_image)
#
    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.imshow(wave_image[:,:,0],norm = norm)

    matplotlib.pyplot.show()
  
    matplotlib.pyplot.figure(4)
    matplotlib.pyplot.imshow(new_image[:,:,0],norm = norm)

    matplotlib.pyplot.show()    
def histeq(im,nbr_bins=256):
    """  Histogram equalization of a grayscale image. """
    # get image histogram
    imhist,bins = numpy.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 1.0 * cdf / cdf[-1] # normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = scipy.interp(im.flatten(),bins[:-1],cdf)
    
    return im2.reshape(im.shape)#, cdf  
def test_SR():
 
    cm = matplotlib.cm.gray
#   load raw data, which is 3D shapp-logan phantom
    raw = numpy.load('phantom_3D_128_128_128.npy')*2.0
#     numpy.save('testfile.npy',raw)
#     raw = numpy.load('testfile.npy')
# demonstrate the 64th slice
    matplotlib.pyplot.imshow(raw[:,:,64],cmap=cm)
    matplotlib.pyplot.show()  
if __name__ == '__main__':
    import cProfile
#     test_wavelet()
#     test_1D()
#     test_prolate()
    test_2D()
#     x = numpy.random.random(1024)
#     print(DFT_point(x, -0))
#     print( numpy.allclose(DFT_slow(x), numpy.fft.fft(x)) )
#     DFT_slow(x)
#     cProfile.run('rFOV_2D()')
#     rFOV_2D()
    
#     import scipy.sparse
#     A=scipy.sparse.lil_matrix((5,4))
#     A[3,0:2] = 3.2
#     print(A[:,:].dot(A[:,:].T).todense())
    
#     test_radial()
#     test_3D()
#     show_3D()
#     test_Dx()
#     cProfile.run('test_3D()') 
#     cProfile.run('test_2D()')    
#     cProfile.run('test_2D_multiprocessing()')
#     test_SR()