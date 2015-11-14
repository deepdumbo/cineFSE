import numpy
import pyfftw

import scipy
import scipy.fftpack
# import scipy.fftpack._fftpack
# import CsSolver
# import CsSolver
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('../CsTransform')

import CsTransform.pynufft 
import matplotlib.pyplot

cmap=matplotlib.cm.gray
norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

class TemporalConstraint(CsTransform.pynufft.pynufft):
    def __init__(self,om, Nd, Kd,Jd):
        CsTransform.pynufft.pynufft.__init__(self,om, Nd, Kd,Jd)
        self.gpu_flag = 0
#         self.compute = 'fast'
#         self.compute = 'reliable'
    '''
    2D cine images + PPG external guided
    
    X-axis is not iterated, so it is only a 2D sparse recovery problem
    data space: sparse ky-t (f, )
    target space: y-f space (u, u0)
                    
    Remark: f is the data (kx,ky,t) which are sparse, and it has been subtracted for 
                   average images.
            u is the y-f space
            u_bar is the DC of y-f space
            u0 the first estimation of 
    
    CsTransform.forward: transform(y-f  ->   ky-t )
    CsTransform.backward: transform(ky-t -> y-f )
    

    self.make_sense: making sensitivity map from TSE images
                        input: x-y-coil 3D data
                        output: x-y-f-coil 4D sensemap
    
    constraints: 1) |u- u_bar|_1: the L1 of |signals - DC| terms
                 2) |part_der_y(u)|_1: the L1 of partial derive of u 
 
    '''
#    def __init__(self,CsTransform, data, mu, LMBD, gamma, nInner, nBreg):
#        CineBaseSolver.__init__(CsTransform, data, mu, LMBD, gamma, nInner, nBreg)
    def data2rho(self, dd_raw,dim_x,dim_y,dim_t,n_coil):
        '''
         dd_raw data in kx-ky-t space
                dd_raw, shape (512(kx), 1056(ky-t), 4coil)
        '''
        if dd_raw.ndim != 3:
            print(' dimension is wrong! ')
        
        dd_raw = numpy.transpose(dd_raw,(1,2,0)) # put the x-axis to the tail
        
        shape_dd_raw = numpy.shape(dd_raw)
        
        dd_raw = numpy.reshape(dd_raw,
                               (shape_dd_raw[0],)+ (shape_dd_raw[1]*shape_dd_raw[2],),
                                order='F')
        print('in data2rho,dd_raw.shape',dd_raw.shape)
        output_data = self.backward(dd_raw) # (dim_y, dim_t*n_coil*dim_x)
        
    
        output_data = numpy.reshape(output_data, (dim_y,dim_t,n_coil,dim_x),order='F')
    
        output_data = numpy.transpose(output_data, (3,0,1,2)) 
        
        return output_data
    
    def FhF(self, dd_raw ,cineObj):
        '''
        2D FFT along y-f axes
        x-axis is not changed
        '''
        if dd_raw.ndim != 4:
            print(' dimension is wrong! ')
        
        dim_x = cineObj.dim_x
        dim_y = self.st['Nd'][0]
        dim_t = self.st['Nd'][1]
        n_coil = cineObj.ncoils
#         CsTransform = self.CsTransform
        
        dd_raw = numpy.transpose(dd_raw,(1,2,3,0)) # put the x-axis to the tail
        
        shape_dd_raw = numpy.shape(dd_raw)
        
        dd_raw = numpy.reshape(dd_raw,
                  (shape_dd_raw[0],)+ (shape_dd_raw[1],)+(shape_dd_raw[2]*shape_dd_raw[3],),
                        order='F')
        output_data = self.forwardbackward(dd_raw) 
    
#        output_data = CsTransform.forward(dd_raw) # (dim_y, dim_t*n_coil*dim_x)
#        
#        output_data = CsTransform.backward(output_data)
        
        #output_data = CsTransform.backward(output_data)
    
        output_data = numpy.reshape(output_data, (dim_y,dim_t,n_coil,dim_x),order='F')
    
        output_data = numpy.transpose(output_data, (3,0,1,2)) 

        return output_data
    
    def fun1(self,cineObj):
        '''
        CsTransform.backward
        from data to kx-y-f
        do I need this? 
        by connecting to data2rho(self, dd_raw,dim_x,dim_y,dim_t,n_coil,CsTransform):
        '''
        output=self.data2rho( cineObj.f , cineObj.dim_x , self.st['Nd'][0],self.st['Nd'][1],cineObj.ncoils)
        print('cineObj.f shape',cineObj.f.shape)
        return output
        
    def fun2(self,m):
        '''
        
        1)shift-kx
        2)ifft: along kx
        3)shift-x
        
        '''
        m = scipy.fftpack.ifftshift(m, axes=(0,)) # because kx energy is at centre
        
        shape_m = numpy.shape(m)

        m= pyfftw.interfaces.scipy_fftpack.ifftn(m, axes=(0,),threads=2, overwrite_x=True)

#         m = scipy.fftpack.ifftn(m,axes=(0,))
          
        m = scipy.fftpack.ifftshift(m, axes=(0,)) # because x-y-f are all shifted at center

        return m        
        
    def fun3(self,m):
        '''
        1)shift-f
        2)fft: along f
        3)(No shift)-t
        '''
        m = scipy.fftpack.fftshift(m, axes=(2,))
        m= pyfftw.interfaces.scipy_fftpack.fftn(m, axes=(2,),threads=2, overwrite_x=True)
        
#         m = scipy.fftpack.fftn(m,axes=(2,)) 

        return m
    
    def fun4(self,m):
        '''
        1D IFFT along temporal/frequency axis 
        inverse of self.fun3
        1)(No shift)-t
        2)fft: along f
        3)shift-f
        '''
        m= pyfftw.interfaces.scipy_fftpack.ifftn(m, axes=(2,),threads=2, overwrite_x=True)
#         m = scipy.fftpack.ifftn(m,axes=(2,)) 
        
        m = scipy.fftpack.ifftshift(m, axes=(2,)) 
     
        return m
#     def fast_fun4_FhF_fun3(self,m, cineObj):
#         m = scipy.fftpack.ifftn(m,axes=(2,)) 
#         
#         m = scipy.fftpack.ifftshift(m, axes=(2,)) 
#      
#         if m.ndim != 4:
#             print(' dimension is wrong! ')
#         
#         dim_x = cineObj.dim_x
#         dim_y = self.st['Nd'][0]
#         dim_t = self.st['Nd'][1]
#         n_coil = cineObj.ncoils
# #         CsTransform = self.CsTransform
#         
#         m = numpy.transpose(m,(1,2,3,0)) # put the x-axis to the tail
#         
#         shape_m = numpy.shape(m)
#         
#         m = numpy.reshape(m,
#                   (shape_m[0],)+ (shape_m[1],)+(shape_m[2]*shape_m[3],),
#                         order='F')
#         m = self.forwardbackward(m) 
#     
# #        output_data = CsTransform.forward(dd_raw) # (dim_y, dim_t*n_coil*dim_x)
# #        
# #        output_data = CsTransform.backward(output_data)
#         
#         #output_data = CsTransform.backward(output_data)
#     
#         m = numpy.reshape(m, (dim_y,dim_t,n_coil,dim_x),order='F')
#     
#         m = numpy.transpose(m, (3,0,1,2)) 
# 
#         m = scipy.fftpack.fftshift(m, axes=(2,))
#         
#         m = scipy.fftpack.fftn(m,axes=(2,)) 
#                    
#         return m
    def do_FhWFm(self,q,cineObj):
        '''
        do_FhWFm convole the input q with the ky-t sampling function 
        firstly, fun4() transform the x-y-t data into x-y-f 
        Secondly,  FhF() perform 2D convolution for x-y-f data
        Thirdly, fun3() transform the x-y-f data back into x-y-t
        
        input:
                q is the x-y-t data
        output:
                output_q is the x-y-t data
        '''
#         if self.compute == 'reliable':
            
        output_q = q
        
        # 1D FFT along temporal/frequency axis
        output_q = self.fun4(output_q)
        
         #  convolved by the 
        output_q = self.FhF(output_q,cineObj)
        output_q = self.fun3(output_q)
#         elif self.compute == 'fast':
#             output_q = q
#             output_q = self.fast_fun4_FhF_fun3(output_q, cineObj)
        return output_q    
    def fun5(self,m):
        '''
        1)shift-f:
        2)shift-x-y:
        2)fft2 along x-y, no shift -> kx ky f
        
        for 3D laplacian operator
        '''
        m = scipy.fftpack.ifftshift(m, axes=(0,1,2,))
        m= pyfftw.interfaces.scipy_fftpack.fftn(m, axes=(0,1,),threads=2, overwrite_x=True)
#         m = scipy.fftpack.fftn(m,axes =  (0,1,))
        
        return m
        
    def fun6(self,m):
        '''
        inverse of fun5
        1)ifft2 along kx-ky 
        2)shift along x & y
        3)shift along f

        '''
        
#         m = scipy.fftpack.ifftn(m,axes =  (0,1,)) 
        m= pyfftw.interfaces.scipy_fftpack.ifftn(m, axes=(0,1,),threads=2, overwrite_x=True)              
        m = scipy.fftpack.fftshift(m, axes=(0,1,2,))    
            
        return m
    

    def do_laplacian(self,q,uker):
        '''
        Laplacian of the input q
        
        '''
#        lapla_Wq=scipy.fftpack.fftshift(q,axes=(2,))
#        lapla_Wq=scipy.fftpack.fftn(lapla_Wq,axes=(0,1,))
#        lapla_Wq=lapla_Wq*uker
#        lapla_Wq=scipy.fftpack.ifftn(lapla_Wq,axes=(0,1,))
#        lapla_Wq=scipy.fftpack.ifftshift(q,axes=(2,))

        lapla_Wq = q
        lapla_Wq = self.fun4(lapla_Wq)
        lapla_Wq = self.fun5(lapla_Wq)
        lapla_Wq = lapla_Wq * uker
        lapla_Wq = self.fun6(lapla_Wq)
        lapla_Wq = self.fun3(lapla_Wq)
          
        return lapla_Wq
  

    def pseudoinverse(self, cineObj, mu, LMBD, gamma, nInner, nBreg): # main function of solver

        self.mu = mu
        self.LMBD = LMBD
        self.gamma = gamma
        self.nInner= nInner
        self.nBreg= nBreg
        u0=numpy.empty((cineObj.dim_x, self.st['Nd'][0], self.st['Nd'][1], cineObj.tse.shape[2]))
        
        print(numpy.shape(cineObj.tse))
        orig_num_ky=numpy.shape(cineObj.tse)[1]
        tse = cineObj.tse[:,orig_num_ky/2 - self.st['Nd'][0]/2 : orig_num_ky/2 + self.st['Nd'][0]/2,:]
        
        tse = CsTransform.pynufft.appendmat(tse,u0.shape[2])
        tse = numpy.transpose(tse,(0,1,3,2))
        print(self.st['Nd'][0])
        print('tse.shape',tse.shape)
        
#===============================================================================
# mask
#===============================================================================
        self.st = self.create_mask(u0)
        print('mask.shape',self.st['mask'].shape)
        
#        for jj in range(0,16):
#            matplotlib.pyplot.subplot(4,4,jj)
#            matplotlib.pyplot.imshow(self.st['mask'][...,jj,0].real)
#        matplotlib.pyplot.show()
        
#===============================================================================
        
        #estimate sensitivity maps by divided by rms images
        self.st = self._make_sense(cineObj.tse[:,orig_num_ky/2 - self.st['Nd'][0]/2 : orig_num_ky/2 + self.st['Nd'][0]/2,:]) # setting up sense map in st['sensemap']
        
        self.st['sensemap'] = CsTransform.pynufft.appendmat(self.st['sensemap'],u0.shape[2])
        self.st['sensemap'] = numpy.transpose(self.st['sensemap'],(0,1,3,2))
        
        #self.st['sensemap'] =self.st['sensemap'] * self.st['mask'] 
        print('self.sense.shape',self.st['sensemap'].shape)

#        for jj in range(0,16):
#            matplotlib.pyplot.subplot(4,4,jj)
#            matplotlib.pyplot.imshow(numpy.abs(self.st['sensemap'][...,jj,0]))
#        matplotlib.pyplot.show()       
         
        self.st['senseflag']=1 # turn-on sense, to get sensemap
        
        (u,uf)=self.kernel( cineObj, self.st , mu, LMBD, gamma, nInner, nBreg)
        return u
    def create_laplacian_kernel2(self,cineObj):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#        Note only the y-axis is used 
#         # related to constraint
#===============================================================================
        uker = numpy.zeros((cineObj.dim_x,)+self.st['Nd'][0:2],dtype=numpy.complex64)
        rows_kd = self.st['Nd'][0] # ky-axis
        #cols_kd = self.st['Kd'][1] # t-axis
#        uker[0,0] = 1.0
        uker[0,0,0] = -4.0
        uker[0,1,0]=1.0
        uker[0,-1,0]=1.0
        uker[1,0,0]=1.0
        uker[-1,0,0]=1.0        
        uker = scipy.fftpack.fftn(uker,axes=(0,1,2,)) # 256x256x16
        return uker
    def create_laplacian_kernel3(self,cineObj):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#        Note only the y-axis is used 
#         # related to constraint
#===============================================================================
        uker = numpy.ones((cineObj.dim_x,)+self.st['Nd'][0:2],dtype=numpy.complex64)
#        rows_kd = self.st['Nd'][0] # ky-axis
#        #cols_kd = self.st['Kd'][1] # t-axis
##        uker[0,0] = 1.0
#        uker[0,0,0] = -2.0
#        uker[0,0,1]=1.0
#        uker[0,0,-1]=1.0   
#             
#        uker = scipy.fftpack.fftn(uker,axes=(0,1,2,)) # 256x256x16
        for pp in range(0,self.st['Nd'][1]):
            
            uker[:,:,pp] =uker[:,:,pp] * ( (pp -  self.st['Nd'][1]/2 )**2 ) / (self.st['Nd'][1]**2) 
        
#        for pp in range(0,16):
#            matplotlib.pyplot.subplot(4,4,pp)
#            matplotlib.pyplot.imshow(numpy.abs(uker[pp,:,:]),interpolation='nearest')
#        matplotlib.pyplot.show() 
        
        
        return uker 
    def create_laplacian_kernel(self,cineObj):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#        Note only the y-axis is used 
#         # related to constraint
#===============================================================================
        uker2 = numpy.zeros((cineObj.dim_x,)+self.st['Nd'][0:2],dtype=numpy.complex64)
        rows_kd = self.st['Nd'][0] # ky-axis
        #cols_kd = self.st['Kd'][1] # t-axis
#        uker[0,0] = 1.0
        uker2[0, 0, 0] = -2.0
        uker2[0, 0, 1]=1.0
        uker2[0, 0,-1]=1.0
      
        uker2 = scipy.fftpack.fftn(uker2,axes=(0,1,2,)) # 256x256x16
        return uker2       

    def create_kspace_sampling_density(self):
            #=======================================================================
            # RTR: k-space sampled density
            #      only diagonal elements are relevant (on k-space grids)
            #=======================================================================
        RTR=self.st['q'] # see __init__() in class "pyNufft"
                            #(prod(Kd),1) 
        
        
        return RTR 
    def kernel(self, cineObj, st , mu, LMBD, gamma, nInner, nBreg):
        self.st['sensemap']=self.st['sensemap']*self.st['mask']
        orig_num_ky=numpy.shape(cineObj.tse)[1]
        tse = cineObj.tse[:,orig_num_ky/2 - self.st['Nd'][0]/2 : orig_num_ky/2 + self.st['Nd'][0]/2,:]
#         tse=cineObj.tse
#        tse=numpy.abs(numpy.mean(self.st['sensemap'],-1))

        tse=CsTransform.pynufft.appendmat(tse,self.st['Nd'][1])
        #tse=Normalize(tse)
        tse=numpy.transpose(tse,(0,1,3,2))
        self.ttse=tse#CsTransform.pynufft.Normalize(tse)
        
        self.tse0 = CsTransform.pynufft.CombineMulti(tse, -1)
        print('line392, shape self.tse0',numpy.shape(self.tse0))
        self.filter= numpy.ones(tse.shape)
        dpss = numpy.kaiser(tse.shape[1], 1.0)*10.0
        for ppp in range(0,tse.shape[1]):
            self.filter[:,ppp,:,:]=self.filter[:,ppp,:,:]*dpss[ppp]
            
        
        
        print('tse.shape',tse.shape)
#        L= numpy.size(f)/st['M'] 
#        image_dim=st['Nd']+(L,)
#         
#        if numpy.ndim(f) == 1:# preventing row vector
#            f=numpy.reshape(f,(numpy.shape(f)[0],1),order='F')
#        f0 = numpy.copy(f) # deep copy to prevent scope f0 to f
##        u = numpy.zeros(image_dim,dtype=numpy.complex64)
        f0=numpy.copy(cineObj.f)
        f=numpy.copy(cineObj.f)

#        u0=self.data2rho(f_internal,  
#                         cineObj.dim_x,
#                         self.st['Nd'][0],
#                         self.st['Nd'][1],
#                         cineObj.ncoils,
#                         self.CsTransform
#                         ) # doing spatial transform
        u0 = self.fun1(cineObj)
        
        pdf = cineObj.pdf
        pdf = CsTransform.pynufft.appendmat(pdf,self.st['Nd'][1])
        pdf = numpy.transpose(pdf,(0,1,3,2))
        
#        u0 = scipy.fftpack.fftn(u0,axes=(1,))
#        u0 = scipy.fftpack.fftshift(u0,axes=(1,))
#        #u0[:,:,u0.shape[2]/2,:] = u0[:,:,u0.shape[2]/2,:]/pdf[:,:,u0.shape[2]/2,:]
#        u0 = u0#/pdf
#        u0 = scipy.fftpack.ifftshift(u0,axes=(1,))
#        u0 = scipy.fftpack.ifftn(u0,axes=(1,))     
        
#        print('cineObj.pdf.shape',cineObj.pdf.shape)
#        for pj in range(0,4):
#            matplotlib.pyplot.imshow(cineObj.pdf[:,:,pj].real)
#            matplotlib.pyplot.show()
        
        u0=self.fun2(u0)
        
        u0=self.fun3(u0)
        
        u0 = u0*self.st['sensemap'].conj()
        
        u0 = CsTransform.pynufft.CombineMulti(u0,-1)
        print('line443, shape u0',numpy.shape(u0))
        #u0 = u0*self.filter 
        
        uker = self.create_laplacian_kernel(cineObj)
        uker = CsTransform.pynufft.appendmat(uker,u0.shape[3])

        
        self.u0 = u0
        
        u = numpy.copy(self.tse0)
        
       
        print('u0.shape',u0.shape)

        (xx,bb,dd)=self.make_split_variables(u)        

        uf = numpy.copy(u)  # only used for ISRA, written here for generality 
          
        murf = numpy.copy(u) # initial values 
#    #===============================================================================
        #u_stack = numpy.empty(st['Nd']+(nBreg,),dtype=numpy.complex)
        for outer in numpy.arange(0,nBreg):
            for inner in numpy.arange(0,nInner):
                # update u
                print('iterating',[inner,outer])
                #===============================================================
#                 update u  # simple k-space deconvolution to guess initial u
                u = self.update_u(murf, u, uker, xx, bb,cineObj)
                
                c = numpy.max(numpy.abs(u[:])) # Rough coefficient
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
                dd=self.update_d(u,dd)

                xx=self.shrink( dd, bb, c*1.0/LMBD/numpy.sqrt(numpy.prod(st['Nd'])))
                
                #===============================================================
            #===================================================================
            # # update b
            #===================================================================

                bb=self._update_b(bb, dd, xx)

            if outer < (nBreg-1): # do not update in the last loop
                (f, uf, murf,u)=self.external_update(u, f, uf, f0, u0) # update outer Split_bregman


#         u = CsTransform.pynufft.Normalize(u)
#         for pp in range(0,u0.shape[2]):
#             matplotlib.pyplot.subplot(numpy.sqrt(u0.shape[2])+1,numpy.sqrt(u0.shape[2])+1,pp)
#             matplotlib.pyplot.imshow(numpy.sum(numpy.abs(u[...,pp,:]),-1),norm=norm,interpolation='nearest')
#         matplotlib.pyplot.show()
#        

        
        return (u,uf)
    def update_u(self,murf,u, uker ,xx, bb,cineObj):
#        u_bar = numpy.copy(u)
#        
#
#        u_bar[...,:u.shape[2]/2,:] = 0.0
#        u_bar[...,(u.shape[2]/2+1):,:] = 0.0

        
        Fhd =   self.u0
#        (self.u0  
#                      - self.FhF(u_bar)#*self.st['sensemap'].conj()
#                        )       
        m = u# - u_bar
 
 
        rhs = Fhd #+ self.constraint(xx,bb) # LMBD/gamma have been added
        
        for jj in range(0,1):
            m = self.cg_step(rhs,m,uker,10,cineObj)
            
        u = m
        return u   
    def cg_step(self,rhs,m,uker,n_iter,cineObj):
        
        # going through the acquisition process
        FhFWm =   self.do_FhWFm(m*self.st['sensemap'],cineObj)*self.st['sensemap'].conj()
        FhFWm = CsTransform.pynufft.CombineMulti(FhFWm,-1)
        
        # Laplacian of the x-y-t data
        lapla_m = self.do_laplacian(m, uker)
        
        # Gradient
        lhs = FhFWm - self.LMBD*lapla_m  + 2.0*self.gamma*m
    
        #C_m= lhs - rhs
        r = rhs - lhs
        p = r
        
        for pp in range(0,n_iter):
          
            Ap = self.do_FhWFm(p*self.st['sensemap'],cineObj)*self.st['sensemap'].conj()
            Ap = CsTransform.pynufft.CombineMulti(Ap,-1)
            
            Ap = Ap - self.LMBD*self.do_laplacian(p, uker) + 2.0*self.gamma*p # a small constraint
            
            residue_square =numpy.sum((r.conj()*r)[:])
            residue_projection = numpy.sum((p.conj()*Ap)[:])
            alfa_k = residue_square/residue_projection
            
            
            print('r',residue_square,'alpha_k',alfa_k)
            #alfa_k = 0.3
            m = m + alfa_k * p
            r2 = r - alfa_k * Ap
            
            beta_k = numpy.sum( (r2.conj()*r2)[:] ) / residue_square
            r = r2
            p = r + beta_k * p
                  

        return m     
#    def update_u(self,murf,u, uker ,xx, bb):
##        u_bar = numpy.copy(u)
##        
##
##        u_bar[...,:u.shape[2]/2,:] = 0.0
##        u_bar[...,(u.shape[2]/2+1):,:] = 0.0
#
#        
#        Fhd =   self.u0
##        (self.u0  
##                      - self.FhF(u_bar)#*self.st['sensemap'].conj()
##                        )       
#        m = u# - u_bar
# 
#       
#        for jj in range(0,6):
#            print('iteratin',jj)
#
#            #for pkpk in range(0,cineObj.dim_x):
#            #    W[pkpk,...]=Normalize(W[pkpk,...])
#            
#            FhFWm =   self.do_FhWFm(m*self.st['sensemap'])*self.st['sensemap'].conj()
#            
#            FhFWm = CsTransform.pynufft.CombineMulti(FhFWm,-1)
#            
#            lapla_m = self.do_laplacian(m, uker)
#
##            lapla_Wq = self.do_laplacian(W*q,uker)
##            
##            
##            constr=self.constraint(xx, bb)
##            constr=scipy.fftpack.ifftn(constr,axes=(2,))
##            constr =scipy.fftpack.ifftshift(constr,axes=(2,))
##            
#            
#            
#            #u = Normalize(u)
#            #FhFWq= Normalize(FhFWq ) 
#            
#            rhs = Fhd #+ self.LMBD*self.constraint(xx,bb)
#            
#            lhs = FhFWm - self.LMBD*lapla_m  #+ self.LMBD*m
#            
#            C_m= lhs - rhs
#            
#            m = m - 0.3*(C_m)
#            
#            
##            if numpy.mod(jj,5) == 0:
##        for pp in range(0,m.shape[2]):
##            matplotlib.pyplot.subplot(4,4,pp)
##            matplotlib.pyplot.imshow(numpy.sum(numpy.abs(m[...,pp,:]),-1),interpolation='nearest')
##        matplotlib.pyplot.show()
#        #q = q/(self.ttse)
#        u = m
#        return u       



    def shrink(self,dd,bb,thrsld):
        '''
        soft-thresholding the edges
        
        '''
  
        output_xx = dd
        return output_xx  #+  output_x2   

    def update_d(self,u,dd):

        out_dd = dd
        return out_dd
    
    def make_split_variables(self,u):
        x=numpy.zeros(u.shape)
        y=numpy.zeros(u.shape)
        tt=numpy.zeros(u.shape)
        gg=numpy.zeros(u.shape)
        
        bx=numpy.zeros(u.shape)
        by=numpy.zeros(u.shape)
        bt=numpy.zeros(u.shape)
        bg=numpy.zeros(u.shape)
        
        dx=numpy.zeros(u.shape)
        dy=numpy.zeros(u.shape)  
        dt=numpy.zeros(u.shape)
        dg=numpy.zeros(u.shape)
        
        xx= (x,y,tt,gg)
        bb= (bx,by,bt,bg)
        dd= (dx,dy,dt,dg)
        
        return(xx, bb, dd)    

    
    def external_update(self,u, f, uf, f0, u0): # overload the update function
        
        CsTransform.pynufft.checkmax(self.st['sensemap'],0)
        tmpuf = u*self.st['sensemap']

        tmpuf = numpy.transpose(tmpuf,(1,2,3,0))
        tmp_shape=tmpuf.shape
        tmpuf = numpy.reshape(tmpuf,tmp_shape[0:2]+(numpy.prod(tmp_shape[2:4]),),order='F')
        tmpuf = self.forwardbackward(tmpuf)
        tmpuf = numpy.reshape(tmpuf ,tmp_shape,order='F')
        tmpuf = numpy.transpose(tmpuf,(3,0,1,2))
        tmpuf = tmpuf*self.st['sensemap'].conj()
        
#        tmpuf=self.st['sensemap'].conj()*(
#                self.CsTransform.forwardbackward(
#                        u*self.st['sensemap']))

        if self.st['senseflag'] == 1:
            tmpuf=CsTransform.pynufft.CombineMulti(tmpuf,-1)

        print('start of ext_update') 

#        checkmax(u)
#        checkmax(tmpuf)
#        checkmax(self.u0)
#        checkmax(uf)

        fact=numpy.sum((self.u0-tmpuf)**2)/numpy.sum((u0)**2)
        fact=numpy.abs(fact.real)
        fact=numpy.sqrt(fact)
        print('fact',fact)
#        fact=1.0/(1.0+numpy.exp(-(fact-0.5)*self.thresh_scale))
#         tmpuf=CsTransform.pynufft.Normalize(tmpuf)*numpy.max(numpy.abs(u0[:]))
        uf = uf+(u0-tmpuf)*1.0#*fact
#         uf =CsTransform.pynufft.Normalize(uf)*numpy.max(numpy.abs(u0[:]))


        CsTransform.pynufft.checkmax(tmpuf,0)
        CsTransform.pynufft.checkmax(u0,0)
        CsTransform.pynufft.checkmax(uf,0)
        
#        for jj in range(0,u.shape[-1]):
#            u[...,jj] = u[...,jj]*self.st['sn']# rescale the final image intensity
   
        print('end of ext_update')        
        murf = uf
        return (f,uf,murf,u)  
    def _update_b(self, bb, dd, xx):
        ndims=len(bb)
        cc=numpy.empty(bb[0].shape)
        out_bb=()
        for pj in range(0,ndims):
            cc=bb[pj]+dd[pj]-xx[pj]
            out_bb=out_bb+(cc,)

        return out_bb
    
    def _make_sense(self,u0):
        st=self.st
        L=numpy.shape(u0)[-1]
        u0dims= numpy.ndim(u0)
        print('in make_sense, u0.shape',u0.shape)
        if u0dims-1 >0:
            rows=numpy.shape(u0)[0]
#             dpss_rows = numpy.kaiser(rows, 100)     
#             dpss_rows = numpy.fft.fftshift(dpss_rows)
#             dpss_rows[3:-3] = 0.0
            dpss_rows = numpy.ones(rows) 
            # replace above sensitivity because
            # Frequency direction is not necessary
            dpss_fil = dpss_rows
            print('dpss shape',dpss_fil.shape)
        if u0dims-1 > 1:
                               
            cols=numpy.shape(u0)[1]
            dpss_cols = numpy.kaiser(cols, 100)            
            dpss_cols = numpy.fft.fftshift(dpss_cols)
            dpss_cols[3:-3] = 0.0
             
            dpss_fil = CsTransform.pynufft.appendmat(dpss_fil,cols)
            dpss_cols  = CsTransform.pynufft.appendmat(dpss_cols,rows)
 
            dpss_fil=dpss_fil*numpy.transpose(dpss_cols,(1,0))
            print('dpss shape',dpss_fil.shape)
        if u0dims-1 > 2:
             
            zag = numpy.shape(u0)[2]
            dpss_zag = numpy.kaiser(zag, 100)            
            dpss_zag = numpy.fft.fftshift(dpss_zag)
            dpss_zag[3:-3] = 0.0
            dpss_fil = CsTransform.pynufft.appendmat(dpss_fil,zag)
                      
            dpss_zag = CsTransform.pynufft.appendmat(dpss_zag,rows)
             
            dpss_zag = CsTransform.pynufft.appendmat(dpss_zag,cols)
             
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
class Cine2DSolver(TemporalConstraint):
    def create_mask(self,u0):
        st=self.st
        print('u0.shape',u0.shape)
        rows=u0.shape[0]
        cols=u0.shape[1]

        kk = numpy.arange(0,rows)
        jj = numpy.arange(0,cols)

        kk = CsTransform.pynufft.appendmat(kk,cols)
        jj = CsTransform.pynufft.appendmat(jj,rows).T
        st['mask']=numpy.ones((rows,cols),dtype=numpy.float32)

        #add circular mask
        sp_rat=(rows**2+cols**2)*1.0
        
#         for jj in numpy.arange(0,cols):
#             for kk in numpy.arange(0,rows):
#                 if ( (kk-rows/2.0)**2+(jj-cols/2.0)**2 )/sp_rat > 1.0/8.0:
#                     st['mask'][kk,jj] = 0.0
        
        if numpy.size(u0.shape) > 2:
            for pp in range(2,numpy.size(u0.shape)):
                st['mask'] = CsTransform.pynufft.appendmat(st['mask'],u0.shape[pp] )
 
        return st
    def update_u(self,murf,u, uker ,xx, bb,cineObj):
#        u_bar = numpy.copy(u)
#        
#
#        u_bar[...,:u.shape[2]/2,:] = 0.0
#        u_bar[...,(u.shape[2]/2+1):,:] = 0.0

        
        Fhd =   self.u0
#        (self.u0  
#                      - self.FhF(u_bar)#*self.st['sensemap'].conj()
#                        )       
        m = u# - u_bar
 
 
        rhs = Fhd + self.constraint(xx,bb) # LMBD/gamma have been added
        
        num_cg_step = 30
        
        m = self.cg_step(rhs,m,uker,num_cg_step,cineObj)
            
        u = m
        return u   

    
    def create_laplacian_kernel(self,cineObj):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#        Note only the y-axis is used 
#         # related to constraint
#===============================================================================
        uker2 = numpy.zeros((cineObj.dim_x,)+self.st['Nd'][0:2],dtype=numpy.complex64)
        rows_kd = self.st['Nd'][0] # ky-axis
        #cols_kd = self.st['Kd'][1] # t-axis
#        uker[0,0] = 1.0
#         rate = 30.0
#         uker2[0,0,0] = -4.0 - 2.0/rate
#         uker2[0,0,1] =1.0
#         uker2[0,0,-1]=1.0
#         uker2[0,1,0] =1.0#/rate
#         uker2[0,-1,0]=1.0#/rate
#         uker2[1,0,0] =1.0/rate
#         uker2[-1,0,0]=1.0/rate  
        rate = 15.0
        uker2[0,0,0] = -2.0 - 4.0/rate
        uker2[0,0,1] =1.0
        uker2[0,0,-1]=1.0
        uker2[0,1,0] =1.0/rate
        uker2[0,-1,0]=1.0/rate
        uker2[1,0,0] =1.0/rate
        uker2[-1,0,0]=1.0/rate        
        uker2 = scipy.fftpack.fftn(uker2,axes=(0,1,2,)) # 256x256x16
        return uker2
    def constraint(self,xx,bb):
        '''
        include TVconstraint and others
        '''

            #tmp_d =get_Diff(u,jj)
        rate = 15.0
        cons = CsTransform.pynufft.TVconstraint(xx[0:2],bb[0:2]) * self.LMBD/rate
        #cons =  CsTransform.pynufft.TVconstraint(xx[0:2],bb[0:2]) * self.LMBD/100.0
        cons = cons + CsTransform.pynufft.TVconstraint(xx[2:3],bb[2:3]) * self.LMBD#/rate
        
        cons = cons + scipy.fftpack.ifftn(xx[3]-bb[3],axes = (2,))* self.LMBD/rate
#         cons = cons + (xx[4]-bb[4])* self.gamma

#         cons = cons + CsTransform.pynufft.TVconstraint(xx[4:5],bb[4:5])* self.LMBD
        #cons = cons + xx[2]-bb[2]
        #print('inside constraint, cons.shpae',cons.shape)
#        cons = cons + freq_gradient_H(xx[3]-bb[3])
        #print('inside constraint 1117, cons.shpae',cons.shape)

        return cons       
    def update_d(self,u,dd):
#        print('inside_update_d ushape',u.shape)
#        print('inside_update_d fre grad ushape',freq_gradient(u).shape)
        out_dd = ()
        for jj in range(0,len(dd)) :
            if jj < 2: # derivative y 
                #tmp_d =get_Diff(u,jj)
                out_dd = out_dd  + (CsTransform.pynufft.get_Diff(u,jj),)
                
            if jj == 2: # derivative y 
                #tmp_d =get_Diff(u,jj)
                out_dd = out_dd  + (CsTransform.pynufft.get_Diff(u,jj),)  
                              
            elif jj == 3: # rho
                tmpu = numpy.copy(u)
                tmpu = scipy.fftpack.fftn(tmpu,axes = (2,))
#                 tmpu[:,:,0,:] = tmpu[:,:,0,:]*0.0
                out_dd = out_dd + (tmpu,)
                
            elif jj == 4:
                average_u = numpy.sum(u,2)
                tmpu= numpy.copy(u)
#                 for jj in range(0,u.shape[2]):
#                     tmpu[:,:,jj,:]= tmpu[:,:,jj,:] - average_u
                out_dd = out_dd + (tmpu,)
#                 out_dd = out_dd + (CsTransform.pynufft.get_Diff(tmpu,),)
#            elif jj == 3:
#                out_dd = out_dd + (freq_gradient(u),)
                
        return out_dd
    def shrink(self,dd,bb,thrsld):
        '''
        soft-thresholding the edges
        
        '''
#        dd2 = ()
#        bb2 = ()
#        for pp in range(0,2):
#            dd2=dd2 + (dd[pp]/100.0,)
#            bb2=bb2+ (bb[pp]/100.0,)
#        dd2 = dd2 +dd[2:]
#        bb2 = bb2 +bb[2:]   
#        tmp_xx=CsTransform.pynufft.shrink( dd2[0:2], bb2[0:2], thrsld)
#        
#        output_xx = ()
#        for pp in range(0,2):
#            output_xx = output_xx + (tmp_xx[pp]*100.0,)
#       
#        output_xx = output_xx + (tmp_xx[2],)
        
        output_xx= CsTransform.pynufft.shrink( dd[0:2], bb[0:2], thrsld)# 3D thresholding
        output_xx=output_xx + CsTransform.pynufft.shrink( dd[2:3], bb[2:3], thrsld)# 3D thresholding 
        output_xx =output_xx + CsTransform.pynufft.shrink( dd[3:4], bb[3:4], thrsld)
#         output_xx =output_xx + CsTransform.pynufft.shrink( dd[4:5], bb[4:5], thrsld)
        
        return output_xx  #+  output_x2  
    
    def make_split_variables(self,u):
        x=numpy.zeros(u.shape)
        y=numpy.zeros(u.shape)
        tt=numpy.zeros(u.shape)
        gg=numpy.zeros(u.shape)
        mm=numpy.zeros(u.shape)
        
        bx=numpy.zeros(u.shape)
        by=numpy.zeros(u.shape)
        bt=numpy.zeros(u.shape)
        bg=numpy.zeros(u.shape)
        bm=numpy.zeros(u.shape)
        
        dx=numpy.zeros(u.shape)
        dy=numpy.zeros(u.shape)  
        dt=numpy.zeros(u.shape)
        dg=numpy.zeros(u.shape)
        dm=numpy.zeros(u.shape)
        
        xx= (x,y,tt,gg)
        bb= (bx,by,bt,bg)
        dd= (dx,dy,dt,dg)
        
        return(xx, bb, dd)  
    
    
            
# class ktfocuss(Cine2DSolver):
#     def kernel(self, f_internal, st , mu, LMBD, gamma, nInner, nBreg):
#         self.st['sensemap']=self.st['sensemap']*self.st['mask']
#         tse=cineObj.tse
# #        tse=numpy.abs(numpy.mean(self.st['sensemap'],-1))
# 
#         tse=CsTransform.pynufft.appendmat(tse,self.st['Nd'][1])
#         #tse=Normalize(tse)
#         tse=numpy.transpose(tse,(0,1,3,2))
#         
#         print('tse.shape',tse.shape)
# #        L= numpy.size(f)/st['M'] 
# #        image_dim=st['Nd']+(L,)
# #         
# #        if numpy.ndim(f) == 1:# preventing row vector
# #            f=numpy.reshape(f,(numpy.shape(f)[0],1),order='F')
# #        f0 = numpy.copy(f) # deep copy to prevent scope f0 to f
# ##        u = numpy.zeros(image_dim,dtype=numpy.complex64)
#         f0=numpy.copy(f_internal)
#         f=numpy.copy(f_internal)
#         v= f
#         u0=self.fun1(f_internal,  
# #                         cineObj.dim_x,
# #                         self.st['Nd'][0],
# #                         self.st['Nd'][1],
# #                         cineObj.ncoils,
# #                         self.CsTransform
#                          ) # doing spatial transform
#         
#         pdf = cineObj.pdf
#         pdf = CsTransform.pynufft.appendmat(pdf,self.st['Nd'][1])
#         pdf = numpy.transpose(pdf,(0,1,3,2))
# #        matplotlib.pyplot.imshow(pdf[:,:,0,0].real)
# #        matplotlib.pyplot.show()
#         u0 = scipy.fftpack.fftn(u0,axes=(1,))
#         u0 = scipy.fftpack.fftshift(u0,axes=(1,))
#         #u0[:,:,u0.shape[2]/2,:] = 2*u0[:,:,u0.shape[2]/2,:]/pdf[:,:,u0.shape[2]/2,:]
#         u0  = u0 /pdf 
#         u0 = scipy.fftpack.ifftshift(u0,axes=(1,))
#         u0 = scipy.fftpack.ifftn(u0,axes=(1,))     
#            
# #        print('cineObj.pdf.shape',cineObj.pdf.shape)
# #        for pj in range(0,4):
# #            matplotlib.pyplot.imshow(cineObj.pdf[:,:,pj].real)
# #            matplotlib.pyplot.show()
#         
#         u0=self.fun2(u0)
# 
#         u0 = u0*self.st['sensemap'].conj()
#         
#         u0 = CsTransform.pynufft.CombineMulti(u0,-1)
#         for pp in range(0,4):
#             matplotlib.pyplot.subplot(2,2,pp)
#             matplotlib.pyplot.imshow(numpy.sum(numpy.abs(u0[...,u0.shape[2]/2+1,:]),-1),norm=norm,interpolation='nearest')
#         matplotlib.pyplot.show()                
#         u = numpy.copy(u0)
#         print('u0.shape',u0.shape)
#         
#         u_bar = numpy.copy(u)
#         u_bar = scipy.fftpack.fftshift(u_bar,axes=(2,))
#         u_bar[...,1:,:] = 0.0
#         u_bar = scipy.fftpack.fftshift(u_bar,axes=(2,))
#         
#         FhFu_bar = self.FhF(u_bar)#*self.st['sensemap'])*self.st['sensemap'].conj()
#         #FhFu_bar = CombineMulti(FhFu_bar,-1)
#         
#         u_ref = u0 - FhFu_bar
#         
#         q = u - u_bar
#         
#         W=numpy.sqrt(numpy.abs(q))
#        
#         for jj in range(0,self.nInner):
#             print('iteratin',jj)
#             #W=numpy.sqrt(numpy.abs(q*W))
#             #W=Normalize(W)
#             
# 
#             FhFWq =   self.FhF(W*q)#*self.st['sensemap'])*self.st['sensemap'].conj()
#             #FhFWq = CombineMulti(FhFWq, -1)
# 
#             C_q= -W*(u_ref - FhFWq) + 0.02*q
#             q=q-0.3*C_q
#             W = numpy.sqrt(numpy.abs(W*q))
#             #q=q#*tse
#             #q=Normalize(q)
#         for pp in range(0,u0.shape[2]):
#             matplotlib.pyplot.subplot(numpy.sqrt(u0.shape[2])+1,numpy.sqrt(u0.shape[2])+1,pp)
#             matplotlib.pyplot.imshow(numpy.sum(numpy.abs(q[...,pp,:]),-1),norm=norm,interpolation='nearest')
#         matplotlib.pyplot.show()
#         u = q*W  + u_bar
# #        for pp in range(0,u0.shape[2]):
# #            matplotlib.pyplot.subplot(4,4,pp)
# #            matplotlib.pyplot.imshow(numpy.sum(numpy.abs(u[...,pp,:]),-1),norm=norm,interpolation='nearest')
# #        matplotlib.pyplot.show()
# #        
# 
#         
#         return (u,u*W)