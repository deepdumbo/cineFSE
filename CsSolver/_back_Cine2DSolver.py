import numpy
import scipy
import scipy.fftpack
import CsSolver
import CsSolver
import matplotlib.pyplot
    #===========================================================================
    # Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver(MyTransform, 
    #                 cineObj, mu, LMBD, gamma, nInner, nBreg)    
    # Cine2DSolver.solve()
    #===========================================================================
cmap=matplotlib.cm.gray
norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

class TemporalConstraint(CsSolver.CsSolver):

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
    def data2rho(self, dd_raw,dim_x,dim_y,dim_t,n_coil,CsTransform):
        if dd_raw.ndim != 3:
            print(' dimension is wrong! ')
        
        dd_raw = numpy.transpose(dd_raw,(1,2,0)) # put the x-axis to the tail
        
        shape_dd_raw = numpy.shape(dd_raw)
        
        dd_raw = numpy.reshape(dd_raw,
                               (shape_dd_raw[0],)+ (shape_dd_raw[1]*shape_dd_raw[2],),
                                order='F')
    
        output_data = CsTransform.backward(dd_raw) # (dim_y, dim_t*n_coil*dim_x)
        
    
        output_data = numpy.reshape(output_data, (dim_y,dim_t,n_coil,dim_x),order='F')
    
        output_data = numpy.transpose(output_data, (3,0,1,2)) 
        
        return output_data
    
    def FhF(self, dd_raw ):
        if dd_raw.ndim != 4:
            print(' dimension is wrong! ')
        
        dim_x = self.f.dim_x
        dim_y = self.st['Nd'][0]
        dim_t = self.st['Nd'][1]
        n_coil = self.f.ncoils
        CsTransform = self.CsTransform
        
        dd_raw = numpy.transpose(dd_raw,(1,2,3,0)) # put the x-axis to the tail
        
        shape_dd_raw = numpy.shape(dd_raw)
        
        dd_raw = numpy.reshape(dd_raw,
                  (shape_dd_raw[0],)+ (shape_dd_raw[1],)+(shape_dd_raw[2]*shape_dd_raw[3],),
                        order='F')
        output_data = CsTransform.forwardbackward(dd_raw) 
    
#        output_data = CsTransform.forward(dd_raw) # (dim_y, dim_t*n_coil*dim_x)
#        
#        output_data = CsTransform.backward(output_data)
        
        #output_data = CsTransform.backward(output_data)
    
        output_data = numpy.reshape(output_data, (dim_y,dim_t,n_coil,dim_x),order='F')
    
        output_data = numpy.transpose(output_data, (3,0,1,2)) 

        return output_data
    
    def fun1(self,input):
        '''
        CsTransform.backward
        from data to kx-y-f
        do I need this? 
        by connecting to data2rho(self, dd_raw,dim_x,dim_y,dim_t,n_coil,CsTransform):
        '''
        output=self.data2rho( input, self.f.dim_x , self.st['Nd'][0],self.st['Nd'][1],self.f.ncoils,self.CsTransform)
        
        return output
        
    def fun2(self,m):
        '''
        
        1)shift-kx
        2)ifft: along kx
        3)shift-x
        
        '''
        m = scipy.fftpack.ifftshift(m, axes=(0,)) # because kx energy is at centre
        
        m = scipy.fftpack.ifftn(m,axes=(0,))  
        
        m = scipy.fftpack.ifftshift(m, axes=(0,)) # because x-y-f are all shifted at center
        
        return m        
        
    def fun3(self,m):
        '''
        1)shift-f
        2)fft: along f
        3)(No shift)-t
        '''
        m = scipy.fftpack.fftshift(m, axes=(2,))
        
        m = scipy.fftpack.fftn(m,axes=(2,)) 
        
        return m
    
    def fun4(self,m):
        '''
        inverse of self.fun3
        1)(No shift)-t
        2)fft: along f
        3)shift-f
        '''

        m = scipy.fftpack.ifftn(m,axes=(2,)) 
        
        m = scipy.fftpack.ifftshift(m, axes=(2,)) 
     
        return m
    
    def fun5(self,m):
        '''
        1)shift-f:
        2)shift-x-y:
        2)fft2 along x-y, no shift -> kx ky f
        
        for 3D laplacian operator
        '''
        m = scipy.fftpack.ifftshift(m, axes=(0,1,2,))
       
        m = scipy.fftpack.fftn(m,axes =  (0,1,))
        
        return m
        
    def fun6(self,m):
        '''
        inverse of fun5
        1)ifft2 along kx-ky 
        2)shift along x & y
        3)shift along f

        '''
        
        m = scipy.fftpack.ifftn(m,axes =  (0,1,)) 
              
        m = scipy.fftpack.fftshift(m, axes=(0,1,2,))    
            
        return m
    

  

    def solve(self): # main function of solver

        u0=numpy.empty((self.f.dim_x, self.st['Nd'][0], self.st['Nd'][1], 4))
        
        tse = self.f.tse
        
        tse = CsSolver.appendmat(tse,u0.shape[2])
        tse = numpy.transpose(tse,(0,1,3,2))
        
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
        self.st = self.make_sense(self.f.tse) # setting up sense map in st['sensemap']
        
        self.st['sensemap'] = CsSolver.appendmat(self.st['sensemap'],u0.shape[2])
        self.st['sensemap'] = numpy.transpose(self.st['sensemap'],(0,1,3,2))
        
        #self.st['sensemap'] =self.st['sensemap'] * self.st['mask'] 
        print('self.sense.shape',self.st['sensemap'].shape)

#        for jj in range(0,16):
#            matplotlib.pyplot.subplot(4,4,jj)
#            matplotlib.pyplot.imshow(numpy.abs(self.st['sensemap'][...,jj,0]))
#        matplotlib.pyplot.show()       
         
        self.st['senseflag']=1 # turn-on sense, to get sensemap
        
        (u,uf)=self.kernel( self.f.f, self.st , self.mu, self.LMBD, self.gamma, self.nInner, self.nBreg)
        self.u = u
    def create_laplacian_kernel2(self):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#        Note only the y-axis is used 
#         # related to constraint
#===============================================================================
        uker = numpy.zeros((self.f.dim_x,)+self.st['Nd'][0:2],dtype=numpy.complex64)
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
    def create_laplacian_kernel3(self):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#        Note only the y-axis is used 
#         # related to constraint
#===============================================================================
        uker = numpy.ones((self.f.dim_x,)+self.st['Nd'][0:2],dtype=numpy.complex64)
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
    def create_laplacian_kernel(self):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#        Note only the y-axis is used 
#         # related to constraint
#===============================================================================
        uker2 = numpy.zeros((self.f.dim_x,)+self.st['Nd'][0:2],dtype=numpy.complex64)
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
    def kernel(self, f_internal, st , mu, LMBD, gamma, nInner, nBreg):
        self.st['sensemap']=self.st['sensemap']*self.st['mask']
        tse=self.f.tse
#        tse=numpy.abs(numpy.mean(self.st['sensemap'],-1))

        tse=CsSolver.appendmat(tse,self.st['Nd'][1])
        #tse=Normalize(tse)
        tse=numpy.transpose(tse,(0,1,3,2))
        self.ttse=CsSolver.Normalize(tse)
        
        self.tse0 = CsSolver.CombineMulti(tse, -1)
        
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
        f0=numpy.copy(f_internal)
        f=numpy.copy(f_internal)

#        u0=self.data2rho(f_internal,  
#                         self.f.dim_x,
#                         self.st['Nd'][0],
#                         self.st['Nd'][1],
#                         self.f.ncoils,
#                         self.CsTransform
#                         ) # doing spatial transform
        u0 = self.fun1(f_internal)
        
        pdf = self.f.pdf
        pdf = CsSolver.appendmat(pdf,self.st['Nd'][1])
        pdf = numpy.transpose(pdf,(0,1,3,2))
        
#        u0 = scipy.fftpack.fftn(u0,axes=(1,))
#        u0 = scipy.fftpack.fftshift(u0,axes=(1,))
#        #u0[:,:,u0.shape[2]/2,:] = u0[:,:,u0.shape[2]/2,:]/pdf[:,:,u0.shape[2]/2,:]
#        u0 = u0#/pdf
#        u0 = scipy.fftpack.ifftshift(u0,axes=(1,))
#        u0 = scipy.fftpack.ifftn(u0,axes=(1,))     
        
#        print('self.f.pdf.shape',self.f.pdf.shape)
#        for pj in range(0,4):
#            matplotlib.pyplot.imshow(self.f.pdf[:,:,pj].real)
#            matplotlib.pyplot.show()
        
        u0=self.fun2(u0)
        
        u0=self.fun3(u0)
        
        u0 = u0*self.st['sensemap'].conj()
        
        u0 = CsSolver.CombineMulti(u0,-1)
        
        #u0 = u0*self.filter 
        
        uker = self.create_laplacian_kernel()
        uker = CsSolver.appendmat(uker,u0.shape[3])

        
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
                u = self.update_u(murf, u, uker, xx, bb)
                
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

                bb=self.update_b(bb, dd, xx)

#            if outer < nBreg: # do not update in the last loop
            (f, uf, murf,u)=self.external_update(u, f, uf, f0, u0) # update outer Split_bregman


        u = CsSolver.Normalize(u)
        for pp in range(0,u0.shape[2]):
            matplotlib.pyplot.subplot(numpy.sqrt(u0.shape[2])+1,numpy.sqrt(u0.shape[2])+1,pp)
            matplotlib.pyplot.imshow(numpy.sum(numpy.abs(u[...,pp,:]),-1),norm=norm,interpolation='nearest')
        matplotlib.pyplot.show()
#        

        
        return (u,uf)
    def update_u(self,murf,u, uker ,xx, bb):
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
        
        for jj in range(0,3):
            m = self.cg_step(rhs,m,uker,3)
            
        u = m
        return u   
    def cg_step(self,rhs,m,uker,n_iter):
        FhFWm =   self.do_FhWFm(m*self.st['sensemap'])*self.st['sensemap'].conj()
        
        FhFWm = CsSolver.CombineMulti(FhFWm,-1)
        
        lapla_m = self.do_laplacian(m, uker)
        
        lhs = FhFWm - self.LMBD*lapla_m  + 2.0*self.gamma*m
    
        #C_m= lhs - rhs
        r = rhs - lhs
        p = r
        
        for pp in range(0,n_iter):
          
            Ap = self.do_FhWFm(p*self.st['sensemap'])*self.st['sensemap'].conj()
            Ap = CsSolver.CombineMulti(Ap,-1)
            
            Ap = Ap - self.LMBD*self.do_laplacian(p, uker) + 2.0*self.gamma*p
            
            upper_ratio =numpy.sum((r.conj()*r)[:])
            lower_ratio = numpy.sum((p.conj()*Ap)[:])
            alfa_k = upper_ratio/lower_ratio
            
            
            print('r',upper_ratio,'alpha_k',alfa_k)
            #alfa_k = 0.3
            m = m + alfa_k * p
            r2 = r - alfa_k * Ap
            
            beta_k = numpy.sum( (r2.conj()*r2)[:] ) / numpy.sum(  (r.conj()*r)[:] )
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
#            #for pkpk in range(0,self.f.dim_x):
#            #    W[pkpk,...]=Normalize(W[pkpk,...])
#            
#            FhFWm =   self.do_FhWFm(m*self.st['sensemap'])*self.st['sensemap'].conj()
#            
#            FhFWm = CsSolver.CombineMulti(FhFWm,-1)
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


    def do_FhWFm(self,q):
        output_q = q
        output_q = self.fun4(output_q)
        output_q = self.FhF(output_q)
        output_q = self.fun3(output_q)
        
        return output_q
    def do_laplacian(self,q,uker):
       
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
        
        CsSolver.checkmax(self.st['sensemap'])
        tmpuf = u*self.st['sensemap']

        tmpuf = numpy.transpose(tmpuf,(1,2,3,0))
        tmp_shape=tmpuf.shape
        tmpuf = numpy.reshape(tmpuf,tmp_shape[0:2]+(numpy.prod(tmp_shape[2:4]),),order='F')
        tmpuf = self.CsTransform.forwardbackward(tmpuf)
        tmpuf = numpy.reshape(tmpuf ,tmp_shape,order='F')
        tmpuf = numpy.transpose(tmpuf,(3,0,1,2))
        tmpuf = tmpuf*self.st['sensemap'].conj()
        
#        tmpuf=self.st['sensemap'].conj()*(
#                self.CsTransform.forwardbackward(
#                        u*self.st['sensemap']))

        if self.st['senseflag'] == 1:
            tmpuf=CsSolver.CombineMulti(tmpuf,-1)

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
        tmpuf=CsSolver.Normalize(tmpuf)*numpy.max(numpy.abs(u0[:]))
        uf = uf+(u0-tmpuf)*1.0#*fact
        uf =CsSolver.Normalize(uf)*numpy.max(numpy.abs(u0[:]))


        CsSolver.checkmax(tmpuf)
        CsSolver.checkmax(u0)
        CsSolver.checkmax(uf)
        
#        for jj in range(0,u.shape[-1]):
#            u[...,jj] = u[...,jj]*self.st['sn']# rescale the final image intensity
   
        print('end of ext_update')        
        murf = uf
        return (f,uf,murf,u)  
    
    
    
class Cine3DSolver(TemporalConstraint):
    def create_mask(self,u0):
        st=self.st

        rows=u0.shape[0]
        cols=u0.shape[1]

        kk = numpy.arange(0,rows)
        jj = numpy.arange(0,cols)

        kk = CsSolver.appendmat(kk,cols)
        jj = CsSolver.appendmat(jj,rows).T
        st['mask']=numpy.ones((rows,cols),dtype=numpy.float32)

        #add circular mask
        sp_rat=(rows**2+cols**2)*1.0
        
#         for jj in numpy.arange(0,cols):
#             for kk in numpy.arange(0,rows):
#                 if ( (kk-rows/2.0)**2+(jj-cols/2.0)**2 )/sp_rat > 1.0/8.0:
#                     st['mask'][kk,jj] = 0.0
        
        if numpy.size(u0.shape) > 2:
            for pp in range(2,numpy.size(u0.shape)):
                st['mask'] = CsSolver.appendmat(st['mask'],u0.shape[pp] )
 
        return st
    def update_u(self,murf,u, uker ,xx, bb):
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
        
        for jj in range(0,2):
            m = self.cg_step(rhs,m,uker,2)
            
        u = m
        return u   

    
    def create_laplacian_kernel(self):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#        Note only the y-axis is used 
#         # related to constraint
#===============================================================================
        uker2 = numpy.zeros((self.f.dim_x,)+self.st['Nd'][0:2],dtype=numpy.complex64)
        rows_kd = self.st['Nd'][0] # ky-axis
        #cols_kd = self.st['Kd'][1] # t-axis
#        uker[0,0] = 1.0
        rate = 40.0
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
        cons = CsSolver.TVconstraint(xx[0:2],bb[0:2]) * self.LMBD/40.0
        #cons =  CsSolver.TVconstraint(xx[0:2],bb[0:2]) * self.LMBD/100.0
        #cons = cons + CsSolver.TVconstraint(xx[2:3],bb[2:3]) * self.LMBD
        cons = cons + scipy.fftpack.ifftn(xx[3]-bb[3],axes = (2,))* self.gamma
        cons = cons + (xx[4]-bb[4])* self.gamma
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
            if jj < 3: # derivative y 
                #tmp_d =get_Diff(u,jj)
                out_dd = out_dd  + (CsSolver.get_Diff(u,jj),)
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
#        tmp_xx=CsSolver.shrink( dd2[0:2], bb2[0:2], thrsld)
#        
#        output_xx = ()
#        for pp in range(0,2):
#            output_xx = output_xx + (tmp_xx[pp]*100.0,)
#       
#        output_xx = output_xx + (tmp_xx[2],)
        
        output_xx= CsSolver.shrink( dd[0:2], bb[0:2], thrsld)# 3D thresholding
        output_xx=output_xx + CsSolver.shrink( dd[2:3], bb[2:3], thrsld)# 3D thresholding 
        output_xx =output_xx + CsSolver.shrink( dd[3:4], bb[3:4], thrsld)
        output_xx =output_xx + CsSolver.shrink( dd[4:5], bb[4:5], thrsld)
        
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
        
        xx= (x,y,tt,gg,mm)
        bb= (bx,by,bt,bg,bm)
        dd= (dx,dy,dt,dg,dm)
        
        return(xx, bb, dd)  
    
    
            
# class ktfocuss(Cine2DSolver):
#     def kernel(self, f_internal, st , mu, LMBD, gamma, nInner, nBreg):
#         self.st['sensemap']=self.st['sensemap']*self.st['mask']
#         tse=self.f.tse
# #        tse=numpy.abs(numpy.mean(self.st['sensemap'],-1))
# 
#         tse=CsSolver.appendmat(tse,self.st['Nd'][1])
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
# #                         self.f.dim_x,
# #                         self.st['Nd'][0],
# #                         self.st['Nd'][1],
# #                         self.f.ncoils,
# #                         self.CsTransform
#                          ) # doing spatial transform
#         
#         pdf = self.f.pdf
#         pdf = CsSolver.appendmat(pdf,self.st['Nd'][1])
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
# #        print('self.f.pdf.shape',self.f.pdf.shape)
# #        for pj in range(0,4):
# #            matplotlib.pyplot.imshow(self.f.pdf[:,:,pj].real)
# #            matplotlib.pyplot.show()
#         
#         u0=self.fun2(u0)
# 
#         u0 = u0*self.st['sensemap'].conj()
#         
#         u0 = CsSolver.CombineMulti(u0,-1)
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