'''
Created on 2013/1/21

@author: sram
'''
import numpy
import numpy.linalg
#import scipy
import scipy.fftpack
#import ctypes
#import spectrum.mtm
import matplotlib.pyplot as mp
import matplotlib.cm
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('..')
import CsTransform.pynufft as pf
cmap=matplotlib.cm.gray
norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
def create_krnl(self,u): # create the negative 3D laplacian kernel of size u.shape[0:3]
    
    krnl = numpy.zeros(numpy.shape(u)[0:3],dtype=numpy.complex64) 
    krnl[0,0,0]=6
    krnl[1,0,0]=-1
    krnl[0,1,0]=-1
    krnl[0,0,1]=-1
    krnl[-1,0,0]=-1
    krnl[0,-1,0]=-1
    krnl[0,0,-1]=-1
    krnl = self.ifft_kkf(krnl)

    return krnl # (256*256*16) 
def output(cc):
    print('max',numpy.max(numpy.abs(cc[:])))
    
def Normalize(D):
    return D/numpy.max(numpy.abs(D[:]))
def checkmax(x):
    print( numpy.max(numpy.abs(x[:])))
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
def shrink(dd, bb,LMBD):
    LMBD=LMBD+1e-15
    if numpy.size(dd) != numpy.size(dd):
        print('size does not match! ')
    if len(dd) != len(dd):
        print('size does not match! ')
     
    n_dims=numpy.shape(dd)[0]
 
    s = numpy.zeros(dd[0].shape)
    

    
    for pj in range(0,n_dims):    
        s = s+ (dd[pj] + bb[pj])*(dd[pj] + bb[pj]).conj()
 
    s = numpy.sqrt(s).real
    
    comp=numpy.zeros(dd[0].shape)
    
    LMBD=LMBD*1.0
    ss = numpy.maximum(s-LMBD,0.0)/(s+1e-7)
    
#    ss = (s-LMBD)*(s > LMBD)
#
#    s = s+(s<LMBD)
#
#    ss = ss/s
    
    xx=()
    
    for pj in range(0,n_dims): 
        
        xx = xx+ (ss*dd[pj]+ss*bb[pj],)
        
    return xx

def TVconstraint(xx,bb):

    n_xx = len(xx)
    n_bb =  len(bb)
    #print('n_xx',n_xx)
    if n_xx != n_bb: 
        print('xx and bb size wrong!')
    
    cons_shape = numpy.shape(xx[0])
    cons=numpy.zeros(cons_shape,dtype=numpy.complex128)
    
    for jj in range(0,n_xx):

        cons =  cons + get_Diff_H( xx[jj] - bb[jj] ,  jj)
    

    return cons 
def Dx(u):
    rows=numpy.size(u)
    d = numpy.zeros(rows,u.dtype)
    d[1:rows-1] = u[1:rows-1]-u[0:rows-2]
    d[0] = u[0]-u[rows-1] 
    return d

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
        xshape=numpy.shape(x)

#        dcy=numpy.empty(numpy.shape(y),dtype=numpy.complex64)
        ShapeProd=numpy.prod(xshape[1:])
        x = numpy.reshape(x,xshape[0:1]+(ShapeProd,),order='F')
        dcx=numpy.empty(numpy.shape(x),dtype=x.dtype)
        
#        dcx=Dx(x)
        for ss in numpy.arange(0,ShapeProd):
            dcx[:,ss] = Dx(x[:,ss]) # Diff operators
#            dcy[:,:,ll] = Dyt(y[:,:,ll]-by[:,:,ll]) # Hermitian of Diff operators
        dcx=numpy.reshape(dcx, xshape ,order='F')
    return dcx 

def CombineMulti(multi_coil_data,axs):
    U=numpy.mean(multi_coil_data,axs)
    U = appendmat(U,multi_coil_data.shape[axs])

    return U

def CopySingle2Multi(single_coil_data,n_tail):
   
    U=numpy.copy(single_coil_data)
    
    U = appendmat(U, n_tail)
    
    
    return U


class CsSolver:
    '''
    interface
    '''
    def __init__(self,CsTransform, data, mu, LMBD, gamma, nInner, nBreg):
#        if numpy.size(LMBD) ==1:
#            LMBD=(LMBD,LMBD,0)
        self.CsTransform=CsTransform # pyNufft object
        self.st=CsTransform.st 
        self.f = data
        self.mu = mu
        self.LMBD = LMBD
        self.gamma = gamma
        self.nInner= nInner
        self.nBreg= nBreg

    def solve(self): # main function of solver

        self.LMBD=self.LMBD*1000.0

        self.st['senseflag']=0 # turn-off sense, to get sensemap
        
        #precompute highly constrainted images to guess the sensitivity maps 
        (u0,dump)=self.kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 1,2)
#===============================================================================
# mask
#===============================================================================
        self.st = self.create_mask(u0)

#===============================================================================
        
        #estimate sensitivity maps by divided by rms images
        self.st = self.make_sense(u0) # setting up sense map in st['sensemap']

        self.st['senseflag']=1 # turn-on sense, to get sensemap
  

        #scale back the constrainted factor LMBD
        self.LMBD=self.LMBD/1000.0
        #CS reconstruction
        (self.u, self.u_stack)=self.kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 
                      self.nInner,self.nBreg)
#        self.u=1.5*self.u/numpy.max(numpy.real(self.u[:]))
    def kernel(self, f, st , mu, LMBD, gamma, nInner, nBreg):
        L= numpy.size(f)/st['M'] 
        image_dim=st['Nd']+(L,)
         
        if numpy.ndim(f) == 1:# preventing row vector
            f=numpy.reshape(f,(numpy.shape(f)[0],1),order='F')
        f0 = numpy.copy(f) # deep copy to prevent scope f0 to f
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
        if st.has_key('mask'):
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
        RTR=self.create_kspace_sampling_density()

#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to constraint
#===============================================================================
        uker = self.create_laplacian_kernel()

        #=======================================================================
        # uker: deconvolution kernel in k-space, 
        #       which will be divided in k-space in iterations
        #=======================================================================

    #===========================================================================
    # initial estimation u, u0, uf
    #===========================================================================

        u = st['sensemap'].conj()*(self.CsTransform.backward(f))
        print('senseflag',st['senseflag'])
        if st['senseflag'] == 1:
            print('combining 8 channels')
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

        uker = self.expand_deconv_kernel_dimension(uker,u.shape[-1])

        RTR = self.expand_RTR(RTR,u.shape[-1])

        uker = self.mu*RTR-LMBD*uker+gamma
        print('uker.shape line 319',uker.shape)
                
        (xx,bb,dd)=self.make_split_variables(u)        

        uf = numpy.copy(u0)  # only used for ISRA, written here for generality 
          
        murf = numpy.copy(u) # initial values 
#    #===============================================================================
        u_stack = numpy.empty(st['Nd']+(nBreg,),dtype=numpy.complex)
        
        for outer in numpy.arange(0,nBreg):
            for inner in numpy.arange(0,nInner):
                # update u
                print('iterating',[inner,outer])
                #===============================================================
#                 update u  # simple k-space deconvolution to guess initial u
                u = self.update_u(murf,u,uker,xx,bb)
                
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
            if st['senseflag']== 1:
                u = appendmat(u[...,0],L)
            
            (f, uf, murf,u)=self.external_update(u, f, uf, f0, u0) # update outer Split_bregman
            if st['senseflag']== 1:
                u = u[...,0:1]
                murf = murf[...,0:1]
            u_stack[...,outer] = Normalize(u[...,0])
#            u_stack[...,outer] =u[...,0] 
            
        return (u,u_stack)
    def external_update(self,u, f, uf, f0, u0): # overload the update function


        tmpuf=self.st['sensemap'].conj()*(
                self.CsTransform.forwardbackward(
                        u*self.st['sensemap']))

        if self.st['senseflag'] == 1:
            tmpuf=CombineMulti(tmpuf,-1)


        r = u0  - tmpuf
        p = r 
        for jj in range(0,1):
            Ap = numpy.copy(p)
            Ap=self.st['sensemap'].conj()*(
                    self.CsTransform.forwardbackward(
                            Ap*self.st['sensemap']))
    
            if self.st['senseflag'] == 1:
                Ap=CombineMulti(Ap,-1)
            
            alpha_k = numpy.sum((r.conj()*r)[:]) /numpy.sum((p.conj()*Ap)[:])
            alpha_k = alpha_k * 1.0
            print('alpha_k')
            uf = uf + alpha_k*p
            r_2= r - alpha_k*Ap
            beta_k = numpy.sum((r_2.conj()*r_2)[:]) /numpy.sum((r.conj()*r)[:])
            
            r = r_2
            p = r + beta_k*p
        #print('start of ext_update') 

#        checkmax(u)
#        checkmax(tmpuf)
#        checkmax(self.u0)
#        checkmax(uf)

#         fact=numpy.sum((self.u0-tmpuf)**2)/numpy.sum((u0)**2)
#         fact=numpy.abs(fact.real)
#         fact=numpy.sqrt(fact)
#         print('fact',fact)
# #        fact=1.0/(1.0+numpy.exp(-(fact-0.5)*self.thresh_scale))
#   
#         uf = uf+(u0-tmpuf)*3.0*fact

        checkmax(tmpuf)
        checkmax(u0)
        checkmax(uf)
        
        for jj in range(0,u.shape[-1]):
            u[...,jj] = u[...,jj]*self.st['sn']# rescale the final image intensity
   
        #print('end of ext_update')        
        murf = uf
        return (f,uf,murf,u)  
    def update_u(self,murf,u,uker,xx,bb):
        #print('inside update_u')
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
               self.LMBD*self.constraint(xx,bb) +      
               self.gamma * u) 
        
        rhs = rhs * self.st['mask'][...,0:u.shape[-1]]
 
#        rhs=Normalize(rhs)
        #=======================================================================
#         Trick: make "flist" for fftn 
        flist = mylist[:-1:1]    
            
        u = self.k_deconv(rhs, uker,self.st,flist,mylist)
        print('max rhs u',numpy.max(numpy.abs(rhs[:])),numpy.max(numpy.abs(u[:])))
        print('max,q',numpy.max(numpy.abs(self.st['q'][:])))
#        for jj in range(0,1):
#            u = u - 0.1*(self.k_deconv(u, 1.0/(RTR+self.LMBD*uker+self.gamma),self.st,flist,mylist) - rhs 
#                         )
#        checkmax(u)
#        checkmax(rhs)
#        checkmax(murf)
        
        #print('leaving update_u')
        return u # normalization    
    def k_deconv(self, u,uker,st,flist,mylist):
        u0=numpy.copy(u)
        
        u=u*st['mask'][...,0:u.shape[-1]]
        
#            u=scipy.fftpack.fftn(u, st['Kd'],flist)

        U=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
        
        for pj in range(0,u.shape[-1]):
            U[...,pj]=self.CsTransform.__emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) / uker[...,pj] # deconvolution
            U[...,pj]=self.CsTransform.emb_ifftn(U[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
         
        u = U[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
        
        # optional: one- additional Conjugated step to ensure the quality
        
        for pp in range(0,1):
            u = self.cg_step(u0,u,uker,st,flist,mylist)
        
        
        u=u*st['mask'][...,0:u.shape[-1]]
      
        return u
    def cg_step(self, rhs, u, uker, st,flist,mylist):
        u=u#*st['mask'][...,0:u.shape[-1]]
#            u=scipy.fftpack.fftn(u, st['Kd'],flist)
        AU=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
#        print('U.shape. line 446',U.shape)
#        print('u.shape. line 447',u.shape)
        for pj in range(0,u.shape[-1]):
            AU[...,pj]=self.CsTransform.__emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
            AU[...,pj]=self.CsTransform.__emb_ifftn(AU[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
            
         
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
                AU[...,pj]=self.CsTransform.__emb_fftn(p[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
                AU[...,pj]=self.CsTransform.__emb_ifftn(AU[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
                
             
            Ap = AU[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
            
            lower_inner =  p.conj()*Ap
            lower_inner = numpy.sum(lower_inner[:])
            
            alfa_k =  upper_inner/ lower_inner
            
            
            u = u + alfa_k * p
            
            r2 = r - alfa_k *Ap
            beta_k = numpy.sum((r2.conj()*r2)[:])/numpy.sum((r.conj()*r)[:])
          
            r = r2
            
            p = r + beta_k*p
            
        return u
        
#    def constraint(self,xx,bb):
#  
#        n_xx = len(xx)
#        n_bb =  len(bb)
#        print('n_xx',n_xx)
#        if n_xx != n_bb: 
#            print('xx and bb size wrong!')
#        
#        cons_shape = numpy.shape(xx[0])
#        cons=numpy.zeros(cons_shape,dtype=numpy.complex128)
#        
#        for jj in range(0,n_xx):
#
#            cons =  cons + get_Diff_H( xx[jj] - bb[jj] ,  jj)
#        
#
#        return cons  
    def constraint(self,xx,bb):
        '''
        include TVconstraint and others
        '''
        cons = TVconstraint(xx[0:2],bb[0:2])
        
        return cons
    
    def shrink(self,dd,bb,thrsld):
        '''
        soft-thresholding the edges
        
        '''
        output_xx=shrink( dd[0:2], bb[0:2], thrsld)# 3D thresholding 
        
        return output_xx
        
    def make_split_variables(self,u):
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
    def make_sense(self,u0):
        st=self.st
        L=numpy.shape(u0)[-1]
        u0dims= numpy.ndim(u0)
        
        rows=numpy.shape(u0)[0]
        cols=numpy.shape(u0)[1]
        # dpss rely on ctypes which are not suitable for cx_freeze 
        #    [dpss_rows, eigens] = spectrum.mtm.dpss(rows, 50, 1)
        #    [dpss_cols, eigens] = spectrum.mtm.dpss(cols, 50, 1)
        dpss_rows = numpy.kaiser(rows, 100)
        dpss_cols = numpy.kaiser(cols, 100)
        
        dpss_rows = numpy.fft.fftshift(dpss_rows)
        dpss_cols = numpy.fft.fftshift(dpss_cols)
        
        dpss_rows = appendmat(dpss_rows,cols)
        dpss_cols  = appendmat(dpss_cols,rows)
        
        
        dpss_fil=dpss_rows*dpss_cols.T # low pass filter
        
        rms=numpy.sqrt(numpy.mean(u0*u0.conj(),-1)) # Root of sum square
        st['sensemap']=numpy.ones(numpy.shape(u0),dtype=numpy.complex64)
        
        #    print('L',L)
        #    print('rms',numpy.shape(rms))
        for ll in numpy.arange(0,L):
            st['sensemap'][:,:,ll]=(u0[:,:,ll]+1e-15)/(rms+1e-15)
            st['sensemap'][:,:,ll]=scipy.fftpack.ifft2(scipy.fftpack.fft2(st['sensemap'][:,:,ll])*dpss_fil)
        return st

    def create_kspace_sampling_density(self):
            #=======================================================================
            # RTR: k-space sampled density
            #      only diagonal elements are relevant (on k-space grids)
            #=======================================================================
        RTR=self.st['q'] # see __init__() in class "pyNufft"
        
        return RTR 
    def create_laplacian_kernel(self):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to constraint
#===============================================================================
        uker = numpy.zeros(self.st['Kd'][0:2],dtype=numpy.complex64)
        rows_kd = self.st['Kd'][0]
        cols_kd = self.st['Kd'][1]
#        uker[0,0] = 1.0
        uker[0,0] = -4.0
        uker[0,1]=1.0
        uker[1,0]=1.0
        uker[rows_kd-1,0]=1.0
        uker[0,cols_kd-1]=1.0 
        uker = scipy.fftpack.fft2(uker,self.st['Kd'][0:2],[0,1])

        return uker
    def expand_deconv_kernel_dimension(self, uker, L):

        if numpy.size(self.st['Kd']) > 2:
            for dd in range(2,numpy.size(self.st['Kd'])):
                uker = appendmat(uker,self.st['Kd'][dd])
        
        uker = appendmat(uker,L)
        
        
        return uker
    def expand_RTR(self,RTR,L):
        if numpy.size(self.st['Kd']) > 2:
            for dd in range(2,numpy.size(self.st['Kd'])):
                RTR = appendmat(RTR,self.st['Kd'][dd])
                
        RTR= numpy.reshape(RTR,self.st['Kd'],order='F')
        
        RTR = appendmat(RTR,L)

        return RTR

    def update_d(self,u,dd):

        out_dd = ()
        for jj in range(0,len(dd)) :
            out_dd = out_dd  + (get_Diff(u,jj),)

        return out_dd
    
    def update_b(self, bb, dd, xx):
        ndims=len(bb)
        cc=numpy.empty(bb[0].shape)
        out_bb=()
        for pj in range(0,ndims):
            cc=bb[pj]+dd[pj]-xx[pj]
            out_bb=out_bb+(cc,)

        return out_bb
  

    def create_mask(self,u0):
        st=self.st
        
#         rows=u0.shape[0]
#         cols=u0.shape[1]
# 
#         kk = numpy.arange(0,rows)
#         jj = numpy.arange(0,cols)
# 
#         kk = appendmat(kk,cols)
#         jj = appendmat(jj,rows).T
#         st['mask']=numpy.ones((rows,cols),dtype=numpy.float32)
# 
#         #add circular mask
#         sp_rat=(rows**2+cols**2)*1.0
#         
#         for jj in numpy.arange(0,cols):
#             for kk in numpy.arange(0,rows):
#                 if ( (kk-rows/2.0)**2+(jj-cols/2.0)**2 )/sp_rat > 1.0/8.0:
#                     st['mask'][kk,jj] = 0.0
#         
#         if numpy.size(u0.shape) > 2:
#             for pp in range(2,numpy.size(u0.shape)):
#                 st['mask'] = appendmat(st['mask'],u0.shape[pp] )
 
        return st

class pySplitBregman(CsSolver):
    def external_update(self,u, f, uf, f0, u0):
        f = f+f0
        f= f-self.CsTransform.forward(u*self.st['sensemap'])
            #        f = f2
                    #murf = numpy.fft.ifft2(mu*R*f)*scale
        murf = self.st['sensemap'].conj()*(self.CsTransform.backward(f))
        murf=CombineMulti(murf,-1)
        murf=Normalize(murf)*self.thresh_scale
        return (f,uf, murf,u)
#     def k_deconv(self, u,uker,st,flist,mylist):
# 
#         u=u*st['mask'][...,0:u.shape[-1]]
# #            u=scipy.fftpack.fftn(u, st['Kd'],flist)
#         U=numpy.empty(st['Kd']+(u.shape[-1],),dtype=u.dtype)
#         for pj in range(0,u.shape[-1]):
#             U[...,pj]=pf.emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) / uker[...,pj] # deconvolution
#             U[...,pj]=pf.emb_ifftn(U[...,pj], st['Kd'], range(0,numpy.size(st['Kd'])))  
#         u = U[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
# 
#         u=u*st['mask'][...,0:u.shape[-1]]
#         return u 
    
class pyPseudoInverse(CsSolver):
    pass
    
class pyCube2D(pyPseudoInverse):
    '''
    2D CS without mask sensitivity maps
    '''
    def create_mask(self,u0):
        pass
    def make_sense(self,u0,st):
        pass
    def solve(self): # main function of solver
#         self.st['senseflag']=0 # turn-off sense, to get sensemap
#         (self.u,tmp)=self.kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 
#                       self.nInner,self.nBreg)
        self.u=self.CsTransform.inverse(self.f, self.mu, self.LMBD, self.gamma, 
                      self.nInner,self.nBreg)
# CineBaseSolver=pyPseudoInverse
class pyCube3D(pyPseudoInverse):
    '''
    2D CS without mask sensitivity maps
    '''
    def create_mask(self,u0):
        pass
    def make_sense(self,u0,st):
        pass
    def solve(self): # main function of solver
        self.st['senseflag']=0 # turn-off sense, to get sensemap
#         (self.u,tmp)=self.kernel(self.f, self.st , self.mu, self.LMBD, self.gamma, 
#                       self.nInner,self.nBreg)
        self.u=self.CsTransform.inverse(self.f, self.mu, self.LMBD, self.gamma, 
                      self.nInner,self.nBreg)