import numpy
import pyfftw

import scipy
import scipy.fftpack
import scipy.fftpack._fftpack
# import CsSolver
# import CsSolver
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('../CsTransform')
sys.path.append('../GeRaw')

import CsTransform.pynufft 
import matplotlib.pyplot
import CsSolver

cm=matplotlib.cm.gray
norm=matplotlib.colors.Normalize(vmin=1e+7, vmax=1e+9)
def makeRandom(input_shape,
               sparsity,
               full_circle_ratio):
    
    import numpy.random

    R = numpy.random.rand(input_shape[0],input_shape[1])
    R[R>=sparsity]=0.0
    R=numpy.ceil(R)
#    mag_num=32
#     full_circle=16 # radius of full-sampling circle
    y_direction = numpy.int(input_shape[0]*full_circle_ratio[0])
    z_direction = numpy.int(input_shape[1]*full_circle_ratio[1])
#     print(y_direction, z_direction)
    for ii in range(0,input_shape[0]/2+1):
        for jj in range(0,input_shape[1]/2+1):
#             print(((1.0*ii/y_direction)**2+(1.0*jj/z_direction)**2))
            if ((1.0*ii/y_direction)**2+(1.0*jj/z_direction)**2) <= 1.0/4.0:
                #central circle            
                R[ ii, jj] = 1.0
                R[ ii,-jj] = 1.0
                R[-ii, jj] = 1.0
                R[-ii,-jj] = 1.0
            elif ((1.0*ii/input_shape[0])**2+(1.0*jj/input_shape[1])**2) > 1.0/4.0:
                #removing 4 corners
                R[ ii, jj] = 0.0
                R[ ii,-jj] = 0.0
                R[-ii, jj] = 0.0
                R[-ii,-jj] = 0.0                
            else:
                pass

    actual_ratio = numpy.sum(numpy.sum(R))/input_shape[0]/input_shape[1]
#     print(actual_ratio)
    return R, actual_ratio
def convert_mask_to_index(R):
    R2=R.copy()
    
    R2=numpy.fft.fftshift(R2,(0,1))

    ind=numpy.where(R2 > 0)
    ind0=numpy.array(ind).T
    ind=numpy.array(ind).T*1.0
#     print(ind)
#     print('R2.shape',R2.shape)
    ind[:,0]=1.0*(ind[:,0]-R2.shape[0]/2)/R2.shape[0]
    ind[:,1]=1.0*(ind[:,1]-R2.shape[1]/2)/R2.shape[1]


    ind=ind*numpy.pi*2.0
    return ind
# class Cartesian3DSolver(CsTransform.pynufft.pynufft):
#     def __init__(self,om, Nd, Kd,Jd):
#         '''
#         Note: om is two dimensional because x dimension is full-sampled
#         Nd, Kd, Jd are also two dimensional
#         '''
#         CsTransform.pynufft.pynufft.__init__(self,om, Nd, Kd,Jd)
#         # creation of computation kernel
#     def _xdim_ft(self,x):
#         return x
#     def _xdim_ift(self,X):
#         return X
#     
#     
#     def forward(self, x):
#         st=self.st
#         Nd = st['Nd']
#         Kd = st['Kd']
#         dims = numpy.shape(x)
#         dd = numpy.size(Nd)
#                  
#         xres = x.shape[0]   
#              
#         if numpy.ndim(x) == 3:
#             Lprod = 1 # coil
#         elif numpy.ndim(x) > dd: # multi-channel data
#             Lprod = numpy.size(x)/numpy.prod(Nd)/xres
#                     
# #         x = scipy.fftpack.fftshift(x, axes = 0 )
# #         x = scipy.fftpack.fftn(x, axes = (0,))
# #         x = scipy.fftpack.fftshift(x, axes = 0)
# 
#         X = numpy.empty( (xres, self.st['M'], Lprod  ),dtype = numpy.complex)
#         for jj in range(0,xres):
#             X[jj,:,:] = CsTransform.pynufft.pynufft.forward(self,x[jj,...])
#              
#         X = numpy.reshape(X, (xres* self.st['M'], Lprod),order='F')
#         print('xres=,',xres)
#         print('M=,',self.st['M'])
#         print('Lprod',Lprod)
#         
#          
#         return X
#      
# #     def backward(self, X):
# #         st=self.st
# # 
# #         Nd = st['Nd']
# #         Kd = st['Kd']
# # 
# # #         checker(X,st['M']) # check X of correct shape
# # 
# #         dims = numpy.shape(X)
# #         Lprod= numpy.prod(dims[1:]) 
# #         # how many channel * slices
# # 
# #         if numpy.size(dims) == 1:
# #             Lprod = 1
# #         else:
# #             Lprod = dims[1]
# #             
# #         xres =  dims[0]/ st['M']
# #         print('xres,',xres)
# #         print('dims,',dims)
# #         print('numpy.prod(Kd)',numpy.prod(Kd))
# #         x= numpy.empty((xres, )+Nd+(Lprod,), dtype = numpy.complex )
# #         X2= X.copy()
# #         
# #         print('xres in backward',xres)
# #         print('st[M] in backward',st['M'])
# #         print('Lprod in backward',Lprod)
# #         X2 = numpy.reshape(X2, (xres,st['M'],Lprod) ,order='F')
# #         for jj in range(0,xres):
# #             x[jj,...]=CsTransform.pynufft.pynufft.backward(self,X2[jj,:])
# # 
# # #         x = scipy.fftpack.fftshift(x, axes = 0 )
# # #         x = scipy.fftpack.ifftn(x, axes = (0,))
# # #         x = scipy.fftpack.fftshift(x, axes = 0)
# #         
# #         return x
#         
#     def inverse(self,data, mu, LMBD, gamma, nInner, nBreg):
#         '''
#         data: parallel 3D data
#         mu: 1.0
#         LMBD: 
#         '''
#         st=self.st
# 
#         Nd = st['Nd']
#         Kd = st['Kd']
# 
# #         checker(X,st['M']) # check X of correct shape
# 
#         dims = numpy.shape(data)
#         Lprod= numpy.prod(dims[1:]) 
#         # how many channel * slices
# 
#         if numpy.size(dims) == 1:
#             Lprod = 1
#         else:
#             Lprod = dims[1]
#             
#         xres =  dims[0]/ st['M']
#         print('data slice shape ',data.shape)        
#         print('xres in inverse',xres)
#         print('Kd',self.st['Kd'])
#         print('Lprod',Lprod)
#         data = numpy.reshape(data,(xres,self.st['M'],Lprod,),order='F')
#         
#         D = numpy.empty((xres,)+self.st['Nd'],dtype = numpy.complex)
# #         data = scipy.fftpack.ifftshift(data, axis = (0,))
# #         data = scipy.fftpack.ifftn(data, axes = (0,))
# #         data = scipy.fftpack.ifftshift(data, axis = (0,))
#         
#         for jj in range(0,xres):
# 
#             tmp_slice = CsTransform.pynufft.pynufft.inverse(self,data[jj,...], mu, LMBD, gamma, nInner, nBreg)
#             D[jj,...] = tmp_slice
#             matplotlib.pyplot.imshow(numpy.abs(tmp_slice),cmap=cm)
#             matplotlib.pyplot.show()
#             
#         return D
def loadRandom(pp):
#     import scipy.io
    folder = 'zerop'
    if pp == 0.1:
        R= numpy.loadtxt(folder+'1')
    elif pp == 0.2:
        R= numpy.loadtxt(folder+'2')
    elif pp == 0.3:
        R= numpy.loadtxt(folder+'3')
    elif pp == 0.4:
        R= numpy.loadtxt(folder+'4')
    elif pp == 0.5:
        R= numpy.loadtxt(folder+'5')
    elif pp == 0.6:
        R= numpy.loadtxt(folder+'6')
    elif pp == 0.7:
        R= numpy.loadtxt(folder+'7')
    elif pp == 0.8:
        R= numpy.loadtxt(folder+'8')
    
    elif pp == 0.9:
        R= numpy.loadtxt(folder+'9')

    R = scipy.fftpack.fftshift(R)
        
    for ii in range(0,224/2+1):
        for jj in range(0,40/2+1):
#             print(((1.0*ii/y_direction)**2+(1.0*jj/z_direction)**2))
#             if ((1.0*ii/224)**2+(1.0*jj/40)**2) <= 1.0/4.0:
#                 #central circle            
#                 R[ ii, jj] = 1.0
#                 R[ ii,-jj] = 1.0
#                 R[-ii, jj] = 1.0
#                 R[-ii,-jj] = 1.0
            if ((1.0*ii/224)**2+(1.0*jj/40)**2) > 1.0/4.0:
                #removing 4 corners
                R[ ii, jj] = 0.0
                R[ ii,-jj] = 0.0
                R[-ii, jj] = 0.0
                R[-ii,-jj] = 0.0                
            else:
                pass        
        
    actual_ratio = numpy.sum(numpy.sum(R))/224.0/40.0
    matplotlib.pyplot.imshow(R)
    matplotlib.pyplot.show()    
    return R,actual_ratio   

def foo4():
    import GeRaw.pfileparser
    filename='/home/sram/Cambridge_2012/DATA_MATLAB/chengcheng/cube_raw_20130704.raw'
    rawdata=GeRaw.pfileparser.geV22(filename)
    
    while len(rawdata.k.shape) > 4:
        rawdata.k= rawdata.k[...,0]    
    
    
    
    print('point size',rawdata.hdr['rdb']['point_size'])
    print(numpy.shape(rawdata.k),numpy.shape(rawdata.k)[1:3])

#     R,ratio=makeRandom(rawdata.k.shape[1:3],0.45,(0.15,0.35))
    
    R,ratio = loadRandom(0.5)
#     import scipy.io
#     R = numpy.loadtxt('zerop3')
#     ratio = sum(sum(R))/224.0/40.0

    print('true ratio',ratio)
    matplotlib.pyplot.imshow(R)
    matplotlib.pyplot.show()
    ind = convert_mask_to_index(R)
    print(ind)
    matplotlib.pyplot.plot(ind[:,1],ind[:,0],'x')
    matplotlib.pyplot.show()
    om = ind
    Nd = (224,40)
    Kd = (224,40)
    Jd = (1,1)
    x = rawdata.k
    x = scipy.fftpack.fftshift(x,axes = (1,2))
    x = pyfftw.interfaces.scipy_fftpack.fftn(x,axes = (1,2),threads=2)
    x = scipy.fftpack.fftshift(x,axes = (1,2))

    x = scipy.fftpack.fftshift(x,axes = (0,))
    x = pyfftw.interfaces.scipy_fftpack.fftn(x,axes = (0,),threads=2)
    x = scipy.fftpack.fftshift(x,axes = (0,))

    MyTransform = CsTransform.pynufft.pynufft( om, Nd,Kd,Jd)
#     Cartesian3DObj = Cartesian3DSolver(om, (224,40), (224,40),(1,1))

    original = x
    recon = numpy.empty((224,224,40,4),dtype = numpy.complex)
    backward = numpy.empty_like(recon)
#     for jj in range(0,224):
#         for kk in range(0,4):
#             print(jj/224.0,kk/224.0)

#     jj = 12
    

#     c=numpy.transpose(c,(1,0))
#     c=c.real/numpy.max(numpy.abs(c[:]))
    f = numpy.empty((MyTransform.st['M'],),dtype = numpy.complex)
    for jj in range(0,224):
        for kk in range(0,4):
            print(jj/224.0, kk/4.0)
            c=x[jj,:,:,kk]
            f=MyTransform.forward(c)
            backward[jj,:,:,kk] = MyTransform.backward(f)[:,:,0]
            Solver1=CsSolver.pyCube2D(MyTransform, f, 1.0, 0.1, 0.001, 4,45)
            Solver1.solve()    
            recon[jj,:,:,kk] = Solver1.u
#     return
#    myu0 = MyTransform.forwardbackward(c)
#    Solver1=CsSolver.CsSolver.isra(MyTransform,f,40)
#     Solver1=CsSolver.pyCube2D(MyTransform, f, 1.0, 0.1, 0.001, 1,15)

    numpy.save('original',original) 

    numpy.save('recon',recon) 

    numpy.save('backward',backward) 
    
def filter_3D(input_x):
    tmp_x = numpy.copy(input_x)
    
    
    tmp_x = scipy.fftpack.ifftshift(tmp_x, axes = (0,1,2))
    tmp_x = pyfftw.interfaces.scipy_fftpack.ifftn(tmp_x, axes = (0,1,2),threads=2)
    tmp_x = scipy.fftpack.ifftshift(tmp_x, axes = (0,1,2))
    
    shape = numpy.shape(input_x)
    xres = shape[0]
    yres = shape[1]
    zres = shape[2]
    window_para= 0.5
    u0dims= 3#numpy.ndim(u0)
    
    if u0dims-1 >0:
        rows=numpy.shape(input_x)[0]
        dpss_rows = numpy.kaiser(rows, window_para)     
#         dpss_rows = numpy.fft.fftshift(dpss_rows)
#         dpss_rows[3:-3] = 0.0
        dpss_fil = dpss_rows
        print('dpss shape',dpss_fil.shape)
    if u0dims-1 > 1:
                          
        cols=numpy.shape(input_x)[1]
        dpss_cols = numpy.kaiser(cols, window_para)            
#         dpss_cols = numpy.fft.fftshift(dpss_cols)
#         dpss_cols[3:-3] = 0.0
        
        dpss_fil = CsTransform.pynufft.appendmat(dpss_fil,cols)
        dpss_cols  = CsTransform.pynufft.appendmat(dpss_cols,rows)

        dpss_fil=dpss_fil*numpy.transpose(dpss_cols,(1,0))
        print('dpss shape',dpss_fil.shape)
    if u0dims-1 > 2:
         
        zag = numpy.shape(input_x)[2]
        dpss_zag = numpy.kaiser(zag, window_para)            
#         dpss_zag = numpy.fft.fftshift(dpss_zag)
#         dpss_zag[3:-3] = 0.0
        dpss_fil = CsTransform.pynufft.appendmat(dpss_fil,zag)
                  
        dpss_zag = CsTransform.pynufft.appendmat(dpss_zag,rows)
         
        dpss_zag = CsTransform.pynufft.appendmat(dpss_zag,cols)
         
        dpss_fil=dpss_fil*numpy.transpose(dpss_zag,(1,2,0)) # low pass filter
        print('dpss shape',dpss_fil.shape)
    #dpss_fil=dpss_fil / 10.0    
#     while numpy.ndim(dpss_fil)< numpy.ndim(input_x):
#         dpss_fil=CsTransform.pynufft.appendmat(dpss_fil,shape[numpy.ndim(dpss_fil)])
    
    for pp in range(0,numpy.ndim(dpss_fil)-3):
        tmp_x[:,:,:,pp] = dpss_fil * tmp_x[:,:,:,pp]
    
    XRES = xres*2
    YRES = yres*2
    ZRES = zres*2
    
    output_x = numpy.zeros((XRES, YRES, ZRES,shape[3]), dtype = numpy.complex64 )
    
    output_x[(XRES/2-xres/2):(XRES/2+xres/2)  ,
             (YRES/2-yres/2):(YRES/2+yres/2) ,
             (ZRES/2-zres/2):(ZRES/2+zres/2) ,...] =tmp_x
    
    output_x = scipy.fftpack.fftshift(output_x, axes = (0,1,2))
    output_x = pyfftw.interfaces.scipy_fftpack.fftn(output_x, axes = (0,1,2),threads=2)
    output_x = scipy.fftpack.fftshift(output_x, axes = (0,1,2))
    
    return output_x
                
def show_recon():
    recon=numpy.load('recon.npy') 
    original=numpy.load('original.npy')
    backward=numpy.load('backward.npy')
     
    print(numpy.shape(recon))
    print(numpy.shape(original))
    print(numpy.shape(backward))
      
    recon=filter_3D(recon)
    original = filter_3D(original)
    backward = filter_3D(backward)
    print(numpy.shape(recon))
    print(numpy.shape(original))
    print(numpy.shape(backward))
     

     
    recon = numpy.mean(recon**2,3)
    original=      numpy.mean(original**2,3)
    backward = numpy.mean(backward**2,3)

    recon=numpy.sqrt(recon)
    original=numpy.sqrt(original)
    backward = numpy.sqrt(backward)

    import scipy.io
    scipy.io.savemat('cube_3D_recon.mat',mdict={'recon': recon})
    scipy.io.savemat('cube_3D_original.mat',mdict={'original': original})
    scipy.io.savemat('cube_3D_backward.mat',mdict ={'backward':backward})
   
#     X = Cartesian3DObj.forward(x) # simulate the data 
    
#     print('Xshape',X.shape)
    
#     x = Cartesian3DObj.backward(X) # regridding of the data
#     mu = 1.0
#     LMBD = 0.05
#     gamma = 1.0e-3
#     nInner = 2
#     nBreg = 50
#      
#     x = Cartesian3DObj.inverse(X, mu, LMBD, gamma, nInner, nBreg)
#     print('xshape',x.shape)

        
#     sdfasdf= recon
#     
# #     sdfasdf=numpy.sum(sdfasdf,2)

    sli=79
    matplotlib.pyplot.subplot(2,2,1)
    matplotlib.pyplot.imshow(original[:,:,sli].real,cmap=cm, norm = norm)
    matplotlib.pyplot.title('full-sample')
    
    matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.imshow(backward[:,:,sli].real,cmap=cm, norm = norm)    
    matplotlib.pyplot.title('40.1% undersampled')
    
    matplotlib.pyplot.subplot(2,2,3)
    matplotlib.pyplot.imshow(recon[:,:,sli].real,cmap=cm, norm  = norm)
    matplotlib.pyplot.title('CS reconstructed')
    matplotlib.pyplot.subplot(2,2,4)
    matplotlib.pyplot.imshow(-original[:,:,sli].real+recon[:,:,sli].real,cmap=cm, norm  = norm)    
    matplotlib.pyplot.title('difference between CS and full')     
    matplotlib.pyplot.show()
    
    print('size of data',recon.shape)

                
if __name__ == "__main__":
#     foo2() # simulation using image space data
#     foo2() # simulation using
    import gc 
    gc.enable()
#     foo4() # cube raw data for simulation   
    import cProfile     
    cProfile.run('show_recon()')
    
        
    