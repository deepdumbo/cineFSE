
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
    from pynufft import *
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
norm=matplotlib.colors.Normalize(vmin=0.0, vmax= 1)
norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=1)
dtype =  numpy.complex64
# try:
#     from numba import jit  
#  
# except:
#     print('numba not supported')
#!/usr/bin/python2.7

import KspaceDesign.Propeller
import CsTransform.pynufft
# import scipy.misc
# import numpy
import cPickle
def pp_load_file_gpu_recon(NufftObj,data, LMBD, gamma,nInner, nBreg ):
    NufftObj.initialize_gpu() 
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
    return image_recon
def pp_load_file_dc(NufftObj,data):
    image_blur = NufftObj.backward2(data)  
    return image_blur
# def pp_load_file_gpu_recon(NufftObj,data, LMBD, gamma,nInner, nBreg ):
#     NufftObj.initialize_gpu() 
#     image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
#     return image_recon
# def pp_load_file_dc(NufftObj,data):
#     image_blur = NufftObj.pseudoinverse2(data)  
#     return image_blur
def pp_recon(NufftObj,my_shift,data, LMBD, gamma,nInner, nBreg ):
    import CsTransform.pynufft
#     NufftObj=CsTransform.pynufft.pynufft(om,Nd,Kd,Jd,n_shift=n_shift )
    NufftObj.linear_phase(  my_shift ) 
    NufftObj.initialize_gpu() 
    image_recon = NufftObj.pseudoinverse(data, 1.0, LMBD, gamma,nInner, nBreg)
    image_dc = NufftObj.pseudoinverse2(data) 
#     result = numpy.zeros(numpy.shape(image_recon)+(2,))
#     print(numpy.shape(image_recon),numpy.shape(image_recon)) 
    return (image_recon, image_dc )
def calculate_dim_index(n_x,n_y,N):
    
    xdim = numpy.ceil(N*1.0/n_x)  
    ydim = numpy.ceil(N*1.0/n_y)  
    
    center_of_tiles  = numpy.zeros((n_x*n_y,2))
    
    for pp in xrange(0,n_x*n_y):
        center_of_tiles[pp,0] = numpy.mod(  xdim * pp + xdim/2 ,  N ) - N/2
        center_of_tiles[pp,1] = numpy.floor( pp*1.0 /n_x ) * ydim + ydim /2 -N/2 
    
    print('center_of_tiles',center_of_tiles)
    
    return xdim, ydim, center_of_tiles
def proto_rFOV_2D():
    '''
    prototype for subprocess
    '''
    import time
    import numpy 
    import matplotlib#.pyplot
    cm = matplotlib.cm.gray
    # load example image    
#     norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
#     norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)
 
    N = 384
    Nd =(N,N) # image space size
    import phantom
    image = phantom.phantom(Nd[0])
    Kd =(2*N,2*N) # k-space size   
    Jd =(6,6) # interpolation size
    import KspaceDesign.Propeller
    nblade = 32#   numpy.round(N*3.1415/24/2)
#     print('nblade',nblade)
    theta=180.0/nblade #111.75 #
    propObj=KspaceDesign.Propeller.Propeller( N ,24,  nblade,   theta,  1,  -1)
    om = propObj.om
 
    NufftObj = pynufft(om, Nd,Kd,Jd )
 
    precondition = 2
    factor = 0.01
    NufftObj.factor = factor
    NufftObj.precondition = precondition
    # simulate "data"
    data= NufftObj.forward(image )
    data_shape = (numpy.shape(data))
    power_of_data= numpy.max(numpy.abs(data))
    data = data + 1.0e-3*power_of_data*(numpy.random.randn(data_shape[0],data_shape[1])+1.0j*numpy.random.randn(data_shape[0],data_shape[1]))
 
    mu= 1
    gamma = 0.001
    LMBD =1
    nInner=2
    nBreg = 25
    # now get the original image
    #reconstruct image with 0.1 constraint1, 0.001 constraint2,
    # 2 inner iterations and 10 outer iterations 
    NufftObj.st['senseflag'] = 1
    
    n_x =1
    n_y =1 
    xdim, ydim, center_of_tiles= calculate_dim_index(n_x,n_y,N)
    xdim = 192
    ydim = 192
    center_of_tiles = (0,0)
#     xdim = numpy.floor(N/n_x) +2  
#     ydim = numpy.floor(N/n_y) +2 #128
    kdim = N*2
    
    NufftObj = pynufft(om, (xdim,ydim),(kdim,kdim),Jd,n_shift=(0,0))
#     NufftObj.linear_phase(om, (32,32), NufftObj.st['M'])# = pynufft(om, (64,64),(512,512),Jd,n_shift=(0,32))
#     NufftObj.linear_phase(om, (-84, 0), NufftObj.st['M'])# = pynufft(om, (64,64),(512,512),Jd,n_shift=(0,32))
#     import cPickle
#     f = open('Nufftobj.pickle','wb')
#     cPickle.dump(NufftObj ,f,2)
#     f.close()
#     f = open('data.pickle','wb')
# 
#     cPickle.dump(data  ,f,2)
#     f.close() 
#     del NufftObj    # confirm that cPickle works
#     del data        # confirm that cPickle works
    
    
#     t = time.time() 
#     image_recon= pp_load_file_gpu_recon(NufftObj, data, LMBD, gamma,nInner, nBreg,(-84,0))
#     image_blur = pp_load_file_dc(NufftObj,data)[...,0]
    import pp
    job_server = pp.Server()
    jobs = []  
    jobs2 = []

    
     
    t = time.time()
    for tt in xrange(0,n_x*n_y):
# #     
#         if tt == 0:
#             my_shift = (0,0)
#         else:
#             my_shift = (-84,0)
#         my_shift = (-center_of_tiles[tt,0], -center_of_tiles[tt,1])
        my_shift = (-84,0)
        NufftObj.linear_phase(  my_shift )
        jobs.append(job_server.submit( pp_load_file_gpu_recon, (NufftObj, data, LMBD, gamma,nInner, nBreg ),
#                                    modules = ('pyfftw','numpy','pynufft','scipy','reikna'),
                             globals = globals()))
        jobs2.append(job_server.submit( pp_load_file_dc, (NufftObj, data),
#                                    modules = ('pyfftw','numpy','pynufft','scipy','reikna'),
                             globals = globals()))
#         jobs.append(job_server.submit(pp_recon,(NufftObj, my_shift,data, LMBD, gamma,nInner, nBreg )))
        
    for tt in range(0,n_x*n_y):
#         if tt ==0:
        image_recon = jobs[tt]()
        image_blur = jobs2[tt]()[...,0]
#         (tmp_imag, tmp_imag2) = jobs[tt]()
        image_recon = Normalize(numpy.real(image_recon))#*  1.1# *1.3
        image_blur=Normalize(numpy.real(image_blur ))#*1.15
#         matplotlib.pyplot.figure(0)
#         fig, ax =matplotlib.pyplot.subplot(n_x,n_y,tt)
#         cax = ax.imshow( numpy.abs(-image_recon.real + image),
# 				cmap= cm,interpolation = 'nearest')
#         cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
#         cbar.ax.set_yticklabels(['< 0', '0.5', '> 1'])# vertically oriented colorbar
        
        
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.subplot(1,2,1)        
        matplotlib.pyplot.imshow( (image_recon.real) ,
                                norm = norm,
                                cmap= cm,interpolation = 'nearest')        
        matplotlib.pyplot.subplot(1,2,2)        
        matplotlib.pyplot.imshow( (image_blur.real) ,
                                norm = norm,
                                cmap= cm,interpolation = 'nearest')       
#         else:
#             image_blur = jobs[tt]()
    job_server.wait()                   
    job_server.destroy() 
    
     
#     image_recon = Normalize(numpy.real(image_recon))#*  1.1# *1.3
#     image_blur=Normalize(numpy.real(image_blur[...,0]))#*1.15
    elapsed = time.time() - t
    print("time is ",elapsed)


    matplotlib.pyplot.show()
    

if __name__ == '__main__':
    import cProfile
#     test_wavelet()
#     test_1D()
#     test_prolate()
#     test_2D()
#     cProfile.run('proto_rFOV_2D()')
    proto_rFOV_2D()
