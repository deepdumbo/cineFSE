 #translated from test_mrics.m by Tom Goldstein  (tagoldst@math.ucla.edu)

import numpy
import numpy.random
import numpy.fft
import matplotlib.pyplot
import matplotlib.cm
import phantom
#from spectrum import *
#import spectrum.window
#from spectrum import window
#import spectrum
#spectrum.window_visu(64, 'hamming')


def myfft2(input_x):
    return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(input_x)))
def myifft2(input_x):
    return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(input_x)))
def Dx(u):
    (rows,cols) = numpy.shape(u) 
    d = numpy.zeros((rows,cols),u.dtype)
    d[:,1:cols-1] = u[:,1:cols-1]-u[:,0:cols-2]
    d[:,0] = u[:,0]-u[:,cols-1]
    return d


def Dxt(u):
    (rows,cols) = numpy.shape(u)
    d = numpy.zeros((rows,cols),u.dtype)
    d[:,0:cols-2] = u[:,0:cols-2]-u[:,1:cols-1]
    d[:,cols-1] = u[:,cols-1]-u[:,0]
    return d
def Dy(u):
    (rows,cols) = numpy.shape(u) 
    d = numpy.zeros((rows,cols),u.dtype)
    d[1:rows-1,:] = u[1:rows-1,:]-u[0:rows-2,:]
    d[0,:] = u[0,:]-u[rows-1,:]  
    return d
def Dyt(u):
    (rows,cols) = numpy.shape(u) 
    d = numpy.zeros((rows,cols),u.dtype)
    d[0:rows-2,:] = u[0:rows-2,:]-u[1:rows-1,:]
    d[rows-1,:] = u[rows-1,:]-u[0,:]
    return d

def shrink2(x,y,LMBD):
    s = numpy.sqrt(x*numpy.conj(x)+y*numpy.conj(y))
    s = (numpy.real(s))
    LMBD=LMBD*1.0
    #ss = s-LMBD
#    tmp_ss=(ss>0)*1.0
#    print(ss)
#     ss = (s-LMBD)*(s > LMBD)
# 
#     s = s+(s<LMBD)
    ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-15)
#     ss = ss/s

    xs = ss*x
    ys = ss*y
    return (xs,ys)



def mrics(R, f, mu, LMBD, gamma, nInner, nBreg):
    (rows,cols) = numpy.shape(f)
    f0 = f
    u = numpy.zeros((rows,cols),dtype=numpy.complex128)
    x = numpy.zeros((rows,cols),dtype=numpy.complex128)
    y = numpy.zeros((rows,cols),dtype=numpy.complex128)
    bx = numpy.zeros((rows,cols),dtype=numpy.complex128)
    by = numpy.zeros((rows,cols),dtype=numpy.complex128)
#     numpy.fft.fftshift
    # build the kernel
    scale = numpy.sqrt(rows*cols)
    murf = numpy.fft.ifft2(mu*(numpy.conj(R)*f))*scale
    
    uker = numpy.zeros((rows,cols),dtype=numpy.complex128)
    # Laplacian oeprator (sram)
    uker[0,0] = 4.0
    uker[0,1]=-1.0
    uker[1,0]=-1.0
    uker[rows-1,0]=-1.0
    uker[0,cols-1]=-1.0 
    
    # K: diagonal operator
    uker = mu*(numpy.conj(R)*R)+LMBD*numpy.fft.fft2(uker)+gamma
#     outer=0 #iterator
#     inner=0 #iterator
    c = numpy.max(murf[...])
    for outer in numpy.arange(0,nBreg):
        for inner in numpy.arange(0,nInner):
            # update u   
            rhs = murf+LMBD*Dxt(x-bx)+LMBD*Dyt(y-by)+gamma*u
            u = numpy.fft.ifft2(numpy.fft.fft2(rhs)/uker)         
#            update x and y
            dx = Dx(u)
            dy  =Dy(u)
            (x,y)=shrink2( dx+bx, dy+by, c*0.1/LMBD)
            # update bregman parameters
            bx = bx+dx-x
            by = by+dy-y
        f = f+f0-R*numpy.fft.fft2(u)/scale
        murf = numpy.fft.ifft2(mu*R*f)*scale
    return u
def sparse2blade_conf(etl,vardense):
    '''
#     input: xres dim
    '''
    size_of_vardense = numpy.size(vardense)
    phase_encode = numpy.empty((etl,))
    for jj in xrange(0,etl):
        phase_encode[jj]  =  vardense[jj-etl/2+size_of_vardense/2]
    print(phase_encode)
    
    positive_max = numpy.max(phase_encode)
    negative_max = numpy.max(-phase_encode)
    
    if positive_max > negative_max+1:
        print('positive_max 1 ',positive_max)
        print('negative_max 1 ',negative_max)
        new_etl = 2*positive_max.astype(int)
    elif positive_max <= negative_max+1:
        print('positive_max 2 ',positive_max)
        print('negative_max 2 ',negative_max)
        new_etl = 2*(negative_max+1).astype(int) 
#         self.etl = new_etl # warning: change self.etl here
        
    print('new_etl',new_etl)
    return new_etl,phase_encode

def sparse2blade_init(raw_blade, etl,new_etl, phase_encode):
    xres= numpy.shape(raw_blade)[0]
    R = numpy.zeros((xres,new_etl))
    for jj in xrange(0,etl):
#         print('phase_encode[jj] = ',jj,phase_encode[jj])
        R[:,phase_encode[jj]-1]= numpy.ones((xres,))

#     matplotlib.pyplot.imshow(raw_blade.real)
#     matplotlib.pyplot.show()
#     matplotlib.pyplot.imshow(R.real)
#     matplotlib.pyplot.show()

        
    F =numpy.zeros((xres,new_etl),dtype = numpy.complex)
    
    for jj in xrange(0,etl):
        F[:,phase_encode[jj]-1] = raw_blade[:,jj]#kspace[:, jj - new_etl/2]
    F= numpy.fft.fftshift(F,axes = 0)   
    return F,R   
def sparse2blade_final(recovered, F,R):
    N = numpy.shape(F)[0]
    new_etl = numpy.shape(F)[1]
    newblade = numpy.fft.fftshift(numpy.fft.fft2(recovered))
    R = numpy.fft.fftshift(R)
    F = numpy.fft.fftshift(F)
    a = numpy.sum(numpy.abs(newblade[N/2-1:N/2+1,new_etl/2-1:new_etl/2+1])[...])
#     print('a',a)
    b = numpy.sum(numpy.abs(F[N/2-1:N/2+1,new_etl/2-1:new_etl/2+1])[...])
#     print('b',b)
    rate = b/a
    
    newblade = newblade*rate#*(1.0-R) + F#numpy.fft.fftshift(F)
    return newblade
if __name__ == "__main__":

    vardense = [ -32,-29,   -28  , -27  , -25  , -24 ,  -21,   -19,   -15 ,  -13  ,  -8 , 
                  -6 ,   -3,
                 -2 ,   -1   ,  0  ,   1  ,   2   ,  3 ,    4   ,  6 ,    8 ,   11,    15,    17 ,   20,
                 23 ,   24 ,  25  ,  27,    30,    32]

    N = 256# The image will be NxN
#     sparsity = 0.1 # use only 25% on the K-Space data for CS 
    mu = 1.0
    etl = 28

    new_etl,phase_encode = sparse2blade_conf(etl,vardense)
    
#     F,R = sparse2blade_conf(input_k, new_etl,phase_encode)

    LMBD = 1.0
    gamma = mu/1000

    image=phantom.phantom(N)
    

    kspace = numpy.fft.fft2(image)
#     R = numpy.zeros((N,new_etl))
    raw_blade = numpy.zeros((N,etl),dtype=numpy.complex)
    for jj in xrange(0,etl):
        print('phase_encode[jj] = ',jj,phase_encode[jj])
#         R[:,phase_encode[jj]-1]= numpy.ones((N,))
        raw_blade[:, jj ] = kspace[:, phase_encode[jj]-1 ]
     # Form the CS data

    raw_blade = numpy.fft.fftshift(raw_blade,axes = 0)
    
    
    F,R = sparse2blade_init(raw_blade, etl,new_etl, phase_encode) 
    
    recovered = mrics(R,F, mu, LMBD, gamma,5, 10)
    newblade = sparse2blade_final(recovered, F,R)
    print('size_of_newblade',numpy.shape(newblade))
#     newblade = numpy.fft.fftshift(numpy.fft.fft2(recovered))
#     R = numpy.fft.fftshift(R)
#     F = numpy.fft.fftshift(F)
#     a = numpy.sum(numpy.abs(newblade[N/2-1:N/2+1,new_etl/2-1:new_etl/2+1])[...])
#     print('a',a)
#     b = numpy.sum(numpy.abs(F[N/2-1:N/2+1,new_etl/2-1:new_etl/2+1])[...])
#     print('a',a)
#     rate = b/a
#     
#     newblade = newblade*rate*(1.0-R) + F#numpy.fft.fftshift(F)
    
    matplotlib.pyplot.imshow(numpy.abs(R))
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(numpy.abs(F))
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(numpy.abs(newblade ),norm=matplotlib.colors.Normalize(vmin=0.0, vmax=200.0))
    matplotlib.pyplot.show()
    
#     R,F 
#     F = R*F#numpy.fft.fft2(image)

    u0= numpy.fft.ifft2(F)
    
    #matplotlib.pyplot.imshow(numpy.real(R))
    #matplotlib.pyplot.show()
    #Recover the image
#     import cProfile

#     cProfile.run('recovered = mrics(R,F, mu, LMBD, gamma,10, 10)')
    recovered=numpy.real(recovered)
    matplotlib.pyplot.subplot(1,4,1)
    matplotlib.pyplot.imshow(recovered, cmap = matplotlib.cm.gray)
    matplotlib.pyplot.subplot(1,4,2)
    matplotlib.pyplot.imshow(image, cmap = matplotlib.cm.gray)
    matplotlib.pyplot.subplot(1,4,3)
    matplotlib.pyplot.imshow(u0.real, cmap = matplotlib.cm.gray)
    matplotlib.pyplot.subplot(1,4,4)
    matplotlib.pyplot.imshow(R, cmap = matplotlib.cm.gray)
    matplotlib.pyplot.show()
    #[my_kaiser,eigens]=spectrum.mtm.dpss(256,2.5,5)
    #my_kaiser=numpy.sum(my_kaiser[:,0:],1)
    #matplotlib.pyplot.plot(abs(numpy.fft.fft(my_kaiser)))
    #matplotlib.pyplot.show()