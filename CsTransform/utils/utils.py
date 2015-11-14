#!/usr/bin/python
'''
A collection of tools
'''
try:
#     from nufft import *
    import numpy
    import scipy.fftpack
#     import numpy.random
#     import matplotlib.pyplot
#     import matplotlib.cm
    import scipy.linalg.lapack
    import pywt # pyWavelets
    
#     import pygasp.dwt.dwt as gasp
    # import matplotlib.numerix 
    # import matplotlib.numerix.random_array
#     import sys
#     import pywt # pyWavelets
#     import utils
except:
    print('faile to import modules')
    print('numpy, scipy, matplotlib, pyWavelets are required')
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
def extract_svd(input_stack,L):
    C= numpy.copy(input_stack) # temporary array
    print('size of input_stack', numpy.shape(input_stack))
    C=C/numpy.max(numpy.abs(C))

    reps_acs = 16 #16
    mysize = 4 #16
    K= 3 # rank of 10 prevent singular? artifacts(certain disruption)
    half_mysize = mysize/2
    dimension = numpy.ndim(C) -1 # collapse coil dimension
    if dimension == 1:
        tmp_stack = numpy.empty((mysize,),dtype = dtype) 
        svd_size = mysize
        C_size = numpy.shape(C)[0]
        data = numpy.empty((svd_size,L*reps_acs),dtype=dtype)
#             for jj in xrange(0,L):
#                 C[:,jj]=tailor_fftn(C[:,jj])
#                 for kk in xrange(0,reps_acs):
#                     tmp_stack = numpy.reshape(tmp_stack,(svd_size,),order = 'F')
#                     data[:,jj] = numpy.reshape(tmp_stack,(svd_size,),order = 'F') 
    elif dimension == 2:
        tmp_stack = numpy.empty((mysize,mysize,),dtype = dtype) 
        svd_size = mysize**2
        data = numpy.empty((svd_size,L*reps_acs),dtype=dtype)
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
        tmp_stack = numpy.empty((mysize,mysize,mysize),dtype = dtype) 
        svd_size = mysize**3
        data = numpy.empty((svd_size,L),dtype=dtype)
        C_size = numpy.shape(C)[0:3]
        for jj in xrange(0,L):
            C[:,:,:,jj]=tailor_fftn(C[:,:,:,jj])
            tmp_stack= C[C_size[0]/2-half_mysize:C_size[0]/2+half_mysize,
                            C_size[1]/2-half_mysize:C_size[1]/2+half_mysize,
                            C_size[2]/2-half_mysize:C_size[2]/2+half_mysize,
                            jj]
            data[:,jj] = numpy.reshape(tmp_stack,(svd_size,),order = 'F')   
             
#         OK, data is the matrix of size (mysize*n, L) for SVD
#         import scipy.linalg
#         import scipy.sparse.linalg            
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
      
    C2 = numpy.zeros((C.shape[0],C.shape[1],L,K),dtype=dtype)
    for jj in xrange(0,L): # coils  
        for kk in xrange(0,K): # rank
            C2[C.shape[0]/2-reps_acs**0.5/2:C.shape[0]/2+reps_acs**0.5/2,
               C.shape[1]/2-reps_acs**0.5/2:C.shape[1]/2+reps_acs**0.5/2,
               jj,kk]=V_para[:,:,jj,kk]
            C2[:,:,jj,kk]=tailor_fftn(C2[:,:,jj,kk])
#         C_value = numpy.empty_like(C)

    for mm in xrange(0,C.shape[0]): # dim 0  
        for nn in xrange(0,C.shape[1]): # dim 1
            tmp_g=C2[mm,nn,:,:]
#                 G =   C2[mm,nn,:,:].T # Transpose (non-conjugated) of G
# #                 Gh = G.conj().T # hermitian
#                 Gh=C2[mm,nn,:,:].conj()
#                 G=
            
            g = numpy.dot(tmp_g.conj(),tmp_g.T)  #construct g matrix for eigen-decomposition  
#                 w,v = scipy.linalg.eig(g.astype(dtype), overwrite_a=True,
#                                        check_finite=False) # eigen value:w, eigen vector: v

#                 print('L=',L,numpy.shape(g))
#                 w,v = scipy.sparse.linalg.eigs(g , 3)
            w,v = myeig(g.astype(dtype))

            ind = numpy.argmax(numpy.abs(w)) # find the maximum 
#                 print('ind=',ind)
#                 the_eig = numpy.abs(w[ind]) # find the abs of maximal eigen value
            tmp_v = v[:,ind]#*the_eig 
#                 ref_angle=(numpy.sum(v[:,ind])/(numpy.abs(numpy.sum(v[:,ind]))))
#                 v[:,ind] = v[:,ind]/ref_angle # correct phase by summed value

            ref_angle=numpy.sum(tmp_v)
            ref_angle=ref_angle/numpy.abs(ref_angle)
            
            C[mm,nn,:] = tmp_v/ref_angle # correct phase by summed value
    C=C/numpy.max(numpy.abs(C))
#         matplotlib.pyplot.figure(1)         
#         for jj in xrange(0,L):
#             matplotlib.pyplot.subplot(2,4,jj+1)  
#             matplotlib.pyplot.imshow(abs(input_stack[...,jj]),
#                                      norm=matplotlib.colors.Normalize(vmin=0.0, vmax=0.2),
#                                      cmap=matplotlib.cm.gray)
#             matplotlib.pyplot.subplot(2,4,jj+1+4)  
#             matplotlib.pyplot.imshow(abs(C[...,jj]),
#                                      norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
#                                      cmap=matplotlib.cm.gray)
#                                   
# #             matplotlib.pyplot.subplot(2,8,jj+1+8)          
# #             matplotlib.pyplot.imshow(numpy.log(C[...,jj]).imag, cmap=matplotlib.cm.gray)
#                          
#         matplotlib.pyplot.show()  

#         for jj in xrange(0,L):   
#             matplotlib.pyplot.subplot(2,2,jj+1)  
#             matplotlib.pyplot.imshow((input_stack[...,jj].real))
#         matplotlib.pyplot.show() 
     
    return C # normalize the coil sensitivities 
def myeig(g):
    '''
    access lapack's cggev which is presumably faster 
     than scipy's eig() 
    '''
    w,vl ,v, info= scipy.linalg.lapack.zgeev(g ,overwrite_a=1)
    return w,v

def low_pass_phase(input_image):
    img_shape = numpy.shape(input_image) # 2D
#     print(img_shape)
    filter_k = numpy.zeros(img_shape)
    for q in xrange(0,img_shape[0]):
        for w in xrange(0,img_shape[1]):
            D2 = (q - img_shape[0]/2.0)**2 +(w - img_shape[1]/2.0)**2
            filter_k[q,w] = numpy.exp(-D2/(2.0*6**2))
            
    kimg = scipy.fftpack.fftshift(scipy.fftpack.fft2(scipy.fftpack.fftshift(input_image)))
    low_pass_img =  scipy.fftpack.fftshift(scipy.fftpack.ifft2(scipy.fftpack.fftshift(kimg*filter_k)))
#     low_pass_img = low_pass_img/numpy.max(numpy.abs(low_pass_img))
#     intensity_img = numpy.abs(low_pass_img)
    low_pass_phase =low_pass_img/numpy.abs(low_pass_img)
    
#     for q in xrange(0,img_shape[0]):
#         for w in xrange(0,img_shape[1]):
# 
#             if (q-img_shape[0]/2)**2 +(w-img_shape[1]/2)**2 > 1.10*( img_shape[1]/2 )**2:
#                 low_pass_phase[q,w] = 1.0  
#     
    output_image = input_image/low_pass_phase
#     intensity_img = numpy.abs(low_pass_img)
#     intensity_img = (intensity_img+numpy.mean(numpy.abs(intensity_img))*0.1)/(intensity_img**2 + numpy.mean(numpy.abs(intensity_img))*0.1)
#     intensity_img  =intensity_img/numpy.max(numpy.abs(intensity_img))
#     output_image = output_image*intensity_img
#     matplotlib.pyplot.imshow(numpy.real(output_image))
#     matplotlib.pyplot.show()
        
    return output_image
def tailor_fftn(X):
    X = scipy.fftpack.fftshift(scipy.fftpack.fftn(scipy.fftpack.fftshift((X))))
    return X
def tailor_ifftn(X):
    X = scipy.fftpack.fftshift(scipy.fftpack.ifftn(scipy.fftpack.ifftshift(X)))
    return X
def output(cc):
    print('max',numpy.max(numpy.abs(cc[:])))
     
def Normalize(D):
    import numpy as np
    (Nx, Ny) = numpy.shape(D)
    for xx in xrange(0, Nx):
        for yy in xrange(0, Ny):
            if ((xx - Nx/2)**2 + (yy - Ny/2)**2 ) >= (  Nx/2)**2  : 
                D[xx, yy] =  D[xx, yy] *0.0
#     hist,bins = np.histogram(D.flatten(),256,[0,256])
# 
#     cdf = hist.cumsum()
#     cdf_normalized = cdf * hist.max()/ cdf.max()
#     cdf_m = np.ma.masked_equal(cdf,0)
#     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
#     cdf = np.ma.filled(cdf_m,0).astype('uint8')
#     D2 = numpy.sort(D.ravel()).searchsorted(D)
    return D/numpy.max(numpy.real(D[:]))
def Normalize3(im):
#     def histeq(im,nbr_bins=256):
    nbr_bins = 256
    from pylab import *
    # get image histogram
    from PIL import Image
    im = im*256
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape)#, cdf
def Normalize2(D,R):
    mean1 = numpy.mean(numpy.real(D))
    mean2 = numpy.mean(numpy.real(R))
    max1=numpy.max(numpy.real(D))
    max2=numpy.max(numpy.real(R))
    
    
    D = (D - mean1 +mean2)*(max2-mean2)/(max1-mean1)
    
    
    
    return D
def checkmax(x,dbg):
    max_val = numpy.max(numpy.abs(x[:]))
    if dbg ==0:
        pass
    else:
        print( max_val)
    return max_val
def appendmat(input_array,L):
    '''
    repeat the input_array L times 
    '''
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
def get_wavelet(image):
    '''
    wavelet transform of the input image
    '''
    if len(numpy.shape( image)) == 2:
        image = numpy.reshape(image,numpy.shape(image)+(1,),order='F')
    else:
        pass
    
    L = numpy.shape( image)[-1]
#     print('L=',L)
    
    wave_image = numpy.empty_like( image)
    for jj in xrange(0, L):
#     import pywt#,numpy,matplotlib.pyplot    
        coeffs = pywt.wavedec2(image[:,:,jj], 'db1',level = 3)
#         coeffs = gasp.dwt2(image[:,:,jj], 'haar',level = 3)
    #     new_coeffs =  (coeffs[0]*0.0,)+coeffs[1:]
        cA, cH1, cH2, cH3 = coeffs
        
#         print('shape of cA',cA.shape)
#         print('shape of cH1',numpy.shape(cH1))
    
    #     new_image = pywt.idwt2(coeffs,'haar')
        
        wave_image0 = numpy.concatenate((cA, cH1[0]),axis = 0)
        wave_image1 = numpy.concatenate((cH1[1],cH1[2]),axis = 0)
        tmp_image =  numpy.concatenate((wave_image0, wave_image1),axis = 1)
        wave_image0 = numpy.concatenate((tmp_image, cH2[0]),axis = 0)
        wave_image1 = numpy.concatenate((cH2[1],cH2[2]),axis = 0)    
        tmp_image =  numpy.concatenate((wave_image0, wave_image1),axis = 1)
        wave_image0 = numpy.concatenate((tmp_image, cH3[0]),axis = 0)
        wave_image1 = numpy.concatenate((cH3[1],cH3[2]),axis = 0)    
        wave_image[:,:,jj] =  numpy.concatenate((wave_image0, wave_image1),axis = 1)   
#         matplotlib.pyplot.imshow(wave_image[:,:,:])
#         matplotlib.pyplot.show
    return wave_image  
def get_wavelet_H(wave_image):
    '''
    inverse wavelet transform of the wavelets image
    '''    
    if len(numpy.shape(wave_image)) == 2:
        wave_image = numpy.reshape(wave_image,numpy.shape(wave_image)+(1,),order='F')
    else:
        pass
    
    L = numpy.shape(wave_image)[-1]
    
    new_image = numpy.empty_like(wave_image,dtype=dtype)
    
    for jj in xrange(0, L):
        N = numpy.shape(wave_image[:,:,jj])[0]
        tmp_image = numpy.abs(wave_image[:,:,jj])
    #     new_image = numpy.empty_like(wave_image) 
        p0=  tmp_image[N/2:, 0:N/2]
        p1 =  tmp_image[0:N/2, N/2:]
        p2 =  tmp_image[N/2:, N/2:]
        cA = tmp_image[0:N/2, 0:N/2]
        cH3 = (p0,p1,p2)
    
        
        
        N = numpy.shape(cA)[0]
        p0=  cA[N/2:, 0:N/2]
        p1 =  cA[0:N/2, N/2:]
        p2 =  cA[N/2:, N/2:]
        cH2 = (p0,p1,p2)    
        cA = cA[0:N/2, 0:N/2]
        
        N = numpy.shape(cA)[0]
        p0=  cA[N/2:, 0:N/2]
        p1 =  cA[0:N/2, N/2:]
        p2 =  cA[N/2:, N/2:]
        cH1 = (p0,p1,p2)    
        cA = cA[0:N/2, 0:N/2]
          
        new_image[:,:,jj] =  pywt.waverec2((cA,cH1, cH2, cH3),'haar')
#         new_image[:,:,jj] =  gasp.idwt2([cA,[cH1, cH2, cH3]],'haar')
##############
             
    return new_image 
# def freq_gradient(x):# zero frequency at centre
#     
#     grad_x = numpy.copy(x)
#      
#     dim_x=numpy.shape(x)
# #     print('freq_gradient shape',dim_x)
#     for pp in xrange(0,dim_x[2]):
#         grad_x[...,pp,:]=grad_x[...,pp,:] * (-2.0*numpy.pi*(pp -dim_x[2]/2.0 )) / dim_x[2]
#  
#     return grad_x
# def freq_gradient_H(x):
#     return -freq_gradient(x)
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
def Wconstraint(xx,bb):
 
#     try:    
#         n_xx = len(xx)
#     print('inside Wconstraint',numpy.shape(xx),numpy.shape(bb))
    cons = get_wavelet_H( numpy.real(xx - bb) ) + 1.0j* get_wavelet_H( numpy.imag(xx - bb) )
#     except:
#         n_xx = len(xx)
#         n_bb =  len(bb)
#         if n_xx != n_bb: 
#             print('xx and bb size wrong!')    
 
    return cons  
def TVconstraint(xx,bb):
 
    try:    
        n_xx = len(xx)
#         n_bb =  len(bb)
#         cons_shape = numpy.shape(xx[0])
#         cons=numpy.zeros(cons_shape,dtype=dtype)
         
    #     cons = sum( get_Diff_H( xx[jj] - bb[jj] ,  jj) 
    #                 for jj in xrange(0,n_xx))
         
#         for jj in xrange(0,n_xx):
#             cons =  cons + get_Diff_H( xx[jj] - bb[jj] ,  jj)
        cons = sum(get_Diff_H( xx[jj] - bb[jj] ,  jj) for jj in xrange(0,n_xx))
#         import matplotlib.pyplot
#         matplotlib.pyplot.figure(5)
# 
#         matplotlib.pyplot.subplot(1,2,1)
#         matplotlib.pyplot.imshow(numpy.abs( get_Diff_H( xx[0] - bb[0] ,  0)), cmap=matplotlib.cm.gray)
#         matplotlib.pyplot.subplot(1,2,2)
#         matplotlib.pyplot.imshow(numpy.abs( get_Diff_H( xx[0] - bb[0] ,  1)), cmap=matplotlib.cm.gray)        
#         matplotlib.pyplot.show()
        
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
  
#        dcy=numpy.empty(numpy.shape(y),dtype=dtype)
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
    '''
    combination of multiple coils
    '''
    
    U=numpy.mean(multi_coil_data,axs)
    U = appendmat(U,multi_coil_data.shape[axs])
 
    return U
 
# def CopySingle2Multi(single_coil_data,n_tail):
#     
#     U=numpy.copy(single_coil_data)
#      
#     U = appendmat(U, n_tail)
#      
#     return U
 