import numpy
import scipy.fftpack
import scipy.linalg
# import scipy.interpolate
import matplotlib.pyplot
# import scipy.signal
import scipy.misc

import matplotlib.cm
import scipy.ndimage.fourier
# import scipy.ndimage
cmap=matplotlib.cm.gray
norm=matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

# from numpy.linalg import lapack_lite
# lapack_routine = lapack_lite.dgesv
# import pyximport
# pyximport.install()
# import Im2Polar
# import AverageHermitian
def complex_imrotate(input_x,angle_x):
#     input_x = input_x+0.0j
    output_x = scipy.misc.imrotate(input_x.real, angle_x,'bicubic') + 1.0j*scipy.misc.imrotate(input_x.imag, angle_x,'bicubic')
   
#     del input_x
    
    return output_x


def make_pyramid(kspace):
    '''
    making 2D pyramid function for filtering, 
    which is used for zero-first order phase correction
    '''
    kshape = numpy.shape(kspace)
    if len(kshape) !=2:
        print('kspace other than 2D is not supported')
        raise()    
    dim_1= kshape[0]
    dim_2= kshape[1]
    filter_1 = numpy.arange(0,dim_1)
    filter_1 = numpy.reshape(filter_1,(dim_1,1))
    
    filter_1 = numpy.abs(filter_1 - (dim_1-1.0)/2.0)
    
    
    filter_1 = filter_1/numpy.max(filter_1)
#     filter_1 = filter_1/4.0
    filter_1 = numpy.clip(filter_1,0,1)
#     matplotlib.pyplot.plot(filter_1)
#     matplotlib.pyplot.show()
    
    filter_1 = 1.0 - filter_1 
    filter_1 = numpy.tile(filter_1,(1,dim_2))
    
    filter_2 = numpy.arange(0,dim_2)
    filter_2 = numpy.reshape(filter_2,(1, dim_2))
    filter_2 = numpy.abs(filter_2 - (dim_2-1.0)/2.0)
    filter_2 = filter_2/numpy.max(filter_2)
#     filter_2 = filter_2/4.0
    filter_2 = numpy.clip(filter_2,0,1)
    filter_2 = 1.0 - filter_2 
    filter_2 = numpy.tile(filter_2,(dim_1,1))
    pyramid_filter = filter_1*filter_2   
             
    return pyramid_filter
def fft2_extrapolate(kspace):
    '''
    2D FFT with 2 times of extrapolation
    '''
    
    kshape = numpy.shape(kspace)
    if len(kshape) !=2:
        print('kspace other than 2D is not supported')
        raise()     
    ispace=kspace
#     extra_shape = (kshape[0]*2, kshape[1]*2,)
#     ispace = numpy.zeros(extra_shape)*0.0j
#     ispace[kshape[0]/2: 3*kshape[0]/2 , kshape[1]/2: 3*kshape[1]/2 ] =kspace
    
    ispace = scipy.fftpack.fftshift(ispace)
    
    ispace = scipy.fftpack.fft2(ispace, overwrite_x=1)
    
    ispace = scipy.fftpack.fftshift(ispace)
        
    return ispace # return image space
def ifft_truncate(ispace):
    ishape = numpy.shape(ispace)
    if len(ishape) !=2:
        print('image space other than 2D is not supported')
        raise()   
    ispace = scipy.fftpack.ifftshift(ispace)        
    kspace = scipy.fftpack.ifft2(ispace, overwrite_x=1)
    kspace = scipy.fftpack.ifftshift(kspace)      
    return kspace#kspace[ishape[0]/4: 3*ishape[0]/4 , ishape[1]/4: 3*ishape[1]/4 ]
def corr_kspace_shift(input_k):
    '''
    correction for k-space shifts due to gradient mismatch or eddy
    currents
    by image space correction
    '''
    
#     output_k = input_k
    
    pyramid_filter = make_pyramid(input_k)

    tmp_k = input_k * pyramid_filter
    reference_image = fft2_extrapolate(tmp_k )
    
    phase_image = reference_image/numpy.abs(reference_image)
    # phase image used for correction of phases
    
    image_space = fft2_extrapolate(input_k)
    image_space = image_space/phase_image
    output_k = ifft_truncate(image_space)

#     import matplotlib.pyplot as mp
#     mp.plot(filter_1)
#     mp.show()
    return output_k
# def corr_kspace_zero_phase(kimg):
#     img_shape = numpy.shape(kimg) # 2D
# #     print(img_shape)
#     filter_k = numpy.zeros(img_shape)
#     for q in xrange(0,img_shape[0]):
#         for w in xrange(0,img_shape[1]):
#             D2 = (q - img_shape[0]/2.0)**2 +(w - img_shape[1]/2.0)**2
#             filter_k[q,w] = numpy.exp(-D2/(2.0*0.5**2))
#               
# #     kimg = scipy.fftpack.fftshift(scipy.fftpack.fft2(scipy.fftpack.fftshift(input_image)))
# #     low_pass_img =  scipy.fftpack.fftshift(scipy.fftpack.ifft2(scipy.fftpack.fftshift(kimg*filter_k)))
#     low_pass_img = kimg*filter_k
#     cnt1 = numpy.sum(low_pass_img)
#     if cnt1 == 0.0:
#         cnt1 = 1.0 # if 
#            
#     cnt1=cnt1/numpy.abs(cnt1)#+1e-12)
#     output_k=kimg/(cnt1)#+1e-12)
#          
# #     low_pass_img =low_pass_img/numpy.abs(low_pass_img)
# #     input_image = scipy.fftpack.fftshift(scipy.fftpack.ifft2(scipy.fftpack.fftshift(kimg)))
# #     output_image = input_image/low_pass_img
# #     output_k = scipy.fftpack.fftshift(scipy.fftpack.fft2(scipy.fftpack.fftshift(output_image)))
# #     matplotlib.pyplot.imshow(numpy.real(output_image))
# #     matplotlib.pyplot.show()
#           
#     return output_k
def corr_kspace_zero_phase(input_k):
    '''
    correction of the zero order phase of k-space
    by dividing the angle of the center
    '''
    radius = 1
    cnt1=numpy.mean(input_k[input_k.shape[0]/2-radius:input_k.shape[0]/2+radius , # not including the last number
              input_k.shape[1]/2-radius:input_k.shape[1]/2+radius]) # not including the last number
        
#     print('cnt1',cnt1, cnt1 == 0.0)
    if cnt1 == 0.0:
        cnt1 = 1.0 # if 
            
    cnt1=cnt1/numpy.abs(cnt1)#+1e-12)
    output_k=input_k/(cnt1)#+1e-12)
#     output_k=AverageHermitian.AverageHermitian(output_k)
#     upper_half = output_k[: , 0:input_k.shape[1]/2]
#     lower_half = output_k[: , input_k.shape[1]/2: ]
# 
#     A = ( numpy.real(upper_half[:,:]) + numpy.real(lower_half[::-1,  ::-1]) ) /2
#     B = ( numpy.imag(upper_half[:,:]) - numpy.imag(lower_half[::-1,  ::-1]) ) /2
#     output_k[: , :input_k.shape[1]/2] = A+1j*B
#     
#     output_k[: , input_k.shape[1]/2: ] = A[::-1,::-1]-1j*B[::-1,::-1]
        
    return output_k
# def _interpolate(imR, xR,yR):
#     '''
#     2D bilinear interpolation
#     auxillary function of Im2Polar of corr_rotation
#     '''    
#     xf = numpy.floor(xR)
#     xc = numpy.ceil(xR)
#     yf = numpy.floor(yR)
#     yc = numpy.ceil(yR)   
#      
#     data_grid = numpy.array([[imR[xf, yf], imR[xf, yc]],
#                              [imR[xc, yf], imR[xc, yc]]])
#      
#     if (xf < xc):
#         C = (data_grid[0,:]*(xc-xR)   +  data_grid[1,:]*(xR-xf) )/(xc -xf) 
#          
#     else:   #xc == xf
#         C = data_grid[0,:]
#          
#          
#     if (yf < yc):
#         C = (data_grid[0,0]*(yc-yR)   +  data_grid[0,1]*(yR-yf) )/(yc -yf) 
#     else:   #xc == xf
#         C = data_grid[0,0]                
#          
#      
# #     print(xR, ' ',yR)
#     #===========================================================================
#     ## slow but accurate version
# 
#     # 
#     #===========================================================================
#     # if (xf == xc) and (yc == yf):
#     #     v = imR[xc,yc]
#     # elif xf == xc:
#     #     v = imR[xf,yf]+(yR-yf)*(imR[xf,yc] - imR[xf,yf])
#     # elif yf == yc:
#     #     v = imR[xf,yf]+(xR -xf)*(imR[xc,yf]- imR[xf,yf])
#     # else:
#     #     A = numpy.array([[xf, yf, xf*yf, 1],
#     #                      [xf, yc, xf*yc, 1],
#     #                      [xc, yf, xc*yf, 1],
#     #                      [xc, yc, xc*yc, 1]]
#     #                     )
#     #     r = numpy.array([imR[xf,yf], imR[xf,yc], imR[xc,yf], imR[xc,yc]]).T      
#     #     a,tmp1,tmp2,tmp3 = scipy.linalg.lstsq(A,r)
#     #     w = numpy.array([xR,yR,xR*yR, 1])
#     #     v = numpy.dot(w,a)
#     #===========================================================================
#     #===========================================================================
#         
#         
#         
#     return C

# def Im2Polar2(imR,rMin, rMax, M, N):
#     return Im2Polar.Im2Polar(imR,rMin, rMax, M, N)
# def Im2Polar(imR, rMin, rMax, M, N):
#     '''
#     convert rectangular to polar image
#     auxillary function of corr_rotation
#     '''
#     shape_imR = numpy.shape(imR)
#     Mr = shape_imR[0]
#     Nr = shape_imR[1]
# #     print('Mr, Nr',Mr, Nr)
#     Om = (Mr - 1)/2.0  # image center 
#     On = (Nr - 1)/2.0 
# #     print('Om, On',Om, On)
#      
#     sx = (Mr - 1)/2.0 # scaling factors
#     sy = (Nr - 1)/2.0
#  
# #     print('sx, sy',sx, sy)
#  
#     imP = numpy.empty((M,N),dtype=numpy.complex64) # define 2D array of Polar coordinate
#      
#     delR = (rMax-rMin)/1.0/(M-1)
#     delT = 2.0*numpy.pi/N
# #             print('ri=', ri,', ti=' ,ti)
#     ########################
# #     ri_0 = xrange(0,M)
# #     ti_0 = xrange(0,N)
# #     
# #     ri,ti =numpy.meshgrid(ri_0,ti_0)
# #     ri = numpy.reshape(ri,(M*N,1))
# #     ti = numpy.reshape(ti,(M*N,1))
# # 
# #     r = rMin + (ri)*delR # (M*N , 1)
# #     t = ti *delT# (M*N , 1)
# # #             print('ri*delT',delR*(ri+1))
# #     x = r*numpy.cos(t + numpy.pi)
# #     y = -r * numpy.sin(t + numpy.pi)
# #     
# #     xR = x*sx + Om; # (M*N , 1)
# #     yR = y*sy + On; # (M*N , 1)
# #      
# #     xf_0 = numpy.floor(xR)
# #     xc_0 = numpy.ceil(xR)
# #     yf_0 = numpy.floor(yR)
# #     yc_0 = numpy.ceil(yR)   
#     ######################   
# #     print('delR, delT', delR, delT)
# #     tmp_imR = numpy.reshape()
# 
#     imR =numpy.real(imR)+(0.0+0.0j) # convert to complex
# #     for jjj in xrange(0,M*N):
# #         xf = xf_0[jjj].astype(int)
# #         xc = xc_0[jjj].astype(int)
# #         yf = yf_0[jjj].astype(int)
# #         yc = yc_0[jjj].astype(int)
# #         data_grid = numpy.array([[imR[xf, yf], imR[xf, yc]],
# #                                       [imR[xc, yf], imR[xc, yc]]])
# #         data_grid2 = (data_grid[0,:]*(xc-xR[jjj]+1e-7)   +  data_grid[1,:]*(xR[jjj]-xf+1e-7) )/(xc -xf+2e-7)
# #         imP[ri, ti] = (data_grid2[0]*(yc-yR[jjj]+1e-7)   +  data_grid2[1]*(yR[jjj]-yf+1e-7) )/(yc -yf+2e-7)        
# #===============================================================================
#     for ri in  xrange(0,M):
#         for ti in  xrange(0,N):
# #             print('ri=', ri,', ti=' ,ti)
#             r = rMin + (ri)*delR
#             t = ti *delT
# #             print('ri*delT',delR*(ri+1))
#             x = r*numpy.cos(t + numpy.pi)
#             y = -r * numpy.sin(t + numpy.pi)
#             xR = x*sx + Om;
#             yR = y*sy + On;
#               
#             xf = numpy.floor(xR)
#             xc = numpy.ceil(xR)
#             yf = numpy.floor(yR)
#             yc = numpy.ceil(yR)   
#                
#             data_grid = numpy.array([[imR[xf, yf], imR[xf, yc]],
#                                      [imR[xc, yf], imR[xc, yc]]])
#             data_grid2 = (data_grid[0,:]*(xc-xR+1e-7)   +  data_grid[1,:]*(xR-xf+1e-7) )/(xc -xf+2e-7)
#             imP[ri, ti] = (data_grid2[0]*(yc-yR+1e-7)   +  data_grid2[1]*(yR-yf+1e-7) )/(yc -yf+2e-7)
# #===============================================================================
#              
# 
#     return imP
             
#===============================================================================
# #             if (xf < xc):
# #                 C = (data_grid[0,:]*(xc-xR)   +  data_grid[1,:]*(xR-xf) )/(xc -xf) 
# #                  
# #             else:   #xc == xf
# #                 C = data_grid[0,:]
# #                  
# #                  
# #             if (yf < yc):
# #                 C = (data_grid[0,0]*(yc-yR)   +  data_grid[0,1]*(yR-yf) )/(yc -yf) 
# #             else:   #xc == xf
# #                 C = data_grid[0,0]    
# #                 
#                  
# #             imP[ri, ti] = C#interpolate(imR.real,xR, yR)+1j*interpolate(imR.imag,xR, yR) 
#===============================================================================

def mycorrelate2d(k1,k2):
    k1 = scipy.fftpack.fftn(k1,axes=(0,1,))
    k2 = scipy.fftpack.fftn(k2,axes=(0,1,))
    result = k1*k2.conj()
    result =  scipy.fftpack.ifftn(result,axes=(1,))
#     result = numpy.real
    return numpy.abs(result)
def radial_correlate2d(radial_kspace):
    
    radial_kspace = scipy.fftpack.fftn(radial_kspace, axes=(0,1,), overwrite_x = 1)
    result = radial_kspace[:,:,:-1]*radial_kspace[:,:,1:].conj()
#     result = numpy.copy(radial_kspace[:,:,1:])
#     for jj in xrange(0,result.shape[2]):
#         result[:,:,jj]=result[:,:,jj]*radial_kspace[:,:,0].conj()
        
    result =  scipy.fftpack.ifftn(result,axes=(0,1,))
#     result = numpy.real
    return result

def find_max_from_3points(input_x):

#     search_center = numpy.size(input_x)/blade
# 
#     search_region = range(search_center/2,3*search_center/2)
    
    

    
    vars = numpy.empty(5)
    inds = numpy.empty(5)
    tmp_x=numpy.copy(input_x)
    for jj in range(0,5):
        tmp_ind =numpy.argmax(tmp_x.real)
        inds[jj]=tmp_ind
        vars[jj]=tmp_x[inds[jj]]
        tmp_x[inds[jj]] = 0
#     for jj in range(0,N):
#         input_x[inds[jj].astype(int)]=vars[jj]

    A=[
      [inds[0]**2,   inds[0],  1 ],
      [inds[1]**2   ,inds[1],  1 ],
      [inds[2]**2,   inds[2],  1 ],
    [inds[3]**2,   inds[3],  1 ],
    [inds[4]**2,   inds[4],  1 ],
       ]
    z = numpy.array([numpy.abs(vars[0]),
                    numpy.abs(vars[1]),
                    numpy.abs(vars[2]),
                    numpy.abs(vars[3]),
                    numpy.abs(vars[4]),
                    ]).T

    a,tmp_a,tmp_b,tmp_c= scipy.linalg.lstsq(A,z)
    averaged_max = a[1]/(-2.0*a[0])
   

        
#     averaged_max = numpy.sum(vars*inds)/numpy.sum(vars)
    
#     averaged_max = averaged_max + search_region[0]
    
    return averaged_max
# def fit_phase_plane(input_plane):
#     delta_x = 0.0
#     delta_y = 0.0 
#     c = 0.0 # constant
#     shape_input_plane = numpy.shape(input_plane)
#     if len(shape_input_plane)!=2:
#         print('linear fitting phase plane is only valid for 2D')
 
     
     
 
# def find_phase_plane(input_plane):
#     '''
#     find the linear phase of a plane
#     '''
#     input_plane = scipy.fftpack.fft2(input_plane)
# #     input_plane[7,14] = 1e+9
#     input_plane = scipy.fftpack.ifft2(input_plane)
#     angle_plane = numpy.angle(input_plane)
#       
#     KX=numpy.array([[0, 1],
#                    [1 ,1],
#                    [2, 1],
#                    [3, 1],
#                    [4,1],
#                    [5,1],
#                    [6,1],
#                    [7,1]])
#     z = angle_plane [input_plane.shape[0]/2-4:input_plane.shape[0]/2+4 ,
#                      input_plane.shape[1]/2]
#     z= numpy.unwrap(z)
#     c,resid,rank,sigma= scipy.linalg.lstsq(KX,z)
#     delta_x = c[0]
#       
#       
#     z=angle_plane[input_plane.shape[0]/2,
#                      input_plane.shape[1]/2-4:input_plane.shape[1]/2+4]
#     z= numpy.unwrap(z)    
#     c,resid,rank,sigma= scipy.linalg.lstsq(KX,z)
#     delta_y = c[0]   
#       
#     print(delta_x, delta_y)
#       
#     return delta_x, delta_y   
def corr_phase_plane(input_plane, delta_x,delta_y):
    '''
    correct the linear phase in kspace according to delta_x and delta_y
    '''

    input_shape = numpy.shape(input_plane)
    output_x, output_y = numpy.meshgrid(range(0,input_shape[0]),range(0,input_shape[1]))
    output_x=output_x.T
    output_y=output_y.T
    output_x =output_x -  (input_shape[0]-1)/2.0
    output_y =output_y -  (input_shape[1]-1)/2.0
    unitary_phase_plane=  numpy.exp( 1j * delta_x* output_x +
                                     1j * delta_y* output_y)
    out_plane =input_plane/unitary_phase_plane
    
    return out_plane

def find_phase_plane(input_plane):
    tmp_plane = scipy.fftpack.fft2(input_plane)
      
#     matplotlib.pyplot.imshow(numpy.abs(tmp_plane))
#     matplotlib.pyplot.show()
    ind_max =numpy.argmax(numpy.abs(tmp_plane))
#     print( ind_max)
    delta_y = numpy.mod( ind_max, input_plane.shape[0])
#     print('delta_y = ',delta_y)
      
    delta_x =  (ind_max -delta_y )/input_plane.shape[0]
      
    # fit x
    A=[
       [(delta_x - 1)**2,   (delta_x-1),  1 ],
       [(delta_x )**2   ,delta_x,  1 ],
       [(delta_x + 1)**2,   (delta_x+1),  1 ] ]
    z = numpy.array([numpy.abs(tmp_plane[delta_x -1,delta_y]),
                    numpy.abs(tmp_plane[delta_x,delta_y]),
                    numpy.abs(tmp_plane[delta_x +1,delta_y])]).T
      
                      
      
    a,resid,rank,sigma = scipy.linalg.lstsq(A,z)
    delta_x = 2.0*numpy.pi*a[1]/(-2.0*a[0]*input_plane.shape[0])
      
    A=[
      [(delta_y - 1)**2,   (delta_y-1),  1 ],
      [(delta_y )**2   ,delta_y,  1 ],
      [(delta_y + 1)**2,   (delta_y+1),  1 ] ]
    z = numpy.array([numpy.abs(tmp_plane[delta_x,delta_y-1]),
                    numpy.abs(tmp_plane[delta_x,delta_y]),
                    numpy.abs(tmp_plane[delta_x,delta_y+1])]).T
       
    a,resid,rank,sigma = scipy.linalg.lstsq(A,z)
    delta_y = 2.0*numpy.pi*a[1]/(-2.0*a[0]*input_plane.shape[1])
      
#     print('delta_x, delta_y', delta_x, delta_y)
#     delta_x = 0.0
#     delta_y = 0.0
    return delta_x, delta_y
def find_rotation(input_k, input_theta, range_theta = None, interp_num = 12): # xres, etl, blade, coils
    
    if range_theta == None:
        range_theta = 0.0
    
    temp_shape = numpy.shape(input_k)
    nblades = temp_shape[2]
    # magnitude of images
    xres = temp_shape[0]
    etl = temp_shape[1]
    
    # copy the array, preventing any change to the original array
    temp_image = numpy.copy(input_k)
    
#     Magnitude images

    temp_image = scipy.fftpack.fftshift(temp_image,axes = (0,1))
    temp_image = scipy.fftpack.ifft2(temp_image,axes = (0,1))
    temp_image = scipy.fftpack.fftshift(temp_image,axes = (0,1))
    
#    Combine multi coils    

    temp_image = numpy.sum(numpy.abs(temp_image*temp_image.conj()),3) # only consider single coil images

#     temp_image =  (numpy.abs(temp_image*temp_image.conj())) # only consider single coil images     
    
    temp_image = scipy.fftpack.fftshift(temp_image,axes = (0,1))
    temp_image = scipy.fftpack.fft2(temp_image,axes = (0,1))
    temp_image = scipy.fftpack.fftshift(temp_image,axes = (0,1)) # now in k-space, coils combined
    
# crop the center of k-space
    
    new_size = etl
    
    temp_k = numpy.zeros((new_size ,new_size ,temp_shape[2]),dtype = numpy.complex64)
 
# crop the center of each strip

    temp_k[ new_size/2 - etl/2 :new_size/2 + etl/2 , 
           new_size/2 - etl/2 :new_size/2 + etl/2  , 
           :] =  temp_image[  xres/2 - etl/2 :xres/2 + etl/2 ,:, : ] 

# FFT2 and increase the size 
    temp_k = scipy.fftpack.fftshift(temp_k,axes = (0,1))
    temp_k = scipy.fftpack.fft2(temp_k,axes = (0,1))
    temp_k = scipy.fftpack.fftshift(temp_k,axes = (0,1)) # now in k-space, coils combined           
           
    temp_image2 = numpy.zeros((etl*2,   etl*2,  nblades),dtype = numpy.complex64)
    temp_image2[etl/2:etl*3/2,  etl/2:etl*3/2  ,:  ]= temp_k 
    temp_k=temp_image2
    new_size = etl*2
    temp_k = scipy.fftpack.fftshift(temp_k,axes = (0,1))
    temp_k = scipy.fftpack.ifft2(temp_k,axes = (0,1))
    temp_k = scipy.fftpack.fftshift(temp_k,axes = (0,1)) # now in k-space, coils combined  
    temp_k = numpy.abs(temp_k).astype(numpy.float32) # only consider magnitudes
       
# zeroing the ref outside the circle, multiply the circle with the square of radii

    for nx in xrange(0,new_size):
        for ny in xrange(0,new_size):
            radii_square = ( (nx-new_size/2)**2 + (ny-new_size/2)**2) 
            if radii_square > (new_size/2)**2:
                temp_k[nx,ny,:]= temp_k[nx,ny,:]*0.0
            else:
                temp_k[nx,ny,:]= temp_k[nx,ny,:]*radii_square
    temp_k = temp_k*1.0/numpy.max(temp_k)
# define the referenced k-space, from the "averaged" and rotated k-space magnitudes

    MA = numpy.zeros( (new_size,new_size),dtype = numpy.float32)  
    
    for n_b in xrange(0,nblades):
        MA = MA + scipy.misc.imrotate(temp_k[:,:,n_b],-1.0*n_b*input_theta,'bicubic')*1.0/nblades
#     matplotlib.pyplot.imshow(MA)
#     matplotlib.pyplot.show()

# Search the angles for each blades
#     Mn = numpy.zeros( (new_size,new_size),dtype = numpy.float32)  
    
    corr_MA_Mn = numpy.zeros((interp_num, ),dtype=numpy.float32)
    angle_MA_Mn = numpy.zeros((interp_num, ),dtype=numpy.float32)
    result_angle = numpy.zeros((nblades, ),dtype=numpy.float32)  
    for n_b in xrange(0,nblades):
        for run_delta in xrange(0,interp_num):
            
            deviate_theta = - range_theta/2.0 + range_theta*(1.0/3)*run_delta
            angle_MA_Mn[run_delta] = deviate_theta - 1.0*n_b*input_theta
            Mn = scipy.misc.imrotate(temp_k[:,:,n_b], angle_MA_Mn[run_delta],'bicubic').astype(numpy.float32)
            
#             print(numpy.max(MA*Mn),numpy.max(Mn))
            corr_MA_Mn[run_delta ] = numpy.mean(  MA*Mn )
            
#         z = numpy.polyfit(angle_MA_Mn, corr_MA_Mn, 2) # degree = 2
#         p = numpy.poly1d(z)
#         ind = numpy.argmax( p(angle_MA_Mn))
        ind = numpy.argmax(corr_MA_Mn)
#         print(ind,angle_MA_Mn[ind],corr_MA_Mn[ind] )
        
        result_angle[n_b] = -angle_MA_Mn[ind]
        
#         matplotlib.pyplot.plot(angle_MA_Mn,corr_MA_Mn)
# #         matplotlib.pyplot.plot(angle_MA_Mn,p(angle_MA_Mn))
#         matplotlib.pyplot.show()
    x_input_theta_list=  numpy.arange(0, nblades) * input_theta  
#     result_angle= numpy.mod( result_angle, 360.0) 
#     print('result_angle,x',  result_angle )
    z2 = numpy.polyfit(x_input_theta_list, result_angle, 2) # degree = 2
#     p2 = numpy.poly1d(z2)
#     result_angle_smooth = result_angle
    print('z2 raw',z2)
    if nblades * input_theta > 180:
        pass
    else:
        z2 = z2 - [ 0 , -0.0030960 , 0.28811] # refinement of the angle estimation
    print('z2 caliberated',z2)    
#     x_input_theta_list=numpy.mod(numpy.arange(0, nblades) * input_theta, 180.0)    
    result_angle =   x_input_theta_list*z2[1] + z2[2] +  z2[0]*x_input_theta_list**2 
#     print('p2',p2)
#     for n_b in xrange(0,nblades):
#         result_angle[n_b] = p2[n_b*1.0]
#

#     print('result_angle,new',result_angle)
    return result_angle

def corr_translation(input_k, input_theta):
    '''
    correction for 
    1) correct k-space translation: fix linear-phase in image space(corr_kspace_shift)
                                        filtering by pyramid, fft2, 
                                        and correction for the phase in image space. 
                                           
    2) zero order phase in kspace(corr_kspace_zero_phase): divided the angle at k-centre
    3) rotation correction: central k-space are mapped to polar coordinate(Im2Polar), 
                            reference k-space[blade 0] are coorelated with n-th blade(radial_correlate2d)
                                through fft of polar coordinate and then multiplication
                                to avoid 180 repeatition, using complex kspace
    4) translation correction: multiplication of the rotated central k-space
                                and then fft to image space to find out the
                                    shifting pixels(find_phase_plane). 
    '''
    


    input_k_shape = numpy.shape(input_k)[0:2]
    
    output_k = numpy.copy(input_k) # copy the stack for output_k space
    
    input_k = scipy.fftpack.fftshift(input_k,axes=(0,1))
    input_k = scipy.fftpack.fft2(input_k,axes=(0,1))
    input_k = scipy.fftpack.fftshift(input_k,axes=(0,1))
    
    for rep in xrange(0,1): # repeat the correction for translation
        input_k = numpy.mean(numpy.abs(input_k*input_k.conj()),3) # collapse the coil dimension
        
        input_k = scipy.fftpack.fftshift(input_k,axes=(0,1))
        input_k = scipy.fftpack.ifft2(input_k,axes=(0,1))
        input_k = scipy.fftpack.fftshift(input_k,axes=(0,1))    
        
        
    #     print('input k shape',input_k_shape)
        dim_of_less= numpy.min(input_k_shape)    
    #     print('input k shape', dim_of_less)
        center_kspace = numpy.empty((dim_of_less,dim_of_less,input_k.shape[2]),
                                    dtype =numpy.complex)
      
        for pp in xrange(0,input_k.shape[2]): # blade
            center_kspace[:,:,pp]= input_k[
                            (input_k_shape[0]/2 - dim_of_less/2 ):(input_k_shape[0]/2 + dim_of_less/2),
                            (input_k_shape[1]/2 - dim_of_less/2 ):(input_k_shape[1]/2 + dim_of_less/2),
                            pp ]    # stacks of center-kspace 
        
        result_angle=input_theta#find_rotation(input_k, input_theta)
        ref_kspace_rotate=numpy.zeros(center_kspace[:,:,0].shape,dtype = numpy.complex128)  
        
        for pp in numpy.arange(0,input_k.shape[2]): # blades
    #         if pp == 0:
            new_kspace_rotate = complex_imrotate(center_kspace[:,:,pp],   -result_angle[pp])
    #             new_kspace_rotate = complex_imrotate(center_kspace[:,:,pp], -1)
            ref_kspace_rotate = ref_kspace_rotate+ new_kspace_rotate*1.0/input_k.shape[2]             
            center_kspace[:,:,pp] = new_kspace_rotate   
    #         else:
    #             
    #             new_kspace_rotate = complex_imrotate(center_kspace[:,:,pp],   -result_angle[pp])
    # #             new_kspace_rotate = complex_imrotate(center_kspace[:,:,pp], -1)
    #             center_kspace[:,:,pp] = new_kspace_rotate   
                  
    #     matplotlib.pyplot.imshow(numpy.real(ref_kspace_rotate))  
    #     matplotlib.pyplot.show()  
        for pp in numpy.arange(0,input_k.shape[2]): # blades 
    
    
            corr_kspace_rotate = ref_kspace_rotate.conj()*center_kspace[:,:,pp]
            
            # find the x-y translation
            delta_x, delta_y = find_phase_plane(corr_kspace_rotate)
            
            for iter in xrange(0,output_k.shape[3]): # coils            
                # correct the x-y translation
                output_k[:,:,pp,iter] = corr_phase_plane(output_k[:,:,pp,iter], delta_x,delta_y)
        input_k = output_k # repeat the correction for motion
            # 
#     center_kspace[:,:,pp]= output_k[
#                 (input_k_shape[0]/2 - dim_of_less/2 ):(input_k_shape[0]/2 + dim_of_less/2),
#                 (input_k_shape[1]/2 - dim_of_less/2 ):(input_k_shape[1]/2 + dim_of_less/2),
#                 pp ]
#     center_kspace[:,:,pp]=complex_imrotate(  center_kspace[:,:,pp] , -result_angle[pp] )
#         center_kspace[:,:,pp]=complex_imrotate(  center_kspace[:,:,pp] ,-45 )


#     DA = numpy.mean(center_kspace,2)                  
#     input_k  = correlation_weighting(input_k ,center_kspace)      
            # weight the k-space strip with correlation
#     Dn = numpy.zeros((dim_of_less,dim_of_less),dtype = numpy.float32)
#     DA = numpy.zeros((dim_of_less,dim_of_less),dtype = numpy.float32)
    
#     for pp in numpy.arange(0,input_k.shape[2]): # blades    
#         center_kspace[:,:,pp]= input_k[
#                     (input_k_shape[0]/2 - dim_of_less/2 ):(input_k_shape[0]/2 + dim_of_less/2),
#                     (input_k_shape[1]/2 - dim_of_less/2 ):(input_k_shape[1]/2 + dim_of_less/2),
#                     pp ] 
#         new_kspace_rotate = input_k[
#                     (input_k_shape[0]/2 - dim_of_less/2 ):(input_k_shape[0]/2 + dim_of_less/2),
#                     (input_k_shape[1]/2 - dim_of_less/2 ):(input_k_shape[1]/2 + dim_of_less/2),
#                     pp ] 
#         new_kspace_rotate = complex_imrotate( new_kspace_rotate, -result_angle[pp] ) 
                      
#         input_k[:,:,pp] = correlation_weighting(input_k[:,:,pp], -result_angle[pp])
#     del ramp_filt       
#     del corr_kspace_rotate     
#     del center_kspace
#     del new_kspace_rotate
#     del ref_kspace_rotate
    
    return output_k#, result_angle# result_radial_correlate#output_k
# def corr_translation(input_k, input_theta):
#     '''
#     correction for 
#     1) correct k-space translation: fix linear-phase in image space(corr_kspace_shift)
#                                         filtering by pyramid, fft2, 
#                                         and correction for the phase in image space. 
#                                            
#     2) zero order phase in kspace(corr_kspace_zero_phase): divided the angle at k-centre
#     3) rotation correction: central k-space are mapped to polar coordinate(Im2Polar), 
#                             reference k-space[blade 0] are coorelated with n-th blade(radial_correlate2d)
#                                 through fft of polar coordinate and then multiplication
#                                 to avoid 180 repeatition, using complex kspace
#     4) translation correction: multiplication of the rotated central k-space
#                                 and then fft to image space to find out the
#                                     shifting pixels(find_phase_plane). 
#     '''
#     
# 
# 
#     input_k_shape = numpy.shape(input_k)[0:2]
#     output_k = numpy.copy(input_k)
# #     print('input k shape',input_k_shape)
#     dim_of_less= numpy.min(input_k_shape)    
# #     print('input k shape', dim_of_less)
#     center_kspace = numpy.empty((dim_of_less,dim_of_less,input_k.shape[2]),
#                                 dtype =numpy.complex)
#   
#     for pp in xrange(0,input_k.shape[2]): # blade
# 
# 
#             center_kspace[:,:,pp]= input_k[
#                         (input_k_shape[0]/2 - dim_of_less/2 ):(input_k_shape[0]/2 + dim_of_less/2),
#                         (input_k_shape[1]/2 - dim_of_less/2 ):(input_k_shape[1]/2 + dim_of_less/2),
#                         pp ]
#             
# 
#     
#     result_angle=input_theta#find_rotation(input_k, input_theta)
#     ref_kspace_rotate=numpy.zeros(center_kspace[:,:,0].shape,dtype = numpy.complex128)  
#     for pp in numpy.arange(0,input_k.shape[2]): # blades
#         if pp == 0:
#             new_kspace_rotate = complex_imrotate(center_kspace[:,:,pp],   -result_angle[pp])
# #             new_kspace_rotate = complex_imrotate(center_kspace[:,:,pp], -1)
#             ref_kspace_rotate = new_kspace_rotate             
#             center_kspace[:,:,pp] = new_kspace_rotate   
#         else:
#             
#             new_kspace_rotate = complex_imrotate(center_kspace[:,:,pp],   -result_angle[pp])
# #             new_kspace_rotate = complex_imrotate(center_kspace[:,:,pp], -1)
#             center_kspace[:,:,pp] = new_kspace_rotate   
#               
# #     matplotlib.pyplot.imshow(numpy.real(ref_kspace_rotate))  
# #     matplotlib.pyplot.show()   
#     for iter in xrange(0,3):
#         for pp in numpy.arange(0,input_k.shape[2]): # blades
#     
#             
#             corr_kspace_rotate = ref_kspace_rotate.conj()*center_kspace[:,:,pp]
#             
#             # find the x-y translation
#             delta_x, delta_y = find_phase_plane(corr_kspace_rotate)
#             
#             # correct the x-y translation
#             output_k[:,:,pp] = corr_phase_plane(input_k[:,:,pp], delta_x,delta_y)
#     
#             # 
#             center_kspace[:,:,pp]= output_k[
#                         (input_k_shape[0]/2 - dim_of_less/2 ):(input_k_shape[0]/2 + dim_of_less/2),
#                         (input_k_shape[1]/2 - dim_of_less/2 ):(input_k_shape[1]/2 + dim_of_less/2),
#                         pp ]
#             center_kspace[:,:,pp]=complex_imrotate(  center_kspace[:,:,pp] , -result_angle[pp] )
# #         center_kspace[:,:,pp]=complex_imrotate(  center_kspace[:,:,pp] ,-45 )
# 
# 
# #     DA = numpy.mean(center_kspace,2)                  
# #     input_k  = correlation_weighting(input_k ,center_kspace)      
#             # weight the k-space strip with correlation
# #     Dn = numpy.zeros((dim_of_less,dim_of_less),dtype = numpy.float32)
# #     DA = numpy.zeros((dim_of_less,dim_of_less),dtype = numpy.float32)
#     
# #     for pp in numpy.arange(0,input_k.shape[2]): # blades    
# #         center_kspace[:,:,pp]= input_k[
# #                     (input_k_shape[0]/2 - dim_of_less/2 ):(input_k_shape[0]/2 + dim_of_less/2),
# #                     (input_k_shape[1]/2 - dim_of_less/2 ):(input_k_shape[1]/2 + dim_of_less/2),
# #                     pp ] 
# #         new_kspace_rotate = input_k[
# #                     (input_k_shape[0]/2 - dim_of_less/2 ):(input_k_shape[0]/2 + dim_of_less/2),
# #                     (input_k_shape[1]/2 - dim_of_less/2 ):(input_k_shape[1]/2 + dim_of_less/2),
# #                     pp ] 
# #         new_kspace_rotate = complex_imrotate( new_kspace_rotate, -result_angle[pp] ) 
#                       
# #         input_k[:,:,pp] = correlation_weighting(input_k[:,:,pp], -result_angle[pp])
# #     del ramp_filt       
# #     del corr_kspace_rotate     
# #     del center_kspace
# #     del new_kspace_rotate
# #     del ref_kspace_rotate
#     
#     return output_k#, result_angle# result_radial_correlate#output_k
def correlation_weighting(input_strips,center_kspace):
    nblades = numpy.shape(input_strips)[2]
    DA = numpy.mean(center_kspace,2)      
    rho = 0.5
    Chi_n = numpy.zeros((nblades,),dtype = numpy.float32)
    for pj in xrange(0,nblades):
        
        Dn = center_kspace[:,:,pj]
        Chi_n[pj]= numpy.sum(numpy.real( DA*Dn.conj()  ).flatten())
        
    max_Chi_n = numpy.max(Chi_n)
    min_Chi_n = numpy.min(Chi_n)
    Chi_n = ( Chi_n - min_Chi_n )/(max_Chi_n -min_Chi_n )
    Pn = Chi_n**rho
    for pj in xrange(0,nblades):
        input_strips[:,:,pj] = input_strips[:,:,pj]*Pn[pj]  
#         input_strips[:,:,pj]=input_strips[:,:,pj]*
   

    return input_strips
if __name__ == "__main__":
    import phantom     
    a=phantom.phantom(34)
    import cProfile
    import Im2Polar
    
    cProfile.run('tt = Im2Polar.Im2Polar(a,0,1,64,512).real')
    
    matplotlib.pyplot.imshow(tt)
    matplotlib.pyplot.show()
    b =numpy.roll(a,-10,axis =0)
    b =numpy.roll(b,10,axis =1)
#     import scipy.misc
#     b = scipy.misc.imrotate(b,15, 'cubic')
    
    
    ka = scipy.fftpack.fftshift(a)
    ka = scipy.fftpack.fft2(ka)
    
    kb = scipy.fftpack.fftshift(b)
    kb = scipy.fftpack.fft2(kb)
    corr_kspace_rotate = ka.conj()*kb
    
    delta_x, delta_y = find_phase_plane(corr_kspace_rotate)
    print('delta_x, delta_y', delta_x, delta_y)
    kb = corr_phase_plane(kb, delta_x,delta_y)
    cnt = kb[0,0]/numpy.abs(kb[0,0])
    kb = kb/cnt
    
    
    
    
    a2=   scipy.fftpack.ifft2(ka)
    a2 = numpy.abs(scipy.fftpack.fftshift(a2))
     
    b2=   scipy.fftpack.ifft2(kb)
    b2 = numpy.abs(scipy.fftpack.fftshift(b2))
    matplotlib.pyplot.imshow(a)
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(b)
    matplotlib.pyplot.show()    
    
    
    matplotlib.pyplot.imshow(a2)
    matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(b2)
    matplotlib.pyplot.show()
    
#     corr_phase_plane(a,0,0)
    # print(numpy.shape(a))
    import cProfile
#     cProfile.run('c=Im2Polar(a,0.2,1,60,521)')
    # print(numpy.shape(c))
       
#     import matplotlib.pyplot
#     matplotlib.pyplot.imshow(a)
#     matplotlib.pyplot.show()
       
#     matplotlib.pyplot.imshow(c)
#     matplotlib.pyplot.show()
    uu = numpy.ones((128,32))
#     sdfa=corr_rotation(uu,uu)
#     b = scipy.fftpack.fft2(a)
#     matplotlib.pyplot.imshow(b.real)
#     matplotlib.pyplot.show()
            
            
            

            
    
    