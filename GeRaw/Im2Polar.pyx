#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    float cos(float theta)
    float sin(float theta)
    int floor(float theta)
    int ceil(float theta)
    
def Im2Polar(np.ndarray imR,
              float rMin, 
              float rMax, 
              int M, 
              int N):
    '''
    convert rectangular to polar image
    auxillary function of corr_rotation
    '''
#     shape_imR = np.shape(imR_real)
    cdef float PI = 3.14159265358979324 
    cdef int Mr = imR.shape[0]
    cdef int Nr = imR.shape[1]
#     print('Mr, Nr',Mr, Nr)
    cdef float Om = (Mr - 1)/2.0  # image center 
    cdef float On = (Nr - 1.0)/2.0 
#     print('Om, On',Om, On)
    
    cdef float sx = (Mr - 1)/2.0 # scaling factors
    cdef float sy = (Nr - 1)/2.0

#     print('sx, sy',sx, sy)

    cdef np.ndarray imP = np.empty((M,N),dtype = np.complex128) # define 2D array of Polar coordinate
    
    cdef float delR = (rMax-rMin)/1.0/(M-1)
    cdef float delT = 2.0*PI/N
    
#     print('delR, delT', delR, delT)
#     imR =np.real(imR)+(0.0+0.0j) # convert to complex
    cdef float r
    cdef float t
    cdef float x
    cdef float y
    cdef float xR
    cdef float yR
    cdef int xf
    cdef int xc
    cdef int yf 
    cdef int yc
#     cdef np.ndarray data_grid=np.zeros((2,2),dtype = np.complex)
    cdef int ri
    cdef int ti
#     cdef double complex C
#     cdef np.ndarray data_grid2=np.zeros((2,),dtype = np.complex)
    cdef float complex data0
    cdef float complex data1
    cdef float complex data2
    cdef float complex data3
    
#     for ri in  xrange(0,M):
#         for ti in  xrange(0,N):
    for ri from 0 <= ri < M:
        for ti from 0 <= ti < N:
         
            r = rMin + (ri)*delR
            t = ti *delT

            x = r*cos(t + PI)
            y = -r * sin(t + PI)
            xR = x*sx + Om;
            yR = y*sy + On;
            
            xf = floor(xR)
            xc = ceil(xR)
            yf = floor(yR)
            yc = ceil(yR)   

            data0=imR[xf, yf]
            data1=imR[xf, yc]
            data2=imR[xc, yf]
            data3=imR[xc, yc]
            
            data0=(data0*(xc-xR+1e-7) + data2*(xR-xf+1e-7) )/(xc -xf+2e-7)
            data1=(data1*(xc-xR+1e-7) + data3*(xR-xf+1e-7) )/(xc -xf+2e-7)
            imP[ri, ti]= (data0*(yc-yR+1e-7)   +  data1*(yR-yf+1e-7) )/(yc -yf+2e-7)

    return imP