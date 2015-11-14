#cython: boundscheck=False
#cython: wraparound=False
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np

# cdef extern from "math.h":
#     float cos(float theta)
#     float sin(float theta)
#     int floor(float theta)
#     int ceil(float theta)
    
def AverageHermitian(np.ndarray input_x):
    '''
    convert rectangular to polar image
    auxillary function of corr_rotation
    '''
#     shape_imR = np.shape(imR_real)
#     cdef float PI = 3.14159265358979324 
    cdef int M = input_x.shape[0]
    cdef int N = input_x.shape[1]
    cdef int m
    cdef int n
    cdef double tmp_real
    cdef double tmp_imag
    
    for m from 0 <= m < M:
        for n from 0 <= n < N/2:
            
            tmp_real = ( input_x[m, n].real + input_x[-m-1, -n-1].real )/2.0
            tmp_imag = ( input_x[m, n].imag - input_x[-m-1, -n-1].imag )/2.0
            input_x[m, n] = tmp_real + 1.0j*tmp_imag
            input_x[-m-1, -n-1] = tmp_real - 1.0j*tmp_imag
    return input_x