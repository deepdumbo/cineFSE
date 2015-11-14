import numpy
cimport numpy
def Dx(numpy.ndarray u):
#     shapes = numpy.shape(u)
    cdef int rows=numpy.shape(u)[0]
#     ndim = numpy.ndim(u)
#     u=numpy.reshape(u,(shapes[0],numpy.prod(shapes[1:])),order='F')
     
#     print('ndims of Dx,=',numpy.ndim(u))
#     shape = numpy.shape(u)
#     dims= len(shape)
#     if dims ==4:
#         return utils.Dx(u)
#     else:
    cdef numpy.ndarray ind1 = numpy.arange(0,rows)
    cdef numpy.ndarray ind2 = numpy.roll(ind1,1) 
#     u2 = u.copy()
    cdef numpy.ndarray u2= u[ind2,...]
    u2[...]= u[...] - u2[...]#array_diff(u,u2)
    return u2#u[ind1,...]-u[ind2,...]
# def Dx(numpy.ndarray u):
#     shapes = numpy.shape(u)
#     cdef int rows=shapes[0]
#     cdef int cols=numpy.prod(shapes[1:])
# #     ndim = numpy.ndim(u)
#     u=numpy.reshape(u,(rows,cols),order='F')
#     
# 
# #     cdef numpy.ndarray ind1 = numpy.arange(0,rows)
# #     cdef numpy.ndarray ind2 = numpy.roll(ind1,1) 
#     cdef numpy.ndarray u2=numpy.empty_like(u)
# #     u2[ind1,...]=u[ind1,...] - u[ind2,...] 
# #     u2 = u.copy()
#     cdef int ii
#     cdef int jj
#     for ii from 0<=ii<rows:
#         for jj from 0<=jj<cols:
#             if ii > 0:
#                 u2[ii,jj] = u[ii,jj] -u[ii-1,jj]
#             else: # ii ==0
#                 u2[ii,jj] = u[ii,jj] -u[-1,jj]
#                   
#     
#     u2=numpy.reshape(u2,shapes,order='F')
#     
# #     u2= u[ind2,...]
# #     u2= u - u2#array_diff(u,u2)
#     return u2#u[ind1,...]-u[ind2,...]