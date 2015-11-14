from nufft import *
import numpy
import numpy.random
import matplotlib.pyplot
import matplotlib.cm

import sys

def output(cc):
    print('max',numpy.max(numpy.abs(cc[:])))
    
def Normalize(D):
    return D/numpy.max(numpy.abs(D[:]))
def checkmax(x):
    max_val = numpy.max(numpy.abs(x[:]))
    print( max_val)
    return max_val
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
#     for pj in range(0,n_dims):    
#         s = s+ (dd[pj] + bb[pj])*(dd[pj] + bb[pj]).conj()   
#     s = numpy.sqrt(s).real
#     ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-7) # shrinkage
#     for pj in range(0,n_dims): 
#         
#         xx = xx+ (ss*(dd[pj]+bb[pj]),)        
#     
#     return xx
def shrink(dd, bb,LMBD):

#     n_dims=numpy.shape(dd)[0]
    n_dims = len(dd)
#     print('n_dims',n_dims)
#     print('dd shape',numpy.shape(dd))
     
    xx=()
#     ss = shrink1(n_dims,dd,bb,LMBD)
#     xx = shrink2(n_dims,xx,dd,bb,ss)
#     return xx
# def shrink1(n_dims,dd,bb,LMBD):
    s = numpy.zeros(dd[0].shape)
    for pj in range(0,n_dims):    
        s = s+ (dd[pj] + bb[pj])*(dd[pj] + bb[pj]).conj()   
    s = numpy.sqrt(s).real
    ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-7) # shrinkage
#     return ss
# def shrink2(n_dims,xx,dd,bb,ss):    
    for pj in range(0,n_dims): 
        
        xx = xx+ (ss*(dd[pj]+bb[pj]),)        
    
    return xx
def TVconstraint(xx,bb):

    n_xx = len(xx)
    n_bb =  len(bb)
    #print('n_xx',n_xx)
    if n_xx != n_bb: 
        print('xx and bb size wrong!')
    
    cons_shape = numpy.shape(xx[0])
    cons=numpy.zeros(cons_shape,dtype=numpy.complex64)
    
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