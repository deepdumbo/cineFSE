import phantom
import scipy.ndimage.interpolation
import scipy.ndimage.fourier
import matplotlib.pyplot
import numpy
# 
def affine_shift(a, shift_tuple):
     
    fa = numpy.fft.fft2(a)
    a= scipy.ndimage.fourier.fourier_shift(fa,shift = shift_tuple)
    a=numpy.fft.ifft2(a).real
     
    return a
# 
# 
N = 128
a=phantom.phantom(N)
# 
theta = 0.0
cosT= numpy.cos(theta*numpy.pi/180)
sinT= numpy.sin(theta*numpy.pi/180)
mymat = [[cosT, -sinT],[sinT, cosT]]
 
print(mymat)
 
b = affine_shift(a, (N/2, N/2))
 
# b=scipy.ndimage.interpolation.affine_transform(numpy.real(a),mymat,mode = 'wrap'  )
#  
# a= affine_shift(a, (-N/2, -N/2))
# b = affine_shift(b, (-N/2, -N/2))
# 

# from numpy import eye, asarray, dot, sum , diag
# from numpy.linalg import svd
# def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
#     p,k = Phi.shape
#     R = eye(k)
#     d=0
#     for i in xrange(q):
#         d_old = d
#         Lambda = dot(Phi, R)
#         u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
#         R = dot(u,vh)
#         d = sum(s)
#         if d_old!=0 and d/d_old < 1 + tol: break
#     return dot(Phi, R)
# from scipy.signal import resample
# from numpy import linspace, sin, pi
# from pylab import plot, show
# 
# # replace 4*pi to see effect of non-periodic function (e.g. at either end) 
# x = linspace(0,4*pi,10,endpoint=False)
# y = sin(x)
# # if you supply t, you get the interpolated t back as well.
# (yy,xx) = resample(y, 100, t = x)
# 
# plot(x,y,'ro', xx, yy)
# show()
# b= varimax(a, gamma = 3.0)
matplotlib.pyplot.subplot(2,1,1)
matplotlib.pyplot.imshow(numpy.abs(a))
matplotlib.pyplot.subplot(2,1,2)
matplotlib.pyplot.imshow(b)
matplotlib.pyplot.show()