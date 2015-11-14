#! /usr/bin/python
from scipy import signal
import numpy as np
# 
# J = 32*32
# K = 1 
# N = int(1.0*J/K) +1
# base_freq = 320
# bpass = signal.remez(N, [0, 0.25*base_freq, 0.4*base_freq, 0.5*base_freq], [1,0 ], [1,10] , Hz=32*base_freq, grid_density = 320,
# maxiter=25)
# print(bpass.shape)
# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# Copyright (c) 2008, Enthought, Inc.
# License: BSD Style.


# freq, response = signal.freqz(bpass)
# ampl = np.abs(response)
import matplotlib.pyplot as plt
# from scipy import special
# import scipy.special._ufuncs as cephes
# from scipy.special import _ufuncs_cxx
# from scipy.special import ellipk
#   
# from scipy.special._testutils import assert_tol_equal, with_special_errors, \
#      assert_func_equal
# import numpy
# # for pp in xrange(3,12):
# (s,sp )  =cephes.obl_rad2(0,1,7, numpy.arange(0,.9,0.01))  
# 
# plt.plot(sp)
# # plt.plot(sp)
# plt.show()
# print(s, sp)
# Prolate spheroidal radial function of the first kind and its derivative
# 
# Computes the prolate sheroidal radial function of the first kind and its derivative (with respect to x) for mode parameters m>=0 and n>=m, spheroidal parameter c and |x| < 1.0.
# 
# Returns:    
# s
# Value of the function
# sp
# Value of the derivative vs x

# print(s)
# plt.plot(s)
# plt.show()
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# #ax1.semilogy(freq/(2*np.pi), ampl, 'b-')  # freq in Hz
# ax1.plot(freq/(2*np.pi),ampl,'b-')
# ax2 = fig.add_subplot(212)
# t= np.arange(0,N)
# t = (t- (N - 1)/2.0  ) /32.0
# 
# 
# bpass = bpass/np.max(bpass)
# import scipy.interpolate
# f = scipy.interpolate.interp1d(t, bpass);
# 
# ax2.plot(t,f(t),'b-')
# print(bpass)
# ax2.plot(t,np.sinc(t ),'r-')
# plt.show()
# 
# plt.figure()
# plt.plot(t,f(t),'k-')
# 
# # plt.plot(t,np.sinc(t )*(0.5+0.5*np.cos(t/3.0 )),'b-')
# # plt.plot(t,np.sin(t),'b-')
# plt.plot(t,np.sinc(t ),'r-')
# plt.show()
import numpy.fft
t=signal.get_window(('chebwin', 6000.0), 8001)
# t=signal.get_window(('slepian', .05), 513)
t=t[1:]
t=t/numpy.max(t)
 
x =  numpy.arange(0, 8000)
x = (x- 3999 ) /10.0
 
print(t)
plt.plot(x,t)
plt.show()
plt.plot(numpy.real(numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(t)))))
plt.show()

# import nitime.algorithms.spectral
# 
# n_bases = 5
# v,e = nitime.algorithms.spectral.dpss_windows(256, 2, n_bases )
# for jj in xrange(0,n_bases):
#     v[jj,:]=v[jj,:]*e[jj]
#     
#     
# plt.plot(v.T)
# plt.show()    
#  
# v2= numpy.sum(v.T, 1)
# v2 = numpy.fft.fftshift(numpy.fft.fft(numpy.fft.fftshift(v2)))
#  
# plt.plot(numpy.real(v2))
# plt.show() 
# return

# from scipy import signal
# from scipy.fftpack import fft, fftshift
# import matplotlib.pyplot as plt
# 
# window = signal.slepian(16, width=1)
# plt.plot(window)
# plt.title("Slepian (DPSS) window (BW=0.3)")
# plt.ylabel("Amplitude")
# plt.xlabel("Sample")
# plt.figure()
# A = fft(window, 2048) / (len(window)/2.0)
# freq = np.linspace(-0.5, 0.5, len(A))
# response = 20 * (np.imag(fftshift(A / abs(A).max())))
# plt.plot(freq, response)
# response = 20 * (np.real(fftshift(A / abs(A).max())))
# plt.plot(freq, response)
# # plt.axis([-0.5, 0.5, -120, 0])
# plt.title("Frequency response of the Slepian window (BW=0.3)")
# plt.ylabel("Normalized magnitude [dB]")
# plt.xlabel("Normalized frequency [cycles per sample]")
# plt.show()