import phantom
import numpy
import scipy.fftpack
import scipy.misc
import matplotlib.pyplot


def test():
    pass

if __name__ == "__main__":
    test()
    a = phantom.phantom(512)*1.0
    d =scipy.misc.imrotate(a, - 23.4, interp = 'bilinear')
    b =scipy.misc.imrotate(d,  23.4, interp = 'bilinear')
    b = b -a*256
    
#     a = scipy.fftpack.fftshift(a)
#     b = scipy.fftpack.fft2(a)
#     b = scipy.fftpack.fftshift(b)
    
    matplotlib.pyplot.imshow(b, norm = matplotlib.pyplot.Normalize(0,256))
    matplotlib.pyplot.show()
    
    