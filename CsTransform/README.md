PYNUFFT
=======

Pythonic Non-uniform Fast Fourier Transform(NUFFT) With Fast Inversion and Total Variation Constraint


Brief
    pynufft is the fast program aims to do constraint inversion
    of irregularly sampled data.
    
    Please cite 
	Jyh-Miin Lin, Andrew Patterson, Hing-Chiu Chang, Tzu-Chao Chuang, Martin J. Graves,
    	"CS-PROPELLER MRI with Parallel Coils Using NUFFT and Split-Bregman Method"(in progress 2013)
    	which is planned to be published soon.
    and
    	J A Fessler, Bradley P Sutton.
    	Nonuniform fast Fourier transforms using min-max interpolation. 
    	IEEE Trans. Sig. Proc., 51(2):560-74, Feb. 2003.  
              
    2. Note the "better" results by min-max interpolator of J.A. Fessler et al
    3. Other relevant works:
	*c-version: http://www-user.tu-chemnitz.de/~potts/nfft/
	    is a c-library with gaussian interpolator
    	*fortran version: http://www.cims.nyu.edu/cmcl/nufft/nufft.html
    		alpha/beta stage
    	* MEX-version http://www.mathworks.com/matlabcentral/fileexchange/25135-nufft-nufft-usffft
     
Overview of the class pynufft

	4 methods of the class pynufft are directly accessible to users. 

        The first method is initiator: pynufft.__init__(). 
        The second one is the forward transform, which is called pynufft.forward(). 
        The third method is the backward transform, called pynufft.backward(), which is the adjoint operator of forward transform, 
        i.e. it reverse the order of operators of forward transform. Finally and more importantly.
        The fourth method is called the pynufft.inverse(). 
        In the following paragraphs the input and output parameters shall be described.

	1.Initiatior:NufftObj = pynufft(om, Nd,Kd,Jd)
	When initiating the pynufft, you need 4 inputs: om, Nd, Kd, and Jd. 

	om is the k-space coordinates denoted by its radians, ranged between [-π and π]. 
        Nd is the dimension of the image/time domain. 
        Kd is the dimension of the k-space/frequency domain. 
        Jd is the number of adjacent points of k-space used for interpolation. 

	For example, if you have a two dimensional image to recover from 131072 data points, 
        om is shown as the following numerical listed as two columns of floating numbers:

           om = numpy.loadtxt('om.txt') # load example text file describing the sampling locations
	   om = 
		[[-3.12932086  0.38042724]      \
		[-3.1047771   0.38042724]	|	
		[-3.08023357  0.38042724]	|	
		………..				131072 pairs of floating numbers
		[-2.9468298   0.97404122] 	|	
		[-2.97090197  0.97882938]	|	
		[-2.99497414  0.98361766]]	/
           Meanwhile, the Nd and Kd are a tuples:
	   Nd = (256,256)
	   Kd=(512,512) # Kd is generally the double of Nd
	   Jd = (6,6) # adjacent 6 points on k-space are used for interpolation

	2.Forward transform: 
	data= NufftObj.forward(image )
	Once NufftObj is created, 
        you can transform any image with the resolution of 256 by 256 grids to non-uniform data points. 
        (which has been defined by om)

	3.Backward transform(regridding + inverse FFT): 
	image_blur = NufftObj.backward(data)
	The adjoint operator is called backward. This calculation is essentially the regridding + inverse FFT, 
        so it suffers from the image blurring denoted by the point-spread function of the sampling pattern. 
        The term of data is the acquired data on non-uniform points described above.

	4.Inversion*: image_recon = NufftObj.inverse(data, mu, LMBD, gamma, nInner, nBreg)
	Yet one of the best model of image recover problem is constraint recovery using L1 norm; 
        this model is the combination of least square minimization(L2 norm) and the minimization of certain constraints for their L1 norm. 
        One of the best constraints is total variation(TV), which measures the summation of the “gradients” of the images. 
        As the algorithm runs, the discrepancy between data and the “recovered images” are gradually minimized, 
        and the TV are also minimized.

        The parameters of the method inverse() are quite long(although the calculation speed was fast). 
        data is the acquired data on non-uniform points described above. 
        mu should be 1.0. 
        LMBD is the strength of constraint, which shall be ranging from 0.001 to 0.3(0.05 is a good value). 
        gamma is another constraint parameter which constraint the image domain strength(0.001 is a good empirical value). 
        nInner is the inner iteration times, and you could use nInner=1~5. 
        nBreg is the iteration times of outer loop, and nBreg = 20~50 is OK.

	*Note: The “inversion” of images recovery is tricky, because literally there is no “true inversion” because of the 
               difficulty to inverse the high dimensional problem. However, one can approximate the inversion based on different models.

Examples of pynufft
	The best way to understand the class pynufft is by looking at these examples. 
	They are problems of 1D, 2D, and 3D reconstruction.

	1.1D case: test_1D()	
	  1D example is the recovery of the original signal demonstrated in Figure 2. 
	  First, the frequency domain spectra are randomly sampled from 256 locations (described by om) 
	  of which the histogram is shown in Figure 1. 
	  
	  After the reconstruction from randomized samples(the inverse calculation), four plots demonstrates 
	  1)the original curve, 2)the IFFT of irregularly sampling, 3)the recovered curve, and 
	  4)the residual signals are demonstrated Figure 2. 
	  
	  Below is the test codes of 1D example.

	  # test codes of 1D problem ===================================
	  # import several modules
	      import numpy 
	      import matplotlib.pyplot
	  #create 1D curve from 2D image
	      image = numpy.loadtxt('phantom_256_256.txt') 
	      image = image[:,128]
	  #determine the location of samples
	      om = numpy.loadtxt('om1D.txt')
	      om = numpy.reshape(om,(numpy.size(om),1),order='F')
	  # reconstruction parameters
	      Nd =(256,) # image space size
	      Kd =(256,) # k-space size
	      
	      Jd =(1,) # interpolation size
	  # initiation of the object
	      NufftObj = pynufft(om, Nd,Kd,Jd)
	  # simulate "data"
	      data= NufftObj.forward(image )
	  #adjoint(reverse) of the forward transform
	      image_blur= NufftObj.backward(data)[:,0]
	  #inversion of data
	      image_recon = NufftObj.inverse(data, 1.0, 1, 0.001,15,16)

	  #Showing histogram of sampling locations
	      matplotlib.pyplot.hist(om,20)
	      matplotlib.pyplot.title('histogram of the sampling locations')
	      matplotlib.pyplot.show()
	  #show reconstruction
	      matplotlib.pyplot.subplot(2,2,1)

	      matplotlib.pyplot.plot(image)
	      matplotlib.pyplot.title('original') 
	      matplotlib.pyplot.ylim([0,1]) 
		    
	      matplotlib.pyplot.subplot(2,2,3)    
	      matplotlib.pyplot.plot(image_recon.real)
	      matplotlib.pyplot.title('recon') 
	      matplotlib.pyplot.ylim([0,1])
		      
	      matplotlib.pyplot.subplot(2,2,2)

	      matplotlib.pyplot.plot(image_blur.real) 
	      matplotlib.pyplot.title('blurred')
	      matplotlib.pyplot.subplot(2,2,4)

	      matplotlib.pyplot.plot(image_recon.real - image) 
	      matplotlib.pyplot.title('residual')
	  #     matplotlib.pyplot.subplot(2,2,4)
	  #     matplotlib.pyplot.plot(numpy.abs(data))  
	      matplotlib.pyplot.show()  
	  #end of test codes of 1D problem ===================================

      2.2D case: test_2D()
	  2D example simulates a data with PROPELLER trajectory, from which the image is restored. 
	  As is shown in Figure 3, the PROPELLER trajectory in k-space are rotating rectangular shaped blades 
	  which are popular in MRI. This PROPELLER sampling would create an blurred appearance. 
	  Figure 4 depict the 1) original image, 2) blurred image with FFT, 3) reconstructed images, and 
	  4) the error of reconstructed image.

