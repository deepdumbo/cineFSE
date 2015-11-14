30c30
< #     import numpy.random
---
>     import numpy.random
35c35
< #     import sys
---
>     import sys
41c41
< dtype = numpy.complex64
---
> 
66,70c66,70
< # try:
< #     from numba import autojit  
< # 
< # except:
< #     print('numba not supported')
---
> try:
>     from numba import autojit  
> 
> except:
>     print('numba not supported')
91d90
<     return X
129c128
<     for pp in xrange(0,dim_x[2]):
---
>     for pp in range(0,dim_x[2]):
148c147
< #     for pj in xrange(0,n_dims):    
---
> #     for pj in range(0,n_dims):    
152c151
< #     for pj in xrange(0,n_dims): 
---
> #     for pj in range(0,n_dims): 
157,169d155
< def shrink2(dd,bb,ss,n_dims):
<     xx = tuple(ss*(dd[pj]+bb[pj]) for pj in xrange(0,n_dims))
<     return xx 
< 
< def shrink1(dd,bb,n_dims):
< #     s = numpy.zeros(numpy.shape(dd[0]),dtype = numpy.float)
< #     c = numpy.empty_like(s) # only real
< #     for pj in xrange(0,n_dims):  
< #         c  = (dd[pj] + bb[pj]).real
< #         s = s+ c**2
<     s = sum((dd[pj] + bb[pj]).real**2 for pj in xrange(0,n_dims))
<     s = s**0.5
<     return s.real
174,178c160,177
< 
<     s = shrink1(dd,bb,n_dims)
<     ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-15)# shrinkage
<        
<     xx = shrink2(dd,bb,ss,n_dims)
---
> #     print('n_dims',n_dims)
> #     print('dd shape',numpy.shape(dd))
>      
>     xx=()
> #     ss = shrink1(n_dims,dd,bb,LMBD)
> #     xx = shrink2(n_dims,xx,dd,bb,ss)
> #     return xx
> # def shrink1(n_dims,dd,bb,LMBD):
>     s = numpy.zeros(dd[0].shape)
>     for pj in range(0,n_dims):    
>         s = s+ (dd[pj] + bb[pj])*(dd[pj] + bb[pj]).conj()   
>     s = numpy.sqrt(s).real
>     ss = numpy.maximum(s-LMBD*1.0 , 0.0)/(s+1e-7) # shrinkage
> #     return ss
> # def shrink2(n_dims,xx,dd,bb,ss):    
>     for pj in range(0,n_dims): 
>         
>         xx = xx+ (ss*(dd[pj]+bb[pj]),)        
184,200c183,195
<     try:    
<         n_xx = len(xx)
< #         n_bb =  len(bb)
< #         cons_shape = numpy.shape(xx[0])
< #         cons=numpy.zeros(cons_shape,dtype=numpy.complex64)
<         
<     #     cons = sum( get_Diff_H( xx[jj] - bb[jj] ,  jj) 
<     #                 for jj in xrange(0,n_xx))
<         
< #         for jj in xrange(0,n_xx):
< #             cons =  cons + get_Diff_H( xx[jj] - bb[jj] ,  jj)
<         cons = sum(get_Diff_H( xx[jj] - bb[jj] ,  jj) for jj in xrange(0,n_xx))
<     except:
<         n_xx = len(xx)
<         n_bb =  len(bb)
<         if n_xx != n_bb: 
<             print('xx and bb size wrong!')    
---
>     n_xx = len(xx)
>     n_bb =  len(bb)
>     #print('n_xx',n_xx)
>     if n_xx != n_bb: 
>         print('xx and bb size wrong!')
>     
>     cons_shape = numpy.shape(xx[0])
>     cons=numpy.zeros(cons_shape,dtype=numpy.complex64)
>     
>     for jj in range(0,n_xx):
> 
>         cons =  cons + get_Diff_H( xx[jj] - bb[jj] ,  jj)
>     
205d199
< 
207,213c201,245
< #     shapes = numpy.shape(u)
< #     rows=shapes[0]
< #     ind1 = xrange(0,rows)
< #     ind2 = numpy.roll(ind1,1,axis=0) 
< #     u2= u[ind2,...]
< #     u2[...]= u[...] - u2[...]
< #     return u2#u[ind1,...]-u[ind2,...]
---
> # # 
> #     u2 = u.copy()
> #     
> #     u2[1:-1,...] = u2[1:-1,...]-u[0:-2,...]
> #     u2[0,...] = u2[0,...]-u[-1,...] 
> #     return u2
> #     return utils.Dx(u)
> # import scipy.weave
> # def Dx2(type,u):
> #     u2 = u.copy()
> #     shape = numpy.shape(u)
> #     dims= len(shape)
> # #     new_array = zeros(shape(a_2d),type)
> # #     NumPy_type = scalar_spec.NumPy_to_blitz_type_mapping[type]
> #     if dims == 4:
> #         code = \
> #         """
> #         for(int i = 0;i < _Nu[0]; i++)
> #             for(int j = 0;  j < _Nu[1]; j++)
> #                 for(int k = 0;  m < _Nu[2]; k++)
> #                     for(int m = 0;  m < _Nu[3]; m++)
> #                         if(i < _Nu[0] -1)
> #                             u2(i,j,k,m) = u2(i,j,k,m)-u(i,j,k,m);
> #                         else
> #                             u2(i,j,k,m) = u2(i,j,k,m)-u(i,j,k,m);
> #         """ #% #NumPy_type
> #     scipy.weave.inline(code,['u','u2'],compiler='gcc')
> #     return u2
> 
> # def array_minux(a,b):
> #     b[...]
> #     return a-b
> # def create_ind(rows):
> #     ind1 = numpy.arange(0,rows)
> #     ind2 = numpy.roll(ind1,1) 
> #     return ind1 , ind2
> # def create_u2(u):
> #     return u
> # def array_diff(u,u2):
> #     u2 = u - u2
> #     return u2
> #===============================================================================
> # import utils
> # def Dx(u):
> #     return utils.Dx(u)
216,219c248,267
<     u2=numpy.concatenate((u,u[0:1,...]),axis=0)
<     u2=numpy.roll(u2,1,axis=0)
<     u2=numpy.diff(u2,n=1,axis=0)
<     return u2
---
> #     u = u.astype(numpy.complex64)
>     shapes = numpy.shape(u)
>     rows=shapes[0]
> #     print('shapes of u=',shapes)
> #     ndim = numpy.ndim(u)
> #     u=numpy.reshape(u,(shapes[0],numpy.prod(shapes[1:])),order='F')
>          
> #     print('ndims of Dx,=',numpy.ndim(u))
> #     shape = numpy.shape(u)
> #     dims= len(shape)
> #     if dims ==4:
> #         return utils.Dx(u)
> #     else:
>     ind1 = numpy.arange(0,rows)
>     ind2 = numpy.roll(ind1,1) 
> #     u2 = u.copy()
>     u2= u[ind2,...]
>     u2[...]= u[...] - u2[...]#array_diff(u,u2)
>     return u2#u[ind1,...]-u[ind2,...]
> 
227c275
<         mylist=list(xrange(0,x.ndim)) 
---
>         mylist=list(numpy.arange(0,x.ndim)) 
253c301
<         mylist=list(xrange(0,x.ndim)) 
---
>         mylist=list(numpy.arange(0,x.ndim)) 
269c317
< #         for ss in xrange(0,ShapeProd):
---
> #         for ss in numpy.arange(0,ShapeProd):
293,297c341,345
< #         self.st['q'] = self.st['p']
< #         self.st['q'] = self.st['q'].conj().multiply(self.st['q'])
< #         self.st['q'] = self.st['q'].sum(0)
< #         self.st['q'] = numpy.array(self.st['q'] )
< #         self.st['q']=numpy.reshape(self.st['q'],(numpy.prod(self.st['Kd']),1),order='F').real
---
>         self.st['q'] = self.st['p']
>         self.st['q'] = self.st['q'].conj().multiply(self.st['q'])
>         self.st['q'] = self.st['q'].sum(0)
>         self.st['q'] = numpy.array(self.st['q'] )
>         self.st['q']=numpy.reshape(self.st['q'],(numpy.prod(self.st['Kd']),1),order='F').real
303,367d350
<     def forwardbackward(self,x):
<         if self.cuda_flag == 0:
<             st=self.st
<             Nd = st['Nd']
<     #         Kd = st['Kd'] # unused
<     #         dims = numpy.shape(x) #unused
<             dd = numpy.size(Nd)
<         #    print('in nufft, dims:dd',dims,dd)
<         #    print('ndim(x)',numpy.ndim(x[:,1]))
<             # checker
<             checker(x,Nd)
<             
<             if numpy.ndim(x) == dd:
<                 Lprod = 1
<                 x = numpy.reshape(x,Nd+(1,),order='F')
<             elif numpy.ndim(x) > dd: # multi-channel data
<                 Lprod = numpy.size(x)/numpy.prod(Nd)
<                 Lprod = Lprod.astype(int)
<             '''
<             Now transform Nd grids to Kd grids(not be reshaped)
<             '''
<             Xk = self.Nd2Kd(x,0) #
<     
<             for ii in xrange(0,Lprod):
<             
<                 Xk[...,ii] = st['q'][...,0]*Xk[...,ii]
<             '''
<             Now transform Kd grids to Nd grids(not be reshaped)
<             '''
<             x= self.Kd2Nd(Xk,0) #
<             
<             checker(x,Nd) # check output
<             return x    
<         elif self.cuda_flag == 1:
<             return self.forwardbackward_gpu(x)
<     def gpu_k_modulate(self):
<         try:
<             self.myfft(self.data_dev, self.data_dev,inverse=False)
<             self.data_dev=self.W_dev*self.data_dev
<             self.myfft(self.data_dev, self.data_dev,inverse=True)
<             return 0
<         except: 
<             return 1       
< #     def gpu_k_demodulate(self):
< #         try:
< #             self.myfft(self.data_dev, self.data_dev,inverse=False)
< #             self.data_dev=self.data_dev/self.W_dev
< #             self.myfft(self.data_dev, self.data_dev,inverse=True)
< #             print('inside gpu_k_demodulate')
< #             return 0
< #         except: 
< #             return 1   
<     def Nd2KdWKd2Nd_gpu(self,x, weight_flag):
<         '''
<         Now transform Nd grids to Kd grids(not be reshaped)
<         
<         '''
<         #print('661 x.shape',x.shape)        
< #         x is Nd Lprod
<         st=self.st
<         Nd = st['Nd']
<         Kd = st['Kd']
< #         dims = numpy.shape(x)
< #         dd = numpy.size(Nd)
<         Lprod = numpy.shape(x)[-1]
369,399c352,353
<         if self.debug==0:
<             pass
<         else:
<             checker(x,Nd)
<             
<         snc = st['sn']
<         output_x=numpy.zeros(Kd, dtype=numpy.complex64)
< #         self.W_dev = self.thr.to_device(self.W.T.astype(dtype))
<         for ll in xrange(0,Lprod):
< 
<             if weight_flag == 0:
<                 pass
<             else:
<                 x[...,ll] = x[...,ll] * snc # scaling factors
<             
<             output_x=output_x*0.0
<         
<             output_x[crop_slice_ind(x[...,ll].shape)] = x[...,ll]
<             self.data_dev = self.thr.to_device(output_x.astype(dtype))
< 
<             if self.gpu_k_modulate()==0:
<                 pass
<             else:
<                 print('gpu_k_modulate error')
<                 break
<             x[...,ll]=self.data_dev.get()[crop_slice_ind(Nd)]
< 
<             if weight_flag == 0:
<                 pass
<             else: #weight_flag =1 scaling factors
<                 x[...,ll] = x[...,ll]*snc.conj() #% scaling factors
---
>                 
>     def forwardbackward(self,x):
401,407d354
<         if self.debug==0:
<             pass # turn off checker
<         else:
<             checker(x,Nd) # checking size of x divisible by Nd
<         return x    
<     def forwardbackward_gpu(self,x):
< #         print('inside forwardbackward_gpu ')
419a367
>             x = numpy.reshape(x,Nd+(1,),order='F')
423d370
<         x = numpy.reshape(x,Nd+(Lprod,),order='F')
427c374
<         x = self.Nd2KdWKd2Nd_gpu(x,0) #
---
>         Xk = self.Nd2Kd(x,0) #
429,432c376,378
< #         for ii in xrange(0,Lprod):
< # #             tmp_Xk = self.Nd2Kd_gpu(x[...,ii],0)
< #             Xk[...,ii] = st['q'][...,0]*Xk[...,ii]
< #             x[...,ii]= self.Kd2Nd_gpu(tmp_Xk,0)
---
>         for ii in range(0,Lprod):
>         
>             Xk[...,ii] = st['q'][...,0]*Xk[...,ii]
436c382
< #         x= self.Kd2Nd(Xk,0) #
---
>         x= self.Kd2Nd(Xk,0) #
439c385,386
<         return x 
---
>         return x    
> 
473c420
<                 for pp in xrange(2,numpy.size(u0.shape)):
---
>                 for pp in range(2,numpy.size(u0.shape)):
475d421
<                     self.st['mask2'] = appendmat(self.st['mask2'],u0.shape[pp] )
479,484c425
<             self.st['sensemap'] = self._make_sense(u0) # setting up sense map in st['sensemap']
<     
< #             for jj in xrange(0,self.st['sensemap'].shape[-1]):
< #                 matplotlib.pyplot.subplot(2,2,jj+1)
< #                 matplotlib.pyplot.imshow(self.st['sensemap'][...,jj].imag)
< #             matplotlib.pyplot.show()
---
>             self.st = self._make_sense(u0) # setting up sense map in st['sensemap']
490c431
<             self.LMBD=self.LMBD*1.0
---
>             self.LMBD=self.LMBD/1.0
497c438
< #         for jj in xrange(0,self.u.shape[-1]):
---
> #         for jj in range(0,self.u.shape[-1]):
536,545c477,479
<         if 'mask' in st: # condition in second step
<             if (numpy.shape(st['mask']) != image_dim) :
<                 st['mask'] = numpy.reshape(st['mask'],image_dim,order='F')
< #                 numpy.ones(image_dim,dtype=numpy.complex64)
<         else: # condition in first step
<             st['mask'] = numpy.ones(image_dim,dtype=numpy.complex64)
<             
<         if 'mask2' in st:
<             if numpy.shape(st['mask2']) != image_dim:
<                 st['mask2'] = numpy.reshape(st['mask2'],image_dim,order='F')
---
>         if 'mask' in st:
>             if numpy.shape(st['mask']) != image_dim:
>                 st['mask'] = numpy.ones(image_dim,dtype=numpy.complex64)
547c481
<             st['mask2'] = numpy.ones(image_dim,dtype=numpy.complex64)
---
>             st['mask'] = numpy.ones(image_dim,dtype=numpy.complex64)
577c511
<         u = self.backward(f)#*self.st['sensemap'].conj()#/(1e-10+self.st['sensemap'].conj())#st['sensemap'].conj()*(self.backward(f))
---
>         u = (self.backward(f))#*st['sensemap'].conj()
580c514
<         for jj in xrange(0,u.shape[-1]):
---
>         for jj in range(0,u.shape[-1]):
627,628c561,562
<         for outer in xrange(0,nBreg):
<             for inner in xrange(0,nInner):
---
>         for outer in numpy.arange(0,nBreg):
>             for inner in numpy.arange(0,nInner):
637c571
<                 for jj in xrange(0,u.shape[-1]):
---
>                 for jj in range(0,u.shape[-1]):
655c589
<                 xx=self._shrink( dd, bb, c/LMBD/(numpy.prod(st['Nd'])**(1.0/len(st['Nd']))))
---
>                 xx=self._shrink( dd, bb, c/LMBD/numpy.sqrt(numpy.prod(st['Nd'])))
662c596
<                 for jj in xrange(0,u.shape[-1]):
---
>                 for jj in range(0,u.shape[-1]):
670c604,622
< 
---
> #             err = (checkmax(tmpuf) -checkmax(u0) )/checkmax(u0)
> #             r = u0  - tmpuf
> # #         r = u0  - tmpuf
> #             p = r 
> # #         err = (checkmax(tmpuf)- checkmax(u0))/checkmax(u0) 
> #             err= numpy.abs(err)
> #             print('err',err,self.err)
> # #         if (err < self.err):
> # #             uf = uf+p*err*0.1            
> #             if err < self.err:
> #                 uf = uf + p*err*0.1*(outer+1)
> #                 self.err = err
> # 
> #                 u_k_1 = u
> #             else: 
> #                 err = self.err
> #                 print('no function')
> #                 u = u_k_1
> #             murf = uf 
676c628
<             u_stack[...,outer] = (u[...,0]*(self.st['sn']))
---
>             u_stack[...,outer] = (u[...,0]*(self.st['sn']**1))
679c631,632
<         for jj in xrange(0,u.shape[-1]):
---
>         for jj in range(0,u.shape[-1]):
> #             u[...,jj] = u[...,jj]*(self.st['sn']**1)# rescale the final image intensity
681,683c634,635
<             u[...,jj] = u[...,jj]*(self.st['sn'])*self.st['mask2'][...,jj]# rescale the final image intensity
< #         matplotlib.pyplot.imshow(self.st['mask2'][:,:,0].real)
< #         matplotlib.pyplot.show()
---
>             u[...,jj] = u[...,jj]*(self.st['sn'])*self.st['mask2']# rescale the final image intensity
> 
693c645
<         mylist = tuple(xrange(0,numpy.ndim(xx[0]))) 
---
>         mylist = tuple(numpy.arange(0,numpy.ndim(xx[0]))) 
711c663
< #        for jj in xrange(0,1):
---
> #        for jj in range(0,1):
726,748c678
< ###
< #         if self.cuda_flag == 1:
< #             tmpU=numpy.zeros(st['Kd'],dtype=u.dtype)
< #             self.W_dev = self.thr.to_device((uker[...,0]).astype(numpy.complex64))
< #             for pj in xrange(0,u.shape[-1]):
< #                 
< #                 tmpU=tmpU*0.0
< #             
< #                 tmpU[crop_slice_ind(st['Nd'])] = u[...,pj]
< #                 self.data_dev = self.thr.to_device(tmpU.astype(numpy.complex64))
< #                 
< #     #             self.myfft(self.data_dev,  self.data_dev,inverse=False)
< #     #             self.data_dev=self.W_dev*self.data_dev
< #     #             self.myfft(self.data_dev, self.data_dev,inverse=True)
< #                 if self.gpu_k_demodulate()==0:
< #                     pass
< #                 else:
< #                     print('gpu_k_modulate error')
< #                     break
< #                 u[...,pj]=self.data_dev.get()[crop_slice_ind(st['Nd'])]
< # #             u = U[[slice(0, st['Nd'][_ss]) for _ss in mylist[:-1]]]
< #             self.W_dev = self.thr.to_device(1.0/uker[...,0].astype(numpy.complex64))
< #         elif self.cuda_flag == 0:
---
> 
750,754c680,683
<         for pj in xrange(0,u.shape[-1]):
<                     
<             U[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) 
<             U[...,pj]=U[...,pj]/uker[...,pj] # deconvolution
<             U[...,pj]=self.emb_ifftn(U[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) 
---
>         
>         for pj in range(0,u.shape[-1]):
>             U[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) / uker[...,pj] # deconvolution
>             U[...,pj]=self.emb_ifftn(U[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
760c689
< #         for pp in xrange(0,3):
---
> #         for pp in range(0,3):
773,775c702,704
<         for pj in xrange(0,u.shape[-1]):
<             AU[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
<             AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) 
---
>         for pj in range(0,u.shape[-1]):
>             AU[...,pj]=self.emb_fftn(u[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
>             AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
784c713
<         for running_count in xrange(0,1):
---
>         for running_count in range(0,1):
792,794c721,723
<             for pj in xrange(0,u.shape[-1]):
<                 AU[...,pj]=self.emb_fftn(p[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
<                 AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], xrange(0,numpy.size(st['Kd']))) 
---
>             for pj in range(0,u.shape[-1]):
>                 AU[...,pj]=self.emb_fftn(p[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) * uker[...,pj] # deconvolution
>                 AU[...,pj]=self.emb_ifftn(AU[...,pj], st['Kd'], range(0,numpy.size(st['Kd']))) 
838c767
<         for jj in xrange(0,n_dims):
---
>         for jj in range(0,n_dims):
954c883
<   
---
>       
955a885,887
>         st=self.st
>         L=numpy.shape(u0)[-1]
>         u0dims= numpy.ndim(u0)
959c891,892
<             coil_sense = self._extract_svd(u0,L)
---
>             st['sensemap'] = self._extract_svd(u0,L)
>             print('run svd')
965,969c898,900
<             return coil_sense
<         except:
<             u0dims= numpy.ndim(u0)
<             beta=100
<             
---
>             return st
>         except: 
>             print('not runing svd')       
972c903
<                 dpss_rows = numpy.kaiser(rows, beta)     
---
>                 dpss_rows = numpy.kaiser(rows, 100)     
983c914
<                 dpss_cols = numpy.kaiser(cols, beta)            
---
>                 dpss_cols = numpy.kaiser(cols, 100)            
998c929
<                 dpss_zag = numpy.kaiser(zag, beta)            
---
>                 dpss_zag = numpy.kaiser(zag, 100)            
1013,1028c944,946
<     
<     #         rms = numpy.mean((coil_sense),-1)
<     
<     #         rms = rms/numpy.max()
<             coil_sense = numpy.copy(u0)
< 
<             rms=(numpy.mean( (coil_sense*coil_sense.conj()),-1))**0.5 # Root of sum square / OLD 
<               
<             for ll in xrange(0,L):
<     #             st['sensemap'][...,ll]=(u0[...,ll]+1e-16)/(rms+1e-16) # / OLD
<                 coil_sense[...,ll]=(coil_sense[...,ll]+1e-16)/(rms+1e-16) # need SVD  
<      
<      
< #             st['sensemap']=coil_sense
<              
<     #         st['sensemap']=numpy.empty(numpy.shape(u0),dtype=numpy.complex64)
---
>             
>             rms=numpy.sqrt(numpy.mean(u0*u0.conj(),-1)) # Root of sum square
>             st['sensemap']=numpy.ones(numpy.shape(u0),dtype=numpy.complex64)
1032c950
< #                 print('sensemap shape',st['sensemap'].shape, L)
---
>                 print('sensemap shape',st['sensemap'].shape, L)
1034,1038c952,956
<      
<      
<             for ll in xrange(0,L):
<     #             st['sensemap'][...,ll]=(u0[...,ll]+1e-16)/(rms+1e-16) # / OLD
<     #             st['sensemap'][...,ll]=coil_sense[...,ll] # need SVD  
---
>     
>             #    print('L',L)
>             #    print('rms',numpy.shape(rms))
>             for ll in numpy.arange(0,L):
>                 st['sensemap'][...,ll]=(u0[...,ll]+1e-16)/(rms+1e-16)
1042c960
<                     print('sensemap shape',coil_sense.shape, L)
---
>                     print('sensemap shape',st['sensemap'].shape, L)
1044c962
<                 coil_sense[...,ll]=  scipy.fftpack.fftshift(coil_sense[...,ll])
---
>                     
1050,1058c968,976
<                     coil_sense[...,ll] = pyfftw.interfaces.scipy_fftpack.fftn(coil_sense[...,ll])#, 
<     #                                                   coil_sense[...,ll].shape,
<     #                                                         range(0,numpy.ndim(coil_sense[...,ll])), 
<     #                                                         threads=self.threads) 
<                     coil_sense[...,ll] = coil_sense[...,ll] * dpss_fil
<                     coil_sense[...,ll] = pyfftw.interfaces.scipy_fftpack.ifftn(coil_sense[...,ll])#, 
<     #                                                   coil_sense[...,ll].shape,
<     #                                                         range(0,numpy.ndim(coil_sense[...,ll])), 
<     #                                                         threads=self.threads)                                                 
---
>                     st['sensemap'][...,ll] = pyfftw.interfaces.scipy_fftpack.fftn(st['sensemap'][...,ll], 
>                                                       st['sensemap'][...,ll].shape,
>                                                             range(0,numpy.ndim(st['sensemap'][...,ll])), 
>                                                             threads=self.threads) 
>                     st['sensemap'][...,ll] = st['sensemap'][...,ll] * dpss_fil
>                     st['sensemap'][...,ll] = pyfftw.interfaces.scipy_fftpack.ifftn(st['sensemap'][...,ll], 
>                                                       st['sensemap'][...,ll].shape,
>                                                             range(0,numpy.ndim(st['sensemap'][...,ll])), 
>                                                             threads=self.threads)                                                 
1060,1070c978,987
<                     coil_sense[...,ll] = scipy.fftpack.fftn(coil_sense[...,ll])#, 
<     #                                                   coil_sense[...,ll].shape,
<     #                                                         range(0,numpy.ndim(coil_sense[...,ll]))) 
<                     coil_sense[...,ll] = coil_sense[...,ll] * dpss_fil
<                     coil_sense[...,ll] = scipy.fftpack.ifftn(coil_sense[...,ll])#, 
<     #                                                   coil_sense[...,ll].shape,
<     #                                                         range(0,numpy.ndim(coil_sense[...,ll])))                             
<     #             coil_sense[...,ll]=scipy.fftpack.ifftn(scipy.fftpack.fftn(coil_sense[...,ll])*dpss_fil)
<     #         coil_sense = Normalize(coil_sense)
<                 coil_sense[...,ll]=  scipy.fftpack.ifftshift(coil_sense[...,ll])
<                 return coil_sense
---
>                     st['sensemap'][...,ll] = scipy.fftpack.fftn(st['sensemap'][...,ll], 
>                                                       st['sensemap'][...,ll].shape,
>                                                             range(0,numpy.ndim(st['sensemap'][...,ll]))) 
>                     st['sensemap'][...,ll] = st['sensemap'][...,ll] * dpss_fil
>                     st['sensemap'][...,ll] = scipy.fftpack.ifftn(st['sensemap'][...,ll], 
>                                                       st['sensemap'][...,ll].shape,
>                                                             range(0,numpy.ndim(st['sensemap'][...,ll])))                             
>     #             st['sensemap'][...,ll]=scipy.fftpack.ifftn(scipy.fftpack.fftn(st['sensemap'][...,ll])*dpss_fil)
>     #         st['sensemap'] = Normalize(st['sensemap'])
>             return st
1111c1028
< #             for dd in xrange(2,numpy.size(self.st['Kd'])):
---
> #             for dd in range(2,numpy.size(self.st['Kd'])):
1120c1037
< #             for dd in xrange(2,numpy.size(self.st['Kd'])):
---
> #             for dd in range(2,numpy.size(self.st['Kd'])):
1130,1134c1047,1051
<         out_dd = tuple(get_Diff(u,jj) for jj in xrange(0,len(dd)))
< #         out_dd = ()
< #         for jj in xrange(0,len(dd)) :
< #             out_dd = out_dd  + (get_Diff(u,jj),)
<         
---
> 
>         out_dd = ()
>         for jj in range(0,len(dd)) :
>             out_dd = out_dd  + (get_Diff(u,jj),)
> 
1141c1058
<         for pj in xrange(0,ndims):
---
>         for pj in range(0,ndims):
1146,1147d1062
<   
< 
1178c1093,1115
<         return st   
---
>         return st     
> 
> #     def _create_mask(self):
> #         st=self.st
> # 
> #         st['mask']=numpy.ones(st['Nd'],dtype=numpy.float64)
> #         n_dims= numpy.size(st['Nd'])
> #  
> #         sp_rat =0.0
> #         for di in range(0,n_dims):
> #             sp_rat = sp_rat + (st['Nd'][di]/2)**2
> #   
> #         x = numpy.ogrid[[slice(0, st['Nd'][_ss]) for _ss in range(0,n_dims)]]
> # 
> #         tmp = 0
> #         for di in range(0,n_dims):
> #             tmp = tmp + ( x[di] - st['Nd'][di]/2 )**2
> #         indx = tmp/sp_rat >=1.0/n_dims
> #             
> #         st['mask'][indx] =0.0       
> #          
> #   
> #         return st   
1183,1184c1120,1122
<         tmpuf=(self.forwardbackward(
<                         u*self.st['sensemap']))#*(self.st['sensemap'])
---
>         tmpuf=(
>                 self.forwardbackward(
>                         u*self.st['sensemap']))#*self.st['sensemap'].conj()
1261c1199
<     MyNufftObj.st['senseflag']=0
---
>     MyNufftObj.st['senseflag']=1
1269,1271c1207,1209
< #     f2=job_server.submit(MyNufftObj.inverse,(numpy.sqrt(K_data)*10+(0.0+0.1j), 1.0, 0.05, 0.01,3, 20),
< #                          modules = ('numpy','pyfftw','pynufft'),globals=globals())
< 
---
> # #     f2=job_server.submit(MyNufftObj.inverse,(numpy.sqrt(K_data)*10+(0.0+0.1j), 1.0, 0.05, 0.01,3, 20),
> # #                          modules = ('numpy','pyfftw','pynufft'),globals=globals())
> # 
1273,1275c1211,1213
< #     image2 = f2()
<     image1=MyNufftObj.inverse(K_data, 1.0, 0.1, 0.01,3,5)
< 
---
> # #     image2 = f2()
>     
>     image1 = MyNufftObj.inverse(K_data, 1.0, 0.1, 0.01,3, 5)
1317,1319c1255
<     image[128,128]= 1.0  
< #     import scipy.misc 
< #     image = scipy.misc.imresize(image,Nd)
---
>     image[128,128]= 1.0   
1329c1265
<     NufftObj.st['senseflag']=1
---
>     
1337,1338c1273,1274
< 
<     image_recon = NufftObj.inverse(data, 1.0, 0.4, 0.01,10, 10)
---
>     NufftObj.st['senseflag'] = 1
>     image_recon = NufftObj.inverse(data, 1.0, 0.05, 0.01,3, 20)
1389c1325
<     image_recon = NufftObj.inverse(data, 1.0, 1, 0.001,15,16)
---
>     image_recon = NufftObj.inverse(data, 1.0, 0.3, 0.001,15,16)
1514,1516c1450,1452
< #     test_1D()
< #     test_2D()
<     test_3D()
---
>     test_1D()
>     test_2D()
> #     test_3D()
1519,1520c1455,1456
< 
< #     cProfile.run('test_3D()')    
---
> #     cProfile.run('test_2D()') 
>     cProfile.run('test_3D()')    
