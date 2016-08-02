#!/usr/bin/python2.7
import GeRaw.pfileparser
import numpy
import matplotlib.pyplot
cm=matplotlib.cm.gray
import scipy.fftpack
import scipy.fftpack._fftpack
import GeRaw.cinePPGVD
import KspaceDesign.Sparse2D            
import CsSolver.Cine2DSolver
# import CsSolver.Cine3DSolver
#import CsSolver.ktfocuss
import CsTransform.pynufft

def fill_missing(x):
    '''
    fill the missing data of trig
    any intervals larger than the average+2*std will be added 
    additional points,
    such that the trigger points are evenly spaced 
    
    '''
    if numpy.size(x) == 1:
        print('scalar value is not permitted')
    if numpy.size(x.shape) >1:
        print('high dimensional input is not permitted')

    
    #===========================================================================
    # First calculate the gradient
    # compare each point to the average of gradient
    #===========================================================================

    grad_x=numpy.gradient(x) 
    c_mean=numpy.mean(grad_x)
    c_std=numpy.std(grad_x)
    
    x3=numpy.array(x[0])
    
    for jj in range(1,numpy.size(x)):
        x_diff= x[jj]-x[jj-1]
        
        if x_diff > c_mean+2*c_std: 
        # if distance is larger than mean+2*std
        
            inter_insert=numpy.round(x_diff/c_mean) 
            # guess how many missing data

            inter_insert=numpy.cast['int'](inter_insert) 
            # cast the data type as int
            
            delta_dist=x_diff*1.0/inter_insert # the distance to be added
            for pp in range(1,inter_insert):
                x3=numpy.append(x3,x[jj-1] + pp*delta_dist) 
            #endfor
            x3=numpy.append(x3,x[jj])
            
            # now update the average/std
            grad_x3=numpy.gradient(x3)
            c_mean=numpy.mean(grad_x3)
            c_std=numpy.std(grad_x3)
        else:
            x3=numpy.append(x3,x[jj])
        
    return x3

def mod_by_trigger(input_trigger,input_seq):
    '''
    find the nearest two triggers(just before and after ) of input_seq
    then measure the relative time 
    
    mod(input_seq[ii], (trigger2-trigger1)) /(trigger2-trigger1) 
    
    '''
    if numpy.size(input_trigger) == 1 or numpy.size(input_seq) == 1:
        print('scalar input_trigger/input_seq may not work!')
        
    if numpy.size(input_trigger.shape) > 1 or numpy.size(input_seq.shape) > 1:  
        print('high dimensional trigger/seq may not work!')
        
    if input_seq[-1] > input_trigger[-1]:
        print('input_seq may be logger than input_trigger, cannot work?')
        
        
    output_seq=[]  
     
    for ii in range(0,numpy.size(input_seq)):
        ref_array=numpy.abs(input_trigger-input_seq[ii])
        ind_ref_sorted=numpy.argsort(ref_array)
        
        # now pick two smallest values:
        (trig1,trig2)=input_trigger[ind_ref_sorted[0:2]]
        if trig2 < trig1:
            (trig2,trig1)=(trig1,trig2)
            
        #print(trig2,trig1,input_seq[ii])
        
        dist=trig2-trig1
        
        output_val= input_seq[ii] -trig1
        output_val= output_val*1.0/dist
        output_seq=numpy.append( output_seq,output_val)    
        
    
    return output_seq

'''
differential and its adjoint operator
'''


#    

def cinefoo():
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    #===============================================================================
    cineObj=GeRaw.pfileparser.cine2DBinary('fsecine_20121212.data',
            'variable_density_sample_with_offset.txt', 0,  (256,256,4,12,44))
#    cineObj=GeRaw.pfileparser.cine2DBinary2('cinefse_20120214_g.data',
#            'variable_density_sample_with_offset.txt', 0,  (256,4,12,44))
    
    print('cineObj.tse.max',numpy.max(numpy.abs(cineObj.tse[:])))
#    matplotlib.pyplot.imshow(numpy.sum(cineObj.tse,2).real,cmap=cm)
#    matplotlib.pyplot.show()
    print(cineObj.f.shape)
    nx=cineObj.f.shape[0]
    ny=cineObj.f.shape[1]
    ncoil=cineObj.f.shape[2]
    f=numpy.reshape(cineObj.f,[nx*ny,ncoil],order='F')
    print(f.shape)
    
    ppgtrig=numpy.loadtxt('PPGTrig_fse_cine_20121212_1212201210_10_00_930')
    
    ppgtrig=fill_missing(ppgtrig) 

    fsetime=numpy.loadtxt('fse.gradx.time')

    TR=2.5
    head_time=(30.0*1000.0/10.0)+ 3.0
    
    no_of_dummy_scan=2 # number of dummy scan
    no_of_effect_scan=44 # number of effective scan
    
    fsepoint=[]
    for pp in range(0,no_of_effect_scan):
        tmp_fsepoint=head_time+(pp+no_of_dummy_scan)*TR*1000/10.0+fsetime/10.0 # two dummy scans
        fsepoint=numpy.append(fsepoint,tmp_fsepoint)

#    matplotlib.pyplot.plot(fsepoint,'x:')
#    matplotlib.pyplot.show()
    # Now determine the pulse time
    
    time_table=mod_by_trigger(ppgtrig,fsepoint)
    
#    matplotlib.pyplot.plot(time_table,'x')
#    matplotlib.pyplot.show()
    
    
    vd_table=numpy.loadtxt('variable_density_sample_with_offset.txt')
    
    
    

    om = numpy.empty((528,2))
    
    print('time_table.shape',time_table.shape)
    print('vd_table.shape',vd_table.shape)
    
#    cineObj.f = numpy.fft.fftn(cineObj.f,axes=(0,))

    
    om[:,1] = numpy.array(time_table-0.5) # t-axis
  
    om[:,0] = (vd_table-128)/256.0
 
    om = om *numpy.pi * 2.0

    Nd=(256,16)
    Jd=(1,2)
    Kd=(256,32)
    nInner = 2
    nBreg  = 100
    LMBD=0.1
    gamma=0.01
    mu=1.0
#    tse=cineObj.tse
    
    #sensemap = make_sense(cineObj.tse,Nd[1])
        
    MyTransform = CsTransform.pyNufft_fast.pyNufft_fast( om, Nd,Kd,Jd)
    #===============================================================================
    kytspace=numpy.reshape(MyTransform.st['q'],MyTransform.st['Kd'],order='F')
   
#    kytspace=numpy.fft.fftn(kytspace,axes=(1,))   
#    matplotlib.pyplot.imshow(numpy.abs(kytspace))
#    
#    matplotlib.pyplot.show()

        
    '''
    Now initialize the first estimation
    '''
#    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver(MyTransform, 
#                    cineObj, mu, LMBD, gamma, nInner, nBreg)
    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver(MyTransform, 
                    cineObj, mu, LMBD, gamma, nInner, nBreg)    
    Cine2DSolver.solve()
    u_in_xyf = Cine2DSolver.u
    print('u_in_xyf',u_in_xyf.shape)
    numpy.save('u_in_xyf',u_in_xyf)
  
#    
def create_fsepoint(TR,no_of_dummy_scan , no_of_effect_scan,fsetime):
#     TR=2.5
    head_time=(30.0*1000.0/10.0)+ 0.0
    
#     no_of_dummy_scan=2 # number of dummy scan
#     no_of_effect_scan=88 # number of effective scan
    
    fsepoint=[]
    for pp in range(0,no_of_effect_scan):
        tmp_fsepoint=head_time+(pp+no_of_dummy_scan)*TR*1000/10.0+fsetime/10.0 # two dummy scans
        fsepoint=numpy.append(fsepoint,tmp_fsepoint)
    return fsepoint

def cine88():
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    #===============================================================================
#     cineObj=GeRaw.pfileparser.cine2DBinary('/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/cinefse_20130411c.data',
#             '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/variable_density_unsort_88.txt',
#              0, 
#               (512,256,4,12,88))

    data_name = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_sorted/cinefse_20130411e.data'
    vd_table_name='/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_sorted/variable_density_sort_88.txt'
    trigger_name = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_sorted/PPGTrig_fse_cine_20130411_0411201315_40_58_369'
    
#     data_name = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/cinefse_20130411c.data'
#     vd_table_name='/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/variable_density_unsort_88.txt'
#     trigger_name = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/PPGTrig_unsorted_20130411'


    grad_table = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/fgre.table' 
    cineObj=GeRaw.pfileparser.cine2DBinary(data_name,
            vd_table_name,
             0, 
              (512,256,4,12,88))  
    
    ppgtrig=numpy.loadtxt(trigger_name )
#     matplotlib.pyplot.plot(ppgtrig)
#     matplotlib.pyplot.show()    
    ppgtrig=fill_missing(ppgtrig)       

#    cineObj=GeRaw.pfileparser.cine2DBinary2('cinefse_20120214_g.data',
#            'variable_density_sample_with_offset.txt', 0,  (256,4,12,44))
    
    print('cineObj.tse.max',numpy.max(numpy.abs(cineObj.tse[:])))
#    matplotlib.pyplot.imshow(numpy.sum(cineObj.tse,2).real,cmap=cm)
#    matplotlib.pyplot.show()
    print(cineObj.f.shape)
    nx=cineObj.f.shape[0]
    ny=cineObj.f.shape[1]
    ncoil=cineObj.f.shape[2]
    f=numpy.reshape(cineObj.f,[nx*ny,ncoil],order='F')
    print(f.shape)
    


    fsetime=numpy.loadtxt(    grad_table)
    fsepoint = create_fsepoint(2.5, 2, 88,fsetime)
    print('fsepointshape',fsepoint.shape)
    print('fsepoint',fsepoint)
#     TR=2.5
#     head_time=(30.0*1000.0/10.0)+ 0.0
#     
#     no_of_dummy_scan=2 # number of dummy scan
#     no_of_effect_scan=88 # number of effective scan
#     
#     fsepoint=[]
#     for pp in range(0,no_of_effect_scan):
#         tmp_fsepoint=head_time+(pp+no_of_dummy_scan)*TR*1000/10.0+fsetime/10.0 # two dummy scans
#         fsepoint=numpy.append(fsepoint,tmp_fsepoint)

#    matplotlib.pyplot.plot(fsepoint,'x:')
#    matplotlib.pyplot.show()
    # Now determine the pulse time
    
    time_table=mod_by_trigger(ppgtrig,fsepoint)
    
#    matplotlib.pyplot.plot(time_table,'x')
#    matplotlib.pyplot.show()
    
    
    vd_table=cineObj.vd#numpy.loadtxt('variable_density_sample_with_offset.txt')
    
    
    

    om = numpy.empty((528*2,2))
    
    print('time_table.shape',time_table.shape)
    print('vd_table.shape',vd_table.shape)
    
#    cineObj.f = numpy.fft.fftn(cineObj.f,axes=(0,))

    
    om[:,1] = numpy.array(time_table-0.5) # t-axis
  
    om[:,0] = (vd_table-128)/256.0
 
    om = om *numpy.pi * 2.0

    Nd=(256,16)
    Jd=(1,2)
    Kd=(256,32)
    nInner = 5
    nBreg  = 1
    LMBD=0.1
    gamma=0.01
    mu=1.0
#    tse=cineObj.tse
    
    #sensemap = make_sense(cineObj.tse,Nd[1])
        
    MyTransform = CsTransform.pynufft.pynufft( om, Nd,Kd,Jd)
    #===============================================================================
    kytspace=numpy.reshape(MyTransform.st['q'],MyTransform.st['Kd'],order='F')
   
#    kytspace=numpy.fft.fftn(kytspace,axes=(1,))   
#    matplotlib.pyplot.imshow(numpy.abs(kytspace))
#    
#    matplotlib.pyplot.show()

        
    '''
    Now initialize the first estimation
    '''
#    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver(MyTransform, 
#                    cineObj, mu, LMBD, gamma, nInner, nBreg)
    #===========================================================================
    # Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver(MyTransform, 
    #                 cineObj, mu, LMBD, gamma, nInner, nBreg)    
    # Cine2DSolver.solve()
    #===========================================================================
    
    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver( om, Nd,Kd,Jd)   
#      
    u_in_xyf =Cine2DSolver.pseudoinverse(cineObj, mu, LMBD, gamma, nInner, nBreg)    
    
#     u_in_xyf = Cine2DSolver.u
    print('u_in_xyf',u_in_xyf.shape)
    numpy.save('u_in_xyf',u_in_xyf)
def obsolete_dante_fsepoint(TR,fsetime,seq_stamp_file , ppgdata_file):
#     TR=2.5
    ppg_data = numpy.loadtxt(ppgdata_file)
    
    physiological_length = numpy.size(ppg_data )
    print('physiological_length',physiological_length)
    seq_stamp = numpy.loadtxt(seq_stamp_file)*1000.0
    no_of_effect_scan = numpy.size(seq_stamp)
    print('no_of_effect_scan',no_of_effect_scan)    
    
    span_time = seq_stamp[-1]-seq_stamp[0] # the first scan to the last scan duration
    print('span_time=',span_time,'(ms)')
    head_to_end_time = span_time + TR*1000.0+200.0 # the first scan to the end of physiological data time,
    #200.0 ms is the fixed time lag between sequence and physiological data
    #says PPG is 200ms longer(more 20 points) than seq time
    print('head_to_end_time=',head_to_end_time,'(ms)')
    
    head_time= physiological_length*10.0 - head_to_end_time
    print('head_time=',head_time,'(ms)')
    
    fsepoint=[]
    for pp in range(0,no_of_effect_scan):
        tmp_fsepoint=head_time/10.0+(seq_stamp[pp]-seq_stamp[0])/10.0+fsetime/10.0 # two dummy scans
        fsepoint=numpy.append(fsepoint,tmp_fsepoint)
    print(fsepoint.shape)
    return fsepoint  
 
def dante_fsepoint(timestamp, ppgdata_file):
#     TR=2.5
    ppg_data = numpy.loadtxt(ppgdata_file)
    
    physiological_length = numpy.size(ppg_data )
    print('physiological_length',physiological_length)
    seq_stamp = numpy.loadtxt(timestamp)*1000.0
    no_of_effect_scan = numpy.size(seq_stamp)
    print('no_of_effect_scan',no_of_effect_scan)    
    
    span_time = seq_stamp[-1]-seq_stamp[0] # the first scan to the last scan duration
    print('span_time=',span_time,'(ms)')
    head_to_end_time = span_time + 225.0 # the first scan to the end of physiological data time,
    #200.0 ms is the fixed time lag between sequence and physiological data
    #says PPG is 200ms longer(more 20 points) than seq time
    print('head_to_end_time=',head_to_end_time,'(ms)')
    
    head_time= physiological_length*10.0 - head_to_end_time
    print('head_time=',head_time,'(ms)')
    
    fsepoint=[]
#     for pp in range(0,no_of_effect_scan):
    tmp_fsepoint=head_time/10.0+(seq_stamp[0:-1]-seq_stamp[0])/10.0 # two dummy scans
    fsepoint=numpy.append(fsepoint,tmp_fsepoint)
        
    print(fsepoint.shape)
    return fsepoint  
def dante_cine_fse():
#     
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    #===============================================================================
#     cineObj=GeRaw.pfileparser.cine2DBinary('/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/cinefse_20130411c.data',
#             '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/variable_density_unsort_88.txt',
#              0, 
#               (512,256,4,12,88))

    data_file = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/cine_fse_dante_20130425d_2nd.data'
    vd_table_file='/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_sorted/variable_density_sort_88.txt'
#     trigger_file = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/PPGTrig_fse_cine_dante_20130411_0425201315_11_52_169_1st'
#     ppgdata_file = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/PPGData_fse_cine_dante_20130411_0425201315_11_52_169_1st'
    trigger_file = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/PPGTrig_fse_cine_dante_20130411_0425201315_25_29_844_2nd'
    ppgdata_file = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/PPGData_fse_cine_dante_20130411_0425201315_25_29_844_2nd'    

    seq_stamp_file = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/seq_timestamp_2.txt'
    grad_table = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/fgre.table' 
    xres = 512
    yres = 256
    etl = 12
    necho=88
    ncoil =4
    cineObj=GeRaw.pfileparser.cine2DBinary(data_file,
            vd_table_file,
             0, 
              (xres,yres,ncoil,etl,necho))  
    
    ppgtrig=numpy.loadtxt(trigger_file )
#     matplotlib.pyplot.plot(ppgtrig)
#     matplotlib.pyplot.show()    
    ppgtrig=fill_missing(ppgtrig)       

#    cineObj=GeRaw.pfileparser.cine2DBinary2('cinefse_20120214_g.data',
#            'variable_density_sample_with_offset.txt', 0,  (256,4,12,44))
    
    print('cineObj.tse.max',numpy.max(numpy.abs(cineObj.tse[:])))
#    matplotlib.pyplot.imshow(numpy.sum(cineObj.tse,2).real,cmap=cm)
#    matplotlib.pyplot.show()
    print(cineObj.f.shape)
#     nx=cineObj.f.shape[0]
#     ny=cineObj.f.shape[1]
#     ncoil=cineObj.f.shape[2]
#     f=numpy.reshape(cineObj.f,[nx*ny,ncoil],order='F')
#     print(f.shape)

    fsetime=numpy.loadtxt(    grad_table) # exact ms of k-lines
   
    fsepoint = obsolete_dante_fsepoint(2.5, fsetime, seq_stamp_file, ppgdata_file)
    

    
    print('fsepointshape',fsepoint.shape)
    print('fsepoint',fsepoint-fsepoint[0])
#     TR=2.5
#     head_time=(30.0*1000.0/10.0)+ 0.0
#     
#     no_of_dummy_scan=2 # number of dummy scan
#     no_of_effect_scan=88 # number of effective scan
#     
#     fsepoint=[]
#     for pp in range(0,no_of_effect_scan):
#         tmp_fsepoint=head_time+(pp+no_of_dummy_scan)*TR*1000/10.0+fsetime/10.0 # two dummy scans
#         fsepoint=numpy.append(fsepoint,tmp_fsepoint)

#    matplotlib.pyplot.plot(fsepoint,'x:')
#    matplotlib.pyplot.show()
    # Now determine the pulse time
    
    time_table=mod_by_trigger(ppgtrig,fsepoint)
    
#    matplotlib.pyplot.plot(time_table,'x')
#    matplotlib.pyplot.show()
    
    
    vd_table=cineObj.vd#numpy.loadtxt('variable_density_sample_with_offset.txt')
    
    
    

    om = numpy.empty((etl*necho,2))
    
    print('time_table.shape',time_table.shape)
    print('vd_table.shape',vd_table.shape)
    
#    cineObj.f = numpy.fft.fftn(cineObj.f,axes=(0,))

    
    om[:,1] = numpy.array(time_table-0.5) # t-axis
  
    om[:,0] = (vd_table-yres/2)/yres
 
    om = om *numpy.pi * 2.0

    Nd=(256,16)
    Jd=(1,2)
    Kd=(256,32)
    nInner = 2
    nBreg  = 10
    LMBD=0.1
    gamma=0.01
    mu=1.0
#    tse=cineObj.tse
    
    #sensemap = make_sense(cineObj.tse,Nd[1])
        
#     MyTransform = CsTransform.pynufft.pynufft( om, Nd,Kd,Jd)
    #===============================================================================
#     kytspace=numpy.reshape(MyTransform.st['q'],MyTransform.st['Kd'],order='F')
   
#    kytspace=numpy.fft.fftn(kytspace,axes=(1,))   
#    matplotlib.pyplot.imshow(numpy.abs(kytspace))
#    
#    matplotlib.pyplot.show()

        
    '''
    Now initialize the first estimation
    '''
#    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver(MyTransform, 
#                    cineObj, mu, LMBD, gamma, nInner, nBreg)
#     Cine2DSolver = CsSolver.Cine3DSolver.Cine3DSolver(MyTransform, 
#                     cineObj, mu, LMBD, gamma, nInner, nBreg)    
#     Cine2DSolver.solve()
    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver( om, Nd,Kd,Jd)   
#   
    print('shape of cineObj.f',numpy.shape(cineObj.f))
####################################################################################
    try:
        import pp
        job_server = pp.Server()
        jobs = []
        ncpu = 16
        if numpy.mod(cineObj.dim_x,ncpu) == 0:
            pass
        else:
            print("ncpu must divid dim_x")
            raise
             
                
        import copy
         
        
        masterObj=()
        for zz in range(0,ncpu):
             
            tmpObj = copy.copy(cineObj)
            tmpObj.dim_x =tmpObj.dim_x/ncpu
             
    #         tmpObj.tse = scipy.fftpack.fftshift(tmpObj.tse,axes=(0,1,))
    #          
    #         tmpObj.tse=scipy.fftpack.fftn(tmpObj.tse,axes=(0,1,))
    #          
    #         tmpObj.tse=scipy.fftpack.fftshift(tmpObj.tse,axes=(0,1,))
             
            tmpObj.tse = tmpObj.tse[tmpObj.dim_x*zz:tmpObj.dim_x*(zz+1)]
             
    #         tmpObj.tse = scipy.fftpack.ifftshift(tmpObj.tse,axes=(0,1,))
    #          
    #         tmpObj.tse=scipy.fftpack.ifftn(tmpObj.tse,axes=(0,1,))
    #          
    #         tmpObj.tse=scipy.fftpack.ifftshift(tmpObj.tse,axes=(0,1,))
             
            tmpObj.f = scipy.fftpack.ifftshift(tmpObj.f,axes=(0,))
            tmpObj.f = scipy.fftpack.ifftn(tmpObj.f,axes=(0,))
            tmpObj.f = scipy.fftpack.ifftshift(tmpObj.f,axes=(0,))
            
            tmpObj.f = tmpObj.f[tmpObj.dim_x*zz:tmpObj.dim_x*(zz+1)]
            
            tmpObj.f = scipy.fftpack.fftshift(tmpObj.f,axes=(0,))
            tmpObj.f = scipy.fftpack.fftn(tmpObj.f,axes=(0,))
            tmpObj.f = scipy.fftpack.fftshift(tmpObj.f,axes=(0,))        
            
            
                     
            masterObj =masterObj +(tmpObj,)
             
             
            jobs.append(job_server.submit(Cine2DSolver.inverse, 
                             (masterObj[zz], mu, LMBD, gamma, nInner, nBreg),
                             modules = ('numpy','pynufft','scipy'),
                             globals = globals()))
            
        u_in_xyf  = numpy.empty((xres,yres,Nd[1],ncoil),dtype = numpy.complex128)
        for zz in range(0,ncpu):
            u_in_xyf[tmpObj.dim_x*zz:tmpObj.dim_x*(zz+1)] = jobs[zz]()#[::-1,::-1].T
    #############################################################################
    except:
        u_in_xyf = Cine2DSolver.pseudoinverse(cineObj, mu, LMBD, gamma, nInner, nBreg)    
#     u_in_xyf = Cine2DSolver.u
    print('u_in_xyf',u_in_xyf.shape)
    u_in_xyf=u_in_xyf/numpy.max(numpy.abs(u_in_xyf))
    numpy.save('/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/test/u_in_xyf',u_in_xyf)  
#        

def para_process_int16(dirname,
                       xres, # x dimension
                       yres, # number of phase encodings, y dim
                       etl, # number of echoes in each train
                       necho, # number of trains
                       ncoil, # number of coils
                       ncpu, # number of threads
                       yrecon # rFOV along phase encoding
                       ): # for mag3
#
    import glob
    
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    #===============================================================================
#     cineObj=GeRaw.pfileparser.cine2DBinary('/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/cinefse_20130411c.data',
#             '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130412/cinefse_20130411_unsorted/variable_density_unsort_88.txt',
#              0, 
#               (512,256,4,12,88))
    data_file = glob.glob(dirname+'dante_cine_*.data')[0]
    print(data_file)
#     vd_table_file=glob.glob(dirname+'*variable_density*')[0]
    vd_table_file='./sort88.txt'
    print(vd_table_file)
#     trigger_file = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/PPGTrig_fse_cine_dante_20130411_0425201315_11_52_169_1st'
#     ppgdata_file = '/home/sram/Cambridge_2012/DATA_MATLAB/Andrew/data_acq_20130425/PPGData_fse_cine_dante_20130411_0425201315_11_52_169_1st'
    trigger_file = glob.glob(dirname+'PPGTrig*')[0]
    print(trigger_file)
    ppgdata_file = glob.glob(dirname+'PPGData*')[0]
    print(ppgdata_file)
    seq_stamp_file = glob.glob(dirname+'cine_SLR_*')[0]
    print(seq_stamp_file)
#     xres = 512
#     yres = 256
#     etl = 12
#     necho=44
#     ncoil =4
    cineObj=GeRaw.pfileparser.cine2DBinary(data_file,
            vd_table_file,
             0, 
              (xres,yres,ncoil,etl,necho)) 
#     import cPickle
#     cinefilename = open(dirname+'rawdata.obj','w')
#     cPickle.dump(cineObj, cinefilename, 2)
    
    
    ppgtrig=numpy.loadtxt(trigger_file )
#     matplotlib.pyplot.plot(ppgtrig)
#     matplotlib.pyplot.show()    
    ppgtrig=fill_missing(ppgtrig)       

#    cineObj=GeRaw.pfileparser.cine2DBinary2('cinefse_20120214_g.data',
#            'variable_density_sample_with_offset.txt', 0,  (256,4,12,44))
    
    print('cineObj.tse.max',numpy.max(numpy.abs(cineObj.tse[:])))
#    matplotlib.pyplot.imshow(numpy.sum(cineObj.tse,2).real,cmap=cm)
#    matplotlib.pyplot.show()
    print(cineObj.f.shape)
    nx=cineObj.f.shape[0]
    ny=cineObj.f.shape[1]
    ncoil=cineObj.f.shape[2]
    f=numpy.reshape(cineObj.f,[nx*ny,ncoil],order='F')
    print(f.shape)

#     fsetime=numpy.loadtxt(    grad_table) # exact ms of k-lines
   
    fsepoint = dante_fsepoint( seq_stamp_file, ppgdata_file)
    

    
    print('fsepointshape',fsepoint.shape)
    print('fsepoint',fsepoint-fsepoint[0])
#     TR=2.5
#     head_time=(30.0*1000.0/10.0)+ 0.0
#     
#     no_of_dummy_scan=2 # number of dummy scan
#     no_of_effect_scan=88 # number of effective scan
#     
#     fsepoint=[]
#     for pp in range(0,no_of_effect_scan):
#         tmp_fsepoint=head_time+(pp+no_of_dummy_scan)*TR*1000/10.0+fsetime/10.0 # two dummy scans
#         fsepoint=numpy.append(fsepoint,tmp_fsepoint)

#    matplotlib.pyplot.plot(fsepoint,'x:')
#    matplotlib.pyplot.show()
    # Now determine the pulse time
    
    time_table=mod_by_trigger(ppgtrig,fsepoint)
    
#    matplotlib.pyplot.plot(time_table,'x')
#    matplotlib.pyplot.show()
    
    
    vd_table=cineObj.vd#numpy.loadtxt('variable_density_sample_with_offset.txt')
    
    
    

    om = numpy.empty((etl*necho,2))
    
    print('time_table.shape',time_table.shape)
    print('vd_table.shape',vd_table.shape)
    
#    cineObj.f = numpy.fft.fftn(cineObj.f,axes=(0,))

    
    om[:,1] = numpy.array(time_table-0.5) # t-axis
  
    om[:,0] = (vd_table[0:numpy.size(om)/2]-128)/256.0
 
    om = om *numpy.pi * 2.0

#     Nd=(yres,16)
    Nd=(yrecon ,16) # rFOV reconstruction
    Jd=(1,1)
    Kd=(yres*Jd[0], Nd[1]*2)
    nInner = 1
    nBreg  = 1
    LMBD=0.1
    gamma=0.01
    mu=1.0
#    tse=cineObj.tse
    
    #sensemap = make_sense(cineObj.tse,Nd[1])
        
#     MyTransform = CsTransform.pynufft.pynufft( om, Nd,Kd,Jd)
    #===============================================================================
#     kytspace=numpy.reshape(MyTransform.st['q'],MyTransform.st['Kd'],order='F')
   
#    kytspace=numpy.fft.fftn(kytspace,axes=(1,))   
#    matplotlib.pyplot.imshow(numpy.abs(kytspace))
#    
#    matplotlib.pyplot.show()

        
    '''
    Now initialize the first estimation
    '''
#    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver(MyTransform, 
#                    cineObj, mu, LMBD, gamma, nInner, nBreg)
#     Cine2DSolver = CsSolver.Cine3DSolver.Cine3DSolver(MyTransform, 
#                     cineObj, mu, LMBD, gamma, nInner, nBreg)    
#     Cine2DSolver.solve()
    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver( om, Nd,Kd,Jd)   
#      
####################################################################################
 
    import pp
#     ppservers = ("10.155.174.185","10.155.174.168",)
    job_server = pp.Server(ncpus = 2)#ppservers=ppservers, ncpus=2)
    jobs = []
#     ncpu = 24
#     if numpy.mod(cineObj.dim_x,ncpu) == 0:
#         pass
#     else:
#         print("ncpu must divid dim_x")
#         raise
         
#     try:   
    import copy
     
    
    masterObj=()
    for zz in range(0,ncpu):
         
        tmpObj = copy.copy(cineObj)
        if zz < ncpu -1:
            tmpObj.dim_x = numpy.ceil((1.0)*cineObj.dim_x/ncpu)
            tmpObj.tse = tmpObj.tse[tmpObj.dim_x*zz:tmpObj.dim_x*(zz+1)]
        else:
            tmpObj.dim_x = cineObj.dim_x - (numpy.ceil((1.0)*cineObj.dim_x/ncpu))*(ncpu - 1)
            tmpObj.tse = tmpObj.tse[- tmpObj.dim_x:]
#         tmpObj.tse = scipy.fftpack.fftshift(tmpObj.tse,axes=(0,1,))
#          
#         tmpObj.tse=scipy.fftpack.fftn(tmpObj.tse,axes=(0,1,))
#          
#         tmpObj.tse=scipy.fftpack.fftshift(tmpObj.tse,axes=(0,1,))
         
        
         
#         tmpObj.tse = scipy.fftpack.ifftshift(tmpObj.tse,axes=(0,1,))
#          
#         tmpObj.tse=scipy.fftpack.ifftn(tmpObj.tse,axes=(0,1,))
#          
#         tmpObj.tse=scipy.fftpack.ifftshift(tmpObj.tse,axes=(0,1,))
         
        tmpObj.f = scipy.fftpack.ifftshift(tmpObj.f,axes=(0,))
        tmpObj.f = scipy.fftpack.ifftn(tmpObj.f,axes=(0,))
        tmpObj.f = scipy.fftpack.ifftshift(tmpObj.f,axes=(0,))
        
        tmpObj.f = tmpObj.f[tmpObj.dim_x*zz:tmpObj.dim_x*(zz+1)]
        
        tmpObj.f = scipy.fftpack.fftshift(tmpObj.f,axes=(0,))
        tmpObj.f = scipy.fftpack.fftn(tmpObj.f,axes=(0,))
        tmpObj.f = scipy.fftpack.fftshift(tmpObj.f,axes=(0,))        
        
        
                 
        masterObj =masterObj +(tmpObj,)
         
         
        jobs.append(job_server.submit(Cine2DSolver.pseudoinverse, 
                         (masterObj[zz], mu, LMBD, gamma, nInner, nBreg),
                         modules = ('numpy','pynufft','scipy'),
                         globals = globals()))
        
    u_in_xyf  = numpy.empty((xres,Nd[0],Nd[1],ncoil),dtype = numpy.complex128)
    for zz in range(0,ncpu):
        if zz < ncpu -1:
            new_x = numpy.ceil((1.0)*cineObj.dim_x/ncpu)
            u_in_xyf[new_x*zz:new_x*(zz+1)] = jobs[zz]()
        else:
            new_x = cineObj.dim_x - (numpy.ceil((1.0)*cineObj.dim_x/ncpu))*(ncpu - 1)
            u_in_xyf[- new_x:]= jobs[zz]()
            
#############################################################################
    
    job_server.wait()
    job_server.destroy()
    
#     u_in_xyf = Cine2DSolver.pseudoinverse(cineObj, mu, LMBD, gamma, nInner, nBreg)    
#     u_in_xyf = Cine2DSolver.u
    print('u_in_xyf',u_in_xyf.shape)
    u_in_xyf=u_in_xyf/numpy.max(numpy.abs(u_in_xyf))
#     except:
#         u_in_xyf =Cine2DSolver.pseudoinverse(cineObj, mu, LMBD, gamma, nInner, nBreg)    
#     u_in_xyf = Cine2DSolver.u
    print('u_in_xyf',u_in_xyf.shape)
    
    import os
    try:
        os.rmdir(dirname+'thread_'+str(ncpu)+'_rfov_'+str(yrecon)+'/')
    except:
        print('cannot remove dir')
        
    os.mkdir(dirname+'thread_'+str(ncpu)+'_rfov_'+str(yrecon)+'/')
#     except:
#         os.mkdir(dirname+'thread_'+str(ncpu)+'_rfov_'+str(yrecon)+'/')
    
    numpy.save(dirname+'thread_'+str(ncpu)+'_rfov_'+str(yrecon)+'/'+'u_in_xyf',u_in_xyf) 
import copy
class atomic_computation2:
    def __init__(self,Cine2DSolver, cineObj):
        self.Cine2DSolver =  Cine2DSolver 
        self.cineObj =  cineObj 
        
        pass
    def create(self,zz,ncpu):
        tmpObj =  self.cineObj 
# #         dividing data along frequency axis
#         tmpObj.f = scipy.fftpack.ifftshift(tmpObj.f,axes=(0,))
#         tmpObj.f = scipy.fftpack.ifftn(tmpObj.f,axes=(0,))
#         tmpObj.f = scipy.fftpack.ifftshift(tmpObj.f,axes=(0,))
#         
#         if zz < ncpu -1:
#             tmpObj.dim_x = numpy.ceil((1.0)*self.cineObj.dim_x/ncpu)
#             tmpObj.tse = tmpObj.tse[tmpObj.dim_x*zz:tmpObj.dim_x*(zz+1)]
#             tmpObj.f = tmpObj.f[tmpObj.dim_x*zz:tmpObj.dim_x*(zz+1)]
#         else:
#             tmpObj.dim_x = self.cineObj.dim_x - (numpy.ceil((1.0)*self.cineObj.dim_x/ncpu))*(ncpu - 1)
#             tmpObj.tse = tmpObj.tse[- tmpObj.dim_x:]
#             tmpObj.f = tmpObj.f[- tmpObj.dim_x:]
 

        
        tmpObj.f = scipy.fftpack.fftshift(tmpObj.f,axes=(0,))
        tmpObj.f = scipy.fftpack.fftn(tmpObj.f,axes=(0,))
        tmpObj.f = scipy.fftpack.fftshift(tmpObj.f,axes=(0,))   
        print('data, type = ',tmpObj.f.dtype)
        self.cineObj = tmpObj
        pass
    def run(self,  zz, mu, LMBD, gamma, nInner, nBreg, ncpu):
        self.create(zz, ncpu)
        return self.Cine2DSolver.pseudoinverse(self.cineObj, mu, LMBD, gamma, nInner, nBreg)
import multiprocessing        
def wrapper_atomic_computation(my_atomic_calc, zz,  mu, LMBD, gamma, nInner, nBreg, ncpu):
    pid= os.getpid()
    pinned_core = zz  % multiprocessing.cpu_count()
    os.system("taskset -p -c %d %d" % ( pinned_core , pid))
    return my_atomic_calc.run( zz, mu, LMBD, gamma, nInner, nBreg, ncpu)
     

def para_process_int32(dirname,
                       xres, # x dimension
                       yres, # number of phase encodings, y dim
                       etl, # number of echoes in each train
                       necho, # number of trains
                       ncoil, # number of coils
                       ncpu, # number of threads
                       yrecon # rFOV along phase encoding
                       ): # for mag3
#
    import glob
    
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
 
    data_file = glob.glob(dirname+'dante_cine_*.data')[0]
    print(data_file)
 
    vd_table_file='./sort88.txt'
    print(vd_table_file)
    trigger_file = glob.glob(dirname+'PPGTrig*')[0]
    print(trigger_file)
    ppgdata_file = glob.glob(dirname+'PPGData*')[0]
    print(ppgdata_file)
    seq_stamp_file = glob.glob(dirname+'cine_SLR_*')[0]
    print(seq_stamp_file)

    cineObj=GeRaw.pfileparser.cine2DBinary_int32(data_file,
            vd_table_file,
             0, 
              (xres,yres,ncoil,etl,necho)) 
#     import cPickle
#     cinefilename = open(dirname+'rawdata.obj','w')
#     cPickle.dump(cineObj, cinefilename, 2)
    
    
    ppgtrig=numpy.loadtxt(trigger_file )
#     matplotlib.pyplot.plot(ppgtrig)
#     matplotlib.pyplot.show()    
    ppgtrig=fill_missing(ppgtrig)       

#    cineObj=GeRaw.pfileparser.cine2DBinary2('cinefse_20120214_g.data',
#            'variable_density_sample_with_offset.txt', 0,  (256,4,12,44))
    
    print('cineObj.tse.max',numpy.max(numpy.abs(cineObj.tse[:])))
#    matplotlib.pyplot.imshow(numpy.sum(cineObj.tse,2).real,cmap=cm)
#    matplotlib.pyplot.show()
    print(cineObj.f.shape)
    nx=cineObj.f.shape[0]
    ny=cineObj.f.shape[1]
    ncoil=cineObj.f.shape[2]
    f=numpy.reshape(cineObj.f,[nx*ny,ncoil],order='F')
    print(f.shape)

#     fsetime=numpy.loadtxt(    grad_table) # exact ms of k-lines
   
    fsepoint = dante_fsepoint( seq_stamp_file, ppgdata_file)
    

    
    print('fsepointshape',fsepoint.shape)
    print('fsepoint',fsepoint-fsepoint[0])
 
    
    time_table=mod_by_trigger(ppgtrig,fsepoint)
 
    
    
    vd_table=cineObj.vd#numpy.loadtxt('variable_density_sample_with_offset.txt')
    
    
    

    om = numpy.empty((etl*necho,2))
    
    print('time_table.shape',time_table.shape)
    print('vd_table.shape',vd_table.shape)
 

    
    om[:,1] = numpy.array(time_table-0.5) # t-axis
  
    om[:,0] = (vd_table[0:numpy.size(om)/2]-128)/256.0
 
    om = om *numpy.pi * 2.0

#     Nd=(yres,16)
    Nd=(yrecon ,16) # rFOV reconstruction
    Jd=(1,1)
    Kd=(yres*Jd[0], Nd[1]*2)
    nInner = 1
    nBreg  = 1
    LMBD=0.1
    gamma=0.01
    mu=1.0

####################################################################################
 
    import multiprocessing
#     import pp  
#    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    
    result = []  

#     ncpu = 24
#     if numpy.mod(cineObj.dim_x,ncpu) == 0:
#         pass
#     else:
#         print("ncpu must divid dim_x")
#         raise
         
#     try:   
    import copy
     
    
#     masterObj=()
 
    Cine2DSolver = CsSolver.Cine2DSolver.Cine2DSolver( om, Nd,Kd,Jd)   
    dim_x = cineObj.dim_x#   


    F = cineObj.f
    
    del cineObj.f
    del cineObj.k
    del cineObj.raw
#     del cineObj.pdf
    del cineObj.f_diff
    del cineObj.k_norm     
#         dividing data along frequency axis
    F = scipy.fftpack.ifftshift(F,axes=(0,))
    F = scipy.fftpack.ifftn(F,axes=(0,))
    F = scipy.fftpack.ifftshift(F,axes=(0,))


    
    my_atomic_calc = []
    import copy
    
    dim_1 = numpy.round((1.0)* dim_x/ncpu).astype(int)
    
    if dim_x > dim_1 * (ncpu - 1):
        dim_1 = numpy.floor((1.0)* dim_x/ncpu).astype(int)
        
    dim_2 =  dim_x - dim_1*(ncpu - 1)
    print('dim1', dim_1,dim_2)
    for zz in range(0,ncpu):

        tmpObj = copy.copy(cineObj)

        
        if zz < ncpu -1:
            tmpObj.dim_x = dim_1 
#             print('dim_x', tmpObj.)
            tmpObj.tse = tmpObj.tse[dim_1 * zz:dim_1*(zz+1)]
            tmpObj.f = F[   dim_1*zz:dim_1*(zz+1)]
        else:
            tmpObj.dim_x = dim_2
            tmpObj.tse = tmpObj.tse[ -dim_2:]
            tmpObj.f = F[ -dim_2:]

        
        my_atomic_calc.append( atomic_computation2(Cine2DSolver, tmpObj) )
        
    pool = multiprocessing.Pool(processes = ncpu)        
    import time
    t0 = time.time()    
    for zz in range(0,ncpu):
          
        result.append(pool.apply_async(wrapper_atomic_computation, 
                         (my_atomic_calc[zz], zz,  mu, LMBD, gamma, nInner, nBreg, ncpu) ) )
#                          modules = ('numpy','pynufft','scipy'),
#                          globals = globals()))
        
    u_in_xyf  = numpy.empty((xres,Nd[0],Nd[1],ncoil),dtype = numpy.complex64)
#         if zz < ncpu -1:
#             new_x = my_atomic_calc[zz].cineObj.dim_x#numpy.ceil((1.0)*dim_x/ncpu).astype(int)
#             u_in_xyf[new_x*zz:new_x*(zz+1)] = result[zz].get()
#         else:
#             new_x = my_atomic_calc[zz].cineObj.dim_x#dim_x - (numpy.ceil((1.0)*dim_x/ncpu))*(ncpu - 1)
#             u_in_xyf[- new_x:]= result[zz].get()
#             
    for zz in range(0,ncpu):
        if zz < ncpu -1:
#             new_x = my_atomic_calc[zz].cineObj.dim_x# numpy.ceil((1.0)*dim_x/ncpu).astype(int)
             
            u_in_xyf[dim_1*zz:  dim_1*(zz+1)] = result[zz].get()
             
        else:
#             new_x = my_atomic_calc[zz].cineObj.dim_x #dim_x - (numpy.ceil((1.0)*dim_x/ncpu))*(ncpu - 1)
             
            u_in_xyf[- dim_2:]= result[zz].get()
#############################################################################
    
#     job_server.wait()
#     job_server.destroy()
    
    pool.close()
    pool.join()     
    import platform
    
    text_file = open("Recon_time_record" + platform.platform() + ".txt", "a")
    text_file.write("%s" % str(yrecon) )
    text_file.write("\t\t\t" )    
    text_file.write("%s" % str(ncpu) )
    text_file.write("\t\t\t" )
    text_file.write("%s" % str(time.time() - t0) )
    text_file.write("\t\t\t" )    
    text_file.write("%s" % time.asctime() )
    text_file.write("\n"  )    
    text_file.close()
#     u_in_xyf = Cine2DSolver.pseudoinverse(cineObj, mu, LMBD, gamma, nInner, nBreg)    
#     u_in_xyf = Cine2DSolver.u
    print('u_in_xyf',u_in_xyf.shape)
    u_in_xyf=u_in_xyf/numpy.max(numpy.abs(u_in_xyf))
#     except:
#         u_in_xyf =Cine2DSolver.pseudoinverse(cineObj, mu, LMBD, gamma, nInner, nBreg)    
#     u_in_xyf = Cine2DSolver.u
    print('u_in_xyf',u_in_xyf.shape)
    
    import os
    try:
        os.rmdir(dirname+'thread_'+str(ncpu)+'_rfov_'+str(yrecon)+'/')
    except:
        print('cannot remove dir')
        
    os.mkdir(dirname+'thread_'+str(ncpu)+'_rfov_'+str(yrecon)+'/')
#     except:
#         os.mkdir(dirname+'thread_'+str(ncpu)+'_rfov_'+str(yrecon)+'/')
    
    numpy.save(dirname+'thread_'+str(ncpu)+'_rfov_'+str(yrecon)+'/'+'u_in_xyf',u_in_xyf) 
         
def showfoo(save_folder_name,low,up):
#     import glob
#     print(glob.glob(save_folder_name+'*.png'))
    u_in_xyf=numpy.load(save_folder_name+'u_in_xyf.npy')
#     for jj in range(0,u_in_xyf.shape[2]):
#         u_in_xyf[...,jj,:]=u_in_xyf[...,jj,:]*numpy.cos((jj - u_in_xyf.shape[2]/2 )/3)
    
    data_in_xy_t =  u_in_xyf#/numpy.max(numpy.abs(u_in_xyf[:]))
    data_in_xy_t[:,:,:,0] = low_pass_filter(data_in_xy_t[:,:,:,0])
    #data_in_xy_t = scipy.fftpack.fftn(u_in_xyf,axes = (2,))
    print('data_in_xy,shape',data_in_xy_t.shape)
    print('u_in_xyf,shape',u_in_xyf.shape)
    cmap=matplotlib.cm.gray
    interpl='nearest'
    norm=matplotlib.colors.Normalize(vmin=low, vmax=up) 
    import scipy.misc
    import scipy.io
    cine2mat = numpy.zeros((512,512,u_in_xyf.shape[2]))
        
    for jj in range(0,u_in_xyf.shape[2]):
#        matplotlib.pyplot.subplot(4,4,jj)
        matplotlib.pyplot.figure(figsize=(20,12))
        #matplotlib.pyplot.imshow(numpy.abs(data_in_xy_t[...,jj,0]),cmap = cmap, interpolation = interpl,norm=norm )
        ccc = (data_in_xy_t[...,jj,0]).T
        if jj == 0:
            base_phase = ccc
        
#         ccc= scipy.fftpack.fft2(ccc)
        ddd = numpy.zeros((512,512))
#         ddd[:,:128]=ccc[:,:128]
#         ddd[:,-128:]=ccc[:,-128:]
# #         ddd[-128:,:128]=ccc[-128:,:128]
# #         ddd[-128:,-128:]=ccc[-128:,-128:]
#         ddd = scipy.fftpack.ifft2(ddd)
        ddd = scipy.misc.imresize(numpy.abs(ccc),(512,512),'bicubic')
        cine2mat[:,:,jj] = ddd
#         ddd.imag = scipy.misc.imresize(ccc.imag,(512,512))
#         if jj == 0: # normalize the image intensity
#             sum_value = numpy.sum(ddd[:])
#         else:
#             tmp_sum_value = numpy.sum(ddd[:])
#             ddd = sum_value*ddd/tmp_sum_value
        #ccc = scipy.fftpack.fftn(ccc,(512,512),axes=(0,1))
        matplotlib.pyplot.imshow(numpy.abs(ddd.T),cmap = cmap, interpolation = interpl, norm =norm 
                                        )
        #fname = '_tmp%03d.png'%jj
        fname ='cine_image_data_%03d.png'%(jj,)
        matplotlib.pyplot.savefig(save_folder_name+fname,#+str(jj)+'.png',
                                  format='png',dpi=100, 
                                  transparent=True, bbox_inches='tight', 
                                  pad_inches=0)
        #matplotlib.pyplot.show()
#    scipy.io.savemat( save_folder_name+'cine_file.mat', mdict={'cine2mat':cine2mat}, oned_as={'column'})
def zero_padding(input_x,size):
#     out_image = input_x
    xres,yres = numpy.shape(input_x)
    if xres > size:
        print('xres > size! Increase size')
    if yres > size:
        print('yres > size! Increase size')
          
    out_x = numpy.zeros((size,size),dtype = numpy.complex)
     
    input_k= (scipy.fftpack.fft2((input_x)))
     
#     matplotlib.pyplot.imshow((input_k.real), cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=200.0))
#     matplotlib.pyplot.show()     
#     out_x[0:xres/2, 0:yres/2] = input_k[0:xres/2, 0:yres/2] 
#     out_x[-xres/2:, 0:yres/2] = input_k[-xres/2, 0:yres/2] 
#     out_x[0:xres/2, -yres/2:] = input_k[0:xres/2, -yres/2:] 
#     out_x[-xres/2:, -yres/2:] = input_k[-xres/2:, -yres/2:]    
    out_x[0:xres/2, 0:yres/2] = input_k[0:xres/2, 0:yres/2] 
    out_x[-xres/2:, 0:yres/2] = input_k[-xres/2:, 0:yres/2]  
    out_x[0:xres/2, -yres/2:] = input_k[0:xres/2, -yres/2:] 
    out_x[-xres/2:, -yres/2:] = input_k[-xres/2:, -yres/2:]  
 
    out_image= scipy.fftpack.ifft2(out_x)*(size*1.0/xres)*(size*1.0/yres)
#     matplotlib.pyplot.imshow(numpy.abs(out_image), cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=100.0))
#     matplotlib.pyplot.show()        
#     matplotlib.pyplot.imshow(numpy.abs(out_x), cmap=cmap)
#     matplotlib.pyplot.show()
    return out_image    
def zero_padding_rfov(input_x,(size_x, size_y)):
#     out_image = input_x
    xres,yres = numpy.shape(input_x)
#     if xres > size:
#         print('xres > size! Increase size')
#     if yres > size:
#         print('yres > size! Increase size')
          
    out_x = numpy.zeros((size_x,size_y),dtype = numpy.complex)
     
    input_k= (scipy.fftpack.fft2((input_x)))
     
#     matplotlib.pyplot.imshow((input_k.real), cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=200.0))
#     matplotlib.pyplot.show()     
#     out_x[0:xres/2, 0:yres/2] = input_k[0:xres/2, 0:yres/2] 
#     out_x[-xres/2:, 0:yres/2] = input_k[-xres/2, 0:yres/2] 
#     out_x[0:xres/2, -yres/2:] = input_k[0:xres/2, -yres/2:] 
#     out_x[-xres/2:, -yres/2:] = input_k[-xres/2:, -yres/2:]    
    out_x[0:xres/2, 0:yres/2] = input_k[0:xres/2, 0:yres/2] 
    out_x[-xres/2:, 0:yres/2] = input_k[-xres/2:, 0:yres/2]  
    out_x[0:xres/2, -yres/2:] = input_k[0:xres/2, -yres/2:] 
    out_x[-xres/2:, -yres/2:] = input_k[-xres/2:, -yres/2:]  
 
    out_image= scipy.fftpack.ifft2(out_x)*(size_x*1.0/xres)*(size_y*1.0/yres)
#     matplotlib.pyplot.imshow(numpy.abs(out_image), cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=100.0))
#     matplotlib.pyplot.show()        
#     matplotlib.pyplot.imshow(numpy.abs(out_x), cmap=cmap)
#     matplotlib.pyplot.show()
    return out_image   
def low_pass_phase(input_image):
    img_shape = numpy.shape(input_image) # 2D
    print(img_shape)
    filter_k = numpy.zeros(img_shape)
    for q in xrange(0,img_shape[0]):
        for w in xrange(0,img_shape[1]):
            D2 = (q - img_shape[0]/2.0)**2 +(w - img_shape[1]/2.0)**2
            filter_k[q,w] = numpy.exp(-D2/(2.0*6**2))
            
    kimg = scipy.fftpack.fftshift(scipy.fftpack.fft2(scipy.fftpack.fftshift(input_image)))
    low_pass_img =  scipy.fftpack.fftshift(scipy.fftpack.ifft2(scipy.fftpack.fftshift(kimg*filter_k)))
#     low_pass_img = low_pass_img/numpy.max(numpy.abs(low_pass_img))
#     intensity_img = numpy.abs(low_pass_img)
    low_pass_phase =low_pass_img/numpy.abs(low_pass_img)
    
    output_image = input_image/low_pass_phase
#     intensity_img = numpy.abs(low_pass_img)
#     intensity_img = (intensity_img+numpy.mean(numpy.abs(intensity_img))*0.1)/(intensity_img**2 + numpy.mean(numpy.abs(intensity_img))*0.1)
#     intensity_img  =intensity_img/numpy.max(numpy.abs(intensity_img))
#     output_image = output_image*intensity_img
#     matplotlib.pyplot.imshow(numpy.real(output_image))
#     matplotlib.pyplot.show()
        
    return output_image
def showfoo_final(save_folder_name,low,up,TRANS):
#     import glob
#     print(glob.glob(save_folder_name+'*.png'))
    u_in_xyf=numpy.load(save_folder_name+'u_in_xyf.npy')
#     for jj in range(0,u_in_xyf.shape[2]):
#         u_in_xyf[...,jj,:]=u_in_xyf[...,jj,:]*numpy.cos((jj - u_in_xyf.shape[2]/2 )/3)
#     min_value = numpy.min(numpy.abs(u_in_xyf[:]))
#     max_value = numpy.max(numpy.abs(u_in_xyf[:]))
    data_in_xy_t =   u_in_xyf# - min_value)/(max_value - min_value)
    data_in_xy_t[:,:,:,0] = low_pass_filter(data_in_xy_t[:,:,:,0])
    for jj in range(0,u_in_xyf.shape[2]):
        data_in_xy_t[:,:,jj,0] = low_pass_phase(data_in_xy_t[:,:,jj,0])
    #data_in_xy_t = scipy.fftpack.fftn(u_in_xyf,axes = (2,))
    print('data_in_xy,shape',data_in_xy_t.shape)
    print('u_in_xyf,shape',u_in_xyf.shape)
    cmap=matplotlib.cm.gray
    interpl='nearest'
    norm=matplotlib.colors.Normalize(vmin=low, vmax=up) 
    import scipy.misc
    import scipy.io
    cine2mat = numpy.zeros((512,512,u_in_xyf.shape[2]),dtype = numpy.complex128)
        
    for jj in range(0,u_in_xyf.shape[2]):
#        matplotlib.pyplot.subplot(4,4,jj)
#         matplotlib.pyplot.figure(figsize=(20,12))
#         matplotlib.pyplot.imshow(numpy.imag(data_in_xy_t[...,jj,0]),cmap = cmap, interpolation = interpl )
#         matplotlib.pyplot.show()
#         cnt = data_in_xy_t[ 255:257 ,127:129,jj,0]
#         cnt = numpy.mean(cnt)
#         if numpy.abs(cnt) < 1e-7:
#             cnt = 1.0
#         else:
#             pass
#          
#         cnt_ang = cnt/numpy.abs(cnt)
        
        ccc = (data_in_xy_t[...,jj,0]).T
        if jj == 0:
            base_phase = ccc
        
#         ccc= scipy.fftpack.fft2(ccc)
        ddd = numpy.zeros((512,512))
#         ddd[:,:128]=ccc[:,:128]
#         ddd[:,-128:]=ccc[:,-128:]
# #         ddd[-128:,:128]=ccc[-128:,:128]
# #         ddd[-128:,-128:]=ccc[-128:,-128:]
#         ddd = scipy.fftpack.ifft2(ddd)
#         ccc = abs(ccc)
#         ddd = scipy.misc.imresize(numpy.real(ccc),(512,512),'bicubic') + 1.0j*scipy.misc.imresize(numpy.imag(ccc),(512,512),'bicubic')
#         ddd = scipy.misc.imresize(numpy.abs(ccc),(512,512),'bicubic')# + 1.0j*scipy.misc.imresize(numpy.imag(ccc),(512,512),'bicubic')
        ddd =  zero_padding(ccc,512)
        
        min_value = numpy.min(numpy.abs(ddd[:]))
        max_value = numpy.max(numpy.abs(ddd[:]))
        ddd =  5.*(ddd - min_value)/(max_value - min_value)
        if TRANS == 1:
            DDD =  (ddd.T).real
        else:
            DDD =  (ddd[:,::-1]).real 
                 
        cine2mat[:,:,jj] =  DDD
#         ddd.imag = scipy.misc.imresize(ccc.imag,(512,512))
#         if jj == 0: # normalize the image intensity
#             sum_value = numpy.sum(ddd[:])
#         else:
#             tmp_sum_value = numpy.sum(ddd[:])
#             ddd = sum_value*ddd/tmp_sum_value
        #ccc = scipy.fftpack.fftn(ccc,(512,512),axes=(0,1))
#         if TRANS == 1:
#             matplotlib.pyplot.imshow( (ddd.T).real,cmap = cmap, interpolation = interpl, norm =norm 
#                                         )
#         else:
        matplotlib.pyplot.imshow( numpy.real(DDD),cmap = cmap, interpolation = interpl, norm =norm 
                                        )
        #fname = '_tmp%03d.png'%jj
        fname ='cine_image_data_%03d.png'%(jj,)
        matplotlib.pyplot.savefig(save_folder_name+fname,#+str(jj)+'.png',
                                  format='png',dpi=100, 
                                  transparent=True, bbox_inches='tight', 
                                  pad_inches=0)
#         matplotlib.pyplot.show()
#    scipy.io.savemat( save_folder_name+'cine_file.mat', mdict={'cine2mat':cine2mat}, oned_as={'column'})   
def showfoo_rfov(save_folder_name,low,up,TRANS, yrecon):
#     import glob
#     print(glob.glob(save_folder_name+'*.png'))
    u_in_xyf=numpy.load(save_folder_name+'u_in_xyf.npy')
#     for jj in range(0,u_in_xyf.shape[2]):
#         u_in_xyf[...,jj,:]=u_in_xyf[...,jj,:]*numpy.cos((jj - u_in_xyf.shape[2]/2 )/3)
#     min_value = numpy.min(numpy.abs(u_in_xyf[:]))
#     max_value = numpy.max(numpy.abs(u_in_xyf[:]))
    data_in_xy_t =   u_in_xyf# - min_value)/(max_value - min_value)
    data_in_xy_t[:,:,:,0] = low_pass_filter(data_in_xy_t[:,:,:,0])
    for jj in range(0,u_in_xyf.shape[2]):
        data_in_xy_t[:,:,jj,0] = low_pass_phase(data_in_xy_t[:,:,jj,0])
    #data_in_xy_t = scipy.fftpack.fftn(u_in_xyf,axes = (2,))
    print('data_in_xy,shape',data_in_xy_t.shape)
    print('u_in_xyf,shape',u_in_xyf.shape)
    cmap=matplotlib.cm.gray
    interpl='nearest'
    norm=matplotlib.colors.Normalize(vmin=low, vmax=up) 
    import scipy.misc
    import scipy.io
    if TRANS == 0: 
        cine2mat = numpy.zeros((yrecon*2,512,u_in_xyf.shape[2]),dtype = numpy.complex128)
    else: 
        cine2mat = numpy.zeros((512,yrecon*2,u_in_xyf.shape[2]),dtype = numpy.complex128)
   
    min_value = numpy.min(numpy.abs(u_in_xyf[:]))
    max_value = numpy.max(numpy.abs(u_in_xyf[:]))   
         
    for jj in range(0,u_in_xyf.shape[2]):
#        matplotlib.pyplot.subplot(4,4,jj)
#         matplotlib.pyplot.figure(figsize=(20,12))
#         matplotlib.pyplot.imshow(numpy.imag(data_in_xy_t[...,jj,0]),cmap = cmap, interpolation = interpl )
#         matplotlib.pyplot.show()
#         cnt = data_in_xy_t[ 255:257 ,127:129,jj,0]
#         cnt = numpy.mean(cnt)
#         if numpy.abs(cnt) < 1e-7:
#             cnt = 1.0
#         else:
#             pass
#          
#         cnt_ang = cnt/numpy.abs(cnt)
        
        ccc = (data_in_xy_t[...,jj,0]).T
        if jj == 0:
            base_phase = ccc
        
#         ccc= scipy.fftpack.fft2(ccc)
        ddd = numpy.zeros((yrecon*2,512),dtype= numpy.complex64)
#         ddd[:,:128]=ccc[:,:128]
#         ddd[:,-128:]=ccc[:,-128:]
# #         ddd[-128:,:128]=ccc[-128:,:128]
# #         ddd[-128:,-128:]=ccc[-128:,-128:]
#         ddd = scipy.fftpack.ifft2(ddd)
#         ccc = abs(ccc)
#         ddd = scipy.misc.imresize(numpy.real(ccc),(512,512),'bicubic') + 1.0j*scipy.misc.imresize(numpy.imag(ccc),(512,512),'bicubic')
#         ddd = scipy.misc.imresize(numpy.abs(ccc),(512,512),'bicubic')# + 1.0j*scipy.misc.imresize(numpy.imag(ccc),(512,512),'bicubic')
        ddd = zero_padding_rfov(ccc,( yrecon*2,512) )
#         ddd.real = scipy.misc.imresize(ccc.real,(yrecon*2,512))
#         ddd.imag = scipy.misc.imresize(ccc.imag,(yrecon*2,512))
#         print('size of ddd',numpy.shape(ddd))

        ddd =  5.*(ddd - min_value)/(max_value - min_value)
        if TRANS == 1:
            DDD =  (ddd.T).real
        else:
            DDD =  (ddd[:,::-1]).real 
#         print('size of DDD',numpy.shape(DDD))
        cine2mat[:,:,jj] =  DDD
#         ddd.real = scipy.misc.imresize(ccc.real,(512,yrecon*2))
#         ddd.imag = scipy.misc.imresize(ccc.imag,(512,yrecon*2))
#         if jj == 0: # normalize the image intensity
#             sum_value = numpy.sum(ddd[:])
#         else:
#             tmp_sum_value = numpy.sum(ddd[:])
#             ddd = sum_value*ddd/tmp_sum_value
        #ccc = scipy.fftpack.fftn(ccc,(512,512),axes=(0,1))
#         if TRANS == 1:
#             matplotlib.pyplot.imshow( (ddd.T).real,cmap = cmap, interpolation = interpl, norm =norm 
#                                         )
#         else:
        matplotlib.pyplot.imshow( numpy.real(DDD),cmap = cmap, interpolation = interpl, norm =norm 
                                        )
        #fname = '_tmp%03d.png'%jj
        fname ='cine_image_data_%03d.png'%(jj,)
        matplotlib.pyplot.savefig(save_folder_name+fname,#+str(jj)+'.png',
                                  format='png',dpi=100, 
                                  transparent=True, bbox_inches='tight', 
                                  pad_inches=0)
#     os.system('convert ' + save_folder_name + '*.png + save_folder_name' + 'cine.gif')
#     os.chdir(save_folder_name)
#     os.system('convert  ' +save_folder_name+'*.png '+save_folder_name+'cine.gif')
#         matplotlib.pyplot.show()
#    scipy.io.savemat( save_folder_name+'cine_file.mat', mdict={'cine2mat':cine2mat}, oned_as={'column'})
def do_convert(save_folder_name):
    os.system('convert  ' +save_folder_name+'*.png '+save_folder_name+'cine.gif')
def low_pass_filter(data_in):
    static_image = numpy.mean(data_in,2)
    spatial_filter = numpy.abs( numpy.mean(data_in,2) )
   
         
    k_filter = numpy.fft.fftshift(spatial_filter)
    k_filter = numpy.fft.fft2( k_filter )
    k_filter = numpy.fft.fftshift( k_filter )
#     k_filter = abs(k_filter)
    k_filter = k_filter/numpy.max(k_filter)
    filter_shape = numpy.shape(spatial_filter)
    for q in xrange(0,filter_shape[0]):
        for w in xrange(0,filter_shape[1]):
            D2 = (q - filter_shape[0]/2.0)**2 +(w - filter_shape[1]/2.0)**2
            k_filter[q,w] = k_filter[q,w]* numpy.exp(-D2/(2.0*8.0**2))
            
    spatial_filter = numpy.fft.fftshift( scipy.fftpack.ifft2(numpy.fft.fftshift(k_filter)))
    spatial_filter = spatial_filter/numpy.max(numpy.abs(spatial_filter))
    
    spatial_filter = spatial_filter/(spatial_filter +3e-1)
    
#     matplotlib.pyplot.imshow(numpy.abs( spatial_filter))
#     matplotlib.pyplot.show()    

    
    Nx=numpy.shape(data_in)[0]
    Ny=numpy.shape(data_in)[1]
    Nt=numpy.shape(data_in)[2]
#     import scipy.ndimage
#     data_out = scipy.ndimage.gaussian_filter(data_out, (256,9999,16,999),0) 
    data_out = numpy.fft.fftshift(data_in,axes=(2, 0,1))
    data_out = numpy.fft.fftn(data_out,axes=(2, 0,1))
    data_out = numpy.fft.fftshift(data_out,axes=(2, 0,1))
#        


#     data_out[:,:,Nt/2] = static_image
#     import matplotlib.pyplot
    norm=matplotlib.colors.Normalize(vmin=0, vmax=64) 
#     matplotlib.pyplot.imshow(numpy.abs(data_out[256,:,:]),norm =norm)
#     matplotlib.pyplot.show()

#     print(data_out.shape)
#     low_pass = numpy.ones(numpy.shape(data_out)[0],numpy.shape(data_out)[2])
#      
#        
#     tmp_fil = spatial_filter/numpy.max(numpy.abs(spatial_filter))

#     data_out[Nx/2-1:Nx/2+1, Ny/2-1:Ny/2+1, :]=0
#     data_out[Nx/2, Ny/2, :]=0
#     data_out[...,Nt/2] = 0
    
    
#     for cc in range(0,Nt):
#         data_out[...,cc]=data_out[...,cc]*(1-k_filter)
#             data_out[...,cc]=data_out[...,cc]*(1-0.4*tmp_fil)#*(
#                             numpy.exp(-( (cc*1.0-Nt/2.0)/(Nt/1.0) )**2))  
       
     
 
#     for xx in xrange(0,Nx):
#         for yy in xrange(0,Ny):
#             for tt in xrange(0,Nt):
#                 D3 = ((xx-Nx/2.0)/( (Nx/2.0)**2))**2 +( (yy-Ny/2.0)/( (Ny/2.0)**2) )**2 +( (tt-Nt/2.0)/( (Nt/2.0)**2))**2
#                 if D3 > 0.1:
#                     data_out[xx,yy,tt] = 0.0
#                 else:
#                     pass
# #                 data_out[xx,yy,tt] = data_out[xx,yy,tt]*numpy.exp(-D3/0.1)
#                 
#     matplotlib.pyplot.imshow(numpy.abs(data_out[256,:,:]),norm = norm)
#     matplotlib.pyplot.show()                   
    data_out = numpy.fft.ifftshift(data_out,axes=(2,0,1 ))
    data_out = numpy.fft.ifftn(data_out,axes=(2, 0,1))
    data_out = numpy.fft.ifftshift(data_out,axes=(2, 0,1))     
    
    for cc in range(0,Nt):
#         if cc == Nt/2:
#             data_out[...,cc]=static_image
#         else:
        data_out[...,cc]=spatial_filter*data_out[...,cc]  

     
    return data_out
def showfoo_1024(save_folder_name,low,up):
    u_in_xyf=numpy.load(save_folder_name+'u_in_xyf.npy')
#     for jj in range(0,u_in_xyf.shape[2]):
#         u_in_xyf[...,jj,:]=u_in_xyf[...,jj,:]*numpy.cos((jj - u_in_xyf.shape[2]/2 )/3)
    
    data_in_xy_t =  u_in_xyf#/numpy.max(numpy.abs(u_in_xyf[:]))
    data_in_xy_t = low_pass_filter(data_in_xy_t)
    
    #data_in_xy_t = scipy.fftpack.fftn(u_in_xyf,axes = (2,))
    print('data_in_xy,shape',data_in_xy_t.shape)
    print('u_in_xyf,shape',u_in_xyf.shape)
    cmap=matplotlib.cm.gray
    interpl='nearest'
    norm=matplotlib.colors.Normalize(vmin=low, vmax=up ) 
    import scipy.misc
    import scipy.io
    cine2mat = numpy.zeros((256,1024,u_in_xyf.shape[2]))
        
    for jj in range(0,u_in_xyf.shape[2]):
#        matplotlib.pyplot.subplot(4,4,jj)
        matplotlib.pyplot.figure(figsize=(20,12))
        #matplotlib.pyplot.imshow(numpy.abs(data_in_xy_t[...,jj,0]),cmap = cmap, interpolation = interpl,norm=norm )
        ccc = (data_in_xy_t[...,jj,0]).T
        
        
        
        
        if jj == 0:
            base_phase = ccc
#         ccc= scipy.fftpack.fft2(ccc)
#         ddd = numpy.zeros((256,1024))
#         ddd[:,:128]=ccc[:,:128]
#         ddd[:,-128:]=ccc[:,-128:]
# #         ddd[-128:,:128]=ccc[-128:,:128]
# #         ddd[-128:,-128:]=ccc[-128:,-128:]
#         ddd = scipy.fftpack.ifft2(ddd)
        ddd = scipy.misc.imresize(numpy.abs(ccc),numpy.shape(cine2mat)[0:2],'bicubic')
        cine2mat[:,:,jj] = ddd
#         ddd.imag = scipy.misc.imresize(ccc.imag,(512,512))
#         if jj == 0: # normalize the image intensity
#             sum_value = numpy.sum(ddd[:])
#         else:
#             tmp_sum_value = numpy.sum(ddd[:])
#             ddd = sum_value*ddd/tmp_sum_value
        #ccc = scipy.fftpack.fftn(ccc,(512,512),axes=(0,1))
        matplotlib.pyplot.imshow(numpy.abs(ddd),cmap = cmap, interpolation = interpl, norm =norm 
                                        )
        #fname = '_tmp%03d.png'%jj
        fname ='cine_image_data_%03d.png'%(jj,)
        matplotlib.pyplot.savefig(save_folder_name+fname,#+str(jj)+'.png',
                                  format='png',dpi=100, 
                                  transparent=True, bbox_inches='tight', 
                                  pad_inches=0)
        #matplotlib.pyplot.show()
    scipy.io.savemat( save_folder_name+'cine_file.mat', mdict={'cine2mat':cine2mat}, oned_as={'column'})    

import sys

class Logger(object):
    def __init__(self,filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  


if __name__ == "__main__":

    import gc
    gc.enable()

    import cProfile

    for subject in ('./rawdata/',  ):
        import glob, os.path
        mydir = glob.glob(subject+'dante_cine_*')
        
        mydir.sort()
        print(mydir)
        num_of_series = numpy.size(mydir)
    #     print(num_of_series)
    #     print(mydir)
        xres = 512
        yres = 256
        ncoil = 8 
        echo_trains = 88
        etl = 12
        nthread=16
        rfov_yres = 32
        import os 
        import platform
        text_file = open("Recon_time_record" + platform.platform() + ".txt", "a")

        text_file.write("%s" % platform.platform()  )
        text_file.write("\n"  )       
        text_file.write("recon FOV") 
        text_file.write("\t" )          
        text_file.write("number of processes"  )
        text_file.write("\t" )
        text_file.write("processing time"  )
        text_file.write("\n"  )
        text_file.close()        
        for jj in xrange(0,num_of_series):
            for rfov_yres in (32,    256, 64, 128, 256,):#xrange(32, 256*2 -32,256-32):
                for nthread in ( 1,2, 4, 8, 16, 32):#, 64, 128, 256,512)+tuple(range(1,41))  :
    #                 jj = 2  
                    folder_for_process = mydir[jj] + '/'
                    print(str(folder_for_process))
                    if os.path.isdir(folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/') is True:
                        print('exist ', folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/')
                        pass
                     
                    else:
                        print('does not exist ', folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/')
                        para_process_int32(folder_for_process,
                                  xres,yres,etl,echo_trains,ncoil,nthread, rfov_yres)
                        showfoo_rfov(folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/',0,2,0,rfov_yres) 
#                    do_convert(folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/')

    for subject in ('./rawdata',):
    #     mydir = os.listdir(subject)
        import glob, os.path
        mydir = glob.glob(subject+'dante_cine_*')
        
        mydir.sort()
        print(mydir)
        num_of_series = numpy.size(mydir)
    #     print(num_of_series)
    #     print(mydir)
        xres = 512
        yres = 256
        ncoil = 8 
        echo_trains = 88
        etl = 12
        nthread=16
        rfov_yres = 32
        import os 
        for jj in xrange(0,num_of_series):
            for nthread in (1,16,):
                for rfov_yres in (32, 256):#xrange(32, 256*2 -32,256-32):
    #                 jj = 2  
                    folder_for_process = mydir[jj] + '/'
                    print(str(folder_for_process))
                    if os.path.isdir(folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/') is True:
                        print('exist ', folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/')
                        pass
                     
                    else:
                        print('does not exist ', folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/')
                        para_process_int16(folder_for_process,
                                  xres,yres,etl,echo_trains,ncoil,nthread, rfov_yres)
                        showfoo_rfov(folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/',0,2,1,rfov_yres)
                    do_convert(folder_for_process+'thread_'+str(nthread)+'_rfov_'+str(rfov_yres)+'/')
   

