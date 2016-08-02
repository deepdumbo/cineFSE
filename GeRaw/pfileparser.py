import matplotlib.pyplot
import numpy
# import correctmotion
#import nitime.algorithms.spectral
#     N=256
#     width = 3
#     n_bases= 30
#     v,e = nitime.algorithms.spectral.dpss_windows(N,width, n_bases )
#     print(v.shape)
import AverageHermitian
# from Im2Polar import *
'''
Created on 2013/1/14

@author: sram
'''
#READ_GEHDR  Read the header of a GE raw data file into a structure
#
#  usage: hdr = read_gehdr(id)
#
#  id:    integer file identifier obtained from fopen()
#
#  hdr:   a structure containing the following elements:
#
#         hdr['rdb'] - a structure equivilant to GE's rdb_hdr_rec
#         hdr.image - a structure equivilant to GE's rdb_hdr_image
#         hdr.series - a structure equivilant to GE's rdb_hdr_series
#         hdr.exam - a structure equivilant to GE's rdb_hdr_exam
#         hdr.data_acq_tab - a structure array equivilant to GE's rdb_hdr_data_acq_tab
#
#         and these elements that are straight storage dumps of their respective
#         GE structures:
#
#         hdr.per_pass
#         hdr.unlock_raw
#         hdr.nex_tab
#         hdr.nex_abort_tab
#         hdr.tool
#
#  e.g.:  >> id = fopen('P12345.7', 'r', 'l');
#         >> hdr = read_gehdr(id);
#
#  note:  Raw file must have RDB_RDBM_REVISION = 14.3F

#  This python class was modified by Jyh-Miin Lin 2012
#  The original MATLAB script was generated from GE source using 'gehdr2matlab' written by
#  DB Clayton
#  This script was further modified by Martin Graves, Addenbrooke's MRIS
#  Unit to work with 22.0 data
#  MJG 270109 Line 796 needs modifying from 32 to 33
#  hdr.image.int_padding = fread(id, 32, 'int');  # Please use this if you are adding any ints
#!/usr/bin/python
#from numpy import *
import numpy
import scipy.fftpack
import matplotlib.pyplot as mp
#def sortProp(KK,prop_shape):
#    prop_shape=tuple(prop_shape[:])
#    
#    prop=numpy.array(numpy.reshape(KK,prop_shape,order='F'))
##    tmp_prop=numpy.array(prop)
##    for pj in range(0,len(prop_shape)-1):
##        tmp_prop=tmp_prop.sum(-1) # force memcpy
##    
##    ind=numpy.argmax(numpy.abs(tmp_prop))
##    print('numpy.argmax',ind, prop_shape[0]/2 )
###    tmp_prop=
##    shift_ind= -ind + prop_shape[0]/2 
##    print('shfit_ind',shift_ind)
##    prop=numpy.roll(prop,shift_ind,axis=0)
#    
#    for pp in range(0,prop.shape[2]):
#        for pj in numpy.arange(0,prop.shape[3]): # coil
##            tmp_prop=numpy.array(prop[:,:,pp,pj])
###            for pj in range(0,len(prop_shape)-1):
##            tmp_prop=tmp_prop.sum(-1) # force memcpy
##            
##            ind=numpy.argmax(numpy.abs(tmp_prop))
##            print('numpy.argmax',ind,prop_shape[0]/2 )
##        #    tmp_prop=
##            shift_ind= -ind + prop_shape[0]/2 
##            print('shfit_ind',shift_ind)
#            prop[:,:,pp,pj]=numpy.roll(prop[:,:,pp,pj],54,axis=0)
#            
#            cnt1=numpy.mean(prop[prop.shape[0]/2-1:prop.shape[0]/2+1,
#                      prop.shape[1]/2-1:prop.shape[1]/2+1,pp,pj])
#            cnt1=cnt1/numpy.abs(cnt1)
#            prop[:,:,pp,pj]=prop[:,:,pp,pj]/cnt1
#    return prop
def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output
 
    Reference
    ---------
 http://leohart.wordpress.com/2006/01/29/hello-world/
http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
 
    '''
    # Special cases
    if alpha <= 0:
        return numpy.ones(window_length) #rectangular window
    elif alpha >= 1:
        return numpy.hanning(window_length)
 
    # Normal case
    x = numpy.linspace(0, 1, window_length)
    w = numpy.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + numpy.cos(2*numpy.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + numpy.cos(2*numpy.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w
def shiftcorr(prop, input_theta,prop_encode):
    import correctmotion
#     result_angle =numpy.zeros((prop.shape[2],prop.shape[3]) )
#     result_angle = correctmotion.find_rotation(prop, input_theta, 2, interp_num = 12) # jml

#     print('size of result_angle',result_angle.shape,prop.shape[2],prop.shape[3])
    result_angle = numpy.arange(0,prop.shape[2])*input_theta
#     prop2  = correctmotion.corr_translation(prop , result_angle )
    prop2 = prop
#     for pj in xrange(0,prop.shape[3]):# coil
# 
#         if prop_encode <= 3: # control by opuser23
# 
#             prop[:,:,:,pj] = correctmotion.corr_translation(prop[:,:,:,pj], result_angle )
#     result_angle = correctmotion.find_rotation(prop, input_theta)
#     
#     for pj in xrange(0,prop.shape[3]):# coil
# 
#         if prop_encode <= 3: # control by opuser23
# 
#             prop[:,:,:,pj] = correctmotion.corr_translation(prop[:,:,:,pj], result_angle )
    
    return (prop2, result_angle)

def zeroPhase(prop, input_theta,prop_encode):
    import correctmotion
#     result_angle =numpy.zeros((prop.shape[2],prop.shape[3]) )
#     import pp
#     job_server = pp.Server()
#     jobs = []
#     for pj in range(0,prop.shape[3]):# coil
#         jobs.append(job_server.submit(corr_phase_rotation_translation, (prop[:,:,:,pj],),
#                          modules = ('numpy','pynufft','scipy','correctmotion','Im2Polar'),
#                          globals = globals()))
# #     
    #===========================================================================
#     Discrete prolate
    n_bases = 4
    dpss_width = 0.0
    
#     if dpss_width == 0.0:
#         
    kaiser_filx = numpy.ones((prop.shape[0],))   # slightly tapering dim 0 (freq)
    kaiser_fily = numpy.ones((prop.shape[1],)) #tukeywin(prop.shape[1],0.9)   # slightly tapering dim 1 (phas)
#         
#     else:
#         v,e = nitime.algorithms.spectral.dpss_windows(prop.shape[0],dpss_width, n_bases )
#         for jj in xrange(0,n_bases):
#             v[jj,:]= v[jj,:] *e[jj]
#               
#         kaiser_filx = numpy.sum(v,0)
#         # 
#         v,e = nitime.algorithms.spectral.dpss_windows(prop.shape[1], dpss_width*prop.shape[1]/prop.shape[0], n_bases )
#         for jj in xrange(0,n_bases):
#             v[jj,:]=v[jj,:]*e[jj]
#               
#         kaiser_fily = numpy.sum(v,0)    
    #===========================================================================
    
#     kaiser_filx = numpy.kaiser(prop.shape[0],0.05)   # slightly tapering dim 0 (freq)
#     kaiser_fily = numpy.kaiser(prop.shape[1],10.0)   # slightly tapering dim 1 (phas)
    
#     kaiser_filx = numpy.ones((prop.shape[0],))   # slightly tapering dim 0 (freq)
#     kaiser_fily = numpy.ones((prop.shape[1],)) #tukeywin(prop.shape[1],0.9)   # slightly tapering dim 1 (phas)

#     kaiser_fily = tukeywin(prop.shape[1],0.5)   # slightly tapering dim 1 (phas)
#     kaiser_filx= tukeywin(prop.shape[0],0.9)   # slightly tapering dim 1 (phas)
   
    
    for run_x in xrange(0,prop.shape[0]):
        for run_y in xrange(0, prop.shape[1]):
            prop[run_x, run_y, ...] =prop[run_x, run_y, ...] *  kaiser_filx[run_x] * kaiser_fily[run_y]
             
#     import pp
#     job_server = pp.Server()
#     jobs = []  
    
    for pj in xrange(0,prop.shape[3]):# coil
#         prop[:,:,:,pj], result_angle[:,pj] = jobs[pj]()
        for pp in xrange(0,prop.shape[2]): #blade
    
                # correct k-space shifts
    #             input_k[:,:,pp] =corr_kspace_shift(input_k[:,:,pp])
                #===================================================================
                #correct zero order phase
                prop[:,:,pp,pj] =correctmotion.corr_kspace_zero_phase(prop[:,:,pp,pj])
#                 if prop_encode == 1:
#                     prop[:,:,pp,pj] =AverageHermitian.AverageHermitian(prop[:,:,pp,pj])
#         if prop_encode <= 2: # control by opuser23
# #             jobs.append(job_server.submit(correctmotion.corr_phase_rotation_translation, (prop[:,:,:,pj], input_theta ),
# #                  modules = ('numpy','correctmotion','scipy'),
# #                  globals = globals()))
#             prop[:,:,:,pj], result_angle[:,pj] = correctmotion.corr_phase_rotation_translation(prop[:,:,:,pj], input_theta )

#     result_angle = numpy.mean(result_angle,1) # average the angles from multi-slices
    (prop, result_angle)=shiftcorr(prop, input_theta,prop_encode)
    for pj in xrange(0,prop.shape[3]):# coil
#         prop[:,:,:,pj], result_angle[:,pj] = jobs[pj]()
        for pp in xrange(0,prop.shape[2]): #blade
    
                # correct k-space shifts
    #             input_k[:,:,pp] =corr_kspace_shift(input_k[:,:,pp])
                #===================================================================
                #correct zero order phase
#                 prop[:,:,pp,pj] =correctmotion.corr_kspace_zero_phase(prop[:,:,pp,pj])
            prop[:,:,pp,pj] =AverageHermitian.AverageHermitian(prop[:,:,pp,pj])
            
    return (prop, result_angle)
import operator
def buildinprod(iterable):
    return reduce(operator.mul, iterable, 1)
# def unpack_uid(uid):
#     """Convert packed PFile UID to standard DICOM UID."""
#     return ''.join([str(i-1) if i < 11 
#                     else '.' for pair in [(ord(c) >> 4, ord(c) & 15) 
#                                           for c in uid] for i in pair if i > 0])

def ParseGeHdrElement(targObj, fid, my_tuple): # recursively adding attributes to desired 
    #targObj={}
    for pj in range(len(my_tuple)):             # looping over my_tuple 
#        ge_value=[0]
        count=0
        
        my_tuple_num=my_tuple[pj]['d_len']
        my_tuple_num=numpy.prod(my_tuple_num) # deal with array
        
#        print(my_tuple_num)
        ge_type=my_tuple[pj]['ge_type']
        if ge_type is 'char':
#            ge_type0='char'
            ge_type='uint8'
            ge_value=''
        elif my_tuple_num>1:
            ge_value=[]
        else:
            pass
#            print(ge_value)
        ge_attr=my_tuple[pj]['ge_attr']
        
        while count < my_tuple_num:  #stacking
            # trick is here: remove the list which numpy.fromfile return
            tmp_ge_value=numpy.fromfile(fid, ge_type, 1)[0] 
            if my_tuple[pj]['ge_type'] is 'char':
                ge_value+=chr(tmp_ge_value)
                ge_value=ge_value.split('\x00')[0] # remove \x00 zeros,
#                
            elif my_tuple_num>1:
                ge_value.append(tmp_ge_value)    
            else:
                #print(ge_value)
                ge_value=tmp_ge_value
#                print(ge_value)
            count=count+1

        if numpy.size(my_tuple[pj]['d_len']) > 1: # deal with array
#            print('not single element ')
            ge_value=numpy.reshape(ge_value,my_tuple[pj]['d_len'],order='F')
#        if 
        
        targObj[ge_attr]=ge_value# assign attribute to targObj
        ge_value=None #clear obj
def gethdr(fid):          
    rdb_tuple=(  
        {'ge_attr':'rdbm_rev','ge_type':'float32','d_len':1},
        {'ge_attr':'run_int','ge_type':'int32','d_len':1},
        {'ge_attr':'scan_seq','ge_type':'int16','d_len':1},
        {'ge_attr':'run_char','ge_type':'char','d_len':6},
        {'ge_attr':'scan_date','ge_type':'char','d_len':10},
        {'ge_attr':'scan_time','ge_type':'char','d_len':8},
        {'ge_attr':'logo','ge_type':'char','d_len':10},
        {'ge_attr':'file_contents','ge_type':'int16','d_len':1},
        {'ge_attr':'lock_mode','ge_type':'int16','d_len':1},
        {'ge_attr':'dacq_ctrl','ge_type':'int16','d_len':1},
        {'ge_attr':'recon_ctrl','ge_type':'int16','d_len':1},
        {'ge_attr':'exec_ctrl','ge_type':'int16','d_len':1},
        {'ge_attr':'scan_type','ge_type':'int16','d_len':1},
        {'ge_attr':'data_collect_type','ge_type':'int16','d_len':1},
        {'ge_attr':'data_format','ge_type':'int16','d_len':1},
        {'ge_attr':'recon','ge_type':'int16','d_len':1},
        {'ge_attr':'datacq','ge_type':'int16','d_len':1},
        {'ge_attr':'npasses','ge_type':'int16','d_len':1},
        {'ge_attr':'npomp','ge_type':'int16','d_len':1},
        {'ge_attr':'nslices','ge_type':'int16','d_len':1},
        {'ge_attr':'nechoes','ge_type':'int16','d_len':1},
        {'ge_attr':'navs','ge_type':'int16','d_len':1},
        {'ge_attr':'nframes','ge_type':'int16','d_len':1},
        {'ge_attr':'baseline_views','ge_type':'int16','d_len':1},
        {'ge_attr':'hnovers','ge_type':'int16','d_len':1},
        {'ge_attr':'frame_size','ge_type':'uint16','d_len':1},
        {'ge_attr':'point_size','ge_type':'int16','d_len':1},
        {'ge_attr':'vquant','ge_type':'int16','d_len':1},
        {'ge_attr':'cheart','ge_type':'int16','d_len':1},
        {'ge_attr':'ctr','ge_type':'float32','d_len':1},
        {'ge_attr':'ctrr','ge_type':'float32','d_len':1},
        {'ge_attr':'initpass','ge_type':'int16','d_len':1},
        {'ge_attr':'incrpass','ge_type':'int16','d_len':1},
        {'ge_attr':'method_ctrl','ge_type':'int16','d_len':1},
        {'ge_attr':'da_xres','ge_type':'uint16','d_len':1},
        {'ge_attr':'da_yres','ge_type':'int16','d_len':1},
        {'ge_attr':'rc_xres','ge_type':'int16','d_len':1},
        {'ge_attr':'rc_yres','ge_type':'int16','d_len':1},
        {'ge_attr':'im_size','ge_type':'int16','d_len':1},
        {'ge_attr':'rc_zres','ge_type':'int32','d_len':1},
    # These variables have been deprecated with the introduction of 64-bit recon.
    # New variables of type n64 have been introduced. They are at the bottom of this
    # struct in order to keep the remainder of the members at the same offset.
    # In order to keep the alignment the same as in 32-bit process, this entire
    # struct is packed to 4-byte boundries for 64-bit vre compilation.                
        {'ge_attr':'raw_pass_size_deprecated','ge_type':'uint32','d_len':1},
        {'ge_attr':'sspsave_deprecated','ge_type':'uint32','d_len':1},
        {'ge_attr':'udasave_deprecated','ge_type':'uint32','d_len':1},
        {'ge_attr':'fermi_radius','ge_type':'float32','d_len':1},
        {'ge_attr':'fermi_width','ge_type':'float32','d_len':1},
        {'ge_attr':'fermi_ecc','ge_type':'float32','d_len':1},
        {'ge_attr':'clip_min','ge_type':'float32','d_len':1},
        {'ge_attr':'clip_max','ge_type':'float32','d_len':1},
        {'ge_attr':'default_offset','ge_type':'float32','d_len':1},
        {'ge_attr':'xoff','ge_type':'float32','d_len':1},
        {'ge_attr':'yoff','ge_type':'float32','d_len':1},
        {'ge_attr':'nwin','ge_type':'float32','d_len':1},
        {'ge_attr':'ntran','ge_type':'float32','d_len':1},
        {'ge_attr':'scalei','ge_type':'float32','d_len':1},
        {'ge_attr':'scaleq','ge_type':'float32','d_len':1},
        {'ge_attr':'rotation','ge_type':'int16','d_len':1},
        {'ge_attr':'transpose','ge_type':'int16','d_len':1},
        {'ge_attr':'kissoff_views','ge_type':'int16','d_len':1},
        {'ge_attr':'slblank','ge_type':'int16','d_len':1},
        {'ge_attr':'gradcoil','ge_type':'int16','d_len':1},
        {'ge_attr':'ddaover','ge_type':'int16','d_len':1},
        {'ge_attr':'sarr','ge_type':'int16','d_len':1},
        {'ge_attr':'fd_tr','ge_type':'int16','d_len':1},
        {'ge_attr':'fd_te','ge_type':'int16','d_len':1},
        {'ge_attr':'fd_ctrl','ge_type':'int16','d_len':1},
        {'ge_attr':'algor_num','ge_type':'int16','d_len':1},
        {'ge_attr':'fd_df_dec','ge_type':'int16','d_len':1},
        {'ge_attr':'buff','ge_type':'int16','d_len':8}, # kluge for type RDB_MULTI_RCV_TYPE
#===============================================================================
#     buff = fread(id, 8, 'short');  
#     hdr.rdb.dab_start_rcv = buff(1:2:end);  % kluge for type RDB_MULTI_RCV_TYPE
#     hdr.rdb.dab_stop_rcv = buff(2:2:end);  % kluge for type RDB_MULTI_RCV_TYPE
#===============================================================================
        
        {'ge_attr':'user0','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user1','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user2','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user3','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user4','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user5','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user6','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user7','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user8','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user9','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user10','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user11','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user12','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user13','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user14','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user15','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user16','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user17','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user18','ge_type':'float32','d_len':1}, 
        {'ge_attr':'user19','ge_type':'float32','d_len':1},                          
        {'ge_attr':'v_type','ge_type':'int32','d_len':1}, 
        {'ge_attr':'v_coefxa','ge_type':'float32','d_len':1},                          
        {'ge_attr':'v_coefxb','ge_type':'float32','d_len':1}, 
        {'ge_attr':'v_coefxc','ge_type':'float32','d_len':1},                          
        {'ge_attr':'v_coefxd','ge_type':'float32','d_len':1}, 
        {'ge_attr':'v_coefya','ge_type':'float32','d_len':1},                          
        {'ge_attr':'v_coefyb','ge_type':'float32','d_len':1}, 
        {'ge_attr':'v_coefyc','ge_type':'float32','d_len':1},                          
        {'ge_attr':'v_coefyd','ge_type':'float32','d_len':1}, 
        {'ge_attr':'v_coefza','ge_type':'float32','d_len':1},                          
        {'ge_attr':'v_coefzb','ge_type':'float32','d_len':1}, 
        {'ge_attr':'v_coefzc','ge_type':'float32','d_len':1},                          
        {'ge_attr':'v_coefzd','ge_type':'float32','d_len':1}, 
        {'ge_attr':'vm_coef1','ge_type':'float32','d_len':1},                          
        {'ge_attr':'vm_coef2','ge_type':'float32','d_len':1}, 
        {'ge_attr':'vm_coef3','ge_type':'float32','d_len':1},                          
        {'ge_attr':'vm_coef4','ge_type':'float32','d_len':1}, 
        {'ge_attr':'v_venc','ge_type':'float32','d_len':1},
                                  
        {'ge_attr':'spectral_width','ge_type':'float32','d_len':1}, 
        {'ge_attr':'csi_dims','ge_type':'int16','d_len':1},                          
        {'ge_attr':'xcsi','ge_type':'int16','d_len':1}, 
        {'ge_attr':'ycsi','ge_type':'int16','d_len':1},                          
        {'ge_attr':'zcsi','ge_type':'int16','d_len':1}, 
        {'ge_attr':'zoilenx','ge_type':'float32','d_len':1},                          
        {'ge_attr':'zoileny','ge_type':'float32','d_len':1}, 
        {'ge_attr':'zoilenz','ge_type':'float32','d_len':1},                          
        {'ge_attr':'roilocx','ge_type':'float32','d_len':1}, 
        {'ge_attr':'roilocy','ge_type':'float32','d_len':1},                          
        {'ge_attr':'roilocz','ge_type':'float32','d_len':1}, 
        {'ge_attr':'numdwell','ge_type':'float32','d_len':1},
                                  
        {'ge_attr':'ps_command','ge_type':'int32','d_len':1}, 
        {'ge_attr':'ps_mps_r1','ge_type':'int32','d_len':1},                          
        {'ge_attr':'ps_mps_r2','ge_type':'int32','d_len':1}, 
        {'ge_attr':'ps_mps_tg','ge_type':'int32','d_len':1},                          
        {'ge_attr':'ps_mps_freq','ge_type':'int32','d_len':1}, 
        {'ge_attr':'ps_aps_r1','ge_type':'int32','d_len':1},                          
        {'ge_attr':'ps_aps_r2','ge_type':'int32','d_len':1},
        {'ge_attr':'ps_aps_tg','ge_type':'int32','d_len':1},
        {'ge_attr':'ps_aps_freq','ge_type':'int32','d_len':1},                          
        {'ge_attr':'ps_scalei','ge_type':'float32','d_len':1}, 
        {'ge_attr':'ps_scaleq','ge_type':'float32','d_len':1},                          
        {'ge_attr':'ps_snr_warning','ge_type':'int32','d_len':1}, 
        {'ge_attr':'ps_aps_or_mps','ge_type':'int32','d_len':1},                          
        {'ge_attr':'ps_mps_bitmap','ge_type':'int32','d_len':1}, 
        {'ge_attr':'ps_powerspec','ge_type':'char','d_len':256},                          
        {'ge_attr':'ps_filler1','ge_type':'int32','d_len':1}, 
        {'ge_attr':'ps_filler2','ge_type':'int32','d_len':1},                          
        {'ge_attr':'obsolete1','ge_type':'float32','d_len':16}, 
        {'ge_attr':'obsolete2','ge_type':'float32','d_len':16},                          
        {'ge_attr':'halfecho','ge_type':'int16','d_len':1},
         
        {'ge_attr':'im_size_y','ge_type':'int16','d_len':1},                          
        {'ge_attr':'data_collect_type1','ge_type':'int32','d_len':1}, 
        {'ge_attr':'freq_scale','ge_type':'float32','d_len':1},                          
        {'ge_attr':'phase_scale','ge_type':'float32','d_len':1}, 
    
        {'ge_attr':'ovl','ge_type':'int16','d_len':1}, #%New fields 02-19-92
         
        {'ge_attr':'pclin','ge_type':'int16','d_len':1},
        {'ge_attr':'pclinnpts','ge_type':'int16','d_len':1},
        {'ge_attr':'pclinorder','ge_type':'int16','d_len':1},
        {'ge_attr':'pclinavg','ge_type':'int16','d_len':1},
        {'ge_attr':'pccon','ge_type':'int16','d_len':1},
        {'ge_attr':'pcconnpts','ge_type':'int16','d_len':1},
        {'ge_attr':'pcconorder','ge_type':'int16','d_len':1},
        {'ge_attr':'pcextcorr','ge_type':'int16','d_len':1},
        {'ge_attr':'pcgraph','ge_type':'int16','d_len':1},
        {'ge_attr':'pcileave','ge_type':'int16','d_len':1},
        {'ge_attr':'hdbestky','ge_type':'int16','d_len':1},
        {'ge_attr':'pcctrl','ge_type':'int16','d_len':1},
        {'ge_attr':'pcthrespts','ge_type':'int16','d_len':1},
        {'ge_attr':'pcdiscbeg','ge_type':'int16','d_len':1},
        {'ge_attr':'pcdiscmid','ge_type':'int16','d_len':1},
        {'ge_attr':'pcdiscend','ge_type':'int16','d_len':1},
        {'ge_attr':'pcthrespct','ge_type':'int16','d_len':1},
        {'ge_attr':'pcspacial','ge_type':'int16','d_len':1},
        {'ge_attr':'pctemporal','ge_type':'int16','d_len':1},
        {'ge_attr':'pcspare','ge_type':'int16','d_len':1},
        {'ge_attr':'ileaves','ge_type':'int16','d_len':1},
        {'ge_attr':'kydir','ge_type':'int16','d_len':1},
        {'ge_attr':'alt','ge_type':'int16','d_len':1},
        {'ge_attr':'reps','ge_type':'int16','d_len':1},
        {'ge_attr':'ref','ge_type':'int16','d_len':1},
        {'ge_attr':'pcconnorm','ge_type':'float32','d_len':1},
        {'ge_attr':'pcconfitwt','ge_type':'float32','d_len':1},
        {'ge_attr':'pclinnorm','ge_type':'float32','d_len':1},
        {'ge_attr':'pclinfitwt','ge_type':'float32','d_len':1},
        {'ge_attr':'pcbestky','ge_type':'float32','d_len':1},
        
        {'ge_attr':'vrgf','ge_type':'int32','d_len':1},
        {'ge_attr':'vrgfxres','ge_type':'int32','d_len':1},
        
        {'ge_attr':'bp_corr','ge_type':'int32','d_len':1}, 
        {'ge_attr':'recv_freq_s','ge_type':'float32','d_len':1}, 
        {'ge_attr':'recv_freq_e','ge_type':'float32','d_len':1},
         
        {'ge_attr':'hniter','ge_type':'int32','d_len':1}, 
        {'ge_attr':'fast_rec','ge_type':'int32','d_len':1},
        {'ge_attr':'refframes','ge_type':'int32','d_len':1},
        {'ge_attr':'refframep','ge_type':'int32','d_len':1},
        {'ge_attr':'scnframe','ge_type':'int32','d_len':1},
        {'ge_attr':'pasframe','ge_type':'int32','d_len':1},
        
        {'ge_attr':'user_usage_tag','ge_type':'uint32','d_len':1},
        {'ge_attr':'user_fill_mapMSW','ge_type':'uint32','d_len':1},
        {'ge_attr':'user_fill_mapLSW','ge_type':'uint32','d_len':1},
        
        {'ge_attr':'user20','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user21','ge_type':'float32','d_len':1},
        {'ge_attr':'user22','ge_type':'float32','d_len':1},
        {'ge_attr':'user23','ge_type':'float32','d_len':1},
        {'ge_attr':'user24','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user25','ge_type':'float32','d_len':1},
        {'ge_attr':'user26','ge_type':'float32','d_len':1},
        {'ge_attr':'user27','ge_type':'float32','d_len':1},
        {'ge_attr':'user28','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user29','ge_type':'float32','d_len':1},
        {'ge_attr':'user30','ge_type':'float32','d_len':1},
        {'ge_attr':'user31','ge_type':'float32','d_len':1},
        {'ge_attr':'user32','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user33','ge_type':'float32','d_len':1},
        {'ge_attr':'user34','ge_type':'float32','d_len':1},
        {'ge_attr':'user35','ge_type':'float32','d_len':1},
        {'ge_attr':'user36','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user37','ge_type':'float32','d_len':1},
        {'ge_attr':'user38','ge_type':'float32','d_len':1},
        {'ge_attr':'user39','ge_type':'float32','d_len':1},
        {'ge_attr':'user40','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user41','ge_type':'float32','d_len':1},
        {'ge_attr':'user42','ge_type':'float32','d_len':1},
        {'ge_attr':'user43','ge_type':'float32','d_len':1},
        {'ge_attr':'user44','ge_type':'float32','d_len':1},                          
        {'ge_attr':'user45','ge_type':'float32','d_len':1},
        {'ge_attr':'user46','ge_type':'float32','d_len':1},
        {'ge_attr':'user47','ge_type':'float32','d_len':1},
        {'ge_attr':'user48','ge_type':'float32','d_len':1},
        
        {'ge_attr':'pcfitorig','ge_type':'int16','d_len':1},
        {'ge_attr':'pcshotfirst','ge_type':'int16','d_len':1},
        {'ge_attr':'pcshotlast','ge_type':'int16','d_len':1},
        {'ge_attr':'pcmultegrp','ge_type':'int16','d_len':1},
        {'ge_attr':'pclinfix','ge_type':'int16','d_len':1},
        
        {'ge_attr':'pcconfix','ge_type':'int16','d_len':1},
        
        {'ge_attr':'pclinslope','ge_type':'float32','d_len':1},
        {'ge_attr':'pcconslope','ge_type':'float32','d_len':1},
        {'ge_attr':'pccoil','ge_type':'int16','d_len':1},
        
        {'ge_attr':'vvsmode','ge_type':'int16','d_len':1},
        {'ge_attr':'vvsaimgs','ge_type':'int16','d_len':1},
        {'ge_attr':'vvstr','ge_type':'int16','d_len':1},
        {'ge_attr':'vvsgender','ge_type':'int16','d_len':1},
        
        {'ge_attr':'zip_factor','ge_type':'int16','d_len':1},
        
        {'ge_attr':'maxcoef1a','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef1b','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef1c','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef1d','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef2a','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef2b','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef2c','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef2d','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef3a','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef3b','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef3c','ge_type':'float32','d_len':1},
        {'ge_attr':'maxcoef3d','ge_type':'float32','d_len':1},
        
        {'ge_attr':'ut_ctrl','ge_type':'int32','d_len':1},
        {'ge_attr':'dp_type','ge_type':'int16','d_len':1},
        
        {'ge_attr':'arw','ge_type':'int16','d_len':1},
        
        {'ge_attr':'vps','ge_type':'int16','d_len':1},
        
        {'ge_attr':'mcReconEnable','ge_type':'int16','d_len':1},
        {'ge_attr':'fov','ge_type':'int32','d_len':1},
        
        {'ge_attr':'te','ge_type':'int32','d_len':1},
        {'ge_attr':'te2','ge_type':'int32','d_len':1},
        {'ge_attr':'dfmrbw','ge_type':'float32','d_len':1},
        {'ge_attr':'dfmctrl','ge_type':'int32','d_len':1},
        {'ge_attr':'raw_nex','ge_type':'int32','d_len':1},
        {'ge_attr':'navs_per_pass','ge_type':'int32','d_len':1},
        {'ge_attr':'dfmxres','ge_type':'int32','d_len':1},
        {'ge_attr':'dfmptsize','ge_type':'int32','d_len':1},
        {'ge_attr':'navs_per_view','ge_type':'int32','d_len':1},
        {'ge_attr':'dfmdebug','ge_type':'int32','d_len':1},
        {'ge_attr':'dfmthreshold','ge_type':'float32','d_len':1},
        
        {'ge_attr':'grid_control','ge_type':'int16','d_len':1},
        {'ge_attr':'b0map','ge_type':'int16','d_len':1},
        {'ge_attr':'grid_tediff','ge_type':'int16','d_len':1},
        {'ge_attr':'grid_motion_comp','ge_type':'int16','d_len':1},
        {'ge_attr':'grid_radius_a','ge_type':'float32','d_len':1},
        {'ge_attr':'grid_radius_b','ge_type':'float32','d_len':1},
        {'ge_attr':'grid_max_gradient','ge_type':'float32','d_len':1},
        {'ge_attr':'grid_max_slew','ge_type':'float32','d_len':1},
        {'ge_attr':'grid_scan_fov','ge_type':'float32','d_len':1},
        {'ge_attr':'grid_a2d_time','ge_type':'float32','d_len':1},
        {'ge_attr':'grid_density_factor','ge_type':'float32','d_len':1},
        {'ge_attr':'grid_display_fov','ge_type':'float32','d_len':1},
        
        {'ge_attr':'fatwater','ge_type':'int16','d_len':1},
        {'ge_attr':'fiestamlf','ge_type':'int16','d_len':1},
        
        {'ge_attr':'app','ge_type':'int16','d_len':1},
        {'ge_attr':'rhncoilsel','ge_type':'int16','d_len':1},
        {'ge_attr':'rhncoillimit','ge_type':'int16','d_len':1},
        {'ge_attr':'app_option','ge_type':'int16','d_len':1},
        {'ge_attr':'grad_mode','ge_type':'int16','d_len':1},
        {'ge_attr':'pfile_passes','ge_type':'int16','d_len':1},
        
        {'ge_attr':'asset','ge_type':'int32','d_len':1},
        {'ge_attr':'asset_calthresh','ge_type':'int32','d_len':1},
        {'ge_attr':'asset_R','ge_type':'float32','d_len':1},
        {'ge_attr':'coilno','ge_type':'int32','d_len':1},
        {'ge_attr':'asset_phases','ge_type':'int32','d_len':1},
        {'ge_attr':'scancent','ge_type':'float32','d_len':1},
        {'ge_attr':'position','ge_type':'int32','d_len':1},
        {'ge_attr':'entry','ge_type':'int32','d_len':1},
        {'ge_attr':'lmhor','ge_type':'float32','d_len':1},
        {'ge_attr':'last_slice_num','ge_type':'int32','d_len':1},
        {'ge_attr':'asset_slice_R','ge_type':'float32','d_len':1},
        {'ge_attr':'asset_slabwrap','ge_type':'float32','d_len':1},
        
        {'ge_attr':'dwnav_coeff','ge_type':'float32','d_len':1},
        {'ge_attr':'dwnav_cor','ge_type':'int16','d_len':1},
        {'ge_attr':'dwnav_view','ge_type':'int16','d_len':1},
        {'ge_attr':'dwnav_corecho','ge_type':'int16','d_len':1},
        {'ge_attr':'dwnav_sview','ge_type':'int16','d_len':1},
        {'ge_attr':'dwnav_eview','ge_type':'int16','d_len':1},
        {'ge_attr':'dwnav_sshot','ge_type':'int16','d_len':1},
        {'ge_attr':'dwnav_eshot','ge_type':'int16','d_len':1},
        
        {'ge_attr':'win3d_type','ge_type':'int16','d_len':1},
        {'ge_attr':'win3d_apod','ge_type':'float32','d_len':1},
        {'ge_attr':'win3d_q','ge_type':'float32','d_len':1},
        
        {'ge_attr':'ime_scic_enable','ge_type':'int16','d_len':1},
        {'ge_attr':'clariview_type','ge_type':'int16','d_len':1},
        {'ge_attr':'ime_scic_edge','ge_type':'float32','d_len':1},
        {'ge_attr':'ime_scic_smooth','ge_type':'float32','d_len':1},
        {'ge_attr':'ime_scic_focus','ge_type':'float32','d_len':1},
        {'ge_attr':'clariview_edge','ge_type':'float32','d_len':1},
        {'ge_attr':'clariview_smooth','ge_type':'float32','d_len':1},
        {'ge_attr':'clariview_focus','ge_type':'float32','d_len':1},
        {'ge_attr':'scic_reduction','ge_type':'float32','d_len':1},
        {'ge_attr':'scic_gauss','ge_type':'float32','d_len':1},
        {'ge_attr':'scic_threshold','ge_type':'float32','d_len':1},
        
        {'ge_attr':'ectricks_no_regions','ge_type':'int32','d_len':1},
        {'ge_attr':'ectricks_input_regions','ge_type':'int32','d_len':1},
        
        {'ge_attr':'psc_reuse','ge_type':'int16','d_len':1},
        
        {'ge_attr':'left_blank','ge_type':'int16','d_len':1},
        {'ge_attr':'right_blank','ge_type':'int16','d_len':1},
        
        {'ge_attr':'acquire_type','ge_type':'int16','d_len':1},
        {'ge_attr':'retro_control','ge_type':'int16','d_len':1},
        {'ge_attr':'etl','ge_type':'int16','d_len':1},
        {'ge_attr':'pcref_start','ge_type':'int16','d_len':1},
        {'ge_attr':'pcref_stop','ge_type':'int16','d_len':1},
        {'ge_attr':'ref_skip','ge_type':'int16','d_len':1},
        {'ge_attr':'extra_frames_top','ge_type':'int16','d_len':1},
        {'ge_attr':'extra_frames_bot','ge_type':'int16','d_len':1},
        {'ge_attr':'multiphase_type','ge_type':'int16','d_len':1},
        {'ge_attr':'nphases','ge_type':'int16','d_len':1},
        {'ge_attr':'pure','ge_type':'int16','d_len':1},
        {'ge_attr':'pure_scale','ge_type':'float32','d_len':1},
        {'ge_attr':'off_data','ge_type':'int32','d_len':1},
        {'ge_attr':'off_per_pass','ge_type':'int32','d_len':1},
        {'ge_attr':'off_unlock_raw','ge_type':'int32','d_len':1},
        {'ge_attr':'off_data_acq_tab','ge_type':'int32','d_len':1},
        {'ge_attr':'off_nex_tab','ge_type':'int32','d_len':1},
        {'ge_attr':'off_nex_abort_tab','ge_type':'int32','d_len':1},
        {'ge_attr':'off_tool','ge_type':'int32','d_len':1},
        {'ge_attr':'off_exam','ge_type':'int32','d_len':1},
        {'ge_attr':'off_series','ge_type':'int32','d_len':1},
        {'ge_attr':'off_image','ge_type':'int32','d_len':1},
    
        {'ge_attr':'off_ps','ge_type':'int32','d_len':1},
        {'ge_attr':'off_spare_b','ge_type':'int32','d_len':1},
        {'ge_attr':'new_wnd_level_flag','ge_type':'int32','d_len':1},
        {'ge_attr':'wnd_image_hist_area','ge_type':'int32','d_len':1},
        {'ge_attr':'wnd_high_hist','ge_type':'float32','d_len':1},
        {'ge_attr':'wnd_lower_hist','ge_type':'float32','d_len':1},
        {'ge_attr':'pure_filter','ge_type':'int16','d_len':1},
        {'ge_attr':'cfg_pure_filter','ge_type':'int16','d_len':1},
        {'ge_attr':'cfg_pure_fit_order','ge_type':'int16','d_len':1},
        {'ge_attr':'cfg_pure_kernelsize_z','ge_type':'int16','d_len':1},
        {'ge_attr':'cfg_pure_kernelsize_xy','ge_type':'int16','d_len':1},
        {'ge_attr':'cfg_pure_weight_radius','ge_type':'int16','d_len':1},
        {'ge_attr':'cfg_pure_intensity_scale','ge_type':'int16','d_len':1},
        {'ge_attr':'cfg_pure_noise_threshold','ge_type':'int16','d_len':1},
        
        {'ge_attr':'wienera','ge_type':'float32','d_len':1},
        {'ge_attr':'wienerb','ge_type':'float32','d_len':1},
        {'ge_attr':'wienert2','ge_type':'float32','d_len':1},
        {'ge_attr':'wieneresp','ge_type':'float32','d_len':1},
        {'ge_attr':'wiener','ge_type':'int16','d_len':1},
        {'ge_attr':'flipfilter','ge_type':'int16','d_len':1},
        {'ge_attr':'dbgrecon','ge_type':'int16','d_len':1},
        {'ge_attr':'ech2skip','ge_type':'int16','d_len':1},
        {'ge_attr':'tricks_type','ge_type':'int32','d_len':1},
        {'ge_attr':'lcfiesta_phase','ge_type':'float32','d_len':1},
        {'ge_attr':'lcfiesta','ge_type':'int16','d_len':1},
        {'ge_attr':'herawflt','ge_type':'int16','d_len':1},
        {'ge_attr':'herawflt_befnwin','ge_type':'int16','d_len':1},
        {'ge_attr':'herawflt_befntran','ge_type':'int16','d_len':1},
        {'ge_attr':'herawflt_befamp','ge_type':'float32','d_len':1},
        {'ge_attr':'herawflt_hpfamp','ge_type':'float32','d_len':1},
        {'ge_attr':'heover','ge_type':'int16','d_len':1},
        {'ge_attr':'pure_correction_threshold','ge_type':'int16','d_len':1},
        {'ge_attr':'swiftenable','ge_type':'int32','d_len':1},
        {'ge_attr':'numslabs','ge_type':'int16','d_len':1},
        {'ge_attr':'swiftcoilnos','ge_type':'uint16','d_len':1},
        {'ge_attr':'ps_autoshim_status','ge_type':'int32','d_len':1},
        
        {'ge_attr':'dynaplan_numphases','ge_type':'int32','d_len':1},
        
        {'ge_attr':'medal_cfg','ge_type':'int16','d_len':1},
        {'ge_attr':'medal_nstack','ge_type':'int16','d_len':1},
        {'ge_attr':'medal_echo_order','ge_type':'int16','d_len':1},
        {'ge_attr':'medal_kernel_up','ge_type':'int16','d_len':1},
        {'ge_attr':'medal_kernel_down','ge_type':'int16','d_len':1},
        {'ge_attr':'medal_kernel_smooth','ge_type':'int16','d_len':1},
        {'ge_attr':'medal_start','ge_type':'int16','d_len':1},
        {'ge_attr':'medal_end','ge_type':'int16','d_len':1},
        {'ge_attr':'rcideal','ge_type':'uint32','d_len':1},
        {'ge_attr':'rcdixproc','ge_type':'uint32','d_len':1},
        {'ge_attr':'df','ge_type':'float32','d_len':1},
        {'ge_attr':'bw','ge_type':'float32','d_len':1},
        {'ge_attr':'te1','ge_type':'float32','d_len':1},
        {'ge_attr':'esp','ge_type':'float32','d_len':1},
        {'ge_attr':'feextra','ge_type':'int32','d_len':1},
        
        {'ge_attr':'raw_pass_size','ge_type':'uint64','d_len':1},
        {'ge_attr':'sspsave','ge_type':'uint64','d_len':1},
        {'ge_attr':'udasave','ge_type':'uint64','d_len':1},
        
        {'ge_attr':'vibrant','ge_type':'int16','d_len':1},
        {'ge_attr':'asset_torso','ge_type':'int16','d_len':1},
        {'ge_attr':'asset_alt_cal','ge_type':'int32','d_len':1},
        
        {'ge_attr':'kacq_uid','ge_type':'int32','d_len':1},
        
        {'ge_attr':'cttEntry','ge_type':'char','d_len':944},
        
        {'ge_attr':'psc_ta','ge_type':'int32','d_len':1},
        {'ge_attr':'disk_acq_ctrl','ge_type':'int32','d_len':1},
        
        {'ge_attr':'asset_localTx','ge_type':'int32','d_len':1},
        
        {'ge_attr':'rh3dscale','ge_type':'float32','d_len':1},
        
        {'ge_attr':'broad_band_select','ge_type':'int32','d_len':1},
        
        {'ge_attr':'scanner_mode','ge_type':'int16','d_len':1},
        
        {'ge_attr':'numbvals','ge_type':'int16','d_len':1},
        {'ge_attr':'numdifdirs','ge_type':'int16','d_len':1},
        {'ge_attr':'difnext2','ge_type':'int16','d_len':1},
        {'ge_attr':'difnextab','ge_type':'int16','d_len':100},
        
        {'ge_attr':'channel_combine_method','ge_type':'int16','d_len':1},
        {'ge_attr':'nexForUnacquiredEncodes','ge_type':'int16','d_len':1},
        {'ge_attr':'excess','ge_type':'int16','d_len':612},
            )
    perpass_tuple=({'ge_attr':'perpass','ge_type':'char','d_len':16384},)
    unlock_tuple=({'ge_attr':'unlock_raw','ge_type':'char','d_len':16384},)
    
    data_acq_tuple=(
        {'ge_attr':'pass_number','ge_type':'int16','d_len':1},
        {'ge_attr':'slice_in_pass','ge_type':'int16','d_len':1},
        {'ge_attr':'gw_point','ge_type':'float32','d_len':(3,3)},
        {'ge_attr':'transpose','ge_type':'int16','d_len':1},
        {'ge_attr':'rotate','ge_type':'int16','d_len':1},
        {'ge_attr':'swiftcoilno','ge_type':'uint32','d_len':1},
                            )
    nex_tab_tuple=({'ge_attr':'nex_tab','ge_type':'char','d_len':2052},)
    nex_abort_tab_tuple=({'ge_attr':'nex_abort_tab','ge_type':'char','d_len':2052},)
    tool_tuple=({'ge_attr':'tool','ge_type':'char','d_len':2048},)
    prescan_tuple=(
        {'ge_attr':'command','ge_type':'int32','d_len':1},
        {'ge_attr':'mps_r1','ge_type':'int32','d_len':1},
        {'ge_attr':'mps_r2','ge_type':'int32','d_len':1},
        {'ge_attr':'mps_tg','ge_type':'int32','d_len':1},
        {'ge_attr':'mps_freq','ge_type':'uint32','d_len':1},
        {'ge_attr':'aps_r1','ge_type':'int32','d_len':1},
        {'ge_attr':'aps_r2','ge_type':'int32','d_len':1},
        {'ge_attr':'aps_tg','ge_type':'int32','d_len':1},
        {'ge_attr':'aps_freq','ge_type':'uint32','d_len':1},
        {'ge_attr':'scalei','ge_type':'float32','d_len':1},
        {'ge_attr':'scaleq','ge_type':'float32','d_len':1},
        {'ge_attr':'snr_warning','ge_type':'int32','d_len':1},
        {'ge_attr':'aps_or_mps','ge_type':'int32','d_len':1},
        {'ge_attr':'mps_bitmap','ge_type':'int32','d_len':1},
        {'ge_attr':'powerspec','ge_type':'char','d_len':256},
        {'ge_attr':'filler1','ge_type':'int32','d_len':1},
        {'ge_attr':'filler2','ge_type':'int32','d_len':1},
        {'ge_attr':'xshim','ge_type':'int16','d_len':1},
        {'ge_attr':'yshim','ge_type':'int16','d_len':1},
        {'ge_attr':'zshim','ge_type':'int16','d_len':1},
        {'ge_attr':'recon_enable','ge_type':'int16','d_len':1},
        {'ge_attr':'autoshim_status','ge_type':'int32','d_len':1},
        {'ge_attr':'rec_std','ge_type':'float32','d_len':128},
        {'ge_attr':'rec_mean','ge_type':'float32','d_len':128},
        {'ge_attr':'line_width','ge_type':'int32','d_len':1},
        {'ge_attr':'ws_flip','ge_type':'int32','d_len':1},
        {'ge_attr':'supp_lvl','ge_type':'int32','d_len':1},
        {'ge_attr':'psc_reuse','ge_type':'int32','d_len':1},
        {'ge_attr':'psc_reuse_string','ge_type':'char','d_len':52},
        {'ge_attr':'psc_ta','ge_type':'int32','d_len':1},
        {'ge_attr':'phase_correction_status','ge_type':'int32','d_len':1},
        {'ge_attr':'broad_band_select','ge_type':'int32','d_len':1},
        {'ge_attr':'buffer','ge_type':'char','d_len':64},
        )
#===============================================================================
#  EXAMDATATYPE rdb_hdr_exam
#===============================================================================
    exam_tuple=(
        {'ge_attr':'firstaxtime','ge_type':'float64','d_len':1},
        {'ge_attr':'double_padding','ge_type':'float64','d_len':31},
        {'ge_attr':'zerocell','ge_type':'float32','d_len':1},
        {'ge_attr':'cellspace','ge_type':'float32','d_len':1},
        {'ge_attr':'srctodet','ge_type':'float32','d_len':1},
        {'ge_attr':'srctoiso','ge_type':'float32','d_len':1},
        {'ge_attr':'float_padding','ge_type':'float32','d_len':32},
        {'ge_attr':'ex_delta_cnt','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_complete','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_seriesct','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_numarch','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_numseries','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_numunser','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_toarchcnt','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_prospcnt','ge_type':'int32','d_len':1}, 
        {'ge_attr':'ex_modelnum','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_modelcnt','ge_type':'int32','d_len':1},
        {'ge_attr':'int_padding1','ge_type':'int32','d_len':32},
        {'ge_attr':'numcells','ge_type':'int32','d_len':1},
        {'ge_attr':'magstrength','ge_type':'int32','d_len':1},
        {'ge_attr':'patweight','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_datetime','ge_type':'int32','d_len':1},
        {'ge_attr':'ex_lastmod','ge_type':'int32','d_len':1},
        {'ge_attr':'int_padding2','ge_type':'int32','d_len':27},
        {'ge_attr':'ex_no','ge_type':'uint16','d_len':1},
        {'ge_attr':'ex_uniq','ge_type':'int16','d_len':1},
        {'ge_attr':'detect','ge_type':'int16','d_len':1},  
        {'ge_attr':'tubetyp','ge_type':'int16','d_len':1},
        {'ge_attr':'dastyp','ge_type':'int16','d_len':1}, 
        {'ge_attr':'num_dcnk','ge_type':'int16','d_len':1},  
        {'ge_attr':'dcn_len','ge_type':'int16','d_len':1}, 
        {'ge_attr':'dcn_density','ge_type':'int16','d_len':1}, 
        {'ge_attr':'dcn_stepsize','ge_type':'int16','d_len':1}, 
        {'ge_attr':'dcn_shiftcnt','ge_type':'int16','d_len':1}, 
        {'ge_attr':'patage','ge_type':'int16','d_len':1}, 
        {'ge_attr':'patian','ge_type':'int16','d_len':1}, 
        {'ge_attr':'patsex','ge_type':'int16','d_len':1}, 
        {'ge_attr':'ex_format','ge_type':'int16','d_len':1}, 
        {'ge_attr':'trauma','ge_type':'int16','d_len':1}, 
        {'ge_attr':'protocolflag','ge_type':'int16','d_len':1}, 
        {'ge_attr':'study_status','ge_type':'int16','d_len':1}, 
        {'ge_attr':'short_padding','ge_type':'int16','d_len':35}, 
        {'ge_attr':'hist','ge_type':'char','d_len':257}, 
        {'ge_attr':'refphy','ge_type':'char','d_len':65}, 
        {'ge_attr':'diagrad','ge_type':'char','d_len':65}, 
        {'ge_attr':'op','ge_type':'char','d_len':65}, 
        {'ge_attr':'ex_desc','ge_type':'char','d_len':65}, 
        {'ge_attr':'ex_typ','ge_type':'char','d_len':3}, 
        {'ge_attr':'ex_sysid','ge_type':'char','d_len':9}, 
        {'ge_attr':'ex_alloc_key','ge_type':'char','d_len':13}, 
        {'ge_attr':'ex_diskid','ge_type':'char','d_len':1}, 
        {'ge_attr':'hospname','ge_type':'char','d_len':33},
        {'ge_attr':'ex_suid','ge_type':'char','d_len':4},
        {'ge_attr':'ex_verscre','ge_type':'char','d_len':2},
        {'ge_attr':'ex_verscur','ge_type':'char','d_len':2},
        {'ge_attr':'uniq_sys_id','ge_type':'char','d_len':16},
        {'ge_attr':'service_id','ge_type':'char','d_len':16},
        {'ge_attr':'mobile_loc','ge_type':'char','d_len':4},
        {'ge_attr':'study_uid','ge_type':'char','d_len':32},
        {'ge_attr':'refsopcuid','ge_type':'char','d_len':32},
        {'ge_attr':'refsopiuid','ge_type':'char','d_len':32},
        {'ge_attr':'patnameff','ge_type':'char','d_len':65}, 
        {'ge_attr':'patidff','ge_type':'char','d_len':65}, 
        {'ge_attr':'reqnumff','ge_type':'char','d_len':17}, 
        {'ge_attr':'dateofbirth','ge_type':'char','d_len':9}, 
        {'ge_attr':'mwlstudyuid','ge_type':'char','d_len':32}, 
        {'ge_attr':'mwlstudyid','ge_type':'char','d_len':16},
        {'ge_attr':'ex_padding','ge_type':'char','d_len':240},
        )
        #=======================================================================
        # SERIESDATATYPE rdb_hdr_series
        #=======================================================================
    series_tuple=(
        {'ge_attr':'double_padding','ge_type':'float64','d_len':32},
        {'ge_attr':'se_pds_a','ge_type':'float32','d_len':1},
        {'ge_attr':'se_pds_c','ge_type':'float32','d_len':1},
        {'ge_attr':'se_pds_u','ge_type':'float32','d_len':1},
        {'ge_attr':'lmhor','ge_type':'float32','d_len':1},
        {'ge_attr':'start_loc','ge_type':'float32','d_len':1},
        {'ge_attr':'end_loc','ge_type':'float32','d_len':1},
        {'ge_attr':'echo1_alpha','ge_type':'float32','d_len':1},
        {'ge_attr':'echo1_beta','ge_type':'float32','d_len':1}, 
        {'ge_attr':'echo2_alpha','ge_type':'float32','d_len':1}, 
        {'ge_attr':'echo2_beta','ge_type':'float32','d_len':1}, 
        {'ge_attr':'echo3_alpha','ge_type':'float32','d_len':1}, 
        {'ge_attr':'echo3_beta','ge_type':'float32','d_len':1}, 
        {'ge_attr':'echo4_alpha','ge_type':'float32','d_len':1},
        {'ge_attr':'echo4_beta','ge_type':'float32','d_len':1},
        {'ge_attr':'echo5_alpha','ge_type':'float32','d_len':1},
        {'ge_attr':'echo5_beta','ge_type':'float32','d_len':1},
        {'ge_attr':'echo6_alpha','ge_type':'float32','d_len':1},
        {'ge_attr':'echo6_beta','ge_type':'float32','d_len':1},
        {'ge_attr':'echo7_alpha','ge_type':'float32','d_len':1},
        {'ge_attr':'echo7_beta','ge_type':'float32','d_len':1},
        {'ge_attr':'echo8_alpha','ge_type':'float32','d_len':1},
        {'ge_attr':'echo8_beta','ge_type':'float32','d_len':1},
        {'ge_attr':'float_padding','ge_type':'float32','d_len':32},

        {'ge_attr':'se_complete','ge_type':'int32','d_len':1}, 
        {'ge_attr':'se_numarch','ge_type':'int32','d_len':1}, 
        {'ge_attr':'se_imagect','ge_type':'int32','d_len':1}, 
        {'ge_attr':'se_numimages','ge_type':'int32','d_len':1}, 
        {'ge_attr':'se_delta_cnt','ge_type':'int32','d_len':1},
        {'ge_attr':'se_numunimg','ge_type':'int32','d_len':1},
        {'ge_attr':'se_toarchcnt','ge_type':'int32','d_len':1},
        {'ge_attr':'int_padding1','ge_type':'int32','d_len':33},
        {'ge_attr':'se_datetime','ge_type':'int32','d_len':1},
        {'ge_attr':'se_actual_dt','ge_type':'int32','d_len':1},
        {'ge_attr':'position','ge_type':'int32','d_len':1},
        {'ge_attr':'entry','ge_type':'int32','d_len':1},
        {'ge_attr':'se_lndmrkcnt','ge_type':'int32','d_len':1},
        {'ge_attr':'se_lastmod','ge_type':'int32','d_len':1},
        {'ge_attr':'ExpType','ge_type':'int32','d_len':1}, 
        {'ge_attr':'TrRest','ge_type':'int32','d_len':1}, 
        {'ge_attr':'TrActive','ge_type':'int32','d_len':1}, 
        {'ge_attr':'DumAcq','ge_type':'int32','d_len':1}, 
        {'ge_attr':'ExptTimePts','ge_type':'int32','d_len':1}, 
        {'ge_attr':'int_padding2','ge_type':'int32','d_len':33},
        {'ge_attr':'se_exno','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo1_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo2_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo3_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo4_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo5_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo6_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo7_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo8_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'echo8_level','ge_type':'int16','d_len':1}, 
        {'ge_attr':'echo7_level','ge_type':'int16','d_len':1}, 
        {'ge_attr':'echo6_level','ge_type':'int16','d_len':1}, 
        {'ge_attr':'echo5_level','ge_type':'int16','d_len':1}, 
        {'ge_attr':'echo4_level','ge_type':'int16','d_len':1}, 
        {'ge_attr':'echo3_level','ge_type':'int16','d_len':1},
        {'ge_attr':'echo2_level','ge_type':'int16','d_len':1},
        {'ge_attr':'echo1_level','ge_type':'int16','d_len':1},
        {'ge_attr':'se_no','ge_type':'int16','d_len':1},
        {'ge_attr':'se_typ','ge_type':'int16','d_len':1},
        {'ge_attr':'se_source','ge_type':'int16','d_len':1},
        {'ge_attr':'se_plane','ge_type':'int16','d_len':1},
        {'ge_attr':'scan_type','ge_type':'int16','d_len':1},
        {'ge_attr':'se_uniq','ge_type':'int16','d_len':1},
        {'ge_attr':'se_contrast','ge_type':'int16','d_len':1},
        {'ge_attr':'se_pseq','ge_type':'int16','d_len':1}, 
        {'ge_attr':'se_sortorder','ge_type':'int16','d_len':1}, 
        {'ge_attr':'se_nacq','ge_type':'int16','d_len':1}, 
        {'ge_attr':'xbasest','ge_type':'int16','d_len':1}, 
        {'ge_attr':'xbaseend','ge_type':'int16','d_len':1}, 
        {'ge_attr':'xenhst','ge_type':'int16','d_len':1},
        {'ge_attr':'xenhend','ge_type':'int16','d_len':1},
        {'ge_attr':'table_entry','ge_type':'int16','d_len':1},
        {'ge_attr':'SwingAngle','ge_type':'int16','d_len':1},
        {'ge_attr':'LateralOffset','ge_type':'int16','d_len':1},
        {'ge_attr':'GradientCoil','ge_type':'int16','d_len':1},
        {'ge_attr':'se_subtype','ge_type':'int16','d_len':1},
        {'ge_attr':'BWRT','ge_type':'int16','d_len':1},
        {'ge_attr':'assetcal_serno','ge_type':'int16','d_len':1},
        {'ge_attr':'assetcal_scnno','ge_type':'int16','d_len':1}, 
        {'ge_attr':'content_qualifn','ge_type':'int16','d_len':1},
        {'ge_attr':'purecal_serno','ge_type':'int16','d_len':1},
        {'ge_attr':'purecal_scnno','ge_type':'int16','d_len':1},
        {'ge_attr':'ideal','ge_type':'int16','d_len':1},
        {'ge_attr':'short_padding','ge_type':'int16','d_len':33},
        {'ge_attr':'se_verscre','ge_type':'char','d_len':2},
        {'ge_attr':'se_verscur','ge_type':'char','d_len':2},
        {'ge_attr':'se_suid','ge_type':'char','d_len':4},
        {'ge_attr':'se_alloc_key','ge_type':'char','d_len':13},
        {'ge_attr':'se_diskid','ge_type':'char','d_len':1},
        {'ge_attr':'se_desc','ge_type':'char','d_len':65},
        {'ge_attr':'pr_sysid','ge_type':'char','d_len':9}, 
        {'ge_attr':'pansysid','ge_type':'char','d_len':9},
        {'ge_attr':'anref','ge_type':'char','d_len':3},
        {'ge_attr':'prtcl','ge_type':'char','d_len':25},
        {'ge_attr':'start_ras','ge_type':'char','d_len':1},
        {'ge_attr':'end_ras','ge_type':'char','d_len':1},
        {'ge_attr':'series_uid','ge_type':'char','d_len':32},
        {'ge_attr':'landmark_uid','ge_type':'char','d_len':32},
        {'ge_attr':'equipmnt_uid','ge_type':'char','d_len':32},
        {'ge_attr':'refsopcuids','ge_type':'char','d_len':32},
        {'ge_attr':'refsopiuids','ge_type':'char','d_len':32},
        {'ge_attr':'schacitval','ge_type':'char','d_len':16},
        {'ge_attr':'schacitdesc','ge_type':'char','d_len':16}, 
        {'ge_attr':'schacitmea','ge_type':'char','d_len':64},
        {'ge_attr':'schprocstdesc','ge_type':'char','d_len':65},
        {'ge_attr':'schprocstid','ge_type':'char','d_len':16},
        {'ge_attr':'reqprocstid','ge_type':'char','d_len':16},
        {'ge_attr':'perprocstid','ge_type':'char','d_len':16},
        {'ge_attr':'perprocstdesc','ge_type':'char','d_len':65},
        {'ge_attr':'reqprocstid2','ge_type':'char','d_len':16},
        {'ge_attr':'reqprocstid3','ge_type':'char','d_len':16},
        {'ge_attr':'schprocstid2','ge_type':'char','d_len':16},
        {'ge_attr':'schprocstid3','ge_type':'char','d_len':16},
        {'ge_attr':'refImgUID1','ge_type':'char','d_len':1*32},
        {'ge_attr':'refImgUID2','ge_type':'char','d_len':3*32},
        {'ge_attr':'PdgmStr','ge_type':'char','d_len':64}, 
        {'ge_attr':'PdgmDesc','ge_type':'char','d_len':256},
        {'ge_attr':'PdgmUID','ge_type':'char','d_len':64},
        {'ge_attr':'ApplName','ge_type':'char','d_len':16},
        {'ge_attr':'ApplVer','ge_type':'char','d_len':16},
        {'ge_attr':'asset_appl','ge_type':'char','d_len':12},
        {'ge_attr':'scic_a','ge_type':'char','d_len':32},
        {'ge_attr':'scic_s','ge_type':'char','d_len':32},
        {'ge_attr':'scic_c','ge_type':'char','d_len':32},
        {'ge_attr':'pure_cfg_params','ge_type':'char','d_len':64},
        {'ge_attr':'se_padding','ge_type':'char','d_len':251},
                )
#===========================================================================
# MRIMAGEDATATYPE rdb_hdr_image
#===========================================================================
    image_tuple=(
        {'ge_attr':'autosubparam','ge_type':'float32','d_len':12},
        {'ge_attr':'double_padding','ge_type':'float64','d_len':32},
        {'ge_attr':'dfov','ge_type':'float32','d_len':1},
        {'ge_attr':'dfov_rect','ge_type':'float32','d_len':1},
        {'ge_attr':'sctime','ge_type':'float32','d_len':1},
        {'ge_attr':'slthick','ge_type':'float32','d_len':1},
        {'ge_attr':'scanspacing','ge_type':'float32','d_len':1},
        {'ge_attr':'loc','ge_type':'float32','d_len':1},
        {'ge_attr':'tbldlta','ge_type':'float32','d_len':1},
        {'ge_attr':'nex','ge_type':'float32','d_len':1},
        {'ge_attr':'reptime','ge_type':'float32','d_len':1},
        {'ge_attr':'saravg','ge_type':'float32','d_len':1},
        {'ge_attr':'sarpeak','ge_type':'float32','d_len':1},
        {'ge_attr':'pausetime','ge_type':'float32','d_len':1},
        {'ge_attr':'vbw','ge_type':'float32','d_len':1},
        {'ge_attr':'user0','ge_type':'float32','d_len':1},
        {'ge_attr':'user1','ge_type':'float32','d_len':1},
        {'ge_attr':'user2','ge_type':'float32','d_len':1},
        {'ge_attr':'user3','ge_type':'float32','d_len':1},
        {'ge_attr':'user4','ge_type':'float32','d_len':1},
        {'ge_attr':'user5','ge_type':'float32','d_len':1},
        {'ge_attr':'user6','ge_type':'float32','d_len':1},
        {'ge_attr':'user7','ge_type':'float32','d_len':1},
        {'ge_attr':'user8','ge_type':'float32','d_len':1},
        {'ge_attr':'user9','ge_type':'float32','d_len':1},
        {'ge_attr':'user10','ge_type':'float32','d_len':1},
        {'ge_attr':'user11','ge_type':'float32','d_len':1},
        {'ge_attr':'user12','ge_type':'float32','d_len':1},
        {'ge_attr':'user13','ge_type':'float32','d_len':1},
        {'ge_attr':'user14','ge_type':'float32','d_len':1},
        {'ge_attr':'user15','ge_type':'float32','d_len':1},
        {'ge_attr':'user16','ge_type':'float32','d_len':1},
        {'ge_attr':'user17','ge_type':'float32','d_len':1},
        {'ge_attr':'user18','ge_type':'float32','d_len':1},
        {'ge_attr':'user19','ge_type':'float32','d_len':1},
        {'ge_attr':'user20','ge_type':'float32','d_len':1},
        {'ge_attr':'user21','ge_type':'float32','d_len':1},  
        {'ge_attr':'user22','ge_type':'float32','d_len':1},
        {'ge_attr':'proj_ang','ge_type':'float32','d_len':1},
        {'ge_attr':'concat_sat','ge_type':'float32','d_len':1},
        {'ge_attr':'user23','ge_type':'float32','d_len':1},
        {'ge_attr':'user24','ge_type':'float32','d_len':1},
        {'ge_attr':'x_axis_rot','ge_type':'float32','d_len':1},
        {'ge_attr':'y_axis_rot','ge_type':'float32','d_len':1},
        {'ge_attr':'z_axis_rot','ge_type':'float32','d_len':1},
        {'ge_attr':'ihtagfa','ge_type':'float32','d_len':1},
        {'ge_attr':'ihtagor','ge_type':'float32','d_len':1},
        {'ge_attr':'ihbspti','ge_type':'float32','d_len':1},
        {'ge_attr':'rtia_timer','ge_type':'float32','d_len':1},
        {'ge_attr':'fps','ge_type':'float32','d_len':1},
        {'ge_attr':'vencscale','ge_type':'float32','d_len':1},
        {'ge_attr':'dbdt','ge_type':'float32','d_len':1},
        {'ge_attr':'dbdtper','ge_type':'float32','d_len':1},
        {'ge_attr':'estdbdtper','ge_type':'float32','d_len':1},
        {'ge_attr':'estdbdtts','ge_type':'float32','d_len':1},
        {'ge_attr':'saravghead','ge_type':'float32','d_len':1},
        {'ge_attr':'neg_scanspacing','ge_type':'float32','d_len':1},
        {'ge_attr':'user25','ge_type':'float32','d_len':1},
        {'ge_attr':'user26','ge_type':'float32','d_len':1},
        {'ge_attr':'user27','ge_type':'float32','d_len':1},
        {'ge_attr':'user28','ge_type':'float32','d_len':1},
        {'ge_attr':'user29','ge_type':'float32','d_len':1},
        {'ge_attr':'user30','ge_type':'float32','d_len':1},
        {'ge_attr':'user31','ge_type':'float32','d_len':1},
        {'ge_attr':'user32','ge_type':'float32','d_len':1},
        {'ge_attr':'user33','ge_type':'float32','d_len':1},
        {'ge_attr':'user34','ge_type':'float32','d_len':1},
        {'ge_attr':'user35','ge_type':'float32','d_len':1},
        {'ge_attr':'user36','ge_type':'float32','d_len':1},
        {'ge_attr':'user37','ge_type':'float32','d_len':1},
        {'ge_attr':'user38','ge_type':'float32','d_len':1},
        {'ge_attr':'user39','ge_type':'float32','d_len':1},
        {'ge_attr':'user40','ge_type':'float32','d_len':1},
        {'ge_attr':'user41','ge_type':'float32','d_len':1},
        {'ge_attr':'user42','ge_type':'float32','d_len':1},
        {'ge_attr':'user43','ge_type':'float32','d_len':1},
        {'ge_attr':'user44','ge_type':'float32','d_len':1},
        {'ge_attr':'user45','ge_type':'float32','d_len':1},
        {'ge_attr':'user46','ge_type':'float32','d_len':1},
        {'ge_attr':'user47','ge_type':'float32','d_len':1},
        {'ge_attr':'user48','ge_type':'float32','d_len':1},
        {'ge_attr':'RegressorVal','ge_type':'float32','d_len':1},
        {'ge_attr':'SliceAsset','ge_type':'float32','d_len':1},
        {'ge_attr':'PhaseAsset','ge_type':'float32','d_len':1},
        {'ge_attr':'sarValues','ge_type':'float32','d_len':4},
        {'ge_attr':'shim_fov','ge_type':'float32','d_len':2},
        {'ge_attr':'shim_ctr_R','ge_type':'float32','d_len':2},
        {'ge_attr':'shim_ctr_A','ge_type':'float32','d_len':2},
        {'ge_attr':'shim_ctr_S','ge_type':'float32','d_len':2},
        {'ge_attr':'dim_X','ge_type':'float32','d_len':1},
        {'ge_attr':'dim_Y','ge_type':'float32','d_len':1},
        {'ge_attr':'pixsize_X','ge_type':'float32','d_len':1},
        {'ge_attr':'pixsize_Y','ge_type':'float32','d_len':1},
        {'ge_attr':'ctr_R','ge_type':'float32','d_len':1},
        {'ge_attr':'ctr_A','ge_type':'float32','d_len':1},
        {'ge_attr':'ctr_S','ge_type':'float32','d_len':1},
        {'ge_attr':'norm_R','ge_type':'float32','d_len':1},
        {'ge_attr':'norm_A','ge_type':'float32','d_len':1},
        {'ge_attr':'norm_S','ge_type':'float32','d_len':1},
        {'ge_attr':'tlhc_R','ge_type':'float32','d_len':1},
        {'ge_attr':'tlhc_A','ge_type':'float32','d_len':1},
        {'ge_attr':'tlhc_S','ge_type':'float32','d_len':1},
        {'ge_attr':'trhc_R','ge_type':'float32','d_len':1},
        {'ge_attr':'trhc_A','ge_type':'float32','d_len':1},
        {'ge_attr':'trhc_S','ge_type':'float32','d_len':1},
        {'ge_attr':'brhc_R','ge_type':'float32','d_len':1},
        {'ge_attr':'brhc_A','ge_type':'float32','d_len':1},
        {'ge_attr':'brhc_S','ge_type':'float32','d_len':1},
        {'ge_attr':'float_padding','ge_type':'float32','d_len':33},
        {'ge_attr':'cal_fldstr','ge_type':'uint32','d_len':1},
        {'ge_attr':'user_usage_tag','ge_type':'uint32','d_len':1},
        {'ge_attr':'user_fill_mapMSW','ge_type':'uint32','d_len':1},
        {'ge_attr':'user_fill_mapLSW','ge_type':'uint32','d_len':1},
        {'ge_attr':'im_archived','ge_type':'int32','d_len':1},
        {'ge_attr':'im_complete','ge_type':'int32','d_len':1},
        {'ge_attr':'int_padding1','ge_type':'int32','d_len':34},
        {'ge_attr':'im_datetime','ge_type':'int32','d_len':1},
        {'ge_attr':'im_actual_dt','ge_type':'int32','d_len':1},
        {'ge_attr':'tr','ge_type':'int32','d_len':1},
        {'ge_attr':'ti','ge_type':'int32','d_len':1},
        {'ge_attr':'te','ge_type':'int32','d_len':1},
        {'ge_attr':'te2','ge_type':'int32','d_len':1},
        {'ge_attr':'tdel','ge_type':'int32','d_len':1},
        {'ge_attr':'mindat','ge_type':'int32','d_len':1},
        {'ge_attr':'obplane','ge_type':'int32','d_len':1},
        {'ge_attr':'slocfov','ge_type':'int32','d_len':1},
        {'ge_attr':'obsolete1','ge_type':'int32','d_len':1},
        {'ge_attr':'obsolete2','ge_type':'int32','d_len':1},
        {'ge_attr':'user_bitmap','ge_type':'int32','d_len':1},
        {'ge_attr':'iopt','ge_type':'int32','d_len':1},
        {'ge_attr':'psd_datetime','ge_type':'int32','d_len':1},
        {'ge_attr':'rawrunnum','ge_type':'int32','d_len':1},
        {'ge_attr':'intr_del','ge_type':'int32','d_len':1},
        {'ge_attr':'im_lastmod','ge_type':'int32','d_len':1},
        {'ge_attr':'im_pds_a','ge_type':'int32','d_len':1},
        {'ge_attr':'im_pds_c','ge_type':'int32','d_len':1},
        {'ge_attr':'im_pds_u','ge_type':'int32','d_len':1},
        {'ge_attr':'thresh_min1','ge_type':'int32','d_len':1},
        {'ge_attr':'thresh_max1','ge_type':'int32','d_len':1},
        {'ge_attr':'thresh_min2','ge_type':'int32','d_len':1},
        {'ge_attr':'thresh_max2','ge_type':'int32','d_len':1},
        {'ge_attr':'numslabs','ge_type':'int32','d_len':1},
        {'ge_attr':'locsperslab','ge_type':'int32','d_len':1},
        {'ge_attr':'overlaps','ge_type':'int32','d_len':1},
        {'ge_attr':'slop_int_4','ge_type':'int32','d_len':1},
        {'ge_attr':'dfax','ge_type':'int32','d_len':1},
        {'ge_attr':'fphase','ge_type':'int32','d_len':1},
        {'ge_attr':'offsetfreq','ge_type':'int32','d_len':1},
        {'ge_attr':'b_value','ge_type':'int32','d_len':1},
        {'ge_attr':'iopt2','ge_type':'int32','d_len':1},
        {'ge_attr':'ihtagging','ge_type':'int32','d_len':1},
        {'ge_attr':'ihtagspc','ge_type':'int32','d_len':1},
        {'ge_attr':'ihfcineim','ge_type':'int32','d_len':1},
        {'ge_attr':'ihfcinent','ge_type':'int32','d_len':1},
        {'ge_attr':'num_seg','ge_type':'int32','d_len':1},
        {'ge_attr':'oprtarr','ge_type':'int32','d_len':1},
        {'ge_attr':'averages','ge_type':'int32','d_len':1},
        {'ge_attr':'station_index','ge_type':'int32','d_len':1},
        {'ge_attr':'station_total','ge_type':'int32','d_len':1},
        {'ge_attr':'iopt3','ge_type':'int32','d_len':1},
        {'ge_attr':'delAcq','ge_type':'int32','d_len':1},
        {'ge_attr':'rxmbloblen','ge_type':'int32','d_len':1},
        {'ge_attr':'rxmblob','ge_type':'int32','d_len':1},
        
        {'ge_attr':'im_no','ge_type':'int32','d_len':1},
        {'ge_attr':'imgrx','ge_type':'int32','d_len':1},
        
#        % ANP additions for R22 (MRE)
        {'ge_attr':'temp_phases','ge_type':'int32','d_len':1},
        {'ge_attr':'MEG_freq','ge_type':'int32','d_len':1},
        {'ge_attr':'driver_amp','ge_type':'int32','d_len':1},
        {'ge_attr':'driverCyc_Trig','ge_type':'int32','d_len':1},
        {'ge_attr':'MEG_dir','ge_type':'int32','d_len':1},
        
        {'ge_attr':'int_padding2','ge_type':'int32','d_len':26},
        
        {'ge_attr':'imatrix_X','ge_type':'int16','d_len':1},
        {'ge_attr':'imatrix_Y','ge_type':'int16','d_len':1},
        {'ge_attr':'im_exno','ge_type':'uint16','d_len':1},
        {'ge_attr':'img_window','ge_type':'uint16','d_len':1},
        {'ge_attr':'img_level','ge_type':'int16','d_len':1},
        {'ge_attr':'numecho','ge_type':'int16','d_len':1},
        {'ge_attr':'echonum','ge_type':'int16','d_len':1},
        {'ge_attr':'im_uniq','ge_type':'int16','d_len':1},
        {'ge_attr':'im_seno','ge_type':'int16','d_len':1},
        {'ge_attr':'contmode','ge_type':'int16','d_len':1},
        {'ge_attr':'serrx','ge_type':'int16','d_len':1},
        {'ge_attr':'screenformat','ge_type':'int16','d_len':1},
        {'ge_attr':'plane','ge_type':'int16','d_len':1},
        {'ge_attr':'im_compress','ge_type':'int16','d_len':1},
        {'ge_attr':'im_scouttype','ge_type':'int16','d_len':1},
        {'ge_attr':'contig','ge_type':'int16','d_len':1},
        {'ge_attr':'hrtrate','ge_type':'int16','d_len':1},
        {'ge_attr':'trgwindow','ge_type':'int16','d_len':1},
        {'ge_attr':'imgpcyc','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete3','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete4','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete5','ge_type':'int16','d_len':1},
        {'ge_attr':'mr_flip','ge_type':'int16','d_len':1},
        {'ge_attr':'cphase','ge_type':'int16','d_len':1},
        {'ge_attr':'swappf','ge_type':'int16','d_len':1},
        {'ge_attr':'pauseint','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete6','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete7','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete8','ge_type':'int16','d_len':1},
        {'ge_attr':'not_used_1','ge_type':'int16','d_len':1},
        {'ge_attr':'imode','ge_type':'int16','d_len':1},
        {'ge_attr':'pseq','ge_type':'int16','d_len':1},
        {'ge_attr':'pseqmode','ge_type':'int16','d_len':1},
        
        {'ge_attr':'ctyp','ge_type':'int16','d_len':1},
        {'ge_attr':'surfctyp','ge_type':'int16','d_len':1},
        {'ge_attr':'surfcext','ge_type':'int16','d_len':1},
        {'ge_attr':'supp_tech','ge_type':'int16','d_len':1},
        {'ge_attr':'slquant','ge_type':'int16','d_len':1},
        {'ge_attr':'gpre','ge_type':'int16','d_len':1},
        {'ge_attr':'satbits','ge_type':'int16','d_len':1},
        {'ge_attr':'scic','ge_type':'int16','d_len':1},
        {'ge_attr':'satxloc1','ge_type':'int16','d_len':1},
        {'ge_attr':'satxloc2','ge_type':'int16','d_len':1},
        {'ge_attr':'satyloc1','ge_type':'int16','d_len':1},
        {'ge_attr':'satyloc2','ge_type':'int16','d_len':1},
        {'ge_attr':'satzloc1','ge_type':'int16','d_len':1},
        {'ge_attr':'satzloc2','ge_type':'int16','d_len':1},
        {'ge_attr':'satxthick','ge_type':'int16','d_len':1},
        {'ge_attr':'satythick','ge_type':'int16','d_len':1},
        {'ge_attr':'satzthick','ge_type':'int16','d_len':1},
        {'ge_attr':'flax','ge_type':'int16','d_len':1},
        {'ge_attr':'venc','ge_type':'int16','d_len':1},
        {'ge_attr':'thk_disclmr','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete9','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete10','ge_type':'int16','d_len':1},
        {'ge_attr':'image_type','ge_type':'int16','d_len':1},
        {'ge_attr':'vas_collapse','ge_type':'int16','d_len':1},
        {'ge_attr':'proj_alg','ge_type':'int16','d_len':1},
        {'ge_attr':'echo_trn_len','ge_type':'int16','d_len':1},
        {'ge_attr':'frac_echo','ge_type':'int16','d_len':1},
        {'ge_attr':'prep_pulse','ge_type':'int16','d_len':1},
        {'ge_attr':'cphasenum','ge_type':'int16','d_len':1},
        {'ge_attr':'var_echo','ge_type':'int16','d_len':1},
        {'ge_attr':'scanactno','ge_type':'int16','d_len':1},
        {'ge_attr':'vasflags','ge_type':'int16','d_len':1},
        {'ge_attr':'integrity','ge_type':'int16','d_len':1},
        {'ge_attr':'freq_dir','ge_type':'int16','d_len':1},
        {'ge_attr':'vas_mode','ge_type':'int16','d_len':1},
        
        {'ge_attr':'pscopts','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete11','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete12','ge_type':'int16','d_len':1},
        {'ge_attr':'obsolete13','ge_type':'int16','d_len':1},
        {'ge_attr':'unoriginal','ge_type':'int16','d_len':1},
        {'ge_attr':'interleaves','ge_type':'int16','d_len':1},
        {'ge_attr':'effechospace','ge_type':'int16','d_len':1},
        {'ge_attr':'viewsperseg','ge_type':'int16','d_len':1},
        {'ge_attr':'irbpm','ge_type':'int16','d_len':1},
        {'ge_attr':'rtpoint','ge_type':'int16','d_len':1},
        {'ge_attr':'rcvrtype','ge_type':'int16','d_len':1},
        {'ge_attr':'sarMode','ge_type':'int16','d_len':1},
        {'ge_attr':'dBdtMode','ge_type':'int16','d_len':1},
        {'ge_attr':'govBody','ge_type':'int16','d_len':1},
        {'ge_attr':'sarDefinition','ge_type':'int16','d_len':1},
        {'ge_attr':'no_shimvol','ge_type':'int16','d_len':1},
        {'ge_attr':'shim_vol_type','ge_type':'int16','d_len':1},
        {'ge_attr':'current_phase','ge_type':'int16','d_len':1},
        {'ge_attr':'art_level','ge_type':'int16','d_len':1},
        {'ge_attr':'slice_group_number','ge_type':'int16','d_len':1},
        {'ge_attr':'number_of_slice_groups','ge_type':'int16','d_len':1},
        {'ge_attr':'show_in_autoview','ge_type':'int16','d_len':1},
#        % ANP additions for R20
        {'ge_attr':'slice_number_inGroup','ge_type':'int16','d_len':1},
        {'ge_attr':'specnuc','ge_type':'int16','d_len':1},
#        % ANP addition for R22
        {'ge_attr':'label_duration','ge_type':'uint16','d_len':1},
        {'ge_attr':'short_padding','ge_type':'int16','d_len':39},
        
        {'ge_attr':'psdname','ge_type':'char','d_len':33},
        {'ge_attr':'proj_name','ge_type':'char','d_len':13},
        {'ge_attr':'psd_iname','ge_type':'char','d_len':13},
        {'ge_attr':'im_diskid','ge_type':'char','d_len':1},
        {'ge_attr':'pdid','ge_type':'char','d_len':14},
        {'ge_attr':'im_suid','ge_type':'char','d_len':4},
        {'ge_attr':'contrastIV','ge_type':'char','d_len':17},
        {'ge_attr':'contrastOral','ge_type':'char','d_len':17},
        {'ge_attr':'loc_ras','ge_type':'char','d_len':1},
        {'ge_attr':'forimgrev','ge_type':'char','d_len':4},
        {'ge_attr':'cname','ge_type':'char','d_len':17},
        {'ge_attr':'im_verscre','ge_type':'char','d_len':2},
        {'ge_attr':'im_verscur','ge_type':'char','d_len':2},
        {'ge_attr':'im_alloc_key','ge_type':'char','d_len':13},
        {'ge_attr':'ref_img','ge_type':'char','d_len':1},
        {'ge_attr':'sum_img','ge_type':'char','d_len':1},
        {'ge_attr':'filter_mode','ge_type':'char','d_len':16},
        {'ge_attr':'slop_str_2','ge_type':'char','d_len':16},
        {'ge_attr':'image_uid','ge_type':'char','d_len':32},
        {'ge_attr':'sop_uid','ge_type':'char','d_len':32},
        {'ge_attr':'GEcname','ge_type':'char','d_len':24},
        {'ge_attr':'usedCoilData','ge_type':'char','d_len':100},
        {'ge_attr':'astcalseriesuid','ge_type':'char','d_len':32},
        {'ge_attr':'purecalseriesuid','ge_type':'char','d_len':32},
        {'ge_attr':'xml_psc_shm_vol','ge_type':'char','d_len':32},
        {'ge_attr':'rxmpath','ge_type':'char','d_len':64},
        {'ge_attr':'psdnameannot','ge_type':'char','d_len':33},
        {'ge_attr':'img_hdr_padding','ge_type':'char','d_len':250},
                 )
    hdr={}
    hdr['rdb']={}
    ParseGeHdrElement(hdr['rdb'], fid, rdb_tuple) 
    ParseGeHdrElement(hdr, fid, perpass_tuple) 
    ParseGeHdrElement(hdr, fid, unlock_tuple)
    
    hdr['data_acq_tab']={}
    for jj in range(0,2048): # stack over slices from 0 to 2047
#        print(jj)
        hdr['data_acq_tab'][jj]={}
        ParseGeHdrElement(hdr['data_acq_tab'][jj], fid, data_acq_tuple)

    ParseGeHdrElement(hdr, fid, nex_tab_tuple)
    ParseGeHdrElement(hdr, fid, nex_abort_tab_tuple)
    ParseGeHdrElement(hdr, fid, tool_tuple)
    
    hdr['prescan']={}
    ParseGeHdrElement(hdr['prescan'], fid, prescan_tuple)
    
    hdr['exam']={}
    ParseGeHdrElement(hdr['exam'], fid, exam_tuple)
    hdr['series']={}
    ParseGeHdrElement(hdr['series'], fid, series_tuple)
    
    hdr['image']={}
    ParseGeHdrElement(hdr['image'], fid, image_tuple) 
    
    hdr['rdb']['dab_start_rcv']=hdr['rdb']['buff'][0::2]
    hdr['rdb']['dab_stop_rcv']=hdr['rdb']['buff'][1::2]
       
    return hdr
    
    #        if not hdr.__hdr:
    #            hdr.__hdr = object.__init__(self)
def openfile(filename):
    try:
        fid = open(filename, "rb") # open filename, r: read, b: binary
    finally:
        pass
#        fid.close()
    return fid
def closefile(fid):
    fid.close()
    return 0
def openhdr(fid):
    hdr=gethdr(fid)
    return hdr
def openraw(fid,offset,data_type):
    import sys
    #print(offset)
    fid.seek(offset)
#     c=fid.read()
#     print(len(c))
#     a = numpy.fromstring( c, dtype = data_type) # must specify the data type
    a= numpy.fromfile(fid,dtype = data_type)
    # check whether the endianess is correct
    if sys.byteorder!='little': # PowerPC/Sparc CPU is not little-end
        a=a.byteswap()          # transform to little end
    return a
#    print 'reading', a[0:8:1]
    #print(a.size)
def raw2complex(a):    
    raw=a[0::2]+1j*a[1::2]
#k=k[0:512]
#    raw={}
    return raw
def raw2float(a):
    raw = a[0::1]
    return raw
def getsize(fileobject):
    fileobject.seek(0,2) # move the pointer to the end of the file
    size = fileobject.tell()
    return size
#class MyClass(object):
#    '''
#    classdocs
#    '''
#
#
#    def __init__(selfparams):
#        '''
#        Constructor
#        '''
class geV22:
    def __init__(self, filename):
        fid=openfile(filename)
        self.hdr=openhdr(fid)
        
        if self.hdr['rdb']['point_size'] == 4:
            data_type = numpy.int32
        elif self.hdr['rdb']['point_size'] == 2:
            data_type = numpy.int16
        
        
        offset=self.hdr['rdb']['off_data']
        print('data type',self.hdr['rdb']['point_size'])
        print('offset', offset)
        print('receiver_weight', self.hdr['prescan']['rec_std'])
        
        self.raw=openraw(fid,offset,data_type)
#         matplotlib.pyplot.plot(self.raw[4489216*4:4489216*4+2*2*256*16*16:1]) #4489216
#         matplotlib.pyplot.plot(self.raw[33619968*0:33619968*0+2*2*1024*32*16:1]) #4489216
#         matplotlib.pyplot.plot(self.raw[4194304*2:4194304*2+3*256*16*16:2])
#         matplotlib.pyplot.show() # rhwawsize
#         cccc = numpy.zeros((16*32,256))
#         for pp in range(0,8):
#             for jj in range(0,32):
#                 cdfd = self.raw[4489216*pp*2 + 2*256*16*jj :          4489216*pp*2 + 2*256*16*(jj+1)]
#                 cdfd = cdfd[0::2]+cdfd[0::2]*1.0j
#                 cccc[ jj*16:(jj+1)*16,:] = numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(numpy.reshape((cdfd[0:256*16]),[16,256]))))
# ##                 matplotlib.pyplot.subplot(32,1,jj+1)
# ##                 matplotlib.pyplot.imshow(numpy.abs(numpy.fft.fft2(numpy.reshape((cdfd[0:256*16]),[16,256]))))
#              
#             matplotlib.pyplot.imshow(numpy.abs(cccc))
#             matplotlib.pyplot.show()
        
        print('self.raw,',self.raw.shape)
        print('raw_pass_size_deprecated',self.hdr['rdb']['raw_pass_size_deprecated'])
        print('dfmptsize',self.hdr['rdb']['dfmptsize'])
        print('nex',self.hdr['image']['nex'])
        
        self.raw = raw2complex(self.raw)
        self.datasize=(getsize(fid) - self.hdr['rdb']['off_data'])/self.hdr['rdb']['point_size']/2
        print(self.datasize)
        print(self.raw.shape)
#        print(getsize(fid) )
#        print(self.hdr['rdb']['off_data'])
#        print(self.size)
#        print('size',self.size)
        ret_val=closefile(fid)
#        self.__shape=list(self.getshape())
        self.datashape=list(self.getshape())
#        self.datashape
        self.k=self.raw2k()
#         self.sortProp((self.datasize[0],  self.datasize[1],    self.datasize[3],  self.datasize[4]))                
        
        if ret_val == 0:
            pass
        else: 
            print('closing problems arisen')
    def getshape(self):

        xres = self.hdr['rdb']['da_xres']
#        xres=numpy.int32(xres)
        
        yres = self.hdr['rdb']['da_yres']-1
#        yres=numpy.int32(yres)
        
        zres = self.hdr['rdb']['nslices']/self.hdr['rdb']['npasses']
#        zres=numpy.int32(zres)
#        print(self.hdr['rdb']['dab_start_rcv'])
#        print(self.hdr['rdb']['dab_stop_rcv'])
        ncoils= (self.hdr['rdb']['dab_stop_rcv'][0] -
            self.hdr['rdb']['dab_start_rcv'][0] +1 ) 
        
        nechoes= self.hdr['rdb']['nechoes']
        print('xres',xres)
        print('yres',self.hdr['rdb']['nframes'])
        print('zres',zres)
        print('ncoils',ncoils)
        print('nechoes',nechoes)
#        print(xres,(yres),zres,ncoils,nechoes)
#        print(xres*(yres)*zres*ncoils)
#        print(numpy.size(self.raw))
#        print(self.datasize/(xres*(yres)*zres*ncoils)  )
        damp=self.datasize/(xres*yres*zres*ncoils)
        print('damp',damp)
        return (xres,yres,zres,ncoils, damp)
    def raw2k(self):
#        if self.shape is None:
#            self.shape=self.getshape()
#        tmpshape=self.shape
#        print('tmpshape',tmpshape)
#        tmpshape[1]=tmpshape[1]+1
        inter_shape=self.datashape
        print(self.datashape)        
        inter_shape[1]=inter_shape[1]+1
#         matplotlib.pyplot.plot(self.raw.real)
#         matplotlib.pyplot.show()
        print(inter_shape)        
        k=numpy.reshape(self.raw,inter_shape,order='F')
        k=k[:,1:,:,:]
        inter_shape[1]=inter_shape[1]-1

        return k  
#     def sortProp(self,prop_shape):
#         '''
#         zero phase correction
#         '''
# 
# #         out_prop = ()#numpy.empty(  prop_shape+(self.nslices,), dtype = numpy.complex)
#         out_angle = ()#numpy.empty( ( self.nblades, self.nslices),dtype = numpy.float)
#         
# #         import pp
# #         job_server = pp.Server()
# # 
# #         f0 = job_server.submit(zeroPhase, (self.k[...,0],),modules=('Im2Polar','numpy','correctmotion'),globals=globals())
# #         f1 = job_server.submit(zeroPhase, (self.k[...,1],),modules=('Im2Polar','numpy','correctmotion'),globals=globals())
# # 
# #         self.prop[...,0], tmp_angle0 = f0()
# #         self.prop[...,1], tmp_angle1 = f1()
# # 
# #         self.result_angle = (tmp_angle0 +tmp_angle1)/2.0
# 
# #         for zz in range(0,self.nslices):
# #             print('slice=',zz,'   ', zz*100.0/self.nslices,'%')
#         (self.k[...], dummy_angle)=  zeroPhase( self.k[...],11.25, 0) 
# #             out_prop=out_prop+(tmp_output,)
# #             self.k[...,zz] =  tmp_output
#         out_angle=out_angle+(dummy_angle,)
#  
# #         tmp_angle =out_angle[0]*0.0
# # #         
# #         for zz in range(0,self.nslices): # average the angles of all slices! 
# # #             self.k[...,zz] = out_prop[zz]
# #             tmp_angle =tmp_angle + out_angle[zz]
#                
#         self.prop = self.k
#         self.result_angle =  out_angle#tmp_angle/ self.nslices
            
    def sortProp(self,prop_shape):
        '''
        zero phase correction
        '''
#        self.prop=sortProp(self.k,prop_shape)
        self.prop=numpy.array(numpy.reshape(self.k,prop_shape,order='F'))
        (self.prop, self.result_angle)=zeroPhase(self.prop, 180.0/prop_shape[2],0)
  
#         for pp in range(0,self.prop.shape[2]): # blade
#             for pj in numpy.arange(0,self.prop.shape[3]): # coil
#                 cnt1=numpy.mean(self.prop[self.prop.shape[0]/2-1:self.prop.shape[0]/2+1,
#                           self.prop.shape[1]/2-1:self.prop.shape[1]/2+1,pp,pj])
#                 cnt1=cnt1/numpy.abs(cnt1)
#                 self.prop[:,:,pp,pj]=self.prop[:,:,pp,pj]/cnt1
#                 self.prop[:,:,pp,pj]=zeroPhase(self.prop[:,:,pp,pj])
    def subSample(self,Ny,SenseFactor):
        pass
class geBinary:
    def __init__(self, filename,offset,(xres,yres,zres,ncoils)):
        fid=openfile(filename)
#        self.hdr=openhdr(fid)
#        offset=self.hdr['rdb']['off_data']
        self.raw=openraw(fid,offset)
        self.raw=raw2complex(self.raw)
        self.datasize=(getsize(fid) - offset)/4.0
        print('number of complex numbers is', self.datasize,xres*yres*zres*ncoils)
#        print(getsize(fid) )
#        print(self.hdr['rdb']['off_data'])
#        print(self.size)
#        print('size',self.size)
        ret_val=closefile(fid)
#        self.__shape=list(self.getshape())
        self.datashape=[xres,yres,zres,ncoils]
#        self.datashape
        self.k=self.raw2k()
                
        
        if ret_val == 0:
            pass
        else: 
            print('problems arisen with closing file')
    def raw2k(self):
#        if self.shape is None:
#            self.shape=self.getshape()
#        tmpshape=self.shape
#        print('tmpshape',tmpshape)
#        tmpshape[1]=tmpshape[1]+1
        inter_shape=self.datashape
#        inter_shape[1]=inter_shape[1]+1
        k=numpy.reshape(self.raw,inter_shape,order='F')
#        k=k[:,1:,:,:]
#        inter_shape[1]=inter_shape[1]-1
        print(self.datashape)
        print(inter_shape)
        return k    
class cine2DBinary:
    def __init__(self, filename,vdfile,offset,(xres,yres,ncoils,ETL,nExcit)):
        self.vd=numpy.loadtxt(vdfile)
        fid=openfile(filename)
#        self.hdr=openhdr(fid)
#        offset=self.hdr['rdb']['off_data']
        data_type = numpy.int16
        self.raw=openraw(fid,offset,data_type)
        self.raw=raw2complex(self.raw)        
        self.datasize=(getsize(fid) - offset)/4.0
        print('number of complex numbers is', self.datasize,xres*ncoils*ETL*nExcit)
        if self.datasize!=xres*ncoils*ETL*nExcit:
            self.bl_flag=1
        else:
            self.bl_flag=0
            
#        print(getsize(fid) )
#        print(self.hdr['rdb']['off_data'])
#        print(self.size)
#        print('size',self.size)
        ret_val=closefile(fid)
#        self.__shape=list(self.getshape())
        self.datashape=[xres,yres, ncoils,ETL,nExcit]
        print('self_datashape',self.datashape)
#         matplotlib.pyplot.plot(self.raw)
#         matplotlib.pyplot.show()
        self.raw2k()
                
        
        if ret_val == 0:
            pass
        else: 
            print('problems arisen with closing file')
            
        self.dim_x=xres
        self.dim_y=yres
        self.ETL=ETL
        self.ncoils=ncoils
        self.nExcit=nExcit   
    def raw2k(self):
#        if self.shape is None:
#            self.shape=self.getshape()
#        tmpshape=self.shape
#        print('tmpshape',tmpshape)
#        tmpshape[1]=tmpshape[1]+1
        if self.bl_flag == 1:
            self.datashape[4]=self.datashape[4]+1
        else:
            pass
        
        inter_shape=self.datashape
#        inter_shape[1]=inter_shape[1]+1
        k=numpy.reshape(self.raw,(inter_shape[0], # xres
                                  inter_shape[2], # ncoils
                                  inter_shape[3], # ETL
                                  inter_shape[4]) # Excits 
                        ,order='F')
        
        if self.bl_flag == 1:        
            b=k[:,:,:,1:]
            self.datashape[4]=self.datashape[4]-1            
        else:
            b =k
            self.datashape[4]=self.datashape[4]
            
        print('datashape',self.datashape)
        b=numpy.reshape(b,[inter_shape[0],inter_shape[2],inter_shape[3]*inter_shape[4]],order='F')
        K=numpy.zeros((inter_shape[0],inter_shape[2],inter_shape[1]))+0.0j
        A=numpy.zeros((inter_shape[0],inter_shape[2],inter_shape[1]))+0.0j
#        k=numpy.zeros((256,256,4))+0.0j
        print('b.shape',b.shape)
        print('K.shape',K.shape)
        print('A.shape',A.shape)

        for pj in numpy.arange(0,self.datashape[3]*self.datashape[4]):
            numpy.size(self.vd)
            ind=int(self.vd[pj])

            K[:,:,ind-1]=K[:,:,ind-1]+b[:,:,pj]
            A[:,:,ind-1]=A[:,:,ind-1]+1.0
            
        self.k=numpy.transpose(K,(0,2,1))
        print('self.k.shape',self.k.shape)
        self.pdf=numpy.transpose(A,(0,2,1))
        print('self.pdf.shape',self.pdf.shape)
        self.f=numpy.transpose(b,[0,2,1])
        print('self.f.shape',self.f.shape)
       
#        mp.imshow(numpy.abs(numpy.fft.fft2((self.k[:,:,0]))))
#        mp.show()
        
        
        norm_k=numpy.zeros((self.k.shape[0],self.k.shape[1],self.k.shape[2]))+0.0j
        norm_k=self.k/self.pdf # temporal averaged k-space
        K2=numpy.array(b)
        for pj in numpy.arange(0,self.datashape[3]*self.datashape[4]):
            ind=int(self.vd[pj])
#             print(pj,ind)        
            K2[:,:,pj]=K2[:,:,pj]-norm_k[:,ind-1,:]
            
        self.f_diff=numpy.transpose(K2,[0,2,1])
#        for pj in range(0,self.k.shape[1]):
#            norm_k[:,:,pj]=numpy.fft.fftshift(
#                scipy.fftpack.ifft2(
#                 numpy.fft.ifftshift(
#                                self.k[:,pj,:]/self.pdf[:,pj,:]
#                                            )
#                                          )
#                                        )
        self.k_norm=norm_k
        self.tse=numpy.array(norm_k)
        self.tse=scipy.fftpack.ifftshift(self.tse,axes=(0,1,))
        
        self.tse=scipy.fftpack.ifftn(self.tse,axes=(0,1,))
        
        self.tse=scipy.fftpack.ifftshift(self.tse,axes=(0,1,))
#         matplotlib.pyplot.imshow(numpy.abs(numpy.sum(self.tse,2)),matplotlib.cm.gray)
#         matplotlib.pyplot.show()
        print('self.tse.shape',self.tse.shape)
class cine2DBinary_int32:
    def __init__(self, filename,vdfile,offset,(xres,yres,ncoils,ETL,nExcit)):
        self.vd=numpy.loadtxt(vdfile)
        fid=openfile(filename)
#        self.hdr=openhdr(fid)
#        offset=self.hdr['rdb']['off_data']
        data_type = numpy.int32
        self.raw=openraw(fid,offset,data_type).astype(numpy.float)
        
        self.raw=raw2complex(self.raw).astype(numpy.complex64)        
        self.datasize=(getsize(fid) - offset)/8.0
        print('number of complex numbers is', self.datasize,xres*ncoils*ETL*nExcit,  self.raw.dtype)
        if self.datasize!=xres*ncoils*ETL*nExcit:
            self.bl_flag=1
        else:
            self.bl_flag=0
            
#        print(getsize(fid) )
#        print(self.hdr['rdb']['off_data'])
#        print(self.size)
#        print('size',self.size)
        ret_val=closefile(fid)
#        self.__shape=list(self.getshape())
        self.datashape=[xres,yres, ncoils,ETL,nExcit]
        print('self_datashape',self.datashape)
#         matplotlib.pyplot.plot(self.raw)
#         matplotlib.pyplot.show()
        self.raw2k()
                
        
        if ret_val == 0:
            pass
        else: 
            print('problems arisen with closing file')
            
        self.dim_x=xres
        self.dim_y=yres
        self.ETL=ETL
        self.ncoils=ncoils
        self.nExcit=nExcit   
    def raw2k(self):
#        if self.shape is None:
#            self.shape=self.getshape()
#        tmpshape=self.shape
#        print('tmpshape',tmpshape)
#        tmpshape[1]=tmpshape[1]+1
        if self.bl_flag == 1:
            self.datashape[4]=self.datashape[4]+1
        else:
            pass
        
        inter_shape=self.datashape
#        inter_shape[1]=inter_shape[1]+1
        k=numpy.reshape(self.raw,(inter_shape[0], # xres
                                  inter_shape[2], # ncoils
                                  inter_shape[3], # ETL
                                  inter_shape[4]) # Excits 
                        ,order='F')
        
        if self.bl_flag == 1:        
            b=k[:,:,:,1:]
            self.datashape[4]=self.datashape[4]-1            
        else:
            b =k
            self.datashape[4]=self.datashape[4]
            
        print('datashape',self.datashape)
        b=numpy.reshape(b,[inter_shape[0],inter_shape[2],inter_shape[3]*inter_shape[4]],order='F')
        K=numpy.zeros((inter_shape[0],inter_shape[2],inter_shape[1]))+0.0j
        A=numpy.zeros((inter_shape[0],inter_shape[2],inter_shape[1]))+0.0j
#        k=numpy.zeros((256,256,4))+0.0j
        print('b.shape',b.shape)
        print('K.shape',K.shape)
        print('A.shape',A.shape)

        for pj in numpy.arange(0,self.datashape[3]*self.datashape[4]):
            numpy.size(self.vd)
            ind=int(self.vd[pj])

            K[:,:,ind-1]=K[:,:,ind-1]+b[:,:,pj]
            A[:,:,ind-1]=A[:,:,ind-1]+1.0
            
        self.k=numpy.transpose(K,(0,2,1))
        print('self.k.shape',self.k.shape)
        self.pdf=numpy.transpose(A,(0,2,1))
        print('self.pdf.shape',self.pdf.shape)
        self.f=numpy.transpose(b,[0,2,1])
        print('self.f.shape',self.f.shape)
       
#        mp.imshow(numpy.abs(numpy.fft.fft2((self.k[:,:,0]))))
#        mp.show()
        
        
        norm_k=numpy.zeros((self.k.shape[0],self.k.shape[1],self.k.shape[2]))+0.0j
        norm_k=self.k/self.pdf # temporal averaged k-space
        K2=numpy.array(b)
        for pj in numpy.arange(0,self.datashape[3]*self.datashape[4]):
            ind=int(self.vd[pj])
#             print(pj,ind)        
            K2[:,:,pj]=K2[:,:,pj]-norm_k[:,ind-1,:]
            
        self.f_diff=numpy.transpose(K2,[0,2,1])
#        for pj in range(0,self.k.shape[1]):
#            norm_k[:,:,pj]=numpy.fft.fftshift(
#                scipy.fftpack.ifft2(
#                 numpy.fft.ifftshift(
#                                self.k[:,pj,:]/self.pdf[:,pj,:]
#                                            )
#                                          )
#                                        )
        self.k_norm=norm_k
        self.tse=numpy.array(norm_k)
        self.tse[:]=scipy.fftpack.ifftshift(self.tse[:],axes=(0,1,))
        
        self.tse[:]=scipy.fftpack.ifftn(self.tse[:],axes=(0,1,))
        
        self.tse[:]=scipy.fftpack.ifftshift(self.tse[:],axes=(0,1,))
#         matplotlib.pyplot.imshow(numpy.abs(numpy.sum(self.tse,2)),matplotlib.cm.gray)
#         matplotlib.pyplot.show()
        print('self.tse.shape',self.tse.shape)        
class cine3DBinary(cine2DBinary):
    def __init__(self, filename,vdfile,offset,(xres,yres,zres, ncoils,ETL,nExcit)):
        self.vd=numpy.loadtxt(vdfile)
        fid=openfile(filename)
#        self.hdr=openhdr(fid)
#        offset=self.hdr['rdb']['off_data']
        self.raw=openraw(fid,offset)
        self.raw=raw2complex(self.raw)        
        self.datasize=(getsize(fid) - offset)/4.0
        print('number of complex numbers is', self.datasize,xres*ncoils*ETL*nExcit)
        if self.datasize!=xres*ncoils*ETL*nExcit:
            self.bl_flag=1
        else:
            self.bl_flag=0
            
#        print(getsize(fid) )
#        print(self.hdr['rdb']['off_data'])
#        print(self.size)
#        print('size',self.size)
        ret_val=closefile(fid)
#        self.__shape=list(self.getshape())
        self.datashape=[xres,yres, zres, ncoils,ETL,nExcit]
#        self.datashape
        self.raw2k()
                
        
        if ret_val == 0:
            pass
        else: 
            print('problems arisen with closing file')
            
        self.dim_x=xres
        self.dim_y=yres
        self.dim_z=zres
        self.ETL=ETL
        self.ncoils=ncoils
        self.nExcit=nExcit   
    def raw2k(self): # 3D cine FSE
#        if self.shape is None:
#            self.shape=self.getshape()
#        tmpshape=self.shape
#        print('tmpshape',tmpshape)
#        tmpshape[1]=tmpshape[1]+1
        if self.bl_flag == 1:
            self.datashape[5]=self.datashape[5]+1
        else:
            pass
        
        inter_shape=self.datashape
#        inter_shape[1]=inter_shape[1]+1
        k=numpy.reshape(self.raw,(inter_shape[0],
                                  inter_shape[3],
                                  inter_shape[4],
                                  inter_shape[5])
                        ,order='F')
        
        if self.bl_flag == 1:        
            b=k[:,:,:,1:]
            self.datashape[5]=self.datashape[5]-1            
        else:
            b =k
            self.datashape[5]=self.datashape[5]
            
        print('datashape',self.datashape)
        b=numpy.reshape(b,[inter_shape[0],inter_shape[3],inter_shape[4]*inter_shape[5]],order='F')
        K=numpy.zeros((inter_shape[0],inter_shape[3],inter_shape[1],inter_shape[2]),dtype =numpy.complex)
        A=numpy.zeros((inter_shape[0],inter_shape[3],inter_shape[1],inter_shape[2]),dtype =numpy.complex)
#        k=numpy.zeros((256,256,4))+0.0j
##################++++++++++++++++++++++++############__________________
        for pj in numpy.arange(0,numpy.size(self.vd)): # pj index of k lines 
            indy=int(self.vd[pj])
            indz=int(self.vd[pj])# check VD table
##################++++++++++++++++++++++++############__________________
#            print(ind)
            K[:,:,indy-1,indz-1]=K[:,:,indy-1,indz-1]+b[:,:,pj]
            A[:,:,indy-1,indz-1]=A[:,:,indy-1,indz-1]+1.0 # probability distribution of k space
            
        self.k=numpy.transpose(K,(0,2,3,1))
        print('self.k.shape',self.k.shape)
        self.pdf=numpy.transpose(A,(0,2,3,1))
        print('self.pdf.shape',self.pdf.shape)
        self.f=numpy.transpose(b,[0,2,3,1])
        print('self.f.shape',self.f.shape)
       
#        mp.imshow(numpy.abs(numpy.fft.fft2((self.k[:,:,0]))))
#        mp.show()
        
        
        norm_k=numpy.zeros((self.k.shape[0],self.k.shape[1],self.k.shape[2],self.k.shape[3]))+0.0j
        norm_k=self.k/self.pdf # temporal averaged k-space with PDF
        K2=numpy.array(b)
##################++++++++++++++++++++++++############__________________        
        for pj in numpy.arange(0,numpy.size(self.vd)):
            indy=int(self.vd[pj])
            indz=int(self.vd[pj])
            print(indy,indz)        
            K2[:,:,pj]=K2[:,:,pj]-norm_k[:,indy-1,indz-1,:]
            
        self.f_diff=numpy.transpose(K2,[0,2,3,1])
#        for pj in range(0,self.k.shape[1]):
#            norm_k[:,:,pj]=numpy.fft.fftshift(
#                scipy.fftpack.ifft2(
#                 numpy.fft.ifftshift(
#                                self.k[:,pj,:]/self.pdf[:,pj,:]
#                                            )
#                                          )
#                                        )
        self.k_norm=norm_k
        self.tse=numpy.array(norm_k)
        self.tse=scipy.fftpack.ifftshift(self.tse,axes=(0,1,2))
        
        self.tse=scipy.fftpack.ifftn(self.tse,axes=(0,1,2))
        
        self.tse=scipy.fftpack.ifftshift(self.tse,axes=(0,1,2))
        matplotlib.pyplot.imshow(numpy.abs(numpy.sum(self.tse,3)),matplotlib.cm.gray)
        matplotlib.pyplot.show()
        print('self.tse.shape',self.tse.shape)    
# class cineBinary2:
#     def __init__(self, filename,vdfile,offset,(xres,ncoils,ETL,nExcit)):
#         self.vd=numpy.loadtxt(vdfile)
#         fid=openfile(filename)
# #        self.hdr=openhdr(fid)
# #        offset=self.hdr['rdb']['off_data']
#         self.raw=openraw(fid,offset)
#         self.raw=raw2complex(self.raw)        
#         self.datasize=(getsize(fid) - offset)/4.0
#         print('number of complex numbers is', self.datasize,xres*ncoils*ETL*nExcit)
# #        print(getsize(fid) )
# #        print(self.hdr['rdb']['off_data'])
# #        print(self.size)
# #        print('size',self.size)
#         ret_val=closefile(fid)
# #        self.__shape=list(self.getshape())
#        
#         tmp_c= numpy.reshape(self.raw[0:256*4*528],(256,4,12*4,11),order='F')
#         
#         print('tmp_c.shape',tmp_c.shape)
# 
#         self.datashape=[xres,ncoils,ETL,nExcit]
# #        self.datashape
#         self.raw2k()
#                 
#         
#         if ret_val == 0:
#             pass
#         else: 
#             print('problems arisen with closing file')
#     def raw2k(self):
# #        if self.shape is None:
# #            self.shape=self.getshape()
# #        tmpshape=self.shape
# #        print('tmpshape',tmpshape)
# #        tmpshape[1]=tmpshape[1]+1
#         self.datashape[3]=self.datashape[3]
#         inter_shape=self.datashape
# #        inter_shape[1]=inter_shape[1]+1
# 
# 
#         
#         k=numpy.reshape(self.raw,inter_shape,order='F')
# 
#         
#         print('intershape',inter_shape)
#         b=k
#         self.datashape[3]=self.datashape[3]
#         print('datashape',self.datashape)
#         b=numpy.reshape(b,[inter_shape[0],inter_shape[1],inter_shape[2]*inter_shape[3]],order='F')
#       
#         
#         K=numpy.zeros((inter_shape[0],inter_shape[1],inter_shape[0]))+0.0j
#         A=numpy.zeros((inter_shape[0],inter_shape[1],inter_shape[0]))+0.0j
# #        k=numpy.zeros((256,256,4))+0.0j
# 
#         for pj in numpy.arange(0,numpy.size(self.vd)):
#             ind=int(self.vd[pj])
#             #print(ind)
#             K[:,:,ind-1]=K[:,:,ind-1]+b[:,:,pj]
#             A[:,:,ind-1]=A[:,:,ind-1]+1.0
#             
# 
#         self.k=numpy.transpose(K,(0,2,1))
# 
#         
#         print('self.k.shape',self.k.shape)
#         self.pdf=numpy.transpose(A,(0,2,1))
#         matplotlib.pyplot.imshow(numpy.abs(numpy.fft.fft2(numpy.sum(self.k/self.pdf,2))))
#         matplotlib.pyplot.show()
#         print('self.pdf.shape',self.pdf.shape)
#         self.f=numpy.transpose(b,[0,2,1])
#         print('self.f.shape',self.f.shape)
#         
#         norm_k=numpy.zeros((self.k.shape[0],self.k.shape[1],self.k.shape[2]))+0.0j
#         norm_k=self.k/self.pdf # temporal averaged k-space
#  
#         K2=numpy.array(b)
#         for pj in numpy.arange(0,numpy.size(self.vd)):
#             ind=int(self.vd[pj])
#             print(ind)        
#             K2[:,:,pj]=K2[:,:,pj]-norm_k[:,ind-1,:]
#         self.f_diff=numpy.transpose(K2,[0,2,1])
# #        for pj in range(0,self.k.shape[1]):
# #            norm_k[:,:,pj]=numpy.fft.fftshift(
# #                scipy.fftpack.ifft2(
# #                 numpy.fft.ifftshift(
# #                                self.k[:,pj,:]/self.pdf[:,pj,:]
# #                                            )
# #                                          )
# #                                        )
#         self.k_norm=norm_k
#         self.tse=numpy.array(norm_k)
#         self.tse=scipy.fftpack.ifftshift(self.tse)
#         self.tse=scipy.fftpack.ifftn(self.tse,[self.k.shape[0],self.k.shape[1]],(0,1))
#         self.tse=scipy.fftpack.ifftshift(self.tse,(0,1))
#         print('self.tse.shape',self.tse.shape)
class propBinary:
    def __init__(self, filename,offset,(xres,ncoils,ETL,nExcit)):
        #self.vd=numpy.loadtxt(vdfile)
        fid=openfile(filename)
#        self.hdr=openhdr(fid)
#        offset=self.hdr['rdb']['off_data']
        self.raw=openraw(fid,offset,numpy.int16)
        self.raw=raw2complex(self.raw)        
        self.datasize=(getsize(fid) - offset)/4.0
        print('number of complex numbers is', self.datasize,xres*ncoils*ETL*nExcit)
#        print(getsize(fid) )
#        print(self.hdr['rdb']['off_data'])
#        print(self.size)
#        print('size',self.size)
        ret_val=closefile(fid)
#        self.__shape=list(self.getshape())
        self.datashape=[xres,ncoils,ETL,nExcit]
#        self.datashape
        self.raw2k()
                
        
        if ret_val == 0:
            pass
        else: 
            print('problems arisen with closing file')
    def raw2k(self):
#        if self.shape is None:
#            self.shape=self.getshape()
#        tmpshape=self.shape
#        print('tmpshape',tmpshape)
#        tmpshape[1]=tmpshape[1]+1
        self.datashape[3]=self.datashape[3]
        inter_shape=self.datashape
        print('inter_shape',inter_shape)
#        inter_shape[1]=inter_shape[1]+1
        print('self.raw.shape',self.raw.shape)
        k=numpy.reshape(self.raw,inter_shape,order='F')
        self.k=k
      
    def sortProp(self,prop_shape,shift):
        self.k=numpy.transpose(self.k,(0,2,3,1))
#    def sortProp(KK,prop_shape):
        KK=self.k
        prop_shape=tuple(prop_shape[:])
        
        prop=numpy.array(numpy.reshape(KK,prop_shape,order='F'))
##        prop=numpy.roll(prop,0,axis=0)
#        tmp_prop=numpy.array(prop[:, prop.shape[1]/2-1:prop.shape[1]/2+1,:,:] )
#        for pj in range(0,len(prop_shape)-1):
#            tmp_prop=tmp_prop.sum(-1) # force memcpy
#        
#        ind=numpy.argmax(numpy.abs(tmp_prop))
#        print('numpy.argmax',ind, prop_shape[0]/2 )
#    ##    tmp_prop=
#        shift_ind= -ind + prop_shape[0]/2 
#        print('shfit_ind',shift_ind)
#        prop=numpy.roll(prop,shift_ind,axis=0)

        if shift == 1:
            #        prop=numpy.roll(prop,0,axis=0)
            tmp_prop=numpy.array(prop[:, prop.shape[1]/2-1:prop.shape[1]/2+1,:,:] )
            for pj in range(0,len(prop_shape)-1):
                tmp_prop=tmp_prop.sum(-1) # force memcpy
            
            ind=numpy.argmax(numpy.abs(tmp_prop))
            print('numpy.argmax',ind, prop_shape[0]/2 )
        ##    tmp_prop=
            shift_ind= -ind + prop_shape[0]/2 
            print('shfit_ind',shift_ind)
            prop=numpy.roll(prop,shift_ind,axis=0)
            prop=scipy.fftpack.fftshift(scipy.fftpack.fft2(prop, axes=(0,1))
                                                   ,1)
            prop = scipy.fftpack.ifft2(prop, axes=(0,1))
        elif shift == 0:
            pass
        
#         (self.prop, self.result_angle)=zeroPhase(prop)
   
class propTSE(propBinary):
    def sortProp(self,prop_shape,shift):
        #self.k=numpy.transpose(self.k,(0,2,3,1))
        self.k=numpy.transpose(self.k,(0,1,3,2))
#    def sortProp(KK,prop_shape):
        KK=self.k
        prop_shape=tuple(prop_shape[:])
        
        tmp_prop=numpy.array(numpy.reshape(KK,prop_shape,order='F'))
        prop=numpy.empty(tmp_prop.shape,dtype=numpy.complex64)
        for pj in range(0,prop_shape[1]):
#            print(pj)
            if numpy.mod(pj,2) == 0:
                ind=prop_shape[1]/2-1-pj/2
#                ind=prop_shape[1]/2+pj/2
                print(ind)
                prop[:,ind,:,:]=tmp_prop[:,pj,:,:]
            elif numpy.mod(pj,2) == 1:
                ind=prop_shape[1]/2-1+(pj+1)/2
#                ind=prop_shape[1]/2-1-(pj-1)/2
                print(ind)
                prop[:,ind,:,:]=tmp_prop[:,pj,:,:]
#        self.prop=prop
#        if shift == 1:
#            #        prop=numpy.roll(prop,0,axis=0)
#            tmp_prop=numpy.array(prop[:, prop.shape[1]/2-1:prop.shape[1]/2+1,:,:] )
#            for pj in range(0,len(prop_shape)-1):
#                tmp_prop=tmp_prop.sum(-1) # force memcpy
#            
#            ind=numpy.argmax(numpy.abs(tmp_prop))
#            print('numpy.argmax',ind, prop_shape[0]/2 )
#        ##    tmp_prop=
#            shift_ind= -ind + prop_shape[0]/2 
#            print('shfit_ind',shift_ind)
##            prop=numpy.roll(prop,shift_ind,axis=0)
#            prop=scipy.fftpack.fftshift(scipy.fftpack.fft2(prop, axes=(0,1))
#                                                   ,1)
#            prop = scipy.fftpack.ifft2(prop, axes=(0,1))
#        elif shift == 0:
#            pass
        #self.prop=(prop)
        (self.prop, )=zeroPhase(prop)
       
#cineObj=cineBinary2('../cinefse_20120214_g.data',
#            '../variable_density_sample_with_offset.txt', 0,  (256,4,12,44))

class AddenbrookesProp:
    def __init__(self,filename):
        fid=openfile(filename)
        self.hdr=openhdr(fid)


        self.datashape=list(self.getshape())       #(xres,yres,zres, num_blades, ncoils)   
        if self.hdr['rdb']['point_size'] == 4:
            data_type = numpy.int32
        elif self.hdr['rdb']['point_size'] == 2:
            data_type = numpy.int16
        
        num_blades = self.hdr['image']['user8']
        theta = self.hdr['image']['user9']
        prop_encode = self.hdr['image']['user23'] # new, encode mode 0=sequential, 1 = centric 2 = sparse
#         etl = self.hdr['rdb']['etl']
        etl = self.hdr['image']['echo_trn_len']
        xres = self.hdr['rdb']['da_xres']
        ncoils= (self.hdr['rdb']['dab_stop_rcv'][0] -
            self.hdr['rdb']['dab_start_rcv'][0] +1 ) 
        nechoes= self.hdr['rdb']['nechoes']
        nslices = self.hdr['rdb']['nslices']
        print('user11',self.hdr['image']['user11'])
        rhrawsize=self.hdr['image']['user11']/self.hdr['rdb']['point_size']/2

        print('rhrawsize=',rhrawsize)
        print('point_size', self.hdr['rdb']['point_size'])
        self.rhrawsize=rhrawsize
        self.nblades=num_blades
        self.theta=theta
        self.etl = etl
        self.xres = xres
        self.ncoils = ncoils
        self.nechoes= nechoes
        self.nslices = nslices
        self.prop_encode = prop_encode
        print('xres',xres)
#         print('yres',self.hdr['rdb']['nframes'])
        print('num_blade', num_blades)
        print('theta',theta)
        print('etl',etl)
        print('nslices',nslices)
        print('ncoils',ncoils)
        print('nechoes',nechoes)
                
        offset=self.hdr['rdb']['off_data']
        print('data type',self.hdr['rdb']['point_size'])
        print('offset', offset)
        print('receiver_weight', self.hdr['prescan']['rec_std'])
        print('passes=',self.hdr['rdb']['npasses'])
        self.raw=openraw(fid,offset,data_type)
        print('rawsize', numpy.size(self.raw)/rhrawsize)
        
        if numpy.mod(numpy.size(self.raw),2) == 0:
            pass
        else: # some of the BAM are not even
            self.raw = self.raw[:-1]

        self.raw = raw2complex(self.raw)

#         self.datasize=(getsize(fid) - self.hdr['rdb']['off_data'])/self.hdr['rdb']['point_size']/2
#         print(self.datasize)
        print(self.datashape)

        
        print('self.raw,',self.raw.shape)
        print('raw_pass_size_deprecated',self.hdr['rdb']['raw_pass_size_deprecated'])
        print('dfmptsize',self.hdr['rdb']['dfmptsize'])
        print('nex',self.hdr['image']['nex'])        
#        print(getsize(fid) )
#        print(self.hdr['rdb']['off_data'])
#        print(self.size)
#        print('size',self.size)
        ret_val=closefile(fid)
#        self.__shape=list(self.getshape())

#        self.datashape
        old_etl = etl # because etl will be changed next line
        old_datashape = self.datashape
        
        self.k, self.datashape, self.etl =self.raw2k()
        ## now everything is saved in self.k self.datashape, self.etl
        
        
        # now change the shape of blades
        
        k = self.k
        old_etl = self.etl
        if self.prop_encode == 1: # centric view order
            k = self.interleave(k) 
            new_etl = self.etl
        elif self.prop_encode == 200: # sparse centric view order
#             k = self.interleave(k)
            vardense =[-32,-29, -28, -27,
                             -25, -24,  -21,  -19,
                             -15, -13,   -8,   -6,
                             -3,  -2 ,   -1,    0,
                              1,   2 ,    3,    4,
                              6,   8 ,   11,   15,
                              17,  20,   23,   24,
                              25,  27,   30,   32]
#             vardense = [ -28,  -27,   -26,    -23, 
#                              -20,   -17,   -13,   -12, 
#                             -8,    -7,  -6,    -5,  
#                             -3,    -2,    -1,     0, 
#                             1,     2,     3,     4, 
#                             5,     6,     7,     8, 
#                             9,    11,    15,    19, 
#                             24,    25,    26,    27]
#             vardense = numpy.arange(0,32) - 16 +1 
            k,self.datashape,new_etl = self.sparse(k,vardense,self.datashape)
            
        elif self.prop_encode == 3: # sparse view order, opuser23
            '''
            from
            http://www.asawicki.info/Download/Productions/Applications/PoissonDiscGenerator/1D.txt
            '''
            vardense =[-32,-29, -28, -27,
                             -25, -24,  -21,  -19,
                             -15, -13,   -8,   -6,
                             -3,  -2 ,   -1,    0,
                              1,   2 ,    3,    4,
                              6,   8 ,   11,   15,
                              17,  20,   23,   24,
                              25,  27,   30,   32]
#             vardense = [ -28,  -27,   -26,    -23, 
#                              -20,   -17,   -13,   -12, 
#                             -8,    -7,  -6,    -5,  
#                             -3,    -2,    -1,     0, 
#                             1,     2,     3,     4, 
#                             5,     6,     7,     8, 
#                             9,    11,    15,    19, 
#                             24,    25,    26,    27]
#             vardense = numpy.arange(0,32) - 16 +1 
            k,self.datashape,new_etl = self.sparse(k,vardense,self.datashape) # doing mrics, replace etl with new etl
#             new_etl = self.etl  
        else: #self.prop_encode == 0:
            new_etl = self.etl # do not change etl here
            pass # do nothing if 0        
        # self.datashape is overwritten as #(xres,yres,num_blades, ncoils, zres)
        # self.k is reshaped as self.datashape=(xres,yres,num_blades, ncoils, zres)
        # self.etl may change if opuser23 == 2: sparse

        if ret_val == 0:
            pass
        else: 
            print('closing problems')
        self.k = k
#         self.etl = new_etl
        self.sortProp((self.xres,   new_etl,    num_blades,   ncoils))
        
        if ( self.prop_encode == 3  ) : #or ( self.prop_encode == 2  ): # sparse view order, opuser23
            vardense =[-32,-29, -28, -27,
                             -25, -24,  -21,  -19,
                             -15, -13,   -8,   -6,
                             -3,  -2 ,   -1,    0,
                              1,   2 ,    3,    4,
                              6,   8 ,   11,   15,
                              17,  20,   23,   24,
                              25,  27,   30,   32]
#             vardense = numpy.arange(0,32) - 16 +1
#             vardense = [ -28,  -27,   -26,    -23,    -20,   -17,   -13,   -12,    -8,    -7,
#                 -6,    -5,    -3,    -2,    -1,     0,     1,     2,     3,     4,     5,     6,     7,
#              8,     9,    11,    15,    19,    24,    25,    26,    27] 
            
            self.k, self.datashape, old_etl = self.restore_sparse(self.k,   vardense,   self.datashape,old_etl)
            
        #end of self.prop_encode == 3: # sparse view order, opuser23
        
        self.prop =self.k
        self.etl = old_etl 
        
    def getshape(self):

        xres = self.hdr['rdb']['da_xres']
#        xres=numpy.int32(xres)
        
#         yres = self.hdr['rdb']['etl']   #self.hdr['rdb']['da_yres']-1
        yres = self.hdr['image']['echo_trn_len'] 
#        yres=numpy.int32(yres)
        
        zres = self.hdr['rdb']['nslices']/self.hdr['rdb']['npasses']
        num_blades = self.hdr['image']['user8']
#         theta = self.hdr['image']['user9']
#         etl = self.hdr['rdb']['etl']        
#        zres=numpy.int32(zres)
#        print(self.hdr['rdb']['dab_start_rcv'])
#        print(self.hdr['rdb']['dab_stop_rcv'])
        ncoils= (self.hdr['rdb']['dab_stop_rcv'][0] -
            self.hdr['rdb']['dab_start_rcv'][0] +1 ) 
        
        nechoes= self.hdr['rdb']['nechoes']
        print('xres',xres)
        print('yres',self.hdr['rdb']['nframes'])
        print('zres',zres)
        print('ncoils',ncoils)
        print('nechoes',nechoes)
#        print(xres,(yres),zres,ncoils,nechoes)
#        print(xres*(yres)*zres*ncoils)
#        print(numpy.size(self.raw))
#        print(self.datasize/(xres*(yres)*zres*ncoils)  )
#         damp=self.datasize/(xres*yres*zres*ncoils)
#         print('damp',damp)
        return (xres,yres,zres, num_blades, ncoils) 
    
    def interleave(self,raw): # convert interleaved data to be sequential
        new_raw = numpy.copy(raw)
        etl = numpy.shape(raw)[1]
        viewtab = numpy.zeros(etl,dtype = numpy.int)
        step_sign = -1
        for jj in xrange(0,numpy.shape(raw)[1]):
            viewtab[jj] = etl/2 + numpy.floor((jj+1)/2)*step_sign -1
            step_sign = (-1)*step_sign
            new_raw[:,viewtab[jj],...]=raw[:,jj,...]
        return new_raw 
    def restore_sparse(self,raw,vardense,newdatashape, oldetl):
        # newdatashape#(xres,yres,num_blades, ncoils, zres)
        import mrics
   
        new_etl,phase_encode = mrics.sparse2blade_conf(oldetl,vardense)
        
        olddatashape = (newdatashape[0],oldetl,newdatashape[2],newdatashape[3],newdatashape[4] )
        new_raw = numpy.empty(olddatashape,dtype = numpy.complex64)
        for mm in xrange(0,olddatashape[1]):
            try:
                new_raw[:,mm,...] = raw[:,phase_encode[mm]+new_etl/2,...]
                print('mm = ',mm, 'phase_encode ',phase_encode[mm])
            except:
                print('failed at mm = ',mm, 'phase_encode ',phase_encode[mm])
        return new_raw,olddatashape,oldetl
    
    def sparse(self,raw,vardense,olddatashape):
        print('inside sparse')
        print('inside sparse')
        print('inside sparse')
        print('inside sparse')
        # newdatashape#(xres,yres,num_blades, ncoils, zres)
        import mrics
        new_raw = raw
        etl = self.etl
        N = self.xres# The image will be NxN
    #     sparsity = 0.1 # use only 25% on the K-Space data for CS 
        mu = 1.0
        LMBD = 0.1
        gamma = mu/1000
        nInner = 5
        nBreg = 10
    
        new_etl,phase_encode = mrics.sparse2blade_conf(etl,vardense)
        
        newdatashape = (olddatashape[0],new_etl,olddatashape[2],olddatashape[3],olddatashape[4] )
        new_raw = numpy.empty(newdatashape,dtype = numpy.complex64)
        for mm in xrange(0,newdatashape[2]):
            for nn in xrange(0,newdatashape[3]):
                for pp in xrange(0,newdatashape[4]):
                    raw_blade = raw[:,:,mm,nn,pp]
                    F,R = mrics.sparse2blade_init(raw_blade, etl,new_etl, phase_encode) 
                 
                    recovered = mrics.mrics(R,F, mu, LMBD, gamma,nInner, nBreg)
#                     newblade = numpy.fft.fftshift(numpy.fft.fft2(recovered))
                    newblade = mrics.sparse2blade_final(recovered, F,R)
                    new_raw[:,:,mm,nn,pp] = newblade
    



        return new_raw,newdatashape,new_etl
    def raw2k(self):
#         import matplotlib.pyplot
#        if self.shape is None:
#            self.shape=self.getshape()
#        tmpshape=self.shape
#        print('tmpshape',tmpshape)
#        tmpshape[1]=tmpshape[1]+1
        rhrawsize=self.rhrawsize
        
#         inter_shape=self.datashape
        ccc= numpy.empty(numpy.prod(self.datashape), dtype = numpy.complex)
#         xres, etl, nslices, ncoils, blades
        for ii in xrange(0, self.datashape[4]): # blades
            ccc[numpy.prod(self.datashape[0:4])*ii : numpy.prod(self.datashape[0:4])*(ii+1) 
                ] = self.raw[rhrawsize*ii : rhrawsize*ii+numpy.prod(self.datashape[0:4])]
            print('coil = ',ii, numpy.prod(self.datashape[0:4]))
            #(xres,yres,zres, num_blades, ncoils) 
#         print('is nan?',numpy.sum(numpy.isnan(ccc)))
#         print('is inf?',numpy.sum(numpy.isinf(ccc)))
#         matplotlib.pyplot.plot(numpy.real(ccc))
   
                
        self.raw = numpy.nan_to_num(ccc)   # remove all NaN  
                    
        print(self.datashape)   #(xres,yres,zres, num_blades, ncoils)   
#         inter_shape[1]=inter_shape[1]
#         matplotlib.pyplot.plot(self.raw.real)
#         matplotlib.pyplot.show()
#         print(inter_shape)        
        k=numpy.reshape(self.raw, self.datashape, order='F')
        #(xres,yres,zres, num_blades, ncoils)  
        for run_coil in xrange(0, self.datashape[4]):
            k[...,run_coil] =k[...,run_coil]/self.hdr['prescan']['rec_std'][run_coil] 
            # normalize the data according to coil noises 
        
        newdatashape=( self.datashape[0],self.datashape[1], self.datashape[3], self.datashape[4], self.datashape[2])
        # self.datashape is overwritten as #(xres,yres,num_blades, ncoils, zres)
        k = numpy.transpose(k,(0,1,3,4,2)) # reaarange the shape of data
#         if self.prop_encode == 1: # centric view order
#             k = self.interleave(k) 
#             new_etl = self.etl
#         elif self.prop_encode == 3: # sparse view order, opuser23
# 
#             '''
#             from
#             http://www.asawicki.info/Download/Productions/Applications/PoissonDiscGenerator/1D.txt
#             '''
#             vardense =[-32,-29, -28, -27,
#                              -25, -24,  -21,  -19,
#                              -15, -13,   -8,   -6,
#                              -3,  -2 ,   -1,    0,
#                               1,   2 ,    3,    4,
#                               6,   8 ,   11,   15,
#                               17,  20,   23,   24,
#                               25,  27,   30,   32]
# #             vardense = numpy.arange(0,32) - 16 +1 
#             print("inside phase_encode == 3 conditions!",self.prop_encode)
#             print("inside phase_encode == 3 conditions!",self.prop_encode)
#             print("inside phase_encode == 3 conditions!")
#             k,newdatashape,new_etl = self.sparse(k,vardense,newdatashape) # doing mrics, replace etl with new etl
# #             new_etl = self.etl  
#         else: #self.prop_encode == 0:
#             new_etl = self.etl # do not change etl here
#             pass # do nothing if 0
        
        
         
        print('numpy.prod(self.datashape)',numpy.prod(self.datashape))
        self.raw = numpy.reshape(k, (numpy.prod(newdatashape),),order='F')

        new_etl = self.etl

        return k, newdatashape, new_etl   
    def sortProp(self,prop_shape):
        '''
        zero phase correction
        '''

#         out_prop = ()#numpy.empty(  prop_shape+(self.nslices,), dtype = numpy.complex)
        out_angle = ()#numpy.empty( ( self.nblades, self.nslices),dtype = numpy.float)
        
#         import pp
#         job_server = pp.Server()
# 
#         f0 = job_server.submit(zeroPhase, (self.k[...,0],),modules=('Im2Polar','numpy','correctmotion'),globals=globals())
#         f1 = job_server.submit(zeroPhase, (self.k[...,1],),modules=('Im2Polar','numpy','correctmotion'),globals=globals())
# 
#         self.prop[...,0], tmp_angle0 = f0()
#         self.prop[...,1], tmp_angle1 = f1()
# 
#         self.result_angle = (tmp_angle0 +tmp_angle1)/2.0

        for zz in range(0,self.nslices):
            print('slice=',zz,'   ', zz*100.0/self.nslices,'%')
            (self.k[...,zz], dummy_angle)=  zeroPhase( self.k[..., zz], self.theta,self.prop_encode) 
#             out_prop=out_prop+(tmp_output,)
#             self.k[...,zz] =  tmp_output
            out_angle=out_angle+(dummy_angle,)
 
#         tmp_angle =out_angle[0]*0.0
# #         
#         for zz in range(0,self.nslices): # average the angles of all slices! 
# #             self.k[...,zz] = out_prop[zz]
#             tmp_angle =tmp_angle + out_angle[zz]
               
        self.prop = self.k
        self.result_angle =  out_angle#tmp_angle/ self.nslices
        
def unpack_uid(uid):
    """Convert packed PFile UID to standard DICOM UID."""
    inter_mark = ''
    
#     for c in uid:
#         print(ord(c))
#         print(ord(c) >> 4, ord(c) & 15)
#         
#     
#     pair_base = [(ord(c) >> 4, ord(c) & 15) for c in uid]
#     print(pair_base)
# #     uid_tmp = []
# #     for pair in pair_base:
# #         for i in pair:
# #             if i >0:
# #                 if i < 11:
# #                     uid_tmp = uid_tmp.append(str(i-1))
# #                 else:
# #                     uid_tmp = uid_tmp.append('.')
#     
    uid_tmp = [str(i-1) if i < 11 else '.' for pair in [(ord(c) >> 4, ord(c) & 15) for c in uid] for i in pair if i > 0]

    dicomuid=inter_mark.join(uid_tmp)
    
    return dicomuid
#     return ''.join([str(i-1) if i < 11 else '.' for pair in [(ord(c) >> 4, ord(c) & 15) for c in uid] for i in pair if i > 0])
        
        
if __name__ == "__main__":
#     myPfile = AddenbrookesProp('/home/sram/Cambridge_2012/DATA_MATLAB/Pfiles/pfile_20140120/P65536.7') # 2014/Jan/20
    myPfile = AddenbrookesProp('/usr/g/mrraw/P82432.7')
#     myPfile = AddenbrookesProp('P59904.7') # 2013/Dec/06
#     myPfile = AddenbrookesProp('P42496.7') # 2013/Dec/06
#     myPfile = AddenbrookesProp('P82944.7') # 2013/Dec/06
#     myPfile=AddenbrookesProp('P68096.7') 
#     myPfile=AddenbrookesProp('P82944.7')
#     myPfile=AddenbrookesProp('P58880.7')
#     myPfile=AddenbrookesProp('P59904.7')
#     matplotlib.pyplot.imshow(myPfile.k)
#     myPfile.sortProp((256,16,16,8))
#     myPfile = geV22('pfile_raw_data_fse_bl16_etl16.pfile')
#     print('header',myPfile.hdr['rdb'])
    print('image uid',unpack_uid(myPfile.hdr['image']['image_uid']))
    print('study uid',unpack_uid(myPfile.hdr['exam']['study_uid']))
    print('series uid',unpack_uid(myPfile.hdr['series']['series_uid']))
    print('ref class uid',unpack_uid(myPfile.hdr['exam']['mwlstudyuid']))
    print('equip uid ',unpack_uid(myPfile.hdr['series']['equipmnt_uid']))
    print('lamrk uid ',unpack_uid(myPfile.hdr['series']['landmark_uid']))
    print('refImgUID1',unpack_uid(myPfile.hdr['series']['refImgUID1']))
    print('refImgUID2',unpack_uid(myPfile.hdr['series']['refImgUID2']))
    print('SOPclassUID',unpack_uid(myPfile.hdr['image']['sop_uid']))
    print('image.freq_dir',(myPfile.hdr['image']['freq_dir']))
    print('image.palne',myPfile.hdr['image']['plane'])
#     print('obsolete1',myPfile.hdr['rdb']['obsolete1'])
#     print('obsolete2',myPfile.hdr['rdb']['obsolete2'])
    print('rec_std',myPfile.hdr['prescan']['rec_std'])
    print('rec_mean',myPfile.hdr['prescan']['rec_mean'])
#     print('ref instance uid',unpack_uid(myPfile.hdr['exam']['refsopiuid']))      
    print('header',myPfile.hdr['image']['GEcname'])
    print('header',myPfile.hdr['rdb']['fatwater'])
    print('fermi radius', myPfile.hdr['rdb']['fermi_radius'])
    print('fermi width', myPfile.hdr['rdb']['fermi_width'])
    print('user9, theta',myPfile.hdr['image']['user9'])
    print('user23, phase_encode',myPfile.hdr['image']['user23'])
    print('scancent',myPfile.hdr['rdb']['scancent'])
    print('position',myPfile.hdr['rdb']['position'])
    print('bw',myPfile.hdr['rdb']['bw'])
    print('gw_point',myPfile.hdr['data_acq_tab'][0]['gw_point'])
    print('gw_point',myPfile.hdr['data_acq_tab'][1]['gw_point'])
    print('gw_point',myPfile.hdr['data_acq_tab'][2]['gw_point'])
    print('gw_point',myPfile.hdr['data_acq_tab'][3]['gw_point'])
    print('gw_point',myPfile.hdr['data_acq_tab'][4]['gw_point'])
    print('gw_point',myPfile.hdr['data_acq_tab'][5]['gw_point'])
    print('gw_point',myPfile.hdr['data_acq_tab'][6]['gw_point'])
    print('refframes',unpack_uid(myPfile.hdr['series']['refImgUID1']))
    print('refframes',unpack_uid(myPfile.hdr['series']['refImgUID2']))
    print('echoes2skip',myPfile.hdr['rdb']['ech2skip'])
