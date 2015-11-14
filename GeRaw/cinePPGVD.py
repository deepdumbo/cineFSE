import numpy
import matplotlib.pyplot
def regTSE(seq_trains,input_indx,VD_table):
    
    
#    if length(seq_trains(:)) ~= length(VD_table(:))
#    error('Sizes of output_trains and VD_table are mismatched!')
#    end
#    print(seq_trains)
    N=numpy.size(seq_trains);
#    print(N)
    T=numpy.zeros((N,2));
    tmp_vd=numpy.reshape(VD_table,(N,1))
    for jj in numpy.arange(0,N): # looping from 0 to N-1
        t=seq_trains[jj]-1; 
        t_floor=numpy.floor(t); 
        t_ceil=numpy.ceil(t);
    
        T[jj,0]= (input_indx[t_floor]*abs(t_ceil-t)+ input_indx[t_ceil]*abs(t-t_floor) )/1;
        # linear interpolator
        
        T[jj,1]=tmp_vd[jj];
        
    return T

def seqTime(ETL, nExcite, TR, refoci,res):
    
    first180=numpy.float64(refoci[0])
    last180=numpy.float64(refoci[1])
    EkSpc=(last180-first180)/ETL
   
    echo_trains=numpy.transpose(numpy.arange(0,ETL)+0.5)*EkSpc+first180
    
    for jj in numpy.arange(0,nExcite):
        if jj == 0:
            output_trains = echo_trains
        else:
            output_trains = numpy.append(output_trains,echo_trains+TR*jj)

    output_trains=output_trains/res

    return output_trains

def FndPks(input_graph, time_window):
    
    locs=[]
    cnt=0 # counter
    last_ind=0
    while last_ind+time_window[1] < len(input_graph)-1:
        if cnt == 0:

            I=numpy.argmax(input_graph[last_ind:last_ind+time_window[1]-1])
            last_ind=I
        else:
            I=numpy.argmax(input_graph[last_ind+time_window[0]:last_ind+time_window[1]])
            last_ind=I+last_ind+time_window[0];
            
        cnt=cnt+1

        locs=locs+[last_ind]
       
    locs=numpy.array(locs,dtype=numpy.int32)

#    pks=input_graph[locs];
    return locs


def regPPG(PPG,time_window):
    PPG=numpy.array(PPG,dtype=numpy.float16) # single precision
    
    gPPG=numpy.gradient(PPG)
    # gradient of PPG. prominant peaks
#    wid_time=time_window[1]-time_window[0]
#    half_wid_time=numpy.round(wid_time/2)
    locs=FndPks(gPPG,time_window)
#    print(pks,locs)
    int_t=numpy.gradient(locs)
    
    #time interval between maximums
    index_PPG = numpy.zeros(PPG.shape)
#    print(len(locs))

    for jj in numpy.arange(0,len(locs)+1): # arange does NOT include end value
        
        if jj == len(locs):
            stop_n=len(PPG)-1
            rgst_interval=int_t[jj-1]
        elif jj <= len(locs):
            stop_n=locs[jj] 
            rgst_interval=int_t[jj]
        # end if
        
        if jj > 0:
            start_n=locs[jj-1]+1
        elif jj == 0:
            start_n= 0
        # end if
        tmp_idx=numpy.arange(1,stop_n-start_n+2)
#        print(jj,tmp_idx.shape, index_PPG[start_n:stop_n+1].shape, start_n,stop_n )
        index_PPG[start_n:stop_n+1]= (jj + tmp_idx*(1/rgst_interval))

#    print(index_PPG)
#    print(int_t)
    return (index_PPG, locs, int_t)

###########################################
class cinePPGVD:
    def __init__(self,PPGfilename,vd_table_filename):
#        PPGfilename='PPGData_fse_cine_20121212_1212201210_10_00_930'
        
        self.PPG=numpy.loadtxt(PPGfilename, dtype=numpy.float16)
        #PPG=PPG.reshape([14523,1],order='F')
        self.PPG=self.PPG[-250*44:]
               
        #print(PPG)
        
        
        time_window=(80,130)
        (self.index_PPG, self.locs, self.int_t)=regPPG(self.PPG,time_window)
        
        
#        print('PPG number',numpy.shape(PPG[-250*44:]))
#        vd_table_filename='variable_density_sample_with_offset.txt'
        
        self.vd_table=numpy.loadtxt(vd_table_filename, 
                                    dtype=numpy.float32)
        
        
        self.ETL=12.0;
        self.nExcite=44.0;
        self.TR=2500.0;
        self.refoci=[8.2258 ,164.0];

        self.output_trains  = seqTime(self.ETL, 
                                      self.nExcite, 
                                      self.TR, 
                                      self.refoci,10.0)


#print(vd_table)
#print(index_PPG[3000:-1])
#print(index_PPG)
#print(index_PPG[3000:])

        om=regTSE(self.output_trains,
                      self.index_PPG[-250*44:],
                      self.vd_table)
        self.om=om[:,::-1]
        self.om[:,0]=( self.om[:,0]-128)/256; # ky
        self.om[:,1]=numpy.mod(self.om[:,1],1)-0.5; # t-axies
#        self.om[:,0:1]=self.om[:,1:0]
#
#obj=cinePPGVD('PPGData_fse_cine_20121212_1212201210_10_00_930',
#              'variable_density_sample_with_offset.txt')
#
#matplotlib.pyplot.subplot(2,2,1)
#matplotlib.pyplot.plot(numpy.gradient(obj.PPG))
#matplotlib.pyplot.title('picked PEAKS of pulse oximetry')
#matplotlib.pyplot.xlabel('points(10ms)')
#matplotlib.pyplot.ylabel('gradient of wave form')
#
#matplotlib.pyplot.plot(obj.locs,numpy.gradient(obj.PPG)[obj.locs],'xr')
#
#
##matplotlib.pyplot.show()
#
#
#matplotlib.pyplot.subplot(2,2,2)
#matplotlib.pyplot.plot(obj.output_trains,'*:')
#matplotlib.pyplot.title('Timing of echoes')
#matplotlib.pyplot.xlabel('Number of echoes')
#matplotlib.pyplot.ylabel('Timing of echoes(10ms)')
#
##matplotlib.pyplot.show()
#matplotlib.pyplot.subplot(2,2,3)
#matplotlib.pyplot.plot(obj.T[:,0],obj.T[:,1],'xr')
#matplotlib.pyplot.title('Normalized sampling patter')
#matplotlib.pyplot.xlabel('T space')
#matplotlib.pyplot.ylabel('Ky space')
#matplotlib.pyplot.show()
#matplotlib.pyplot.hold





    