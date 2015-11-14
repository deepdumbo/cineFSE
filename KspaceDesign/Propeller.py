'''
Created on 2013/1/21

@author: JYH-MIIN LIN jyhmiinlin@gmail.com
'''
import numpy
def sparse2blade_conf(etl,vardense):
    '''
#     input: xres dim
    '''
    size_of_vardense = numpy.size(vardense)
    phase_encode = numpy.empty((etl,))
    for jj in xrange(0,etl):
        phase_encode[jj]  =  vardense[jj-etl/2+size_of_vardense/2]
    print(phase_encode)
    
    positive_max = numpy.max(phase_encode)
    negative_max = numpy.max(-phase_encode)
    
    if positive_max > negative_max+1:
        print('positive_max 1 ',positive_max)
        print('negative_max 1 ',negative_max)
        new_etl = 2*positive_max.astype(int)
    elif positive_max <= negative_max+1:
        print('positive_max 2 ',positive_max)
        print('negative_max 2 ',negative_max)
        new_etl = 2*(negative_max+1).astype(int) 
#         self.etl = new_etl # warning: change self.etl here
        
    print('new_etl',new_etl)
    return new_etl,phase_encode
class Propeller:
    '''
    classdocs
    '''
    def __init__(self, Nx, Ny, NumBlade, result_angle, SenseFactor, ClockWiseNess):
        '''
        Constructor
        '''
        if numpy.size(result_angle) == 1: # turn off motion correction
            corr_motion = 0
        else:
            corr_motion = 1
            if Ny <=16: 
                corr_motion = 0 


        self.Nx=Nx
        self.Ny=Ny
        self.NumBlade=NumBlade
#         self.PerAng=PerAng
        self.SenseFactor=SenseFactor
        self.ClockWiseNess=ClockWiseNess
        A=numpy.arange(0, Nx, dtype=numpy.float32)
        A=numpy.tile(A,(Ny,1)).T
        
        B=numpy.arange(0,Ny, dtype=numpy.float32)
        B=numpy.tile(B,(Nx,1))
        A=A- Nx/2.0+0.5 # 0.5 is to shift grid so centre is between
        B=B- Ny/2.0+0.5 # two adjacent k-lines
#         A=-A
        B=-B
#        A=A*2.0*numpy.pi/Nx
#        B=B*2.0*numpy.pi/Nx
        # self.blade=A+1.0j*B
        #self.blade = (self.blade -( Nx/2.0-0.5) - 1.0j*(Ny/2.0-0.5))
        #SenseFactor = numpy.round(SenseFactor) # int only
        if Ny*SenseFactor*1 > Nx:
            print('warning: SenseFactor too big!')
            return
        self.blade=(A+1.0j*B*SenseFactor)*2.0*numpy.pi/Nx
        #ClockWiseNess=(ClockWiseNess+1e-15)/numpy.abs(ClockWiseNess+1e-15)
        PerAng = result_angle#180.0/NumBlade
        d_ang=PerAng*numpy.pi/180.0 # interval of angles
        
        if corr_motion == 0:
            Ang=numpy.arange(0,NumBlade)*d_ang

#             print('accurate angl',Ang)
        elif corr_motion == 1:
            Ang = result_angle*numpy.pi/180.0
#         Ang = - Ang
#             print('angles from estimation',Ang)
        self.BLADES=numpy.empty((Nx,Ny,NumBlade),dtype=numpy.complex128)
        for pj in numpy.arange(0,NumBlade):
            
            self.BLADES[:,:,pj]=self.blade*numpy.exp(1.0j*Ang[pj])       
        om=self.BLADES#.flatten(order='F')
        om = numpy.reshape(om,(numpy.size(om),),order='F')
        
        om=numpy.tile(om,[1,1]).T
#        self.om
        self.om=numpy.concatenate((om.real,om.imag),1)
#        self.om=[numpy.real(self.om) numpy.imag(self.om)]
#         print('sampling dimension: [M(number of samples), dims(dimensionality)]= ',numpy.shape(self.om))


class Propeller2:
    '''
    classdocs
    '''
    def __init__(self, Nx, Ny, NumBlade, result_angle, SenseFactor, ClockWiseNess,vardense):
        '''
        Constructor
        '''
        if numpy.size(result_angle) == 1: # turn off motion correction
            corr_motion = 0
        else:
            corr_motion = 1
            if Ny <=16: 
                corr_motion = 0 


        self.Nx=Nx
        self.Ny=Ny
        self.NumBlade=NumBlade
#         self.PerAng=PerAng
        self.SenseFactor=SenseFactor
        self.ClockWiseNess=ClockWiseNess
        A=numpy.arange(0, Nx, dtype=numpy.float32)
        A=numpy.tile(A,(Ny,1)).T
        new_etl, phase_encode = sparse2blade_conf(Ny,vardense)
        print('phase_encode',phase_encode)
#         B=numpy.arange(0,Ny, dtype=numpy.float32)
        B = phase_encode
        
        B=numpy.tile(B,(Nx,1))
        A=A- Nx/2.0+0.5 # 0.5 is to shift grid so centre is between
        B=B + 0.5 # two adjacent k-lines
#         A=-A
        B=-B
#        A=A*2.0*numpy.pi/Nx
#        B=B*2.0*numpy.pi/Nx
        # self.blade=A+1.0j*B
        #self.blade = (self.blade -( Nx/2.0-0.5) - 1.0j*(Ny/2.0-0.5))
        #SenseFactor = numpy.round(SenseFactor) # int only
        if Ny*SenseFactor*1 > Nx:
            print('warning: SenseFactor too big!')
            return
        self.blade=(A+1.0j*B*SenseFactor)*2.0*numpy.pi/Nx
        #ClockWiseNess=(ClockWiseNess+1e-15)/numpy.abs(ClockWiseNess+1e-15)
        PerAng = result_angle#180.0/NumBlade
        d_ang=PerAng*numpy.pi/180.0 # interval of angles
        
        if corr_motion == 0:
            Ang=numpy.arange(0,NumBlade)*d_ang

            print('accurate angl',Ang)
        elif corr_motion == 1:
            Ang = result_angle*numpy.pi/180.0
#         Ang = - Ang
            print('angles from estimation',Ang)
        self.BLADES=numpy.empty((Nx,Ny,NumBlade),dtype=numpy.complex128)
        for pj in numpy.arange(0,NumBlade):
            
            self.BLADES[:,:,pj]=self.blade*numpy.exp(1.0j*Ang[pj])       
        om=self.BLADES#.flatten(order='F')
        om = numpy.reshape(om,(numpy.size(om),),order='F')
        
        om=numpy.tile(om,[1,1]).T
#        self.om
        self.om=numpy.concatenate((om.real,om.imag),1)
#        self.om=[numpy.real(self.om) numpy.imag(self.om)]
#         print('sampling dimension: [M(number of samples), dims(dimensionality)]= ',numpy.shape(self.om))

