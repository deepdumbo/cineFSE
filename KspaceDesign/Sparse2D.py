'''
Created on 2013/1/26

@author: sram
'''
import numpy
def getGrid(Obj,Nd):
    C = numpy.array(Obj.om)
#    Obj.omgrid=Obj.om
    for dimid in numpy.arange(0,numpy.size(Nd)):
        C[:,dimid]=numpy.round(C[:,dimid]*Nd[dimid]/2.0/numpy.pi)*1.0
        C[:,dimid]=C[:,dimid]*2.0*numpy.pi/Nd[dimid]
    return C


class Sparse2D:
    '''
    classdocs
    '''
    def __init__(self,om,Nd):
        '''
        Constructor
        '''
        #=======================================================================
        # om_grids randoms pattern between -0.5 to 0.5
        # otherwise, you must shift om_grids to other 
        #=======================================================================
#        C=om*2.0*numpy.pi
#        for dimid in numpy.arange(0,numpy.size(Nd)):
#            self.om[:,dimid]=2.0*numpy.pi*(om_grids[:,dimid]+0.5)*1.0
        self.om=om*2.0*numpy.pi
#        self.C=C
        self.omgrid=getGrid(self,Nd)

            
class Sparse2DStack(Sparse2D):
    '''
    classdocs
    '''
    def __init__(self,om,Nd):
        '''
        Constructor
        '''
        Sparse2D.__init__(self,om,Nd[1:3])
        
        M=self.om.shape[0]
        
        x_vec=((numpy.arange(0,Nd[0])+0.5)*1.0/Nd[0] -0.5 )*2*numpy.pi
        
        self.om = numpy.reshape(self.om,[1, 2*M],order='F') # [1, 2*M]
        self.om = numpy.tile(self.om,[Nd[0],1]).flatten(order='F')
        self.om = numpy.reshape(self.om, [Nd[0]*M, 2], order='F')

        x_vec = numpy.tile(x_vec.T,[1,M]).flatten(order='F') # [Nd[0] 2*M]
#        self.om = numpy.reshape(self.om,[M*Nd[0],3 ],order='F')
        x_vec=numpy.reshape(x_vec,[Nd[0]*M,1],order='F')
        

        self.om = numpy.concatenate((x_vec,self.om),axis=1)
#        print('self.om.shape',self.om.shape)
#        print(Nd[0])