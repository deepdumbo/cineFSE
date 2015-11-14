# Remove redundant data from pfile
# rhrawsize=self.hdr['image']['user11']/self.hdr['rdb']['point_size']/2
# before rdb.nslices=68
# before rdb.nechoes=70
# before rdb.point_size=82
# before rdb.da_xres = 102
# before rdb.dab_start_rcv=216
# before rdb.dab_stop_rcv=216
# before rdb.off_data = 1468
# before image.user8=147728
# before image.user9=147732
# before image.user11=147740
# before image.echo_trn_len=148824
# http://docs.python.org/2/library/struct.html
# x    pad byte    no value          
# c    char    string of length 1    1     
# b    signed char    integer    1    (3)
# B    unsigned char    integer    1    (3)
# ?    _Bool    bool    1    (1)
# h    short    integer    2    (3)
# H    unsigned short    integer    2    (3)
# i    int    integer    4    (3)
# I    unsigned int    integer    4    (3)
# l    long    integer    4    (3)
# L    unsigned long    integer    4    (3)
# q    long long    integer    8    (2), (3)
# Q    unsigned long long    integer    8    (2), (3)
# f    float    float    4    (4)
# d    double    float    8    (4)
# s    char[]    string          
# p    char[]    string          
# P    void *    integer         (5), (3)

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

# import array
# values = array.array('int') # array of long integers

def grep_rhrawsize(filename,outfilename):
    
    import struct
    
    fid = openfile(filename)
    # fid.seek(1468)
    fid.seek(68)
    nslices=struct.unpack('h', fid.read(2))[0]
        
    
    print(nslices)
    
    fid.seek(70)
    nechoes=struct.unpack('h', fid.read(2))[0]    
    print(nechoes)
    
    fid.seek(1468)
    off_data=struct.unpack('i', fid.read(4))[0]    
    print(off_data)
    fid.seek(200)
    dab_start_rcv=struct.unpack('h', fid.read(2))[0]    
    print(dab_start_rcv)
    fid.seek(202)
    dab_stop_rcv=struct.unpack('h', fid.read(2))[0]
    print(dab_stop_rcv)    
    
    fid.seek(102)
    
    da_xres=struct.unpack('H', fid.read(2))[0]
    print(da_xres) 
    fid.seek(82)
    point_size=struct.unpack('h', fid.read(2))[0]
    
    print(point_size)    
    
#     147728
# before image.user9=147732
# before image.user11=147740
    fid.seek(147728)
    user8=struct.unpack('f', fid.read(4))[0]
    fid.seek(147732)
    user9=struct.unpack('f', fid.read(4))[0]
    fid.seek(147740)
    user11=struct.unpack('f', fid.read(4))[0]
    
    print(user8, user9, user11)
    fid.seek(148824)
    echo_trn_len=struct.unpack('h', fid.read(2))[0]
    fid.seek(116)
    raw_pass_size_deprecated = struct.unpack('L',fid.read(8))[0]
    print('raw_pass_size_dreprecated =',raw_pass_size_deprecated)
    

    
    ncoils = dab_stop_rcv +1 - dab_start_rcv
    byte_size = da_xres * echo_trn_len * user8 *nslices * nechoes* point_size *2

    print('byte_size',byte_size)
    print('rhrawsize',user11)
    
    new_filename =  outfilename
    fid2 = open(new_filename,'wb')
   
   # now copy data from data1 to data2
   
    fid.seek(0)
    tmp_data = fid.read(off_data)
    fid2.write(tmp_data) # write_header

    
     
    for run_coils in xrange(0,ncoils):
        fid.seek(int(off_data +run_coils*user11))
        tmp_data = fid.read(int(byte_size))
        fid2.write(tmp_data) # write_header
        
    tmp_data=struct.pack('f', byte_size)
    fid2.seek(147740)
    fid2.write(tmp_data)
    tmp_data=struct.pack('L', byte_size*ncoils)    
    fid2.seek(1660)
    fid2.write(tmp_data)   
     
    fid.close()
    fid2.close()
# http://stackoverflow.com/questions/2363483/python-slicing-a-very-large-binary-file
#     
#     fid2.close()
#     
#     fid = openfile(filename)
#     fid.seek(116)
#     raw_pass_size_deprecated = struct.unpack('L',fid.read(8))[0]
#     print('raw_pass_size_dreprecated =',raw_pass_size_deprecated)

    fid2 = openfile(new_filename)
    fid2.seek(147740)
    raw_pass_size_new = struct.unpack('f',fid2.read(4))[0]
    fid2.seek(1660)
    raw_pass_size = struct.unpack('L',fid2.read(8))[0]
    print('raw_pass_size_new =',raw_pass_size_new)
    print('raw_pass_size =',raw_pass_size)
    
    
# file_size  = values[0]
from optparse import OptionParser

#title of program
# help_text='python script to shrink big pfile'
# sign_off='author:Jyh-Miin Lin, jml86@cam.ac.uk'
# parser=argparse.ArgumentParser(description=help_text, epilog=sign_off)
parser = OptionParser()
#add user defined arguments
# parser.add_argument('--limit', '-x', dest='limit', action='store', type=float, default=1.0, help='Range of values of X', metavar='X')
# parser.add_argument('--lower', '-m', dest='lower', action='store', type=int, default=1, help='Minimum order of polynomial',metavar='M1')
# parser.add_argument('--upper', '-n', dest='upper', action='store', type=int, default=3, help='Maximum order of polynomial',metavar='M2')
# parser.add_argument('--npts', '-k', dest='npts', action='store', type=int, default=512, help='Number of points to plot',metavar='N')
# parser.add_argument('--filename', '-f', dest='filename', action='store', type=str, default='', help='Title of graph', metavar='T')
# parser.add_argument('--verbose', '-v', dest='verbosity', action='count', default=0, help='repeat this value to be even more verbose')
parser.add_option('-f', '--file', dest='filename',
                   type=str, default='', help='Title of graph', metavar='T')

parser.add_option('-o', '--outputfile', dest='outfilename',
                   type=str, default='', help='Title of graph', metavar='T')

arguments=parser.parse_args()
print(arguments[0].filename)


# grep_rhrawsize('P64512.7')
grep_rhrawsize(arguments[0].filename,arguments[0].outfilename)
    
